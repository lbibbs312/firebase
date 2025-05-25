import abc
import enum
import json # For JSONDecodeError
import logging
import math
import os
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator, # For MultiProvider, not strictly this file
    Awaitable,
    Callable,
    ClassVar,
    Generic,
    Iterable, # For function_specs_from_commands
    Mapping,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
    cast,
    get_args,
)

import sentry_sdk # Keep conditional usage
import tenacity
import tiktoken
from openai import AsyncOpenAI # Moved to where it's used
from openai._exceptions import APIConnectionError, APIStatusError, RateLimitError
from openai.types import CreateEmbeddingResponse, EmbeddingCreateParams
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam, # Used for casting
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    CompletionCreateParams,
)
from openai.types.shared_params import FunctionDefinition

from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError # Added SecretStr

# Assuming forge.json.parsing.json_loads is available
# from forge.json.parsing import json_loads
# Stub for forge.json.parsing.json_loads if not available
def json_loads(s: str) -> Any:
    return json.loads(s)

# Assuming forge.models.config.UserConfigurable and SystemConfiguration are available
# from forge.models.config import UserConfigurable, SystemConfiguration
# Stubs if not available:
class SystemConfiguration(BaseModel):
    pass

def UserConfigurable(default: Any = None, from_env: Any = None): # type: ignore
    from pydantic import Field
    # This is a simplified stub. Real implementation would handle env loading.
    return Field(default=default)


# --- Start: Content from schema.py (or relevant parts) ---

_T = TypeVar("_T") # Generic type
_P = ParamSpec("_P") # Parameter specification for Callables

# ModelName type variable for generic providers
_ModelName = TypeVar("_ModelName", bound=str)
# ModelProviderSettings type variable for generic providers
_ModelProviderSettings = TypeVar("_ModelProviderSettings", bound="ModelProviderSettings")


class ModelProviderName(str, enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic" # For MultiProvider context
    GROQ = "groq"           # For MultiProvider context
    LLAMAFILE = "llamafile" # For MultiProvider context
    GEMINI = "gemini"       # Added for Gemini


class ModelProviderService(str, enum.Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    # Add other services as needed


class ModelTokenizer(Generic[_T]): # From user's OpenAIProvider
    def encode(self, text: str) -> list[_T]: ...
    def decode(self, tokens: list[_T]) -> str: ...


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    name: Optional[str] = None  # For tool role if function name, or assistant name
    tool_calls: Optional[list[Any]] = None # Placeholder for AssistantToolCall or similar structure
    tool_call_id: Optional[str] = None # For tool role

    model_config = ConfigDict(extra="allow") # Allow additional fields if needed


class AssistantFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class AssistantToolCall(BaseModel):
    id: str
    type: str = "function"  # Usually "function"
    function: AssistantFunctionCall


class AssistantChatMessage(ChatMessage):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[AssistantToolCall]] = None


class ModelProviderCredentials(SystemConfiguration):
    def unmasked(self) -> dict: # From user's schema.py
        unmasked_fields = {}
        for field_name, _ in self.model_fields.items():
            value = getattr(self, field_name)
            if isinstance(value, SecretStr):
                unmasked_fields[field_name] = value.get_secret_value()
            else:
                unmasked_fields[field_name] = value
        return unmasked_fields

    model_config = ConfigDict( # From user's schema.py
        json_encoders={
            SecretStr: lambda v: v.get_secret_value() if v else None,
            # SecretBytes: lambda v: v.get_secret_value() if v else None, # Not used here
            # Secret: lambda v: v.get_secret_value() if v else None, # Not used here
        }
    )


class ModelProviderConfiguration(SystemConfiguration):
    raw: ClassVar[str] = os.getenv("EXTRA_REQUEST_HEADERS", "{}")
    parsed: ClassVar[dict[str, str]] = json.loads(raw)

    extra_request_headers: dict[str, str] = UserConfigurable(
        default=parsed
    )


class ModelUsage(BaseModel): # Example, can be more detailed
    input_tokens: int = 0
    output_tokens: int = 0

class ModelProviderBudget(SystemConfiguration, Generic[_T]): # From user's schema.py
    total_budget: float = UserConfigurable(default=math.inf)
    total_cost: float = 0.0
    remaining_budget: float = UserConfigurable(default=math.inf) # Should be calculated
    # usage: _T # This was generic, let's make it specific or a dict for simplicity
    usage_details: dict[str, ModelUsage] = UserConfigurable(default_factory=dict) # model_name -> ModelUsage

    def model_post_init(self, __context: Any) -> None:
        self.remaining_budget = self.total_budget - self.total_cost

    def update_usage_and_cost(
        self,
        model_info: "ChatModelInfo | EmbeddingModelInfo",
        input_tokens_used: int,
        output_tokens_used: int = 0, # Embeddings have 0 output tokens
    ) -> float:
        """Update the usage and cost of the provider."""
        cost = 0.0
        if isinstance(model_info, ChatModelInfo):
            cost = (
                model_info.prompt_token_cost * input_tokens_used
            ) + (model_info.completion_token_cost * output_tokens_used)
        elif isinstance(model_info, EmbeddingModelInfo):
            cost = model_info.token_cost * input_tokens_used
        
        self.total_cost += cost
        self.remaining_budget = self.total_budget - self.total_cost

        # Update usage_details
        model_name_str = str(model_info.name)
        if model_name_str not in self.usage_details:
            self.usage_details[model_name_str] = ModelUsage()
        self.usage_details[model_name_str].input_tokens += input_tokens_used
        self.usage_details[model_name_str].output_tokens += output_tokens_used
        
        if self.total_cost > self.total_budget:
            logging.warning(
                f"Budget exceeded! Current cost: ${self.total_cost:.4f}, "
                f"Budget: ${self.total_budget:.4f}"
            )
        return cost


class ModelProviderSettings(SystemConfiguration):
    name: ModelProviderName = ModelProviderName.OPENAI # Default, will be overridden by specific provider settings
    description: str = "LLM Provider Settings"
    credentials: Optional[ModelProviderCredentials] = None
    budget: Optional[ModelProviderBudget] = None
    configuration: ModelProviderConfiguration = ModelProviderConfiguration()


class ModelInfo(BaseModel, Generic[_ModelName]):
    name: _ModelName
    provider_name: ModelProviderName
    service: ModelProviderService
    max_tokens: int  # Max total tokens (input + output) or context window size

    model_config = ConfigDict(frozen=True) # Make them hashable for dict keys if needed


class ChatModelInfo(ModelInfo[_ModelName]):
    service: ModelProviderService = ModelProviderService.CHAT
    prompt_token_cost: float  # Cost per 1k prompt tokens
    completion_token_cost: float  # Cost per 1k completion tokens
    has_function_call_api: bool = False


class EmbeddingModelInfo(ModelInfo[_ModelName]):
    service: ModelProviderService = ModelProviderService.EMBEDDING
    token_cost: float  # Cost per 1k tokens
    embedding_dimensions: int


Embedding = list[float] # From user's schema.py


class ChatModelResponse(BaseModel, Generic[_T]):
    response: AssistantChatMessage
    parsed_result: Optional[_T] = None
    llm_info: ChatModelInfo # Information about the LLM used
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EmbeddingModelResponse(BaseModel):
    embedding: Embedding
    llm_info: EmbeddingModelInfo
    prompt_tokens_used: int = 0
    # completion_tokens_used is 0 for embeddings

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CompletionParameter(BaseModel):
    type: str
    description: Optional[str] = None
    required: bool = False
    enum: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class CompletionModelFunction(BaseModel):
    name: str
    description: str
    parameters: dict[str, CompletionParameter]

    def validate_call(
        self, call_info: AssistantFunctionCall
    ) -> tuple[bool, list[Any]]: # Simplified error type
        # Placeholder for actual JSON schema validation logic if needed
        # For now, just check if all required parameters are present
        is_valid = True
        errors = []
        for param_name, param_spec in self.parameters.items():
            if param_spec.required and param_name not in call_info.arguments:
                is_valid = False
                errors.append(
                    ValidationError( # This is not how Pydantic ValidationError is typically used
                        f"Missing required parameter: {param_name}",
                        loc=("arguments", param_name) # type: ignore
                    )
                )
        return is_valid, errors


class BaseModelProvider(abc.ABC, Generic[_ModelName, _ModelProviderSettings]):
    MODELS: ClassVar[Mapping[_ModelName, ChatModelInfo[_ModelName] | EmbeddingModelInfo[_ModelName]]]

    default_settings: ClassVar[_ModelProviderSettings]

    _settings: _ModelProviderSettings
    _credentials: Optional[ModelProviderCredentials] # More specific than Any
    _configuration: ModelProviderConfiguration
    _budget: Optional[ModelProviderBudget]
    _logger: logging.Logger
    _client: Any # The API client (e.g., AsyncOpenAI)

    def __init__(
        self,
        settings: Optional[_ModelProviderSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._settings = settings or self.default_settings.model_copy(deep=True)
        self._credentials = self._settings.credentials
        self._configuration = self._settings.configuration
        self._budget = self._settings.budget
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        # _client initialization is deferred to concrete base or provider

    @abc.abstractmethod
    async def get_available_models(
        self,
    ) -> Sequence[ChatModelInfo[_ModelName] | EmbeddingModelInfo[_ModelName]]: ...

    @abc.abstractmethod
    def get_token_limit(self, model_name: _ModelName) -> int: ...

    @abc.abstractmethod
    def count_tokens(self, text: str, model_name: _ModelName) -> int: ...

    @abc.abstractmethod
    def get_tokenizer(self, model_name: _ModelName) -> ModelTokenizer[Any]: ...


class BaseChatModelProvider(BaseModelProvider[_ModelName, _ModelProviderSettings]):
    CHAT_MODELS: ClassVar[Mapping[_ModelName, ChatModelInfo[_ModelName]]]

    @abc.abstractmethod
    async def get_available_chat_models(self) -> Sequence[ChatModelInfo[_ModelName]]: ...

    @abc.abstractmethod
    def count_message_tokens(
        self, messages: ChatMessage | list[ChatMessage], model_name: _ModelName
    ) -> int: ...

    @abc.abstractmethod
    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: _ModelName,
        completion_parser: Callable[[AssistantChatMessage], _T],
        functions: Optional[list[CompletionModelFunction]],
        max_output_tokens: Optional[int],
        prefill_response: str, # Note: prefill_response is custom
        **kwargs: Any,
    ) -> ChatModelResponse[_T]: ...


class BaseEmbeddingModelProvider(BaseModelProvider[_ModelName, _ModelProviderSettings]):
    EMBEDDING_MODELS: ClassVar[Mapping[_ModelName, EmbeddingModelInfo[_ModelName]]]

    @abc.abstractmethod
    async def get_available_embedding_models(self) -> Sequence[EmbeddingModelInfo[_ModelName]]: ...

    @abc.abstractmethod
    async def create_embedding(
        self,
        text: str,
        model_name: _ModelName,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs: Any,
    ) -> EmbeddingModelResponse: ...

# --- End: Content from schema.py ---


# --- Start: Content from utils.py ---
if TYPE_CHECKING:
    # from forge.command.command import Command # Not directly used in this file
    pass

class InvalidFunctionCallError(Exception): # From user's utils.py
    def __init__(self, name: str, arguments: dict[str, Any], message: str):
        self.message = message
        self.name = name
        self.arguments = arguments
        super().__init__(message)

    def __str__(self) -> str:
        return f"Invalid function call for {self.name}: {self.message}"


def validate_tool_calls(
    tool_calls: list[AssistantToolCall], functions: list[CompletionModelFunction]
) -> list[InvalidFunctionCallError]: # From user's utils.py
    errors: list[InvalidFunctionCallError] = []
    for tool_call in tool_calls:
        function_call = tool_call.function

        if function := next(
            (f for f in functions if f.name == function_call.name),
            None,
        ):
            is_valid, validation_errors = function.validate_call(function_call)
            if not is_valid:
                fmt_errors = []
                for f_err in validation_errors: # type: ignore
                    path_str = '.'.join(str(p) for p in f_err.loc) if hasattr(f_err, 'loc') else ""
                    msg_str = f_err.msg if hasattr(f_err, 'msg') else str(f_err)
                    fmt_errors.append(f"{path_str}: {msg_str}" if path_str else msg_str)

                errors.append(
                    InvalidFunctionCallError(
                        name=function_call.name,
                        arguments=function_call.arguments,
                        message=(
                            "The set of arguments supplied is invalid:\n"
                            + "\n".join(fmt_errors)
                        ),
                    )
                )
        else:
            errors.append(
                InvalidFunctionCallError(
                    name=function_call.name,
                    arguments=function_call.arguments,
                    message=f"Unknown function {function_call.name}",
                )
            )
    return errors


def function_specs_from_commands( # From user's utils.py
    commands: Iterable[Any], # "Command" type stubbed with Any
) -> list[CompletionModelFunction]:
    """Get LLM-consumable function specs for the agent's available commands."""
    return [
        CompletionModelFunction(
            name=command.names[0],
            description=command.description,
            parameters={param.name: param.spec for param in command.parameters},
        )
        for command in commands
    ]
# --- End: Content from utils.py ---


class _BaseOpenAIProvider(BaseModelProvider[OpenAIModelName, "OpenAISettings"]): # Forward reference OpenAISettings
    """Base class for LLM providers with OpenAI-like APIs (OpenAI, Gemini via OpenAI SDK)"""

    # MODELS ClassVar is defined in the concrete OpenAIProvider

    def __init__(
        self,
        settings: Optional["OpenAISettings"] = None, # Forward reference
        logger: Optional[logging.Logger] = None,
    ):
        if not getattr(self, "MODELS", None): # MODELS is set on OpenAIProvider
            raise ValueError(f"{self.__class__.__name__}.MODELS is not set")

        # settings is guaranteed to be populated by OpenAIProvider before calling super()
        super().__init__(settings=settings, logger=logger) # type: ignore

        if not getattr(self, "_client", None):
            client_init_kwargs: dict[str, Any] = {}
            provider_name_setting = self._settings.name # type: ignore

            if provider_name_setting == ModelProviderName.GEMINI:
                api_key = os.getenv("GEMINI_API_KEY")
                if self._credentials and hasattr(self._credentials, 'api_key') and self._credentials.api_key: # type: ignore
                    api_key_val = cast(SecretStr, self._credentials.api_key).get_secret_value() # type: ignore
                    if api_key_val:
                        api_key = api_key_val
                
                if not api_key:
                    raise ValueError(
                        "GEMINI_API_KEY not found in environment or credentials for Gemini provider."
                    )
                
                client_init_kwargs = {
                    "api_key": api_key,
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
                }
                self._logger.info("Initializing Gemini client via OpenAI SDK compatibility layer.")
            
            elif provider_name_setting == ModelProviderName.OPENAI or \
                 (self._credentials and hasattr(self._credentials, 'api_type') and self._credentials.api_type and # type: ignore
                  cast(SecretStr, self._credentials.api_type).get_secret_value() == "azure"): # type: ignore
                if not self._credentials:
                    raise ValueError(f"Credentials not configured for {provider_name_setting or 'OpenAI/Azure'} provider.")
                
                # Assuming self._credentials is OpenAICredentials and has get_api_access_kwargs
                client_init_kwargs = cast("OpenAICredentials", self._credentials).get_api_access_kwargs()
                self._logger.info(f"Initializing {provider_name_setting or 'OpenAI/Azure'} client.")
            else:
                raise ValueError(f"Unsupported provider name '{provider_name_setting}' or credentials configuration error.")

            self._client = AsyncOpenAI(**client_init_kwargs)

    async def get_available_models(
        self,
    ) -> Sequence[ChatModelInfo[OpenAIModelName] | EmbeddingModelInfo[OpenAIModelName]]:
        try:
            response_models = (await self._client.models.list()).data
            available_and_configured = [
                self.MODELS[cast(OpenAIModelName, m.id)] 
                for m in response_models 
                if m.id in self.MODELS and self.MODELS[cast(OpenAIModelName, m.id)].provider_name == self._settings.name # type: ignore
            ]
            if not available_and_configured and self._settings.name == ModelProviderName.GEMINI: # type: ignore
                self._logger.warning(
                    "Could not list models from Gemini endpoint matching configuration, or API error. "
                    "Falling back to configured Gemini models."
                )
                return [model_info for model_id, model_info in self.MODELS.items() # type: ignore
                        if model_info.provider_name == ModelProviderName.GEMINI]
            return available_and_configured # type: ignore
        except Exception as e:
            self._logger.error(f"Failed to list available models from API for {self._settings.name}: {e}. " # type: ignore
                                "Falling back to statically configured models for this provider.")
            current_provider_name = self._settings.name # type: ignore
            return [model_info for model_id, model_info in self.MODELS.items() # type: ignore
                    if model_info.provider_name == current_provider_name]

    def get_token_limit(self, model_name: OpenAIModelName) -> int:
        return self.MODELS[model_name].max_tokens # type: ignore

    def count_tokens(self, text: str, model_name: OpenAIModelName) -> int:
        try:
            tokenizer = self.get_tokenizer(model_name)
            encoded = tokenizer.encode(text)
            return len(encoded)
        except (NotImplementedError, AttributeError, Exception) as e:
            self._logger.debug(
                f"Tokenizer for {model_name} not available or failed (Error: {e}). "
                "Using character-based estimation (len(text) // 4)."
            )
            return len(text) // 4

    def get_tokenizer(self, model_name: OpenAIModelName) -> ModelTokenizer[Any]:
         raise NotImplementedError("get_tokenizer must be implemented by concrete provider.")

    def _retry_api_request(self, func: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]: # Ensure Awaitable for async
        return tenacity.retry(
            retry=(tenacity.retry_if_exception_type(APIConnectionError)
                   | tenacity.retry_if_exception(
                        lambda e: isinstance(e, APIStatusError) and e.status_code >= 500
                    )
                   | tenacity.retry_if_exception_type(RateLimitError) # Added RateLimitError
            ),
            wait=tenacity.wait_exponential(),
            stop=tenacity.stop_after_attempt(self._configuration.retries_per_request),
            after=tenacity.after_log(self._logger, logging.DEBUG),
        )(func)

    def __repr__(self):
        return f"{self.__class__.__name__}(provider='{self._settings.name if self._settings else 'Unknown'}')" # type: ignore


class BaseOpenAIChatProvider(
    _BaseOpenAIProvider, # Defaults to OpenAIModelName, OpenAISettings
    BaseChatModelProvider[OpenAIModelName, "OpenAISettings"],
):
    # CHAT_MODELS ClassVar is defined in the concrete OpenAIProvider

    def __init__(
        self,
        settings: Optional["OpenAISettings"] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not getattr(self, "CHAT_MODELS", None):
            raise ValueError(f"{self.__class__.__name__}.CHAT_MODELS is not set")
        # MODELS is fully set by OpenAIProvider. This update is minor.
        # if not getattr(self, "MODELS", None): self.MODELS = {} -> This will be an issue if MODELS is not on self
        # For multiple inheritance, OpenAIProvider should handle setting self.MODELS before super() calls.
        # However, _BaseOpenAIProvider is called first by OpenAIProvider's MRO for __init__.
        # This logic is tricky with multiple inheritance. Let's rely on OpenAIProvider setting ClassVars.
        super().__init__(settings=settings, logger=logger) # type: ignore


    async def get_available_chat_models(self) -> Sequence[ChatModelInfo[OpenAIModelName]]:
        all_available_models = await self.get_available_models()
        return [
            model for model in all_available_models
            if isinstance(model, ChatModelInfo) # service check removed, ChatModelInfo implies service=CHAT
        ]

    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: OpenAIModelName,
    ) -> int:
        if isinstance(messages, ChatMessage):
            messages = [messages]
        
        # OpenAI's recommended way to count tokens for chat messages is more complex.
        # This is a simplified version. For tiktoken, it can be more accurate.
        # For Gemini (char count), it remains an estimate.
        model_info = self.MODELS[model_name] # type: ignore
        if model_info.provider_name == ModelProviderName.OPENAI:
            # A more accurate way for OpenAI, though tiktoken's direct support is better
            # This is a placeholder for a more complex message tokenization scheme
            # See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
            num_tokens = 0
            tokenizer = self.get_tokenizer(model_name) # type: tiktoken.Encoding
            for message in messages:
                num_tokens += 4  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
                for key, value in message.model_dump(exclude_none=True).items():
                    if value is None: continue # Should be excluded by exclude_none
                    if key == "tool_calls": # Special handling for tool_calls array if complex
                        # This part needs very careful implementation based on actual format
                        # For simplicity, we'll count the string representation
                        num_tokens += len(tokenizer.encode(json.dumps(value)))
                    elif isinstance(value, str):
                        num_tokens += len(tokenizer.encode(value))
                    if key == "name":  # If there's a name, the role is omitted
                        num_tokens -= 1  # Role is always required and always 1 token
            num_tokens += 2  # Every reply is primed with <|start|>assistant
            return num_tokens
        else: # Gemini or other non-OpenAI using the char/4 estimate
            text_content = "\n\n".join(
                f"{m.role.upper()}: {m.content or ''}" +
                (f" TOOL_CALLS: {m.tool_calls}" if hasattr(m, 'tool_calls') and m.tool_calls else "") +
                (f" NAME: {m.name}" if hasattr(m, 'name') and m.name else "")
                for m in messages
            )
            return self.count_tokens(text_content, model_name)


    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: OpenAIModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: cast(_T, None),
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        prefill_response: str = "", # Custom param, not used by OpenAI/Gemini SDK
        **kwargs, # For passthrough like 'reasoning_effort'
    ) -> ChatModelResponse[_T]:
        (
            api_messages, # Renamed from openai_messages for clarity
            completion_kwargs,
            parse_kwargs, # Currently empty, but kept for structure
        ) = self._get_chat_completion_args(
            prompt_messages=model_prompt,
            model=model_name,
            functions=functions,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

        total_cost = 0.0
        attempts = 0
        current_api_messages = list(api_messages) # Work with a copy for retry modifications

        while True:
            # Ensure 'messages' is part of completion_kwargs for each attempt
            # The OpenAI SDK expects 'messages' at the top level of create()
            final_call_kwargs = {**completion_kwargs, "messages": current_api_messages} # type: ignore

            _response_obj, _cost, t_input, t_output = await self._create_chat_completion( # type: ignore
                model=model_name, completion_kwargs=final_call_kwargs # type: ignore
            )
            total_cost += _cost
            attempts += 1
            parse_errors: list[Exception] = []
            
            if not _response_obj.choices:
                # This can happen with content filtering or other API issues
                self._logger.error(f"API response for {model_name} had no choices. Response: {_response_obj.model_dump_json(indent=2)}")
                # Depending on finish_reason, may need specific error
                finish_reason = _response_obj.choices[0].finish_reason if _response_obj.choices and _response_obj.choices[0].finish_reason else "unknown_error_no_choices"
                raise APIStatusError(message=f"No choices returned by API. Finish reason: {finish_reason}", response=None, body=None) # type: ignore
                
            _assistant_msg_from_api = _response_obj.choices[0].message

            tool_calls, _errors = self._parse_assistant_tool_calls(
                _assistant_msg_from_api, **parse_kwargs
            )
            parse_errors.extend(_errors)

            if not parse_errors and tool_calls and functions:
                parse_errors.extend(validate_tool_calls(tool_calls, functions))

            # Create our internal AssistantChatMessage object
            assistant_chat_message_obj = AssistantChatMessage(
                content=_assistant_msg_from_api.content or "",
                tool_calls=tool_calls or None,
                # Potentially copy 'name' if assistant can have a name, role is fixed
            )
            parsed_result: _T = cast(_T, None)
            if not parse_errors:
                try:
                    parsed_result = completion_parser(assistant_chat_message_obj)
                except Exception as e:
                    self._logger.error(f"Completion parser failed: {e}", exc_info=True)
                    parse_errors.append(e)
            
            if not parse_errors:
                if attempts > 1:
                    self._logger.debug(f"Total cost for {attempts} attempts: ${round(total_cost, 5)}")
                return ChatModelResponse(
                    response=assistant_chat_message_obj,
                    parsed_result=parsed_result,
                    llm_info=self.CHAT_MODELS[model_name], # type: ignore
                    prompt_tokens_used=t_input,
                    completion_tokens_used=t_output,
                )
            else: # Parsing failed
                self._logger.debug(f"Parsing failed on response: '''{_assistant_msg_from_api}'''")
                parse_errors_fmt = "\n\n".join(f"{e.__class__.__name__}: {e}" for e in parse_errors)
                self._logger.warning(f"Parsing attempt #{attempts} for {model_name} failed: {parse_errors_fmt}")
                
                for e in parse_errors:
                    if sentry_sdk and sentry_sdk.Hub.current.client:
                         sentry_sdk.capture_exception(
                              error=e,
                              extras={"assistant_msg": _assistant_msg_from_api.model_dump(exclude_none=True), "i_attempt": attempts},
                         )
                if attempts < self._configuration.fix_failed_parse_tries:
                    # Add API's response and error message to the message history for retry
                    current_api_messages.append(cast(ChatCompletionMessageParam, _assistant_msg_from_api.model_dump(exclude_none=True)))
                    current_api_messages.append(
                        cast(ChatCompletionMessageParam, {
                            "role": "system", # Or "user" if system prompts cause issues with some models
                            "content": f"ERROR PARSING YOUR PREVIOUS RESPONSE:\n\n{parse_errors_fmt}\n\nPlease try again, ensuring your output is valid.",
                        })
                    )
                    # Optionally reduce max_tokens for the retry if it seems like a truncation issue
                    if "max_tokens" in completion_kwargs and completion_kwargs["max_tokens"]: # type: ignore
                        completion_kwargs["max_tokens"] = int(completion_kwargs["max_tokens"] * 0.9) # type: ignore
                    continue
                else:
                    self._logger.error(f"Exhausted parse-fixing retries ({self._configuration.fix_failed_parse_tries}) for {model_name}.")
                    raise parse_errors[0] # Raise the first encountered parse error

    def _get_chat_completion_args(
        self,
        prompt_messages: list[ChatMessage],
        model: OpenAIModelName,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs, # Catches 'reasoning_effort', etc.
    ) -> tuple[
        list[ChatCompletionMessageParam], CompletionCreateParams, dict[str, Any]
    ]:
        # kwargs here are CompletionCreateParams plus potential extras
        final_kwargs: dict[str, Any] = {**kwargs} # Start with passthrough kwargs

        if max_output_tokens:
            final_kwargs["max_tokens"] = max_output_tokens
        
        if functions:
            # Use self.format_function_def_for_openai (defined in OpenAIProvider)
            final_kwargs["tools"] = [
                {"type": "function", "function": self.format_function_def_for_openai(f)} # type: ignore
                for f in functions
            ]
            if len(functions) == 1:
                 final_kwargs["tool_choice"] = {
                     "type": "function",
                     "function": {"name": functions[0].name},
                 }
            # else: tool_choice="auto" is default if tools are present

        if extra_h := self._configuration.extra_request_headers:
            final_kwargs["extra_headers"] = {**final_kwargs.get("extra_headers", {}), **extra_h}

        # Prepare messages for the API
        api_messages: list[ChatCompletionMessageParam] = [
            message.model_dump( # type: ignore
                include={"role", "content", "tool_calls", "tool_call_id", "name"},
                exclude_none=True,
            )
            for message in prompt_messages
        ]
        
        # If 'messages' was passed in kwargs (e.g. from a retry loop), merge carefully
        # However, the current retry loop in create_chat_completion handles message append internally.
        # So, 'messages' in kwargs here would be from the initial call if user passed it, which is unusual.
        if "messages" in final_kwargs:
             passed_messages = final_kwargs.pop("messages") # Remove to avoid conflict
             if isinstance(passed_messages, list):
                  api_messages.extend(cast(list[ChatCompletionMessageParam], passed_messages))
             else:
                  self._logger.warning("kwargs['messages'] was not a list, ignoring.")

        # The OpenAI client's create method expects 'model' and 'messages' as top-level params.
        # Other standard OpenAI params (max_tokens, tools, etc.) are also top-level.
        # Custom params like 'reasoning_effort' will be passed through if base_url is custom.
        # So, final_kwargs should contain these. 'model' is added by _create_chat_completion.
        # 'messages' is added by create_chat_completion loop.
        return api_messages, cast(CompletionCreateParams, final_kwargs), {}


    @_BaseOpenAIProvider._retry_api_request # type: ignore # Apply retry logic from the base
    async def _create_chat_completion(
        self,
        model: OpenAIModelName,
        completion_kwargs: CompletionCreateParams, # May contain extra keys like reasoning_effort
    ) -> tuple[ChatCompletion, float, int, int]:
        
        final_call_kwargs = {**completion_kwargs} # type: ignore
        final_call_kwargs["model"] = model # Set model for the call
        final_call_kwargs["stream"] = False # Ensure no streaming

        response_obj = await self._client.chat.completions.create(**final_call_kwargs) # type: ignore
        
        prompt_tokens_used = 0
        completion_tokens_used = 0
        if hasattr(response_obj, "usage") and response_obj.usage is not None:
            prompt_tokens_used = response_obj.usage.prompt_tokens or 0
            completion_tokens_used = response_obj.usage.completion_tokens or 0
        
        cost = 0.0
        model_info = self.CHAT_MODELS.get(model) # type: ignore

        if self._budget and model_info:
            cost = self._budget.update_usage_and_cost(
                    model_info=model_info,
                    input_tokens_used=prompt_tokens_used,
                    output_tokens_used=completion_tokens_used,
            )
        elif self._budget and not model_info:
            self._logger.warning(f"Model '{model}' not found in CHAT_MODELS. Cannot calculate cost.")
        
        self._logger.debug(
            f"{model.value} completion usage: {prompt_tokens_used} input, "
            f"{completion_tokens_used} output - ${round(cost, 5)}"
        )
        if self._budget:
            self._logger.info(
                f"💰 Cost this call: ${cost:.5f} | "
                f"Total cost: ${self._budget.total_cost:.5f} | "
                f"Budget left: ${self._budget.remaining_budget:.5f}"
            )
        else:
             self._logger.info(f"💰 Cost this call: ${cost:.5f} (Budget tracking disabled)")

        return response_obj, cost, prompt_tokens_used, completion_tokens_used

    def _parse_assistant_tool_calls(
        self, assistant_message_from_api: ChatCompletionMessage, **kwargs
    ) -> tuple[list[AssistantToolCall], list[Exception]]:
        parsed_tool_calls: list[AssistantToolCall] = []
        parse_errors: list[Exception] = []

        if assistant_message_from_api.tool_calls:
            for tc_from_api in assistant_message_from_api.tool_calls:
                if not (tc_from_api.id and tc_from_api.type == "function" and tc_from_api.function and tc_from_api.function.name):
                     parse_errors.append(ValueError(f"Malformed tool call object received: {tc_from_api}"))
                     continue
                try:
                    arguments_str = tc_from_api.function.arguments
                    # json_loads should handle empty string or valid JSON. OpenAI SDK ensures it's a string.
                    parsed_args = json_loads(arguments_str) if arguments_str else {}

                    parsed_tool_calls.append(
                        AssistantToolCall(
                            id=tc_from_api.id,
                            type=tc_from_api.type, # Should be "function"
                            function=AssistantFunctionCall(
                                name=tc_from_api.function.name,
                                arguments=parsed_args,
                            ),
                        )
                    )
                except (json.JSONDecodeError, TypeError) as e: # json.JSONDecodeError from standard json
                    err_msg = f"Decoding arguments for tool '{tc_from_api.function.name}' failed: {e}"
                    self._logger.warning(f"{err_msg} Raw arguments: '{tc_from_api.function.arguments}'")
                    parse_errors.append(ValueError(err_msg).with_traceback(e.__traceback__))
                except Exception as e:
                     err_msg = f"Unexpected error parsing tool '{tc_from_api.function.name}': {e}"
                     self._logger.error(f"{err_msg} Raw tool call: {tc_from_api}", exc_info=True)
                     parse_errors.append(type(e)(err_msg, *e.args[1:]).with_traceback(e.__traceback__))
        
        return parsed_tool_calls, parse_errors


class BaseOpenAIEmbeddingProvider(
    _BaseOpenAIProvider, # Defaults to OpenAIModelName, OpenAISettings
    BaseEmbeddingModelProvider[OpenAIModelName, "OpenAISettings"],
):
    # EMBEDDING_MODELS ClassVar is defined in the concrete OpenAIProvider

    def __init__(
        self,
        settings: Optional["OpenAISettings"] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not getattr(self, "EMBEDDING_MODELS", None):
            raise ValueError(f"{self.__class__.__name__}.EMBEDDING_MODELS is not set")
        super().__init__(settings=settings, logger=logger) # type: ignore

    async def get_available_embedding_models(self) -> Sequence[EmbeddingModelInfo[OpenAIModelName]]:
        all_available_models = await self.get_available_models()
        return [model for model in all_available_models if isinstance(model, EmbeddingModelInfo)]

    async def create_embedding(
        self,
        text: str,
        model_name: OpenAIModelName,
        embedding_parser: Callable[[Embedding], Embedding] = lambda x: x,
        **kwargs,
    ) -> EmbeddingModelResponse:
        embedding_creation_kwargs = self._get_embedding_kwargs(input=text, model=model_name, **kwargs)

        response_obj = await self._create_embedding_call(embedding_creation_kwargs) # Renamed to avoid recursion if overridden

        prompt_tokens = 0
        if hasattr(response_obj, 'usage') and response_obj.usage is not None:
             prompt_tokens = response_obj.usage.prompt_tokens or 0

        cost = 0.0
        model_info = self.EMBEDDING_MODELS.get(model_name) # type: ignore
        if self._budget and model_info:
            cost = self._budget.update_usage_and_cost(
                    model_info=model_info,
                    input_tokens_used=prompt_tokens,
                    output_tokens_used=0, # Embeddings don't have completion tokens
            )
            self._logger.info(
                f"💰 Embedding Cost: ${cost:.5f} | Total Cost: ${self._budget.total_cost:.5f} | Budget Left: ${self._budget.remaining_budget:.5f}"
            )
        elif self._budget and not model_info:
            self._logger.warning(f"Model '{model_name}' not found in EMBEDDING_MODELS. Cannot calculate cost.")
        
        if not response_obj.data or not response_obj.data[0] or not response_obj.data[0].embedding:
             raise ValueError("Embedding response did not contain valid embedding data.")
        
        raw_embedding_data: list[float] = response_obj.data[0].embedding
        return EmbeddingModelResponse(
            embedding=embedding_parser(raw_embedding_data),
            llm_info=model_info or self.MODELS[model_name], # type: ignore # Fallback to general MODELS
            prompt_tokens_used=prompt_tokens,
        )

    def _get_embedding_kwargs(
        self, input: str | list[str], model: OpenAIModelName, **kwargs
    ) -> EmbeddingCreateParams:
        final_kwargs: dict[str, Any] = {**kwargs}
        final_kwargs["input"] = input
        # 'model' is added by _create_embedding_call

        if extra_h := self._configuration.extra_request_headers:
             final_kwargs["extra_headers"] = {**final_kwargs.get("extra_headers", {}), **extra_h}
        
        return cast(EmbeddingCreateParams, final_kwargs)

    @_BaseOpenAIProvider._retry_api_request # type: ignore
    async def _create_embedding_call( # Renamed from _create_embedding
        self, embedding_kwargs: EmbeddingCreateParams
    ) -> CreateEmbeddingResponse:
        # 'model' must be in embedding_kwargs here
        return await self._client.embeddings.create(**embedding_kwargs)


# --- Concrete OpenAIProvider (handles OpenAI & Gemini) ---

class OpenAIModelName(str, enum.Enum):
    # OpenAI Embedding Models
    EMBEDDING_V2 = "text-embedding-ada-002"
    EMBEDDING_V3_S = "text-embedding-3-small"
    EMBEDDING_V3_L = "text-embedding-3-large"

    # OpenAI GPT-3.5 Models
    GPT3_TURBO = "gpt-3.5-turbo" # Rolling alias, often 16k context

    # OpenAI GPT-4 Models
    GPT4_TURBO = "gpt-4-turbo" # Rolling alias for latest turbo (e.g., gpt-4-turbo-2024-04-09)
    GPT4_O = "gpt-4o" # Rolling alias for latest gpt-4o (e.g., gpt-4o-2024-05-13)
    
    # Gemini Models (via OpenAI SDK compatibility)
    GEMINI_1_5_FLASH_LATEST = "gemini-1.5-flash-latest"
    GEMINI_1_5_PRO_LATEST = "gemini-1.5-pro-latest"
    GEMINI_2_5_PRO_PREVIEW_05_06 = "gemini-2.5-pro-preview-05-06" # User requested model

# Update _ModelName for this provider
_ConcreteModelName = OpenAIModelName


OPEN_AI_CHAT_MODELS_DEFINITIONS: dict[_ConcreteModelName, ChatModelInfo[_ConcreteModelName]] = {
    info.name: info for info in [
        ChatModelInfo(
            name=_ConcreteModelName.GPT3_TURBO, provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0005 / 1000, completion_token_cost=0.0015 / 1000,
            max_tokens=16385, has_function_call_api=True,
        ),
        ChatModelInfo(
            name=_ConcreteModelName.GPT4_TURBO, provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.01 / 1000, completion_token_cost=0.03 / 1000,
            max_tokens=128000, has_function_call_api=True,
        ),
        ChatModelInfo(
            name=_ConcreteModelName.GPT4_O, provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.005 / 1000, completion_token_cost=0.015 / 1000,
            max_tokens=128000, has_function_call_api=True,
        ),
        # Gemini Models - Costs and max_tokens are illustrative
        # Gemini pricing: https://ai.google/pricing/
        # Gemini Flash 1.5M tokens: $0.35/1M input, $1.05/1M output (for context > 128k)
        # ($0.00000035/token input, $0.00000105/token output for large context)
        # ($0.00035/1k input, $0.00105/1k output for large context)
        # For <=128K context: $0.70/1M input, $2.10/1M output
        # ($0.0007/1k input, $0.0021/1k output)
        ChatModelInfo(
            name=_ConcreteModelName.GEMINI_1_5_FLASH_LATEST, provider_name=ModelProviderName.GEMINI,
            prompt_token_cost=0.0007 / 1000, completion_token_cost=0.0021 / 1000, # For <=128K tokens
            max_tokens=1048576, has_function_call_api=True, # Model supports 1M, practical limits may apply
        ),
        # Gemini Pro 1.5 1M tokens: $3.50/1M input, $10.50/1M output (for context > 128k)
        # For <=128K context: $7.00/1M input, $21.00/1M output
        # ($0.007/1k input, $0.021/1k output)
        ChatModelInfo(
            name=_ConcreteModelName.GEMINI_1_5_PRO_LATEST, provider_name=ModelProviderName.GEMINI,
            prompt_token_cost=0.007 / 1000, completion_token_cost=0.021 / 1000, # For <=128K tokens
            max_tokens=1048576, has_function_call_api=True,
        ),
        # Gemini 2.5 Pro Preview - Assuming similar to 1.5 Pro for now, adjust when public pricing available
         ChatModelInfo(
            name=_ConcreteModelName.GEMINI_2_5_PRO_PREVIEW_05_06, provider_name=ModelProviderName.GEMINI,
            prompt_token_cost=0.007 / 1000, completion_token_cost=0.021 / 1000, # Placeholder
            max_tokens=2097152, has_function_call_api=True, # Gemini Advanced can have 2M context
        ),
    ]
}

OPEN_AI_EMBEDDING_MODELS_DEFINITIONS: dict[_ConcreteModelName, EmbeddingModelInfo[_ConcreteModelName]] = {
    info.name: info for info in [
        EmbeddingModelInfo(
            name=_ConcreteModelName.EMBEDDING_V2, provider_name=ModelProviderName.OPENAI,
            token_cost=0.0001 / 1000, max_tokens=8191, embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=_ConcreteModelName.EMBEDDING_V3_S, provider_name=ModelProviderName.OPENAI,
            token_cost=0.00002 / 1000, max_tokens=8191, embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=_ConcreteModelName.EMBEDDING_V3_L, provider_name=ModelProviderName.OPENAI,
            token_cost=0.00013 / 1000, max_tokens=8191, embedding_dimensions=3072,
        ),
        # Gemini embedding models are typically not accessed via OpenAI SDK compatibility.
        # If they were, they'd be added here.
    ]
}

ALL_PROVIDER_MODELS_FLAT: Mapping[
    _ConcreteModelName, ChatModelInfo[_ConcreteModelName] | EmbeddingModelInfo[_ConcreteModelName]
] = {
    **OPEN_AI_CHAT_MODELS_DEFINITIONS, # type: ignore
    **OPEN_AI_EMBEDDING_MODELS_DEFINITIONS, # type: ignore
}


class OpenAICredentials(ModelProviderCredentials): # Based on user's openai_provider.py
    api_key: Optional[SecretStr] = UserConfigurable(from_env="OPENAI_API_KEY", default=None)
    api_base: Optional[SecretStr] = UserConfigurable(from_env="OPENAI_API_BASE_URL", default=None)
    organization: Optional[SecretStr] = UserConfigurable(from_env="OPENAI_ORGANIZATION", default=None)

    api_type: Optional[SecretStr] = UserConfigurable(default=None, from_env=lambda: SecretStr(os.getenv("OPENAI_API_TYPE") or ("azure" if os.getenv("USE_AZURE") == "True" else os.getenv("OPENAI_API_TYPE")))) # type: ignore
    api_version: Optional[SecretStr] = UserConfigurable(default=None, from_env="OPENAI_API_VERSION")
    azure_endpoint: Optional[SecretStr] = UserConfigurable(default=None, from_env="AZURE_OPENAI_ENDPOINT") # Example
    azure_model_to_deploy_id_map: Optional[dict[str, str]] = UserConfigurable(default=None) # Loaded from config

    @classmethod
    def from_env(cls) -> "OpenAICredentials":
        # This is a simplified from_env. UserConfigurable would handle complex loading.
        # For Gemini, GEMINI_API_KEY is handled directly by _BaseOpenAIProvider if no creds.
        # This method primarily serves OpenAI/Azure.
        data = {}
        if key := os.getenv("OPENAI_API_KEY"): data["api_key"] = SecretStr(key)
        if base := os.getenv("OPENAI_API_BASE_URL"): data["api_base"] = SecretStr(base)
        if org := os.getenv("OPENAI_ORGANIZATION"): data["organization"] = SecretStr(org)
        
        api_type_val = os.getenv("OPENAI_API_TYPE") or ("azure" if os.getenv("USE_AZURE") == "True" else None)
        if api_type_val: data["api_type"] = SecretStr(api_type_val)
        
        if ver := os.getenv("OPENAI_API_VERSION"): data["api_version"] = SecretStr(ver)

        if data.get("api_type") and data["api_type"].get_secret_value() == "azure": # type: ignore
            if azure_ep := os.getenv("AZURE_OPENAI_ENDPOINT"): data["azure_endpoint"] = SecretStr(azure_ep)
            # azure_model_to_deploy_id_map usually loaded from a YAML or config file
        
        try:
            return cls(**data)
        except ValidationError as e:
            # Log this error appropriately
            logging.getLogger(__name__).warning(f"Validation error creating OpenAICredentials from env: {e}")
            # Decide if to raise or return partially valid/empty. Let's raise if essential like api_key missing for OpenAI
            if "api_key" not in data and (not data.get("api_type") or data["api_type"].get_secret_value() != "azure"): # type: ignore
                 raise ValueError("OPENAI_API_KEY is required for OpenAI provider unless using Azure with other auth.") from e
            return cls(**data) # Or return a default if that's preferred

    def get_api_access_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.api_key: kwargs["api_key"] = self.api_key.get_secret_value()
        if self.api_base: kwargs["base_url"] = self.api_base.get_secret_value()
        if self.organization: kwargs["organization"] = self.organization.get_secret_value()
        
        is_azure = self.api_type and self.api_type.get_secret_value() == "azure"
        if is_azure:
            if self.azure_endpoint: kwargs["azure_endpoint"] = self.azure_endpoint.get_secret_value()
            if self.api_version: kwargs["api_version"] = self.api_version.get_secret_value()
            # Azure specific: remove api_key if using managed identity, etc.
            # This example assumes api_key is still used for Azure.
        return kwargs

    # load_azure_config method from user's code would go here if needed

class OpenAISettings(ModelProviderSettings):
    name: ModelProviderName = ModelProviderName.OPENAI # Default is OpenAI, can be changed to GEMINI
    description: str = "Settings for OpenAI or Gemini (via OpenAI SDK) provider."
    credentials: Optional[OpenAICredentials] = None
    budget: Optional[ModelProviderBudget] = Field(default_factory=ModelProviderBudget) # type: ignore
    configuration: ModelProviderConfiguration = Field(default_factory=ModelProviderConfiguration) # type: ignore


class OpenAIProvider(
    BaseOpenAIChatProvider, # MRO: BaseOpenAIChatProvider -> _BaseOpenAIProvider -> BaseChatModelProvider -> BaseModelProvider -> ABC -> object
    BaseOpenAIEmbeddingProvider, # MRO: BaseOpenAIEmbeddingProvider -> _BaseOpenAIProvider (shared) -> BaseEmbeddingModelProvider -> BaseModelProvider (shared) -> ABC -> object
):
    MODELS = ALL_PROVIDER_MODELS_FLAT # type: ignore
    CHAT_MODELS = OPEN_AI_CHAT_MODELS_DEFINITIONS # type: ignore
    EMBEDDING_MODELS = OPEN_AI_EMBEDDING_MODELS_DEFINITIONS # type: ignore
    
    default_settings = OpenAISettings() # Default to OpenAI

    def __init__(
        self,
        settings: Optional[OpenAISettings] = None,
        logger: Optional[logging.Logger] = None, # Consistent name
    ):
        effective_logger = logger or logging.getLogger(self.__class__.__name__)
        final_settings = settings or self.default_settings.model_copy(deep=True)

        # Ensure credentials are set up based on provider type if not explicitly passed
        if not final_settings.credentials:
            if final_settings.name == ModelProviderName.OPENAI:
                try:
                    final_settings.credentials = OpenAICredentials.from_env()
                except ValueError as e: # from_env might raise if key vars missing
                    effective_logger.warning(f"Could not load OpenAI credentials from environment: {e}. Provider might be unusable.")
                    # Create empty creds to satisfy type, but operations will likely fail
                    final_settings.credentials = OpenAICredentials()
            elif final_settings.name == ModelProviderName.GEMINI:
                # For Gemini, _BaseOpenAIProvider handles GEMINI_API_KEY from env if no creds object.
                # If user WANTS to pass Gemini key via credentials object:
                if os.getenv("GEMINI_API_KEY"):
                     final_settings.credentials = OpenAICredentials(api_key=SecretStr(os.getenv("GEMINI_API_KEY"))) # type: ignore
                else:
                     effective_logger.info("No GEMINI_API_KEY in env and no explicit credentials for Gemini. Operations may fail if key not provided elsewhere.")
                     # _BaseOpenAIProvider will raise error if key not found.
        
        # Super __init__ call needs to happen once for shared bases like _BaseOpenAIProvider
        # Python's MRO handles this. _BaseOpenAIProvider.__init__ has a guard.
        super().__init__(settings=final_settings, logger=effective_logger)

    def get_tokenizer(self, model_name: OpenAIModelName) -> ModelTokenizer[Any]:
        model_info = self.MODELS.get(model_name)
        
        if model_info and model_info.provider_name == ModelProviderName.GEMINI:
            self._logger.debug(
                f"Using character-based estimation (len(text) // 4) for Gemini model '{model_name.value}' token counting."
            )
            class GeminiEstTokenizer: # Dummy tokenizer for estimation
                def encode(self, text: str) -> list[int]:
                    return [0] * (len(text) // 4) 
                def decode(self, tokens: list[int]) -> str: # Not used for counting
                    return f"<estimated_decode_for_{len(tokens)}_tokens>"
            return GeminiEstTokenizer() # type: ignore
        
        # OpenAI models: Use tiktoken
        try:
            return tiktoken.encoding_for_model(model_name.value) # type: ignore
        except KeyError:
            self._logger.warning(
                f"No direct tiktoken encoder for model '{model_name.value}'. Attempting fallback."
            )
            # Fallback logic from user's original OpenAIProvider
            if "gpt-4o" in model_name.value or \
               "gpt-4" in model_name.value or \
               model_name.value == _ConcreteModelName.GPT3_TURBO.value: # GPT-3.5-Turbo also uses cl100k_base
                return tiktoken.get_encoding("cl100k_base") # type: ignore
            else: # Should not happen if model names are curated
                self._logger.error(f"Unknown model '{model_name.value}' for tiktoken fallback. Using cl100k_base.")
                return tiktoken.get_encoding("cl100k_base") # type: ignore

    def format_function_def_for_openai(self, function: CompletionModelFunction) -> FunctionDefinition:
        # This method is called by BaseOpenAIChatProvider
        return {
            "name": function.name,
            "description": function.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: param.to_dict() for name, param in function.parameters.items()
                },
                "required": [
                    name for name, param in function.parameters.items() if param.required
                ],
            },
        }
    
    # __repr__ is inherited from _BaseOpenAIProvider


if __name__ == "__main__":
    # Example Usage (requires async context for API calls)
    import asyncio

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("main_test")

    async def test_provider():
        # Test with OpenAI
        print("\n--- Testing OpenAI Provider (GPT-4o) ---")
        # Ensure OPENAI_API_KEY is set in your environment
        openai_settings = OpenAISettings(name=ModelProviderName.OPENAI)
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set. Skipping OpenAI test.")
        else:
            try:
                openai_provider = OpenAIProvider(settings=openai_settings, logger=logger)
                available_openai_models = await openai_provider.get_available_chat_models()
                print(f"Available OpenAI models: {[m.name.value for m in available_openai_models]}")

                if OpenAIModelName.GPT4_O in openai_provider.CHAT_MODELS:
                    prompt = [ChatMessage(role="user", content="Hello, GPT-4o! What is 2+2?")]
                    response = await openai_provider.create_chat_completion(
                        model_prompt=prompt,
                        model_name=OpenAIModelName.GPT4_O,
                        max_output_tokens=50
                    )
                    print(f"GPT-4o Response: {response.response.content}")
                    print(f"Tokens used: Input {response.prompt_tokens_used}, Output {response.completion_tokens_used}")
                else:
                    print("GPT-4o not configured/available for test.")
            except Exception as e:
                print(f"Error testing OpenAI: {e}")


        # Test with Gemini
        print("\n--- Testing Gemini Provider (gemini-2.5-pro-preview-05-06) ---")
        # Ensure GEMINI_API_KEY is set in your environment
        gemini_settings = OpenAISettings(name=ModelProviderName.GEMINI)
        if not os.getenv("GEMINI_API_KEY"):
            print("GEMINI_API_KEY not set. Skipping Gemini test.")
        else:
            try:
                gemini_provider = OpenAIProvider(settings=gemini_settings, logger=logger)
                available_gemini_models = await gemini_provider.get_available_chat_models()
                print(f"Available Gemini models (via SDK): {[m.name.value for m in available_gemini_models]}")

                test_gemini_model = OpenAIModelName.GEMINI_2_5_PRO_PREVIEW_05_06
                if test_gemini_model in gemini_provider.CHAT_MODELS: # Check if configured
                    prompt_gemini = [ChatMessage(role="user", content="Hello, Gemini! What is the capital of France? Explain in one sentence.")]
                    # Example with reasoning_effort for Gemini (if supported by the SDK pass-through)
                    response_gemini = await gemini_provider.create_chat_completion(
                        model_prompt=prompt_gemini,
                        model_name=test_gemini_model,
                        max_output_tokens=100,
                        # reasoning_effort="low" # Example Gemini-specific param
                    )
                    print(f"Gemini ({test_gemini_model.value}) Response: {response_gemini.response.content}")
                    print(f"Tokens used: Input {response_gemini.prompt_tokens_used}, Output {response_gemini.completion_tokens_used}")

                    # Test token counting for Gemini
                    gemini_text = "This is a test sentence for Gemini token counting."
                    gemini_tokens = gemini_provider.count_tokens(gemini_text, test_gemini_model)
                    print(f"Estimated tokens for '{gemini_text}' with Gemini: {gemini_tokens} (char/4 estimate)")

                else:
                    print(f"{test_gemini_model.value} not configured/available for test.")
            except Exception as e:
                print(f"Error testing Gemini: {e}")

    asyncio.run(test_provider())