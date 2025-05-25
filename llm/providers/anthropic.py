from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, ParamSpec, Sequence, TypeVar

import sentry_sdk
import tenacity
import tiktoken
from anthropic import APIConnectionError, APIStatusError
from pydantic import SecretStr

from forge.models.config import UserConfigurable

from .schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    BaseChatModelProvider,
    ChatMessage,
    ChatModelInfo,
    ChatModelResponse,
    CompletionModelFunction,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
    ToolResultMessage,
)
from .utils import validate_tool_calls

if TYPE_CHECKING:
    from anthropic.types.beta.tools import MessageCreateParams
    from anthropic.types.beta.tools import ToolsBetaMessage as Message
    from anthropic.types.beta.tools import ToolsBetaMessageParam as MessageParam

_T = TypeVar("_T")
_P = ParamSpec("_P")

class AnthropicModelName(str, enum.Enum):
    CLAUDE3_OPUS_v1 = "claude-3-opus-20240229"
    CLAUDE3_SONNET_v1 = "claude-3-sonnet-20240229"
    CLAUDE3_5_SONNET_v1 = "claude-3-5-sonnet-20240620"
    CLAUDE3_HAIKU_v1 = "claude-3-haiku-20240307"
    CLAUDE3_7_SONNET_v1 = "claude-3-7-sonnet-20250219" # Confirmed by user


ANTHROPIC_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=AnthropicModelName.CLAUDE3_OPUS_v1,
            provider_name=ModelProviderName.ANTHROPIC,
            prompt_token_cost=15 / 1e6,
            completion_token_cost=75 / 1e6,
            max_tokens=200000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=AnthropicModelName.CLAUDE3_SONNET_v1,
            provider_name=ModelProviderName.ANTHROPIC,
            prompt_token_cost=3 / 1e6,
            completion_token_cost=15 / 1e6,
            max_tokens=200000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=AnthropicModelName.CLAUDE3_5_SONNET_v1,
            provider_name=ModelProviderName.ANTHROPIC,
            prompt_token_cost=3 / 1e6,
            completion_token_cost=15 / 1e6,
            max_tokens=200000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=AnthropicModelName.CLAUDE3_HAIKU_v1,
            provider_name=ModelProviderName.ANTHROPIC,
            prompt_token_cost=0.25 / 1e6,
            completion_token_cost=1.25 / 1e6,
            max_tokens=200000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=AnthropicModelName.CLAUDE3_7_SONNET_v1, # Add model info
            provider_name=ModelProviderName.ANTHROPIC,
            prompt_token_cost=3 / 1e6, # Assuming same as other Sonnet models
            completion_token_cost=15 / 1e6, # Assuming same as other Sonnet models
            max_tokens=200000,
            has_function_call_api=True,
        ),
    ]
}

class AnthropicCredentials(ModelProviderCredentials):
    """Credentials for Anthropic."""

    api_key: SecretStr = UserConfigurable(from_env="ANTHROPIC_API_KEY") # type: ignore
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="ANTHROPIC_API_BASE_URL"
    )

    def get_api_access_kwargs(self) -> dict[str, str]:
        return {
            k: v.get_secret_value()
            for k, v in {
                "api_key": self.api_key,
                "base_url": self.api_base,
            }.items()
            if v is not None
        }


class AnthropicSettings(ModelProviderSettings):
    credentials: Optional[AnthropicCredentials] # type: ignore
    budget: ModelProviderBudget # type: ignore


class AnthropicProvider(BaseChatModelProvider[AnthropicModelName, AnthropicSettings]):
    default_settings = AnthropicSettings(
        name="anthropic_provider",
        description="Provides access to Anthropic's API.",
        configuration=ModelProviderConfiguration(),
        credentials=None,
        budget=ModelProviderBudget(),
    )

    _settings: AnthropicSettings
    _credentials: AnthropicCredentials
    _budget: ModelProviderBudget
    _client: AsyncAnthropic # Add type hint for the async client

    def __init__(
        self,
        settings: Optional[AnthropicSettings] = None,
        logger: Optional[logging.Logger] = None, # Use the passed logger or default
    ):
        # Use the provided logger or get a default one for this module
        self._logger = logger or logging.getLogger(__name__)

        if not settings:
            settings = self.default_settings.model_copy(deep=True)
        if not settings.credentials:
            settings.credentials = AnthropicCredentials.from_env()

        # Call super init *after* setting logger, but *before* using credentials
        super(AnthropicProvider, self).__init__(settings=settings, logger=self._logger)

        # Initialize the async client using fetched credentials
        self._client = AsyncAnthropic(
            **self._credentials.get_api_access_kwargs() # type: ignore
        )

    async def get_available_models(self) -> Sequence[ChatModelInfo[AnthropicModelName]]:
        # Anthropic SDK might not have a model list endpoint readily available.
        # Falling back to the static list defined above.
        return await self.get_available_chat_models()

    async def get_available_chat_models(
        self,
    ) -> Sequence[ChatModelInfo[AnthropicModelName]]:
        # Return the statically defined list of models
        return list(ANTHROPIC_CHAT_MODELS.values())

    def get_token_limit(self, model_name: AnthropicModelName) -> int:
        """Get the token limit for a given model from the static definition."""
        return ANTHROPIC_CHAT_MODELS[model_name].max_tokens

    def get_tokenizer(self, model_name: AnthropicModelName) -> ModelTokenizer[Any]:
        """
        DEPRECATED: Anthropic token counting should use the client's method,
        not a separate tokenizer object like tiktoken.
        """
        self._logger.warning(
            "get_tokenizer is deprecated for AnthropicProvider. "
            "Use count_tokens or count_message_tokens instead."
        )
        raise NotImplementedError(
            "Anthropic token counting is done via client.count_tokens, not a separate tokenizer object."
        )

    async def count_tokens(self, text: str, model_name: Optional[AnthropicModelName] = None) -> int:
        """
        Accurately count tokens for the given text using the Anthropic client's
        built-in counter. Added Optional model_name for potential future use,
        but current client.count_tokens might not require it.
        """
        try:
            # Use await as the client is async
            token_count = await self._client.count_tokens(text=text)
            return token_count
        except AttributeError:
            self._logger.exception( # Use exception logger to include traceback
                "self._client.count_tokens method not found. "
                "Please check your 'anthropic' library version and documentation."
            )
            raise NotImplementedError(
                "Accurate token counting requires 'anthropic' library's count_tokens method."
            )
        except Exception as e:
            self._logger.exception(f"Error counting tokens with Anthropic client: {e}")
            # Re-raise the exception so the calling code knows counting failed.
            raise

    async def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: AnthropicModelName,
    ) -> int:
        """
        Count tokens for a list of ChatMessages using the Anthropic client.
        Currently uses an approximation by formatting messages to text, as a
        direct message list counting method might not be available or requires
        specific formatting not implemented here.
        """
        if isinstance(messages, ChatMessage):
            messages = [messages]

        # Approximate by formatting and counting raw text.
        # This structure might need refinement to better match Anthropic's internal counting.
        text_representation = ""
        last_role = None
        for msg in messages:
            # Add extra newline between turns of different roles
            if last_role and msg.role != last_role:
                 text_representation += "\n"
            text_representation += f"{msg.role.upper()}:\n{msg.content}"
            if isinstance(msg, AssistantChatMessage) and msg.tool_calls:
                # Add representation for tool calls if needed for counting
                # This is a placeholder format, actual token cost might differ
                tool_calls_repr = ", ".join([f"{tc.function.name}({tc.function.arguments})" for tc in msg.tool_calls])
                text_representation += f"\nTOOL_CALLS: [{tool_calls_repr}]"
            elif isinstance(msg, ToolResultMessage):
                # Add representation for tool results
                text_representation += f"\nTOOL_RESULT ID {msg.tool_call_id}: {msg.content}"

            text_representation += "\n\n" # Add separation between messages
            last_role = msg.role

        # Use the accurate (but text-based) counter
        # Pass model_name if count_tokens implementation requires it in the future
        return await self.count_tokens(text_representation.strip(), model_name)

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: AnthropicModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        prefill_response: str = "",
        **kwargs,
    ) -> ChatModelResponse[_T]:
        """Create a completion using the Anthropic API."""
        anthropic_messages, completion_kwargs = self._get_chat_completion_args(
            prompt_messages=model_prompt,
            functions=functions,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

        total_cost = 0.0
        attempts = 0
        while True:
            # Copy messages for this attempt, potentially adding prefill
            current_messages = anthropic_messages.copy()
            if prefill_response:
                # Anthropic expects prefill as the start of an assistant message
                current_messages.append(
                    {"role": "assistant", "content": prefill_response}
                )

            completion_kwargs["messages"] = current_messages

            try:
                (
                    _response_message, # Renamed from _assistant_msg to avoid confusion
                    cost,
                    t_input,
                    t_output,
                ) = await self._create_chat_completion(model_name, completion_kwargs)
                total_cost += cost
                self._logger.debug(
                    f"Completion usage: {t_input} input, {t_output} output "
                    f"- ${round(cost, 5)}"
                )

                # Merge prefill into generated response if prefill was used
                if prefill_response:
                    # Find the first text block and prepend the prefill
                    merged_content_blocks = []
                    prefill_applied = False
                    for block in _response_message.content:
                        if not prefill_applied and block.type == "text":
                             # Create a new text block with merged content
                            merged_block = block.model_copy()
                            merged_block.text = prefill_response + block.text
                            merged_content_blocks.append(merged_block)
                            prefill_applied = True
                        else:
                            merged_content_blocks.append(block)
                    # If no text block was found (unlikely), handle appropriately
                    if not prefill_applied and prefill_response:
                         self._logger.warning("Prefill requested but no text block found in response content.")
                         # Decide how to handle: maybe add prefill as a new text block?
                         # For now, we'll just use the original blocks if prefill failed
                         final_content_blocks = _response_message.content
                    else:
                         final_content_blocks = merged_content_blocks
                else:
                    final_content_blocks = _response_message.content


                # Parse into internal schema
                assistant_msg = AssistantChatMessage(
                    content="\n\n".join(
                        b.text for b in final_content_blocks if b.type == "text"
                    ),
                    tool_calls=self._parse_assistant_tool_calls(_response_message), # Use original response for parsing tool calls
                )

            except Exception as api_or_budget_error:
                # Catch errors during the API call itself (incl. BadRequestError for size limits)
                # or budget update errors.
                self._logger.exception(f"API call or cost update failed: {api_or_budget_error}")
                # Re-raise the original error to be handled by the caller
                raise api_or_budget_error


            # If parsing the response fails, append the error to the prompt, and let the
            # LLM fix its mistake(s).
            attempts += 1
            tool_call_errors = []
            parsed_result : _T = None # type: ignore
            try:
                # Validate tool calls
                if assistant_msg.tool_calls and functions:
                    tool_call_errors = validate_tool_calls(
                        assistant_msg.tool_calls, functions
                    )
                    if tool_call_errors:
                        # Combine errors into a single exception message
                        error_msg = "Invalid tool use(s):\n" + "\n".join(str(e) for e in tool_call_errors)
                        raise ValueError(error_msg) # Raise specific error for handling below

                # Try parsing the content with the provided parser
                parsed_result = completion_parser(assistant_msg)

                # If we got here without exceptions, break the loop
                break

            except Exception as e:
                self._logger.debug(
                    f"Parsing or validation failed on response: '''{_response_message}'''"
                )
                self._logger.warning(f"Attempt #{attempts} parsing/validation failed: {e}")
                sentry_sdk.capture_exception(
                    error=e,
                    extras={"assistant_msg_raw": _response_message.model_dump(), "i_attempt": attempts},
                )
                if attempts < self._configuration.fix_failed_parse_tries:
                    # Append the raw assistant message that failed
                    anthropic_messages.append(
                         _response_message.model_dump(include={"role", "content"}) # type: ignore # noqa
                    )
                    # Append a user message explaining the error(s)
                    error_feedback_content = []

                    # Add tool results indicating errors if tool validation failed
                    if assistant_msg.tool_calls: # Check if there were tool calls attempted
                         for tc in assistant_msg.tool_calls:
                             # Find the corresponding error, if any
                             specific_error = next((tce for tce in tool_call_errors if hasattr(tce, 'name') and tce.name == tc.function.name), None)
                             error_text = "Error during validation or parsing."
                             if specific_error:
                                 error_text = str(specific_error)
                             elif isinstance(e, ValueError) and "Invalid tool use" in str(e):
                                 # If it was a general tool validation error not tied to a specific call error object
                                 error_text = str(e) # Use the main exception message

                             error_feedback_content.append({
                                 "type": "tool_result",
                                 "tool_use_id": tc.id,
                                 "is_error": True,
                                 "content": [{"type": "text", "text": error_text}]
                             })

                    # Add the general error message
                    error_feedback_content.append({
                         "type": "text",
                         "text": f"ERROR PARSING OR VALIDATING YOUR LAST RESPONSE (Attempt {attempts}):\n{e.__class__.__name__}: {e}\nPlease correct the issue and provide the response again."
                    })

                    anthropic_messages.append({
                        "role": "user",
                        "content": error_feedback_content
                    })
                else:
                    self._logger.error(f"Failed to parse/validate response after {attempts} attempts.")
                    raise e # Re-raise the last exception if max retries exceeded

        if attempts > 1:
            self._logger.info( # Use info level for multi-attempt success
                f"Successfully parsed response after {attempts} attempts. Total cost: ${round(total_cost, 5)}"
            )

        return ChatModelResponse(
            response=assistant_msg,
            parsed_result=parsed_result,
            llm_info=ANTHROPIC_CHAT_MODELS[model_name],
            prompt_tokens_used=t_input,
            completion_tokens_used=t_output,
        )

    def _get_chat_completion_args(
        self,
        prompt_messages: list[ChatMessage],
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> tuple[list[MessageParam], MessageCreateParams]:
        """Prepare arguments for message completion API call."""
        if functions:
            kwargs["tools"] = [
                {
                    "name": f.name,
                    "description": f.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            name: param.to_dict()
                            for name, param in f.parameters.items()
                        },
                        "required": [
                            name
                            for name, param in f.parameters.items()
                            if param.required
                        ],
                    },
                }
                for f in functions
            ]
            # Anthropic doesn't have a direct equivalent to OpenAI's tool_choice='required'
            # or forcing a specific single function. You might need to adjust prompts
            # or handle cases where the model doesn't use a tool when expected.

        # Set max_tokens, defaulting if not provided
        kwargs["max_tokens"] = max_output_tokens or 4096 # Default to 4096, adjust if needed

        if extra_headers := self._configuration.extra_request_headers:
            kwargs["extra_headers"] = kwargs.get("extra_headers", {})
            kwargs["extra_headers"].update(extra_headers.copy())

        # Separate system prompt
        system_messages_content = [
            m.content for m in prompt_messages if m.role == ChatMessage.Role.SYSTEM
        ]
        if len(system_messages_content) > 1:
            self._logger.warning(
                f"Prompt has {len(system_messages_content)} system messages; Anthropic supports only 1. Merging them."
            )
        kwargs["system"] = "\n\n".join(system_messages_content)

        # Format remaining messages
        messages: list[MessageParam] = []
        for message in prompt_messages:
            if message.role == ChatMessage.Role.SYSTEM:
                continue # Handled above

            elif message.role == ChatMessage.Role.USER:
                # Check if last message was also user, if so, merge content
                if messages and messages[-1]["role"] == "user":
                     last_content = messages[-1]["content"]
                     new_content = [{"type": "text", "text": message.content}] # Start with new content as text block

                     if isinstance(last_content, str):
                         # Previous was simple string, merge into list
                         messages[-1]["content"] = [{"type": "text", "text": last_content}] + new_content
                     elif isinstance(last_content, list):
                         # Previous was already a list, append new text block
                         last_content.extend(new_content)
                     # Handle potential image content here if needed in the future
                else:
                     # Previous message was not user or no previous message
                     messages.append({"role": "user", "content": message.content}) # Simple string content for now

            elif message.role == ChatMessage.Role.ASSISTANT:
                assistant_content_blocks = []
                if message.content:
                    assistant_content_blocks.append({"type": "text", "text": message.content})

                if isinstance(message, AssistantChatMessage) and message.tool_calls:
                    for tc in message.tool_calls:
                         assistant_content_blocks.append({
                             "type": "tool_use",
                             "id": tc.id,
                             "name": tc.function.name,
                             "input": tc.function.arguments,
                         })

                if assistant_content_blocks: # Only add assistant message if there's content or tool calls
                     messages.append({
                         "role": "assistant",
                         "content": assistant_content_blocks
                     })

            elif isinstance(message, ToolResultMessage):
                messages.append({
                    "role": "user", # Tool results are added as user role messages
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": [{"type": "text", "text": message.content}], # Content needs to be list for tool_result
                            "is_error": message.is_error,
                        }
                    ],
                })

        # Ensure correct types before returning
        typed_kwargs = MessageCreateParams(**kwargs)
        return messages, typed_kwargs

    async def _create_chat_completion(
        self, model: AnthropicModelName, completion_kwargs: MessageCreateParams
    ) -> tuple[Message, float, int, int]:
        """Create a chat completion using the Anthropic API with retry handling."""

        @self._retry_api_request
        async def _create_chat_completion_with_retry() -> Message:
            # Make the API call using the formatted arguments
            return await self._client.beta.tools.messages.create(
                model=model, **completion_kwargs # Pass pre-validated kwargs
            )

        response: Message = await _create_chat_completion_with_retry()

        # Check for valid usage data
        if not response.usage:
             self._logger.warning(f"Anthropic response for model {model} missing usage data.")
             input_tokens, output_tokens = 0, 0 # Default if missing
        else:
             input_tokens = response.usage.input_tokens
             output_tokens = response.usage.output_tokens

        # Calculate cost using budget tracker
        cost = self._budget.update_usage_and_cost(
            model_info=ANTHROPIC_CHAT_MODELS[model],
            input_tokens_used=input_tokens,
            output_tokens_used=output_tokens,
        )
        # Return the raw response message, cost, and token counts
        return response, cost, input_tokens, output_tokens

    def _parse_assistant_tool_calls(
        self, assistant_message: Message
    ) -> list[AssistantToolCall]:
        """Parses tool_use blocks from Anthropic response into internal schema."""
        tool_calls = []
        for content_block in assistant_message.content:
            if content_block.type == "tool_use":
                tool_calls.append(
                    AssistantToolCall(
                        id=content_block.id,
                        type="function", # Assuming all tool_use maps to function type internally
                        function=AssistantFunctionCall(
                            name=content_block.name,
                            arguments=content_block.input, # Arguments are already parsed dict by Anthropic SDK
                        ),
                    )
                )
        return tool_calls

    def _retry_api_request(self, func: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
        """Decorator for retrying API requests with exponential backoff."""
        return tenacity.retry(
            retry=(
                tenacity.retry_if_exception_type(APIConnectionError)
                | tenacity.retry_if_exception(
                    lambda e: isinstance(e, APIStatusError) and e.status_code >= 500
                )
            ),
            wait=tenacity.wait_exponential(),
            stop=tenacity.stop_after_attempt(self._configuration.retries_per_request),
            # Log before retrying
            before_sleep=tenacity.before_sleep_log(self._logger, logging.WARNING),
             # Reraise the exception after the last attempt
            reraise=True,
        )(func)

    def __repr__(self):
        return f"AnthropicProvider(budget={self._budget})" # Add budget info for better representation
