import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Mapping,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
    cast,
    get_args,
)

import sentry_sdk
import tenacity
from openai._exceptions import APIConnectionError, APIStatusError
from openai.types import CreateEmbeddingResponse, EmbeddingCreateParams
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    CompletionCreateParams,
)
from openai.types.shared_params import FunctionDefinition

from forge.json.parsing import json_loads

from .schema import (  # Assuming schema.py is in the same directory or accessible
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    BaseChatModelProvider,
    BaseEmbeddingModelProvider,
    BaseModelProvider,
    ChatMessage,
    ChatModelInfo,
    ChatModelResponse,
    CompletionModelFunction,
    Embedding,
    EmbeddingModelInfo,
    EmbeddingModelResponse,
    ModelProviderService,
    _ModelName,
    _ModelProviderSettings,
)
from .utils import validate_tool_calls # Assuming utils.py is in the same directory or accessible

_T = TypeVar("_T")
_P = ParamSpec("_P")


class _BaseOpenAIProvider(BaseModelProvider[_ModelName, _ModelProviderSettings]):
    """Base class for LLM providers with OpenAI-like APIs"""

    MODELS: ClassVar[
        Mapping[_ModelName, ChatModelInfo[_ModelName] | EmbeddingModelInfo[_ModelName]]
    ]

    def __init__(
        self,
        settings: Optional[_ModelProviderSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not getattr(self, "MODELS", None):
            raise ValueError(f"{self.__class__.__name__}.MODELS is not set")

        if not settings:
            settings = self.default_settings.model_copy(deep=True)
        if not settings.credentials:
            settings.credentials = get_args(
                self.default_settings.model_fields["credentials"].annotation
            )[0].from_env()

        super(_BaseOpenAIProvider, self).__init__(settings=settings, logger=logger)

        if not getattr(self, "_client", None):
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                **self._credentials.get_api_access_kwargs()  # type: ignore
            )

    async def get_available_models(
        self,
    ) -> Sequence[ChatModelInfo[_ModelName] | EmbeddingModelInfo[_ModelName]]:
        _models = (await self._client.models.list()).data
        return [
            self.MODELS[cast(_ModelName, m.id)] for m in _models if m.id in self.MODELS
        ]

    def get_token_limit(self, model_name: _ModelName) -> int:
        """Get the maximum number of input tokens for a given model"""
        return self.MODELS[model_name].max_tokens

    def count_tokens(self, text: str, model_name: _ModelName) -> int:
        # Placeholder: Implement actual token counting if needed, e.g., using tiktoken
        # return len(self.get_tokenizer(model_name).encode(text))
        # For now, returning a rough estimate based on characters / 4
        return len(text) // 4 # Replace with proper tokenizer

    def get_tokenizer(self, model_name: _ModelName) -> Any:
         # Placeholder: Implement tokenizer retrieval if needed
         raise NotImplementedError("Tokenizer retrieval not implemented in this base class.")


    def _retry_api_request(self, func: Callable[_P, _T]) -> Callable[_P, _T]:
        return tenacity.retry(
            retry=(tenacity.retry_if_exception_type(APIConnectionError)
                   | tenacity.retry_if_exception(
                        lambda e: isinstance(e, APIStatusError) and e.status_code >= 500
                    )),
            wait=tenacity.wait_exponential(),
            stop=tenacity.stop_after_attempt(self._configuration.retries_per_request),
            after=tenacity.after_log(self._logger, logging.DEBUG),
        )(func)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class BaseOpenAIChatProvider(
    _BaseOpenAIProvider[_ModelName, _ModelProviderSettings],
    BaseChatModelProvider[_ModelName, _ModelProviderSettings],
):
    CHAT_MODELS: ClassVar[dict[_ModelName, ChatModelInfo[_ModelName]]]

    def __init__(
        self,
        settings: Optional[_ModelProviderSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not getattr(self, "CHAT_MODELS", None):
            raise ValueError(f"{self.__class__.__name__}.CHAT_MODELS is not set")
        # Ensure MODELS includes CHAT_MODELS for the base class constructor
        if not getattr(self, "MODELS", None):
             self.MODELS = {} # Initialize if not present
        self.MODELS.update(self.CHAT_MODELS)

        super(BaseOpenAIChatProvider, self).__init__(settings=settings, logger=logger)


    async def get_available_chat_models(self) -> Sequence[ChatModelInfo[_ModelName]]:
        all_available_models = await self.get_available_models()
        return [
            model
            for model in all_available_models
            if model.service == ModelProviderService.CHAT
        ]

    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: _ModelName,
    ) -> int:
        if isinstance(messages, ChatMessage):
            messages = [messages]
        # Placeholder: Implement actual message token counting if needed
        # For now, concatenating content and using character count / 4
        text_content = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
        return self.count_tokens(text_content, model_name) # Use the class's count_tokens

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: _ModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        prefill_response: str = "",
        **kwargs,
    ) -> ChatModelResponse[_T]:
        (
            openai_messages,
            completion_kwargs,
            parse_kwargs,
        ) = self._get_chat_completion_args(
            prompt_messages=model_prompt,
            model=model_name,
            functions=functions,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

        total_cost = 0.0
        attempts = 0
        while True:
            completion_kwargs["messages"] = openai_messages
            _response, _cost, t_input, t_output = await self._create_chat_completion(
                model=model_name, completion_kwargs=completion_kwargs
            )
            total_cost += _cost
            attempts += 1
            parse_errors: list[Exception] = []
            _assistant_msg = _response.choices[0].message

            tool_calls, _errors = self._parse_assistant_tool_calls(
                _assistant_msg, **parse_kwargs
            )
            parse_errors += _errors

            if not parse_errors and tool_calls and functions:
                parse_errors += validate_tool_calls(tool_calls, functions)

            assistant_msg = AssistantChatMessage(
                content=_assistant_msg.content or "",
                tool_calls=tool_calls or None,
            )
            parsed_result: _T = None  # type: ignore
            if not parse_errors:
                try:
                    parsed_result = completion_parser(assistant_msg)
                except Exception as e:
                    parse_errors.append(e)
            if not parse_errors:
                if attempts > 1:
                    self._logger.debug(
                        f"Total cost for {attempts} attempts: ${round(total_cost, 5)}"
                    )
                return ChatModelResponse(
                    response=AssistantChatMessage(
                        content=_assistant_msg.content or "",
                        tool_calls=tool_calls or None,
                    ),
                    parsed_result=parsed_result,
                    llm_info=self.CHAT_MODELS[model_name],
                    prompt_tokens_used=t_input,
                    completion_tokens_used=t_output,
                )
            else:
                self._logger.debug(f"Parsing failed on response: '''{_assistant_msg}'''")
                parse_errors_fmt = "\n\n".join(
                    f"{e.__class__.__name__}: {e}" for e in parse_errors
                )
                self._logger.warning(
                    f"Parsing attempt #{attempts} failed: {parse_errors_fmt}"
                )
                for e in parse_errors:
                    if sentry_sdk: # Check if sentry is available
                         sentry_sdk.capture_exception(
                              error=e,
                              extras={"assistant_msg": _assistant_msg.model_dump(exclude_none=True), "i_attempt": attempts},
                         )
                if attempts < self._configuration.fix_failed_parse_tries:
                    openai_messages.append(
                        _assistant_msg.model_dump(exclude_none=True)
                    )
                    openai_messages.append(
                        {
                            "role": "system",
                            "content": (
                                f"ERROR PARSING YOUR RESPONSE:\n\n{parse_errors_fmt}"
                            ),
                        }
                    )
                    continue
                else:
                    raise parse_errors[0]

    def _get_chat_completion_args(
        self,
        prompt_messages: list[ChatMessage],
        model: _ModelName,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> tuple[
        list[ChatCompletionMessageParam], CompletionCreateParams, dict[str, Any]
    ]:
        kwargs = cast(CompletionCreateParams, kwargs)
        if max_output_tokens:
            kwargs["max_tokens"] = max_output_tokens
        if functions:
            kwargs["tools"] = [
                {"type": "function", "function": format_function_def_for_openai(f)}
                for f in functions
            ]
            # Decide on tool_choice logic (e.g., auto, specific function, etc.)
            # Example: Force calling the first function if only one is provided
            if len(functions) == 1:
                 kwargs["tool_choice"] = {
                     "type": "function",
                     "function": {"name": functions[0].name},
                 }
            # else: # Or let the model choose:
            #    kwargs["tool_choice"] = "auto"

        if extra_headers := self._configuration.extra_request_headers:
            kwargs["extra_headers"] = kwargs.get("extra_headers", {})
            kwargs["extra_headers"].update(extra_headers.copy())

        prepped_messages: list[ChatCompletionMessageParam] = [
            message.model_dump(
                include={"role", "content", "tool_calls", "tool_call_id", "name"},
                exclude_none=True,
            )
            for message in prompt_messages
        ]
        if "messages" in kwargs:
             # Combine explicitly passed messages with prompt_messages if needed
             # Ensure roles and structure are correct
             passed_messages = kwargs["messages"]
             if isinstance(passed_messages, list):
                  prepped_messages += passed_messages
             else:
                  # Handle case where kwargs["messages"] might not be a list
                  self._logger.warning("kwargs['messages'] was not a list, ignoring.")
             del kwargs["messages"] # Remove from kwargs after processing

        return prepped_messages, kwargs, {}

    async def _create_chat_completion(
        self,
        model: _ModelName,
        completion_kwargs: CompletionCreateParams,
    ) -> tuple[ChatCompletion, float, int, int]:
        # Set the model and disable streaming
        completion_kwargs["model"] = completion_kwargs.get("model") or model
        completion_kwargs["stream"] = False

        @self._retry_api_request
        async def _create_chat_completion_with_retry() -> tuple[ChatCompletion, float, int, int]:
            # Await the complete response (no streaming)
            response = await self._client.chat.completions.create(**completion_kwargs)
            prompt_tokens_used = 0
            completion_tokens_used = 0
            if hasattr(response, "usage") and response.usage is not None:
                prompt_tokens_used = response.usage.prompt_tokens
                completion_tokens_used = response.usage.completion_tokens
            # Note: Cost calculation moved outside the retry loop

            # Return 0 for cost initially, it will be calculated after the call succeeds
            return response, 0, prompt_tokens_used, completion_tokens_used

        result = await _create_chat_completion_with_retry()
        if result is None:
            # This should ideally not happen if retry logic is sound, but handle defensively
            raise ValueError("Completion returned None. Check your API client configuration and retry logic.")
        completion, _initial_cost_placeholder, prompt_tokens_used, completion_tokens_used = result

        cost = 0 # Initialize cost
        if self._budget:
            # Ensure the model exists in CHAT_MODELS before calculating cost
            if model in self.CHAT_MODELS:
                 cost = self._budget.update_usage_and_cost(
                      model_info=self.CHAT_MODELS[model],
                      input_tokens_used=prompt_tokens_used,
                      output_tokens_used=completion_tokens_used,
                 )
            else:
                 self._logger.warning(f"Model '{model}' not found in CHAT_MODELS. Cannot calculate cost.")
        else:
             # Cost remains 0 if budget tracking is disabled
             pass # Explicitly do nothing


        # Original debug log
        self._logger.debug(
            f"{model} completion usage: {prompt_tokens_used} input, "
            f"{completion_tokens_used} output - ${round(cost, 5)}"
        )

        # <<<--- ADDED LOGGING CODE ---<<<
        # Added logging for cost and budget
        if self._budget:
            total_cost = self._budget.total_cost  # Or self._budget.get_incurred_cost()
            remaining_budget = self._budget.remaining_budget # Or self._budget.get_remaining_budget()
            self._logger.info(
                f"💰 Cost this call: ${cost:.5f} | "
                f"Total cost: ${total_cost:.5f} | "
                f"Budget left: ${remaining_budget:.5f}"
            )
        else:
             self._logger.info(f"💰 Cost this call: ${cost:.5f} (Budget tracking disabled)")
        # >>>--- END OF ADDED LOGGING --- >>>

        return completion, cost, prompt_tokens_used, completion_tokens_used


    def _parse_assistant_tool_calls(
        self, assistant_message: ChatCompletionMessage, **kwargs
    ) -> tuple[list[AssistantToolCall], list[Exception]]:
        tool_calls: list[AssistantToolCall] = []
        parse_errors: list[Exception] = []
        if assistant_message.tool_calls:
            for _tc in assistant_message.tool_calls:
                # Check if function attribute exists and has name/arguments
                if not (hasattr(_tc, 'function') and hasattr(_tc.function, 'name') and hasattr(_tc.function, 'arguments')):
                     parse_errors.append(ValueError(f"Malformed tool call object received: {_tc}"))
                     continue # Skip this malformed tool call

                try:
                    # Ensure arguments is a string before parsing
                    if not isinstance(_tc.function.arguments, str):
                         raise TypeError(f"Expected arguments to be a string, got {type(_tc.function.arguments)}")

                    parsed_arguments = json_loads(_tc.function.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    err_message = (
                        f"Decoding arguments for tool '{_tc.function.name}' failed: " + str(e)
                    )
                    # Create a new exception with more context
                    parse_errors.append(
                        ValueError(err_message).with_traceback(e.__traceback__)
                    )
                    continue # Skip tool call if arguments parsing fails
                except Exception as e: # Catch other potential errors during parsing
                     err_message = (
                          f"Unexpected error decoding arguments for tool '{_tc.function.name}': {e}"
                     )
                     parse_errors.append(
                          type(e)(err_message, *e.args[1:]).with_traceback(e.__traceback__)
                     )
                     continue

                # Construct the AssistantToolCall object
                tool_calls.append(
                    AssistantToolCall(
                        id=_tc.id if hasattr(_tc, 'id') else 'unknown_id', # Handle missing ID
                        type=_tc.type if hasattr(_tc, 'type') else 'function', # Assume function if type missing
                        function=AssistantFunctionCall(
                            name=_tc.function.name,
                            arguments=parsed_arguments,
                        ),
                    )
                )
            # Clear parse_errors only if all tool calls were successfully parsed
            # Note: This might hide earlier errors if later ones succeed. Consider accumulating errors instead.
            # if len(tool_calls) == len(assistant_message.tool_calls):
            #    parse_errors = [] # Reconsider if partial success should clear all errors
        return tool_calls, parse_errors


class BaseOpenAIEmbeddingProvider(
    _BaseOpenAIProvider[_ModelName, _ModelProviderSettings],
    BaseEmbeddingModelProvider[_ModelName, _ModelProviderSettings],
):
    EMBEDDING_MODELS: ClassVar[dict[_ModelName, EmbeddingModelInfo[_ModelName]]]

    def __init__(
        self,
        settings: Optional[_ModelProviderSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not getattr(self, "EMBEDDING_MODELS", None):
            raise ValueError(f"{self.__class__.__name__}.EMBEDDING_MODELS is not set")
         # Ensure MODELS includes EMBEDDING_MODELS for the base class constructor
        if not getattr(self, "MODELS", None):
             self.MODELS = {} # Initialize if not present
        self.MODELS.update(self.EMBEDDING_MODELS)

        super(BaseOpenAIEmbeddingProvider, self).__init__(settings=settings, logger=logger)


    async def get_available_embedding_models(
        self,
    ) -> Sequence[EmbeddingModelInfo[_ModelName]]:
        all_available_models = await self.get_available_models()
        return [
            model for model in all_available_models if model.service == ModelProviderService.EMBEDDING
        ]

    async def create_embedding(
        self,
        text: str,
        model_name: _ModelName,
        embedding_parser: Callable[[Embedding], Embedding] = lambda x: x, # Default parser returns embedding as is
        **kwargs,
    ) -> EmbeddingModelResponse:
        embedding_kwargs = self._get_embedding_kwargs(input=text, model=model_name, **kwargs)

        # Await the result of the coroutine
        response = await self._create_embedding(embedding_kwargs)

        prompt_tokens = 0
        if hasattr(response, 'usage') and response.usage is not None and hasattr(response.usage, 'prompt_tokens'):
             prompt_tokens = response.usage.prompt_tokens

        # Budget update for embedding costs
        cost = 0
        if self._budget:
            # Embedding models typically only have input costs
            if model_name in self.EMBEDDING_MODELS:
                 cost = self._budget.update_usage_and_cost(
                      model_info=self.EMBEDDING_MODELS[model_name],
                      input_tokens_used=prompt_tokens,
                      output_tokens_used=0, # Embedding models don't have completion tokens
                 )
                 self._logger.info(
                     f"💰 Embedding Cost: ${cost:.5f} | Total Cost: ${self._budget.total_cost:.5f} | Budget Left: ${self._budget.remaining_budget:.5f}"
                 )
            else:
                 self._logger.warning(f"Model '{model_name}' not found in EMBEDDING_MODELS. Cannot calculate cost.")


        # Ensure embedding data is present before accessing
        if not response.data or not response.data[0].embedding:
             raise ValueError("Embedding response did not contain embedding data.")


        return EmbeddingModelResponse(
            embedding=embedding_parser(response.data[0].embedding),
            llm_info=self.EMBEDDING_MODELS[model_name], # Assumes model_name is valid
            prompt_tokens_used=prompt_tokens,
            # completion_tokens_used is implicitly 0 for embeddings
        )

    def _get_embedding_kwargs(
        self, input: str | list[str], model: _ModelName, **kwargs
    ) -> EmbeddingCreateParams:
        # Cast kwargs for type checking, although it might be redundant if using **kwargs directly
        embedding_params: EmbeddingCreateParams = {"input": input, "model": model, **kwargs} # type: ignore

        if extra_headers := self._configuration.extra_request_headers:
             # Ensure extra_headers in params is a dict before updating
             if "extra_headers" not in embedding_params:
                  embedding_params["extra_headers"] = {}
             elif not isinstance(embedding_params["extra_headers"], dict):
                  # Handle case where extra_headers might be set incorrectly
                  self._logger.warning("extra_headers in embedding_params was not a dict, overwriting.")
                  embedding_params["extra_headers"] = {}

             # Safely update the dictionary
             embedding_params["extra_headers"].update(extra_headers.copy())

        return embedding_params


    def _create_embedding(
        self, embedding_kwargs: EmbeddingCreateParams
    ) -> Awaitable[CreateEmbeddingResponse]:
        @self._retry_api_request
        async def _create_embedding_with_retry() -> CreateEmbeddingResponse:
            # Call the async client method
            response = await self._client.embeddings.create(**embedding_kwargs)
            return response
        # Return the awaitable coroutine
        return _create_embedding_with_retry()


def format_function_def_for_openai(func: CompletionModelFunction) -> FunctionDefinition:
    """Returns an OpenAI-consumable function definition"""
    return {
        "name": func.name,
        "description": func.description,
        "parameters": {
            "type": "object",
            "properties": {name: param.to_dict() for name, param in func.parameters.items()},
            "required": [name for name, param in func.parameters.items() if param.required],
        },
    }