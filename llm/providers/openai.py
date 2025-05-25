import enum
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, ParamSpec, TypeVar, cast

import tenacity
import tiktoken
import yaml
from openai._exceptions import APIStatusError, RateLimitError
from openai.types import EmbeddingCreateParams
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    CompletionCreateParams,
)
from pydantic import SecretStr

from forge.json.parsing import json_loads
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema

from ._openai_base import BaseOpenAIChatProvider, BaseOpenAIEmbeddingProvider
from .schema import (
    AssistantToolCall,
    AssistantToolCallDict,
    ChatMessage,
    ChatModelInfo,
    CompletionModelFunction,
    Embedding,
    EmbeddingModelInfo,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")

logger = logging.getLogger(__name__)

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]


class OpenAIModelName(str, enum.Enum):
    EMBEDDING_v2 = "text-embedding-ada-002"
    EMBEDDING_v3_S = "text-embedding-3-small"
    EMBEDDING_v3_L = "text-embedding-3-large"

    GPT3_v1 = "gpt-3.5-turbo-0301"
    GPT3_v2 = "gpt-3.5-turbo-0613"
    GPT3_v2_16k = "gpt-3.5-turbo-16k-0613"
    GPT3_v3 = "gpt-3.5-turbo-1106"
    GPT3_v4 = "gpt-3.5-turbo-0125"

    # Rolling / alias references for GPT-3.5
    GPT3_ROLLING = "gpt-3.5-turbo"       # rolling base
    GPT3_ROLLING_16k = "gpt-3.5-turbo-16k" # rolling 16k
    GPT3 = GPT3_ROLLING
    GPT3_16k = GPT3_ROLLING_16k

    # GPT-4 variants
    GPT4_v1 = "gpt-4-0314"
    GPT4_v1_32k = "gpt-4-32k-0314"
    GPT4_v2 = "gpt-4-0613"
    GPT4_v2_32k = "gpt-4-32k-0613"
    GPT4_v3 = "gpt-4-1106-preview"
    GPT4_v3_VISION = "gpt-4-1106-vision-preview"
    GPT4_v4 = "gpt-4-0125-preview"
    GPT4_v5 = "gpt-4-turbo-2024-04-09"
    GPT4_ROLLING = "gpt-4"
    GPT4_ROLLING_32k = "gpt-4-32k"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT4_VISION = "gpt-4-vision-preview"
    GPT4_O_v1 = "gpt-4o-2024-05-13"
    GPT4_O_ROLLING = "gpt-4o"

    # Aliases
    GPT4 = GPT4_ROLLING
    GPT4_32k = GPT4_ROLLING_32k
    GPT4_O = GPT4_O_ROLLING

    # CUSTOM ENTRIES
    GPT4_5_PREVIEW = "gpt-4.5-preview-2025-02-27"
    GPT4_1 = "gpt-4.1"


OPEN_AI_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=OpenAIModelName.GPT3_ROLLING,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.001 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_16k,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.003 / 1000,
            completion_token_cost=0.004 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_ROLLING,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.03 / 1000,
            completion_token_cost=0.06 / 1000,
            max_tokens=8191,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_TURBO,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.01 / 1000,
            completion_token_cost=0.03 / 1000,
            max_tokens=128000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_O,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.005 / 1000,
            completion_token_cost=0.015 / 1000,
            max_tokens=128000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_1, 
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.01 / 1000, 
            completion_token_cost=0.03 / 1000,
            max_tokens=128000,  
            has_function_call_api=True,
        ),
    
        ChatModelInfo(
            name=OpenAIModelName.GPT4_5_PREVIEW,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.01 / 1000,
            completion_token_cost=0.03 / 1000,
            max_tokens=128000,
            has_function_call_api=True,
        ),
    ]
}

OPEN_AI_EMBEDDING_MODELS = {
    info.name: info
    for info in [
        EmbeddingModelInfo(
            name=OpenAIModelName.EMBEDDING_v2,
            provider_name=ModelProviderName.OPENAI,
            token_cost=0.0001 / 1000,  # $0.0001 / 1K tokens
            max_tokens=8191,
            embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=OpenAIModelName.EMBEDDING_v3_S,
            provider_name=ModelProviderName.OPENAI,
            token_cost=0.00002 / 1000,  # $0.00002 / 1K tokens
            max_tokens=8191,
            embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=OpenAIModelName.EMBEDDING_v3_L,
            provider_name=ModelProviderName.OPENAI,
            token_cost=0.00013 / 1000,  # $0.00013 / 1K tokens
            max_tokens=8191,
            embedding_dimensions=3072,
        ),
    ]
}

OPEN_AI_MODELS: Mapping[
    OpenAIModelName,
    ChatModelInfo[OpenAIModelName] | EmbeddingModelInfo[OpenAIModelName],
] = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


class OpenAICredentials(ModelProviderCredentials):
    """Credentials for OpenAI."""
    api_key: SecretStr = UserConfigurable(from_env="OPENAI_API_KEY")  # type: ignore
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="OPENAI_API_BASE_URL"
    )
    organization: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="OPENAI_ORGANIZATION"
    )

    api_type: Optional[SecretStr] = UserConfigurable(
        default=None,
        from_env=lambda: cast(
            SecretStr | None,
            (
                "azure"
                if os.getenv("USE_AZURE") == "True"
                else os.getenv("OPENAI_API_TYPE")
            ),
        ),
    )
    api_version: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="OPENAI_API_VERSION"
    )
    azure_endpoint: Optional[SecretStr] = None
    azure_model_to_deploy_id_map: Optional[dict[str, str]] = None

    def get_api_access_kwargs(self) -> dict[str, str]:
        kwargs = {
            k: v.get_secret_value()
            for k, v in {
                "api_key": self.api_key,
                "base_url": self.api_base,
                "organization": self.organization,
                "api_version": self.api_version,
            }.items()
            if v is not None
        }
        if self.api_type == SecretStr("azure"):
            assert self.azure_endpoint, "Azure endpoint not configured"
            kwargs["azure_endpoint"] = self.azure_endpoint.get_secret_value()
        return kwargs

    def get_model_access_kwargs(self, model: str) -> dict[str, str]:
        kwargs = {"model": model}
        if self.api_type == SecretStr("azure") and model:
            azure_kwargs = self._get_azure_access_kwargs(model)
            kwargs.update(azure_kwargs)
        return kwargs

    def load_azure_config(self, config_file: Path) -> None:
        with open(config_file) as file:
            config_params = yaml.load(file, Loader=yaml.SafeLoader) or {}

        try:
            assert config_params.get(
                "azure_model_map", {}
            ), "Azure model->deployment_id map is empty"
        except AssertionError as e:
            raise ValueError(*e.args)

        self.api_type = config_params.get("azure_api_type", "azure")
        self.api_version = config_params.get("azure_api_version", None)
        self.azure_endpoint = config_params.get("azure_endpoint")
        self.azure_model_to_deploy_id_map = config_params.get("azure_model_map")

    def _get_azure_access_kwargs(self, model: str) -> dict[str, str]:
        """Get the kwargs for the Azure API."""
        if not self.azure_model_to_deploy_id_map:
            raise ValueError("Azure model deployment map not configured")

        if model not in self.azure_model_to_deploy_id_map:
            raise ValueError(f"No Azure deployment ID configured for model '{model}'")
        deployment_id = self.azure_model_to_deploy_id_map[model]

        return {"model": deployment_id}


class OpenAISettings(ModelProviderSettings):
    name: str = "openai"
    description: str = "OpenAI API provider for GPT models"
    credentials: Optional[OpenAICredentials] = None
    budget: Optional[ModelProviderBudget] = None
    configuration: ModelProviderConfiguration = ModelProviderConfiguration()


class OpenAIProvider(
    BaseOpenAIChatProvider[OpenAIModelName, OpenAISettings],
    BaseOpenAIEmbeddingProvider[OpenAIModelName, OpenAISettings],
):
    MODELS = OPEN_AI_MODELS
    CHAT_MODELS = OPEN_AI_CHAT_MODELS
    EMBEDDING_MODELS = OPEN_AI_EMBEDDING_MODELS
    
    default_settings = OpenAISettings()

    def __init__(
        self,
        settings: Optional[OpenAISettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        # Initialize settings if not provided
        settings = settings or self.default_settings.model_copy(deep=True)
        # Initialize OpenAI credentials from environment if not set
        if not settings.credentials:
            settings.credentials = OpenAICredentials.from_env()
            
        # Initialize the parent classes with the settings
        BaseOpenAIChatProvider.__init__(self, settings=settings, logger=logger)
        BaseOpenAIEmbeddingProvider.__init__(self, settings=settings, logger=logger)

    def get_tokenizer(self, model_name: OpenAIModelName) -> ModelTokenizer[int]:
        """Get tokenizer for the specified model
        
        Args:
            model_name: The model name to get a tokenizer for
            
        Returns:
            A tokenizer suitable for the model
        """
        try:
            # Try to get the standard tokenizer for common models
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            # For new models not yet in tiktoken's registry, use a suitable alternative
            if model_name == OpenAIModelName.GPT4_5_PREVIEW:
                # Use cl100k_base encoding which is used for GPT-4 and GPT-3.5 models
                # This is likely the same encoding used by GPT-4.5
                return tiktoken.get_encoding("cl100k_base")
            elif "gpt-4" in model_name:
                # For other GPT-4 variants, use the base GPT-4 tokenizer
                return tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model_name:
                # For GPT-3.5 variants
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default fallback to a common encoding
                logger.warning(f"No specific tokenizer for {model_name}, using cl100k_base as fallback")
                return tiktoken.get_encoding("cl100k_base")

    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: OpenAIModelName,
    ) -> int:
        if isinstance(messages, ChatMessage):
            messages = [messages]
        return self.count_tokens(
            "\n\n".join(f"{m.role.upper()}: {m.content}" for m in messages), model_name
        )

    def _get_chat_completion_args(
        self,
        prompt_messages: list[ChatMessage],
        model: OpenAIModelName,
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
                {"type": "function", "function": self.format_function_def_for_openai(f)}
                for f in functions
            ]
            if len(functions) == 1:
                # force the model to call the only specified function
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": functions[0].name},
                }

        if self._configuration and self._configuration.extra_request_headers:
            extra_headers = self._configuration.extra_request_headers
            # 'extra_headers' is not on CompletionCreateParams, but is on chat.create()
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
            prepped_messages += kwargs["messages"]
            del kwargs["messages"]

        return prepped_messages, kwargs, {}

    def _parse_assistant_tool_calls(
        self, assistant_message: ChatCompletionMessage, **kwargs
    ) -> tuple[list[AssistantToolCall], list[Exception]]:
        tool_calls: list[AssistantToolCall] = []
        parse_errors: list[Exception] = []

        if assistant_message.tool_calls:
            for _tc in assistant_message.tool_calls:
                try:
                    parsed_arguments = json_loads(_tc.function.arguments)
                except Exception as e:
                    err_message = (
                        f"Decoding arguments for {_tc.function.name} failed: "
                        + str(e.args[0])
                    )
                    parse_errors.append(
                        type(e)(err_message, *e.args[1:]).with_traceback(
                            e.__traceback__
                        )
                    )
                    continue

                tool_calls.append(
                    AssistantToolCall(
                        id=_tc.id,
                        type=_tc.type,
                        function=AssistantFunctionCall(
                            name=_tc.function.name,
                            arguments=parsed_arguments,
                        ),
                    )
                )

            # If parsing of all tool calls succeeds in the end, we ignore any issues
            if len(tool_calls) == len(assistant_message.tool_calls):
                parse_errors = []

        return tool_calls, parse_errors

    def _get_embedding_kwargs(
        self, input: str | list[str], model: OpenAIModelName, **kwargs
    ) -> EmbeddingCreateParams:
        kwargs = cast(EmbeddingCreateParams, kwargs)

        kwargs["input"] = input
        kwargs["model"] = model

        if self._configuration and self._configuration.extra_request_headers:
            extra_headers = self._configuration.extra_request_headers
            # 'extra_headers' is not on EmbeddingCreateParams, but is on embeddings.create()
            kwargs["extra_headers"] = kwargs.get("extra_headers", {})
            kwargs["extra_headers"].update(extra_headers.copy())

        return kwargs

    def format_function_def_for_openai(self, function: CompletionModelFunction) -> Any:
        """Returns an OpenAI-consumable function definition"""
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

    def __repr__(self):
        return f"{self.__class__.__name__}()"