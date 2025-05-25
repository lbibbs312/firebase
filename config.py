from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()    # ⬅️ ensure os.environ includes .env immediately

import logging
import os
import re
from pathlib import Path
from typing import Optional, Union

import forge
from forge.config.base import BaseConfig
from forge.llm.providers import CHAT_MODELS, ModelName
from forge.llm.providers.openai import OpenAICredentials, OpenAIModelName
from forge.logging.config import LoggingConfig
from forge.models.config import Configurable, UserConfigurable
from pydantic import SecretStr, ValidationInfo, field_validator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(forge.__file__).parent.parent
AZURE_CONFIG_FILE = Path("azure.yaml")

GPT_4_MODEL = OpenAIModelName.GPT4
GPT_3_MODEL = OpenAIModelName.GPT3

class AppConfig(BaseConfig):
    """Default configuration for the Auto-GPT application."""
    name: str = "Auto-GPT configuration"
    description: str = "Default configuration for the Auto-GPT application."
    version: str = "dev"
    agent_protocol_version: Optional[str] = None
    # New field: unique identifier for the agent
    agent_id: str = UserConfigurable(default="default_agent", from_env="AGENT_ID")
    debug_mode: bool = False
    frontend_static_path: Optional[Path] = Path(
        os.environ.get(
            "FRONTEND_STATIC_PATH",
            r"D:\testautogpt - Copy (2)\MyAutoGPT\classic\frontend\build\web"
        )
    )


    class Config:
        env_file = ".env"
    # Application Settings
    project_root: Path = PROJECT_ROOT
    app_data_dir: Path = project_root / "data"
    skip_news: bool = False
    skip_reprompt: bool = True
    authorise_key: str = UserConfigurable(default="y", from_env="AUTHORISE_COMMAND_KEY")
    exit_key: str = UserConfigurable(default="n", from_env="EXIT_KEY")
    noninteractive_mode: bool = False
    logging: LoggingConfig = LoggingConfig()
    component_config_file: Optional[Path] = UserConfigurable(
        default=None, from_env="COMPONENT_CONFIG_FILE"
    )

    # Agent Control Settings
    fast_llm: ModelName = UserConfigurable(
        default=OpenAIModelName.GPT3,
        from_env="FAST_LLM",
    )
    smart_llm: ModelName = UserConfigurable(
       default=OpenAIModelName.GPT4_TURBO,
       from_env="SMART_LLM_MODEL"
    )
    temperature: float = UserConfigurable(default=0, from_env="TEMPERATURE")
    openai_functions: bool = UserConfigurable(
        default=False,
        from_env=lambda: os.getenv("OPENAI_FUNCTIONS", "False") == "True"
    )
    embedding_model: str = UserConfigurable(
        default="text-embedding-3-small", from_env="EMBEDDING_MODEL"
    )

    # Run loop configuration
    continuous_mode: bool = True
    continuous_limit: int = 0
    one_shot_mode: bool = False

    # Commands
    disabled_commands: list[str] = UserConfigurable(
        default_factory=list,
        from_env=lambda: _safe_split(os.getenv("DISABLED_COMMANDS")),
    )
    restrict_to_workspace: bool = UserConfigurable(
        default=False,
        from_env=lambda: os.getenv("RESTRICT_TO_WORKSPACE", "True") == "True",
    )

    # Credentials
    openai_credentials: Optional[OpenAICredentials] = None
    azure_config_file: Optional[Path] = UserConfigurable(
        default=AZURE_CONFIG_FILE, from_env="AZURE_CONFIG_FILE"
    )

    @field_validator("openai_functions")
    def validate_openai_functions(cls, value: bool, info: ValidationInfo):
        if value:
            smart_llm = info.data["smart_llm"]
            assert CHAT_MODELS[smart_llm].has_function_call_api, (
                f"Model {smart_llm} does not support tool calling. "
                "Please disable OPENAI_FUNCTIONS or choose a suitable model."
            )
        return value


class ConfigBuilder(Configurable[AppConfig]):
    default_settings = AppConfig()

    @classmethod
    def build_config_from_env(cls, project_root: Path = PROJECT_ROOT) -> AppConfig:
        """Initialize the Config class"""
        config = cls.build_agent_configuration()
        config.project_root = project_root

        # Make relative paths absolute
        # Note: If FRONTEND_STATIC_PATH can also be relative and needs to be made absolute
        # based on project_root, it should be added to this loop or handled similarly.
        # For now, it's assumed to be an absolute path as provided.
        for k in {"azure_config_file"}:
            # Check if the attribute exists and is not None before trying to join
            attr_value = getattr(config, k, None)
            if attr_value is not None:
                setattr(config, k, project_root / attr_value)


        if (
            config.openai_credentials
            and config.openai_credentials.api_type == SecretStr("azure")
            and (config_file := config.azure_config_file)
        ):
            config.openai_credentials.load_azure_config(config_file)

        return config


async def assert_config_has_required_llm_api_keys(config: AppConfig) -> None:
    """
    Check if API keys (if required) are set for the configured SMART_LLM and FAST_LLM.
    """
    from forge.llm.providers.anthropic import AnthropicModelName
    from forge.llm.providers.groq import GroqModelName
    from pydantic import ValidationError

    if set((config.smart_llm, config.fast_llm)).intersection(AnthropicModelName):
        from forge.llm.providers.anthropic import AnthropicCredentials

        try:
            credentials = AnthropicCredentials.from_env()
        except ValidationError as e:
            if "api_key" in str(e):
                logger.error(
                    "Set your Anthropic API key in .env or as an environment variable"
                )
                logger.info(
                    "For further instructions: "
                    "https://docs.agpt.co/classic/original_autogpt/setup/#anthropic"
                )
            raise ValueError("Anthropic is unavailable: can't load credentials") from e

        key_pattern = r"^sk-ant-api03-[\w\-]{95}"

        if not re.search(key_pattern, credentials.api_key.get_secret_value()):
            logger.warning(
                "Possibly invalid Anthropic API key! "
                f"Configured Anthropic API key does not match pattern '{key_pattern}'. "
                "If this is a valid key, please report this warning to the maintainers."
            )

    if set((config.smart_llm, config.fast_llm)).intersection(GroqModelName):
        from forge.llm.providers.groq import GroqProvider
        from groq import AuthenticationError # type: ignore

        try:
            groq = GroqProvider()
            await groq.get_available_models()
        except ValidationError as e:
            if "api_key" not in str(e):
                raise

            logger.error("Set your Groq API key in .env or as an environment variable")
            logger.info(
                "For further instructions: "
                + "https://docs.agpt.co/classic/original_autogpt/setup/#groq"
            )
            raise ValueError("Groq is unavailable: can't load credentials")
        except AuthenticationError as e:
            logger.error("The Groq API key is invalid!")
            logger.info(
                "For instructions to get and set a new API key: "
                "https://docs.agpt.co/classic/original_autogpt/setup/#groq"
            )
            raise ValueError("Groq is unavailable: invalid API key") from e

    if set((config.smart_llm, config.fast_llm)).intersection(OpenAIModelName):
        from forge.llm.providers.openai import OpenAIProvider
        from openai import AuthenticationError

        try:
            openai = OpenAIProvider()
            await openai.get_available_models()
        except ValidationError as e:
            if "api_key" not in str(e): # Check if the error is specifically about the API key
                raise # If not, re-raise the original error

            logger.error(
                "Set your OpenAI API key in .env or as an environment variable"
            )
            logger.info(
                "For further instructions: "
                "https://docs.agpt.co/classic/original_autogpt/setup/#openai"
            )
            raise ValueError("OpenAI is unavailable: can't load credentials") from e
        except AuthenticationError as e: # Catch authentication errors specifically
            logger.error(f"OpenAI API key is invalid: {e}")
            logger.info(
                "For further instructions: "
                "https://docs.agpt.co/classic/original_autogpt/setup/#openai"
            )
            raise ValueError("OpenAI is unavailable: invalid API key") from e


def _safe_split(s: Union[str, None], sep: str = ",") -> list[str]:
    """Split a string by a separator. Return an empty list if the string is None."""
    if s is None:
        return []
    return s.split(sep)
