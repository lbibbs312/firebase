from __future__ import annotations

import logging # Make sure logging is imported to use logger
from typing import Callable, Iterator, Optional, Any as AnyProposal # Use AnyProposal as specified

from pydantic import BaseModel

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import AfterExecute, AfterParse, MessageProvider
from forge.llm.prompting.utils import indent
from forge.llm.providers import ChatMessage, MultiProvider
from forge.llm.providers.multi import ModelName
from forge.llm.providers.openai import OpenAIModelName
from forge.llm.providers.schema import ToolResultMessage

from .model import ActionResult, Episode, EpisodicActionHistory # Removed AnyProposal from here as it's defined above

# Define logger for this module
logger = logging.getLogger(__name__)


class ActionHistoryConfiguration(BaseModel):
    llm_name: ModelName = OpenAIModelName.GPT3 # Default was OpenAIModelName.GPT3 in your original
    """Name of the llm model used to compress the history"""
    max_tokens: int = 1024
    """Maximum number of tokens to use up with generated history messages"""
    spacy_language_model: str = "en_core_web_sm"
    """Language model used for summary chunking using spacy"""
    full_message_count: int = 4
    """Number of latest non-summarized messages to include in the history"""


class ActionHistoryComponent(
    MessageProvider,
    AfterParse[AnyProposal],
    AfterExecute,
    ConfigurableComponent[ActionHistoryConfiguration],
):
    """Keeps track of the event history and provides a summary of the steps."""

    config_class = ActionHistoryConfiguration

    def __init__(
        self,
        event_history: EpisodicActionHistory[AnyProposal],
        count_tokens: Callable[[str], int],
        llm_provider: MultiProvider,
        config: Optional[ActionHistoryConfiguration] = None,
    ) -> None:
        ConfigurableComponent.__init__(self, config)
        self.event_history = event_history
        self.count_tokens = count_tokens
        self.llm_provider = llm_provider

    def get_messages(self) -> Iterator[ChatMessage]:
        messages: list[ChatMessage] = []
        step_summaries: list[str] = []
        tokens: int = 0
        n_episodes = len(self.event_history.episodes)

        # Include a summary for all except a few latest steps
        for i, episode in enumerate(reversed(self.event_history.episodes)):
            # Use full format for a few steps, summary or format for older steps
            if i < self.config.full_message_count:
                # Ensure episode.action and episode.action.raw_message are not None
                if episode.action and episode.action.raw_message:
                    messages.insert(0, episode.action.raw_message)
                    tokens += self.count_tokens(str(messages[0]))  # HACK
                else:
                    logger.warning(f"Episode {n_episodes - i} missing action or raw_message. Skipping for history.")

                if episode.result:
                    result_message = self._make_result_message(episode, episode.result)
                    messages.insert(1, result_message) # Insert after the action message
                    tokens += self.count_tokens(str(result_message))  # HACK
                continue
            elif episode.summary is None:
                step_content = indent(episode.format(), 2).strip()
            else:
                step_content = episode.summary

            step = f"* Step {n_episodes - i}: {step_content}"

            if self.config.max_tokens and self.count_tokens:
                step_tokens = self.count_tokens(step)
                if tokens + step_tokens > self.config.max_tokens:
                    break
                tokens += step_tokens

            step_summaries.insert(0, step)

        if step_summaries:
            step_summaries_fmt = "\n\n".join(step_summaries)
            yield ChatMessage.system(
                f"## Progress on your Task so far\n"
                "Here is a summary of the steps that you have executed so far, "
                "use this as your consideration for determining the next action!\n"
                f"{step_summaries_fmt}"
            )

        yield from messages

    # --- CORRECTED after_parse METHOD ---
    def after_parse(self, result: AnyProposal) -> None:
        """
        Registers the proposed action (result of LLM parsing) into the event history.
        Handles potential duplicate registrations within the same reasoning cycle.
        """
        try:
            self.event_history.register_action(result)
            logger.debug(f"Action registered in history. Proposal: {str(result)[:100]}")
        except ValueError as e:
            if "Action for current cycle already set" in str(e):
                logger.warning(
                    f"Swallowed 'Action for current cycle already set' in ActionHistoryComponent.after_parse. "
                    f"This might indicate a state issue or rapid/duplicate non-streaming calls. Proposal: {str(result)[:100]}"
                )
                # Silently ignore this specific error, as the action is likely already registered for this cycle.
                pass
            else:
                # Re-raise other ValueErrors that are not about duplicate actions.
                logger.error(f"ValueError during action registration: {e}. Proposal: {str(result)[:100]}", exc_info=True)
                raise
        except Exception as other_exception:
            # Catch any other unexpected exceptions during registration
            logger.error(
                f"Unexpected error in ActionHistoryComponent.after_parse during register_action: {other_exception}. Proposal: {str(result)[:100]}",
                exc_info=True
            )
            # Depending on desired robustness, you might choose to re-raise or handle gracefully.
            # For now, re-raising to make sure it's not silently failing on other issues.
            raise
    # --- END OF CORRECTION ---


    async def after_execute(self, result: ActionResult) -> None:
        self.event_history.register_result(result)
        logger.debug(f"Result registered in history: {type(result).__name__} - Summary: {result.summary() if hasattr(result, 'summary') else str(result)[:100]}")
        try:
            await self.event_history.handle_compression(
                self.llm_provider, self.config.llm_name, self.config.spacy_language_model
            )
            logger.debug("Action history compression handled.")
        except Exception as e_compress:
            logger.error(f"Error during action history compression: {e_compress}", exc_info=True)


    @staticmethod
    def _make_result_message(episode: Episode, result: ActionResult) -> ChatMessage:
        # Ensure episode.action, raw_message, and tool_calls exist before accessing them
        has_tool_call = (
            episode.action 
            and episode.action.raw_message 
            and episode.action.raw_message.tool_calls 
            and len(episode.action.raw_message.tool_calls) > 0
        )
        tool_name_from_action = "unknown_tool"
        if episode.action and episode.action.use_tool:
            tool_name_from_action = episode.action.use_tool.name


        if result.status == "success":
            return (
                ToolResultMessage(
                    content=str(result.outputs) if result.outputs is not None else "", # Ensure content is string
                    tool_call_id=episode.action.raw_message.tool_calls[0].id,
                )
                if has_tool_call
                else ChatMessage.user( # Fallback if no direct tool_call_id from raw_message
                    f"{tool_name_from_action} returned: "
                    + (
                        f"```\n{result.outputs}\n```"
                        if "\n" in str(result.outputs)
                        else f"`{result.outputs}`"
                    )
                )
            )
        elif result.status == "error":
            return (
                ToolResultMessage(
                    content=f"{result.reason}\n\n{result.error or ''}".strip(),
                    is_error=True,
                    tool_call_id=episode.action.raw_message.tool_calls[0].id,
                )
                if has_tool_call
                else ChatMessage.user( # Fallback
                    f"{tool_name_from_action} raised an error: ```\n"
                    f"{result.reason}\n"
                    f"{result.error or ''}"
                    "```"
                )
            )
        else: # Typically for feedback like "INTERRUPTED_BY_HUMAN"
            return ChatMessage.user(result.feedback or "Action produced feedback.")


    def _compile_progress( # This method seems unused in the provided get_messages, but keeping it
        self,
        episode_history: list[Episode[AnyProposal]],
        max_tokens: Optional[int] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
    ) -> str:
        if max_tokens and not count_tokens:
            logger.error("_compile_progress requires count_tokens if max_tokens is set.")
            raise ValueError("count_tokens is required if max_tokens is set")

        steps: list[str] = []
        tokens: int = 0
        n_episodes = len(episode_history)

        for i, episode in enumerate(reversed(episode_history)):
            if i < self.config.full_message_count or episode.summary is None:
                step_content = indent(episode.format(), 2).strip()
            else:
                step_content = episode.summary

            step = f"* Step {n_episodes - i}: {step_content}"

            if max_tokens and count_tokens:
                step_tokens = count_tokens(step)
                if tokens + step_tokens > self.config.max_tokens:
                    break
                tokens += step_tokens

            steps.insert(0, step)

        return "\n\n".join(steps)