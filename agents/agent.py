# autogpt/agents/agent.py
from __future__ import annotations

import inspect # Keep for other potential uses, though not directly used in this agent fixed version much
import logging
import time 
import asyncio 
import orjson 
from typing import TYPE_CHECKING, Any, ClassVar, Optional, List, Dict, Union, AsyncGenerator

# Sentry SDK import
try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None 

# Forge imports
try:
    from forge.agent.base import BaseAgent, BaseAgentConfiguration, BaseAgentSettings
    from forge.agent.protocols import (
        AfterExecute,
        AfterParse,
        CommandProvider,
        DirectiveProvider,
        MessageProvider,
    )
    from forge.command.command import Command
    from forge.components.action_history import (
        ActionHistoryComponent,
        EpisodicActionHistory,
    )
    from forge.components.action_history.action_history import ActionHistoryConfiguration
    from forge.components.code_executor.code_executor import (
        CodeExecutorComponent,
        CodeExecutorConfiguration,
    )
    from forge.components.context.context import AgentContext, ContextComponent
    from forge.components.file_manager import FileManagerComponent
    from forge.components.git_operations import GitOperationsComponent
    from forge.components.image_gen import ImageGeneratorComponent
    from forge.components.system import SystemComponent
    from forge.components.user_interaction import UserInteractionComponent
    from forge.components.watchdog import WatchdogComponent
    from forge.components.web import WebSearchComponent, WebCSEComponent

    # Local components (potentially in project's 'components' folder)
    try:
        from components.google_vision import GoogleVisionComponent # type: ignore
    except ImportError:
        GoogleVisionComponent = None 
        # logger.debug("Optional GoogleVisionComponent not found or import path issue.")
    try:
        from components.vision_search import VisionSearchComponent # type: ignore
    except ImportError:
        VisionSearchComponent = None 
        # logger.debug("Optional VisionSearchComponent not found or import path issue.")

    from forge.file_storage.base import FileStorage
    from forge.llm.prompting.schema import ChatPrompt 
    from forge.llm.prompting.utils import dump_prompt
    from forge.llm.providers import (
        AssistantFunctionCall,
        ChatMessage,
        ChatModelResponse,
        MultiProvider,
        # ChatModel was the problematic import, we now use ChatModelInfo
    )
    from forge.llm.providers.schema import ChatModelInfo, ModelProviderName # <<< IMPORTED ChatModelInfo and ModelProviderName
    from forge.llm.providers.utils import function_specs_from_commands
    from forge.models.action import (
        ActionErrorResult,
        ActionInterruptedByHuman,
        ActionResult,
        ActionSuccessResult,
        ActionProposal, 
    )
    from forge.models.config import Configurable
    from forge.utils.exceptions import (
        AgentException,
        AgentTerminated,
        CommandExecutionError,
        UnknownCommandError,
    )
    from pydantic import Field, PrivateAttr 
    from forge.logging.utils import print_attribute 
    from colorama import Fore, Style 

    from .prompt_strategies.one_shot import (
        OneShotAgentActionProposal,
        OneShotAgentPromptStrategy,
    )

except ImportError as e:
    # This is a critical error, ensure it's visible
    # Using a temporary logger or print if main logger isn't configured yet
    _log_temp = logging.getLogger("AGENT_BOOTSTRAP_IMPORT_ERROR")
    _log_temp.critical(f"CRITICAL IMPORT ERROR in agent.py: {e}. Check forge library paths/dependencies.", exc_info=True)
    raise

AgentConfiguration = BaseAgentConfiguration 

if TYPE_CHECKING:
    from autogpt.app.config import AppConfig 
    # No explicit import for a 'ChatModel' type alias is needed from providers if ChatModelInfo is used directly

logger = logging.getLogger(__name__)

AgentProgressEvent = Dict[str, Any]


class AgentSettings(BaseAgentSettings):
    config: BaseAgentConfiguration = Field(default_factory=BaseAgentConfiguration)
    history: EpisodicActionHistory[OneShotAgentActionProposal] = Field(
        default_factory=EpisodicActionHistory[OneShotAgentActionProposal]
    )
    context: AgentContext = Field(default_factory=AgentContext)


class Agent(BaseAgent[OneShotAgentActionProposal], Configurable[AgentSettings]):
    default_settings: ClassVar[AgentSettings] = AgentSettings(
        name="Agent",
        description="Base class for an AutoGPT agent using a one-shot prompting strategy."
    )
    
    _active_chat_model_info: Optional[ChatModelInfo] = PrivateAttr(default=None) # <<< RENAMED & TYPED

    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: MultiProvider, 
        file_storage: FileStorage,
        app_config: AppConfig,
    ):
        super().__init__(settings) 
        self.app_config = app_config
        self.llm_provider = llm_provider 
        self.event_history = settings.history
        
        # Initialize _active_chat_model_info
        retrieved_model_info: Optional[ChatModelInfo] = None
        if self.app_config and self.app_config.smart_llm:
            target_model_name_str = self.app_config.smart_llm
            try:
                logger.debug(f"Fetching ChatModelInfo for smart_llm: '{target_model_name_str}'...")
                # 1. Get the specific sub-provider (e.g. OpenAIProvider) from MultiProvider
                #    The model_name here should be a key in multi.CHAT_MODELS to find the provider_name
                specific_llm_provider = self.llm_provider.get_model_provider(target_model_name_str) # type: ignore
                
                provider_name_for_log = getattr(specific_llm_provider, 'NAME', type(specific_llm_provider).__name__)
                logger.debug(f"Obtained specific provider '{provider_name_for_log}' for model '{target_model_name_str}'.")

                # 2. Access CHAT_MODELS (or MODELS) on the specific_llm_provider to get ChatModelInfo
                #    These are usually ClassVar dictionaries on providers like OpenAIProvider.
                models_dict_on_provider = None
                if hasattr(specific_llm_provider, 'CHAT_MODELS') and isinstance(specific_llm_provider.CHAT_MODELS, dict):
                    models_dict_on_provider = specific_llm_provider.CHAT_MODELS
                elif hasattr(specific_llm_provider, 'MODELS') and isinstance(specific_llm_provider.MODELS, dict): # Fallback for some structures
                    models_dict_on_provider = specific_llm_provider.MODELS
                
                if models_dict_on_provider and target_model_name_str in models_dict_on_provider:
                    candidate_info = models_dict_on_provider[target_model_name_str]
                    if isinstance(candidate_info, ChatModelInfo):
                        retrieved_model_info = candidate_info
                        logger.info(f"Successfully set active model info: {retrieved_model_info.name} "
                                    f"(Provider: {retrieved_model_info.provider_name}, Max Tokens: {retrieved_model_info.max_tokens})")
                    else:
                        logger.error(f"Model '{target_model_name_str}' found in provider's dictionary, but it's not ChatModelInfo (type: {type(candidate_info)}).")
                else:
                    logger.error(f"Model name '{target_model_name_str}' NOT FOUND in CHAT_MODELS/MODELS of provider '{provider_name_for_log}'.")

            except KeyError: # smart_llm not in MultiProvider.CHAT_MODELS (used to route to get_model_provider)
                logger.error(f"Smart LLM '{target_model_name_str}' not found in MultiProvider's CHAT_MODELS global registry. Cannot determine specific provider.", exc_info=True)
            except AttributeError as e_attr:
                 logger.error(f"AttributeError while getting model info for '{target_model_name_str}': {e_attr}. This might be a structural issue in your Forge LLM providers.", exc_info=True)
            except Exception as e_init_model:
                logger.error(
                    f"Unexpected exception during active model info initialization for '{target_model_name_str}': {e_init_model}",
                    exc_info=True
                )
        
        self._active_chat_model_info = retrieved_model_info

        if not self._active_chat_model_info:
            logger.critical( # Changed to CRITICAL as it's quite severe
                "CRITICAL: Agent's _active_chat_model_info (ChatModelInfo) is NOT SET after initialization. "
                "LLM-dependent operations (proposing actions, counting tokens specifically for the smart_llm) will likely fail or use incorrect defaults. "
                "Check smart_llm configuration and provider definitions."
            )

        # Configure Prompt Strategy
        prompt_strategy_config = OneShotAgentPromptStrategy.default_configuration.model_copy(deep=True)
        if self.state.config: 
            service_provider_name = self._active_chat_model_info.provider_name if self._active_chat_model_info else None
            prompt_strategy_config.use_functions_api = (
                self.state.config.use_functions_api
                and self._active_chat_model_info is not None 
                # Compare with ModelProviderName enum members
                and service_provider_name != ModelProviderName.ANTHROPIC 
            )
        else:
            prompt_strategy_config.use_functions_api = False
            logger.warning("Agent self.state.config not available during __init__ for prompt strategy. Functions API might be misconfigured.")
        self.prompt_strategy = OneShotAgentPromptStrategy(prompt_strategy_config, logger)

        # Initialize Commands and System Component
        self.commands: List[Command] = []
        self.system = SystemComponent()

        # Configure Action History
        default_max_tokens_for_history = 4000 
        max_history_tokens = default_max_tokens_for_history
# ... inside Agent.__init__ a few lines below where self._active_chat_model_info is assigned ...
        if self.state and self.state.config: 
            configured_send_token_limit = self.state.config.send_token_limit
            if isinstance(configured_send_token_limit, int):
                max_history_tokens = configured_send_token_limit
            elif configured_send_token_limit is None: 
                if self._active_chat_model_info and isinstance(self._active_chat_model_info.max_tokens, int):
                    max_history_tokens = self._active_chat_model_info.max_tokens * 3 // 4
                    # --- CORRECTED LOGGING LINE ---
                    logger.debug(f"Calculated max_history_tokens: {max_history_tokens} "
                                 f"(75% of {self._active_chat_model_info.name}'s {self._active_chat_model_info.max_tokens})")
                    # --- END CORRECTION ---
                else:
                    logger.debug(f"Using default_max_tokens_for_history: {default_max_tokens_for_history} as active model info or max_tokens is unavailable.")
        else: # Ensure max_history_tokens has a value if self.state.config is not present
            max_history_tokens = default_max_tokens_for_history
            logger.debug(f"self.state.config not present, using default_max_tokens_for_history: {default_max_tokens_for_history}")

        action_history_llm_name = self._active_chat_model_info.name if self._active_chat_model_info else self.app_config.fast_llm
        action_history_config_obj = ActionHistoryConfiguration(       
            llm_name=action_history_llm_name,
            max_tokens=max_history_tokens
        )
# ...
        self.history = ActionHistoryComponent(
            settings.history,
            # Lambda for token counting, using active model's name or fallback fast_llm
            lambda text_to_count: self.llm_provider.count_tokens(text_to_count, action_history_llm_name),
            self.llm_provider, 
            action_history_config_obj,
        )

        # Initialize other components
        if not self.app_config.noninteractive_mode:
            self.user_interaction = UserInteractionComponent()
        self.file_manager = FileManagerComponent(file_storage, settings)
        code_exec_name = f"{settings.agent_id}_sandbox" if settings.agent_id else "default_sandbox_agent"
        self.code_executor = CodeExecutorComponent(self.file_manager.workspace, CodeExecutorConfiguration(docker_container_name=code_exec_name))
        self.git_ops = GitOperationsComponent()
        self.image_gen = ImageGeneratorComponent(self.file_manager.workspace)
        self.web_search = WebSearchComponent()
        app_data_path_str = str(self.app_config.app_data_dir if hasattr(self.app_config, 'app_data_dir') else './app_data')
        self.web_cse = WebCSEComponent(self.llm_provider, app_data_path_str)
        self.context = ContextComponent(self.file_manager.workspace, settings.context)
        self.watchdog = WatchdogComponent(settings.config, settings.history) 
        
        # Guarded initialization for Vision Components
        self.google_vision = GoogleVisionComponent(self.file_manager.workspace) if GoogleVisionComponent else None
        if self.google_vision: logger.info("GoogleVisionComponent initialized.")
        
        self.vision_search = VisionSearchComponent(self.file_manager.workspace) if VisionSearchComponent else None
        if self.vision_search: logger.info("VisionSearchComponent initialized.")

    async def propose_action(
        self, user_input: Optional[str] = None
    ) -> AsyncGenerator[Union[AgentProgressEvent, ActionProposal], None]:
        if hasattr(self, 'reset_trace') and callable(self.reset_trace): self.reset_trace()

        yield {"type": "progress", "stage": "pre_processing", "message": "Initializing action proposal..."}
        await asyncio.sleep(0.01)

        if user_input:
            # (user input handling as before)
            logger.info(f"Agent User Input: '{user_input}'")
            if hasattr(self.event_history, 'add_event') and callable(self.event_history.add_event):
                self.event_history.add_event(event_type="user_message", data={"content": user_input}) 
                yield {"type": "progress", "stage": "user_input_handling", "message": f"Incorporated user input: '{user_input[:50]}...'"}
                await asyncio.sleep(0.01)
            else:
                logger.warning("Could not add user_input to event_history.")


        yield {"type": "progress", "stage": "directives_compilation", "message": "Gathering directives and constraints..."}
        await asyncio.sleep(0.01)
        # (directive pipeline as before)
        resources = await self.run_pipeline(DirectiveProvider.get_resources)
        constraints = await self.run_pipeline(DirectiveProvider.get_constraints)
        best_practices = await self.run_pipeline(DirectiveProvider.get_best_practices)
        if self.state and self.state.directives: 
            current_directives = self.state.directives.model_copy(deep=True)
            current_directives.resources.extend(r for r in resources if r not in current_directives.resources)
            current_directives.constraints.extend(c for c in constraints if c not in current_directives.constraints)
            current_directives.best_practices.extend(bp for bp in best_practices if bp not in current_directives.best_practices)
            self.state.directives = current_directives
        else:
            raise AgentException("Agent self.state.directives not initialized.")

        yield {"type": "progress", "stage": "commands_loading", "message": "Loading available tools..."}
        await asyncio.sleep(0.01)
        # (command pipeline as before)
        self.commands = await self.run_pipeline(CommandProvider.get_commands)
        self._remove_disabled_commands() 

        yield {"type": "progress", "stage": "history_compilation", "message": "Compiling conversation history..."}
        await asyncio.sleep(0.01)
        # (message pipeline as before)
        messages_for_prompt = await self.run_pipeline(MessageProvider.get_messages)


        yield {"type": "progress", "stage": "prompt_construction", "message": "Constructing prompt for AI."}
        await asyncio.sleep(0.01)
        if not self._active_chat_model_info:
             logger.critical("CRITICAL: _active_chat_model_info is None in propose_action. Cannot proceed.")
             raise AgentException("Agent active chat model/info configuration error.")
        if not (self.state and self.state.task and self.state.ai_profile and self.state.directives):
             raise AgentException("Agent state (task, ai_profile, directives) is not fully initialized for prompt generation.")
        
        # (include_os_info and current_prompt construction as before)
        include_os_info = self.code_executor.config.execute_local_commands if hasattr(self, "code_executor") and self.code_executor.config else False
        current_prompt = self.prompt_strategy.build_prompt(
            messages=messages_for_prompt,
            task=self.state.task,
            ai_profile=self.state.ai_profile,
            ai_directives=self.state.directives,
            commands=function_specs_from_commands(self.commands),
            include_os_info=include_os_info,
        )
        if self.app_config.debug_mode: logger.debug(f"LLM Prompt (first 500 chars):\n{dump_prompt(current_prompt)[:500]}")


        yield {"type": "progress", "stage": "llm_request", "message": "Sending request to AI model..."}
        await asyncio.sleep(0.01) 

        parsed_llm_action_proposal: OneShotAgentActionProposal
        try:
           parsed_llm_action_proposal = await self.complete_and_parse(current_prompt)
        except Exception as e_llm:
            logger.error(f"LLM call or parsing failed critically in propose_action: {e_llm}", exc_info=True)
            yield {"type": "error", "stage": "llm_processing_fatal", "message": f"AI interaction severely failed: {str(e_llm)}"}
            raise AgentException(f"Fatal LLM processing error during action proposal: {str(e_llm)}") from e_llm
        
        yield {"type": "progress", "stage": "llm_response_parsing_complete", "message": "AI's plan interpreted."}
        await asyncio.sleep(0.01)
        
        if self.state and self.state.config and hasattr(self.state.config, 'cycle_count'): 
             self.state.config.cycle_count += 1

        yield parsed_llm_action_proposal # This is the final ActionProposal


    async def complete_and_parse(
        self, prompt: ChatPrompt, exception: Optional[Exception] = None
    ) -> OneShotAgentActionProposal:
        if exception: 
            prompt.messages.append(ChatMessage.system(f"System Note: Error occurred before LLM call: {str(exception)}"))
        
        if not self._active_chat_model_info:
            logger.critical("CRITICAL: _active_chat_model_info is None in complete_and_parse. Cannot make LLM call.")
            raise AgentException("Agent's active model information is not set. Cannot create chat completion.")

        # Call create_chat_completion using the .name from _active_chat_model_info
        llm_response_obj: ChatModelResponse[OneShotAgentActionProposal] = await self.llm_provider.create_chat_completion(
            prompt.messages,
            model_name=self._active_chat_model_info.name, # Crucial: use .name
            completion_parser=self.prompt_strategy.parse_response_content,
            functions=prompt.functions, 
            prefill_response=prompt.prefill_response, 
        )
        parsed_result = llm_response_obj.parsed_result
        await self.run_pipeline(AfterParse.after_parse, parsed_result) # Post-parse hooks
        return parsed_result

    # The methods execute, _execute_tool, do_not_execute, _get_command, 
    # _remove_disabled_commands, find_obscured_commands remain
    # EXACTLY as they were in the "Full Code with Fixes Applied" version.
    # Ensure they use self._active_chat_model_info.name for token counting or model name passing.
    # ... PASTE THOSE METHODS HERE, VERIFYING _active_chat_model_info USAGE ...
    async def execute(
        self,
        proposal: OneShotAgentActionProposal,
        user_feedback: str = "", 
    ) -> AsyncGenerator[Union[AgentProgressEvent, ActionResult], None]:
        if not hasattr(proposal, 'use_tool') or proposal.use_tool is None:
            logger.warning("execute called but no 'use_tool' in proposal.")
            final_result = ActionSuccessResult(outputs="No tool/command was proposed for execution.")
            yield {"type": "progress", "stage": "no_tool_to_execute", "message": "No executable tool in proposal."}
            await asyncio.sleep(0.01)
            yield final_result
            return

        tool_to_execute = proposal.use_tool

        yield {"type": "progress", "stage": "pre_tool_command_refresh", "message": "Refreshing tool list before execution..."}
        await asyncio.sleep(0.01)
        self.commands = await self.run_pipeline(CommandProvider.get_commands)
        self._remove_disabled_commands()

        action_result: ActionResult
        tool_output_data: Any = None
        tool_execution_error: Optional[Exception] = None
        
        try:
            async for item in self._execute_tool(tool_to_execute):
                if isinstance(item, dict) and item.get("type") == "progress":
                    item["stage"] = f"tool_execution_internal/{item.get('stage', 'sub_process')}"
                    yield item
                elif isinstance(item, dict) and item.get("type") == "tool_output":
                    tool_output_data = item["data"]
                    break 
                elif isinstance(item, dict) and item.get("type") == "tool_error":
                    error_message_from_tool = item.get("message", "Tool execution failed with unspecified_erro_message")
                    tool_execution_error = CommandExecutionError(error_message_from_tool) 
                    yield {"type": "error", "stage": "tool_reported_error", "message": error_message_from_tool}
                    break 
                else:
                    logger.warning(f"Tool '{tool_to_execute.name}' yielded an unexpected item type: {type(item)}. Assuming as output.")
                    tool_output_data = item 
                    break

            if tool_execution_error:
                action_result = ActionErrorResult.from_exception(tool_execution_error)
            elif tool_output_data is not None or (tool_output_data is None and not tool_execution_error):
                action_result = ActionSuccessResult(outputs=tool_output_data)
            else:
                unexpected_state_msg = f"Tool '{tool_to_execute.name}' concluded without signaling explicit output or error. Defaulting to error."
                logger.error(unexpected_state_msg)
                action_result = ActionErrorResult(reason=unexpected_state_msg, error=CommandExecutionError(unexpected_state_msg))

        except AgentTerminated:
            logger.info(f"AgentTerminated signal received during execution of tool '{tool_to_execute.name}'.")
            raise
        except AgentException as ag_exc:
            action_result = ActionErrorResult.from_exception(ag_exc)
            logger.warning(f"AgentException in 'execute' wrapper for tool '{tool_to_execute.name}': {ag_exc}", exc_info=True)
            if sentry_sdk: sentry_sdk.capture_exception(ag_exc)
            yield {"type": "error", "stage": "execute_wrapper_agent_exception", "message": str(ag_exc)}
        except Exception as unexp_exc:
            action_result = ActionErrorResult.from_exception(
                CommandExecutionError(f"System error during execution wrapper for {tool_to_execute.name}: {unexp_exc}")
            )
            logger.error(f"Unexpected Exception in 'execute' wrapper for tool '{tool_to_execute.name}': {unexp_exc}", exc_info=True)
            if sentry_sdk: sentry_sdk.capture_exception(unexp_exc)
            yield {"type": "error", "stage": "execute_wrapper_system_error", "message": f"System fault during tool call: {str(unexp_exc)}"}

        if not self._active_chat_model_info: # Check ChatModelInfo instance
             logger.warning("Agent's _active_chat_model_info not set; cannot validate result token length.")
        elif action_result and self.state.config:
            _send_token_limit = self.state.config.send_token_limit
            _limit_for_check = float('inf')

            if isinstance(_send_token_limit, int):
                _limit_for_check = _send_token_limit
            elif _send_token_limit is None and self._active_chat_model_info and isinstance(self._active_chat_model_info.max_tokens, int):
                 _limit_for_check = self._active_chat_model_info.max_tokens * 3 // 4 # Using max_tokens from ChatModelInfo
            
            if _limit_for_check != float('inf'):
                _result_summary_str = str(action_result.summary() if hasattr(action_result, 'summary') else action_result)
                _tokens_in_result = self.llm_provider.count_tokens(_result_summary_str, self._active_chat_model_info.name) # Using .name
                
                if _tokens_in_result > _limit_for_check / 3:
                    _warning_msg = f"Output from tool '{tool_to_execute.name}' is large: {_tokens_in_result} tokens (limit context based on {_limit_for_check})."
                    logger.warning(_warning_msg)
                    yield {"type": "progress", "stage": "result_token_length_check", "message": _warning_msg, "level": "warning"}
        
        yield {"type": "progress", "stage": "post_tool_action_pipeline", "message": "Finalizing executed action lifecycle."}
        await asyncio.sleep(0.01)
        await self.run_pipeline(AfterExecute.after_execute, action_result)

        if hasattr(self, 'trace') and self.trace is not None: logger.debug("Full Execution Trace:\n" + "\n".join(map(str, self.trace)))
        yield action_result


    async def _execute_tool(
        self, tool_call: AssistantFunctionCall
    ) -> AsyncGenerator[Dict[str, Any], None]: 
        if not hasattr(tool_call, 'name') or not tool_call.name:
             raise AgentException("Tool call object is invalid: 'name' attribute is missing.")
        tool_arguments = tool_call.arguments or {} 

        try:
            command_to_run = self._get_command(tool_call.name)
        except UnknownCommandError as e_cmd_not_found:
            logger.error(f"Command '{tool_call.name}' not found: {e_cmd_not_found}", exc_info=True)
            yield {"type": "tool_error", "message": str(e_cmd_not_found), "is_agent_exception": True, "reason": "command_not_found"}
            return 

        yield {
            "type": "progress", "stage": "tool_execution_started",
            "message": f"Initiating tool: {tool_call.name}",
            "details": {"tool": tool_call.name, "arguments": tool_arguments}
        }
        await asyncio.sleep(0.05) 
        
        try:
            command_output_or_generator = command_to_run(**tool_arguments)
            
            tool_final_output: Any = None 
            if inspect.isasyncgen(command_output_or_generator): 
                logger.debug(f"Tool '{tool_call.name}' is an async generator. Iterating...")
                async for item_from_gen in command_output_or_generator:
                    if isinstance(item_from_gen, dict) and item_from_gen.get("type") == "progress": 
                        item_from_gen["stage"] = f"tool_internal_progress/{tool_call.name}/{item_from_gen.get('stage', 'step')}"
                        yield item_from_gen 
                    else: 
                        tool_final_output = item_from_gen 
            elif inspect.isawaitable(command_output_or_generator): 
                tool_final_output = await command_output_or_generator
            else: 
                tool_final_output = command_output_or_generator

            yield {
                "type": "progress", "stage": "tool_execution_finished",
                "message": f"Tool {tool_call.name} completed its execution.",
                "details": {"output_preview": str(tool_final_output)[:100] if tool_final_output is not None else "No output (None)"}
            }
            await asyncio.sleep(0.01)
            yield {"type": "tool_output", "data": tool_final_output} 

        except AgentException as e_agent_in_tool: 
            error_message = f"Known AgentException from tool '{tool_call.name}': {e_agent_in_tool}"
            logger.warning(error_message, exc_info=self.app_config.debug_mode if hasattr(self.app_config, 'debug_mode') else False)
            yield {"type": "tool_error", "message": error_message, "is_agent_exception": True}
        except Exception as e_unexpected_in_tool: 
            error_message = f"Unexpected system error during execution of tool '{tool_call.name}': {e_unexpected_in_tool}"
            logger.error(error_message, exc_info=True) 
            yield {"type": "tool_error", "message": error_message, "is_agent_exception": False}


    async def do_not_execute(
        self, denied_proposal: OneShotAgentActionProposal, user_feedback: str
    ) -> AsyncGenerator[Union[AgentProgressEvent, ActionResult], None]:
        yield {"type": "progress", "stage": "action_denial_initiated", "message": "User denied action. Processing feedback given..."}
        await asyncio.sleep(0.01)
        action_interrupted_result = ActionInterruptedByHuman(feedback=user_feedback)
        await self.run_pipeline(AfterExecute.after_execute, action_interrupted_result) 
        if hasattr(self, 'trace') and self.trace: 
            logger.debug("DNX (Do Not Execute) Action Trace:\n" + "\n".join(map(str, self.trace)))
        yield {"type": "progress", "stage": "action_denial_completed", "message": "Feedback from denial processed and recorded."}
        await asyncio.sleep(0.01)
        yield action_interrupted_result 


    def _get_command(self, command_name: str) -> Command:
        if not hasattr(self, 'commands') or not self.commands:
            raise AgentException("Command list is empty or not initialized. Unable to find command.")
        for cmd_instance in reversed(self.commands): 
             if hasattr(cmd_instance, 'names') and command_name in cmd_instance.names:
                return cmd_instance 
        raise UnknownCommandError(f"Command '{command_name}' is not recognized or enabled.")


    def _remove_disabled_commands(self) -> None:
        if not hasattr(self, 'commands'): self.commands = []; return
        disabled_cmds_from_config = set()
        if hasattr(self.app_config, 'disabled_commands') and self.app_config.disabled_commands is not None:
            if isinstance(self.app_config.disabled_commands, (list, set, tuple)):
                 disabled_cmds_from_config = set(self.app_config.disabled_commands)
            else:
                logger.warning(f"app_config.disabled_commands type is {type(self.app_config.disabled_commands)}. Expected list/set. No commands disabled.")
        if not disabled_cmds_from_config: return 
        self.commands = [
            cmd_obj for cmd_obj in self.commands
            if hasattr(cmd_obj, 'names') and not any(name in disabled_cmds_from_config for name in cmd_obj.names)
        ]

    def find_obscured_commands(self) -> List[Command]:
        if not hasattr(self, 'commands') or not self.commands: return []
        names_already_registered = set() 
        list_of_obscured_commands = [] 
        for current_command_object in reversed(self.commands): 
             if not hasattr(current_command_object, 'names') or not current_command_object.names: 
                 logger.debug(f"Command object {current_command_object} lacks 'names' attribute. Skipping in obscurity check.")
                 continue
             is_this_command_fully_obscured = True 
             newly_provided_names_by_this_command = []
             for alias_or_name in current_command_object.names:
                 if alias_or_name not in names_already_registered:
                     is_this_command_fully_obscured = False 
                     newly_provided_names_by_this_command.append(alias_or_name) 
             if is_this_command_fully_obscured: 
                if current_command_object.names: 
                    list_of_obscured_commands.append(current_command_object)
             else: 
                   for unique_name_provided in newly_provided_names_by_this_command:
                       names_already_registered.add(unique_name_provided)
        return list(reversed(list_of_obscured_commands))