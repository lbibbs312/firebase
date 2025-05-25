# autogpt/app/main.py

from __future__ import annotations

import asyncio
import datetime
import enum
import logging
import math
import os
import re
import signal
import sys
import json 
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Optional, Tuple, List, Union, Any

from colorama import Fore, Style
from dotenv import load_dotenv, find_dotenv

# Assuming these are the correct paths for your project structure
from forge.agent_protocol.database import AgentDB 
from forge.components.code_executor.code_executor import (
    is_docker_available,
    we_are_running_in_a_docker_container,
)
from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.file_storage import FileStorageBackendName, get_storage
from forge.llm.providers import MultiProvider
from forge.llm.providers.openai import OpenAIModelName 
from forge.logging.config import configure_logging
from forge.logging.utils import speak
from forge.models.action import ActionInterruptedByHuman, ActionProposal, ActionResult, ActionSuccessResult, ActionErrorResult
from forge.models.utils import ModelWithSummary 
from forge.utils.const import FINISH_COMMAND
from forge.utils.exceptions import AgentTerminated, InvalidAgentResponseError

from autogpt.agent_factory.configurators import configure_agent_with_state, create_agent
from autogpt.agents.agent_manager import AgentManager
# Assuming AgentProgressEvent is a Dict[str, Any] or similar for typing if used
from autogpt.agents.agent import Agent as CoreAutoGPTAgent, AgentProgressEvent # Renamed to CoreAutoGPTAgent for clarity
from autogpt.agents.prompt_strategies.one_shot import AssistantThoughts, OneShotAgentActionProposal

from autogpt.app.config import (
    AppConfig,
    ConfigBuilder,
    assert_config_has_required_llm_api_keys,
)

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent # This is your CoreAutoGPTAgent, type hint for Agent var
    from autogpt.app.agent_protocol_server import AgentProtocolServer

# Assuming these are in autogpt/app/
from .configurator import apply_overrides_to_config
from .input import clean_input
from .setup import apply_overrides_to_ai_settings, interactively_revise_ai_settings
from .utils import (
    coroutine,
    get_legal_warning,
    markdown_to_ansi_style,
    remove_ansi_escape, # Assuming this is in your utils
    print_motd,
    print_git_branch_info,
    print_python_version_info
)

# --- .env Loading (as in your script) ---
SPECIFIC_ENV_PATH = Path(r"D:\testautogpt - Copy (2)\MyAutoGPT\classic\original_autogpt\.env").resolve()
if SPECIFIC_ENV_PATH.is_file():
    if load_dotenv(dotenv_path=SPECIFIC_ENV_PATH, override=True):
        print(f"INFO: Successfully loaded environment variables from: {SPECIFIC_ENV_PATH}")
    else:
        print(f"WARNING: Attempted to load specific .env, but load_dotenv indicated no variables were loaded from: {SPECIFIC_ENV_PATH}")
else:
    print(f"WARNING: Specific .env file not found at: {SPECIFIC_ENV_PATH}. Falling back to default .env discovery or existing environment variables.")
    default_env_path = find_dotenv(".env", usecwd=True) 
    if default_env_path and Path(default_env_path).is_file():
        if load_dotenv(dotenv_path=default_env_path, override=True):
            print(f"INFO: Loaded default .env file from: {default_env_path}")
        else:
            print(f"INFO: Found default .env at {default_env_path}, but no variables loaded.")
    else:
        print("INFO: No default .env file found either in CWD or parent directories.")
# --- End .env Loading ---

logger = logging.getLogger("autogpt.app.main")

def get_time_based_greeting() -> str:
    now = datetime.datetime.now()
    hour = now.hour
    if 5 <= hour < 12: return "Good morning"
    elif 12 <= hour < 17: return "Good afternoon"
    else: return "Good evening"

@coroutine
async def run_auto_gpt(
    continuous: bool = False,
    continuous_limit: Optional[int] = None,
    skip_reprompt: bool = True,
    speak: bool = False, 
    debug: bool = True,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file_format: Optional[str] = None,
    skip_news: bool = True, 
    install_plugin_deps: bool = False,
    override_ai_name: Optional[str] = None,
    override_ai_role: Optional[str] = None,
    resources: Optional[list[str]] = None,
    constraints: Optional[list[str]] = None,
    best_practices: Optional[list[str]] = None,
    override_directives: bool = False,
    component_config_file: Optional[Path] = None,
    task: Optional[str] = None,
):
    config = ConfigBuilder.build_config_from_env()
    
    data_path_for_storage = config.app_data_dir / "autogpt_workspace" 
    data_path_for_storage.mkdir(parents=True, exist_ok=True)
    
    is_local_backend = hasattr(config, 'file_storage_backend') and \
                       config.file_storage_backend == FileStorageBackendName.LOCAL
    
    restrict_to_root_calculated = not is_local_backend or \
                                  (hasattr(config, 'restrict_to_workspace') and config.restrict_to_workspace)
    
    file_storage = get_storage(
        config.file_storage_backend,
        root_path=data_path_for_storage, 
        restrict_to_root=restrict_to_root_calculated,
    )
    file_storage.initialize()

    # This logger instance will be passed to utils
    # It's configured after configure_logging sets up the handlers and levels.
    run_logger = logging.getLogger("autogpt.app.main.run") 

    if speak: 
        if hasattr(config, 'tts_config') and config.tts_config is not None:
            config.tts_config.speak_mode = True
        else:
            run_logger.warning("TTS config not found in AppConfig, 'speak' mode may not function.")

    effective_log_level = log_level
    if not effective_log_level:
        effective_log_level = "DEBUG" if debug else "INFO"
    
    logging_config_obj = getattr(config, 'logging', None)
    tts_config_obj = getattr(config, 'tts_config', None)  

    configure_logging(
        debug=debug, # Corrected from debug_mode
        level=effective_log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        config=logging_config_obj, 
        tts_config=tts_config_obj,  
    )
    
    await assert_config_has_required_llm_api_keys(config)

    # Apply runtime overrides to AppConfig (ensure AppConfig fields exist for these)
    config.continuous_mode = continuous if continuous is not None else config.continuous_mode
    config.continuous_limit = continuous_limit if continuous_limit is not None else config.continuous_limit
    config.skip_reprompt = skip_reprompt if skip_reprompt is not None else config.skip_reprompt
    
    llm_provider = _configure_llm_provider(config)
    agent_manager = AgentManager(file_storage)
    
    agent: Optional[Agent] = None # Use the type hint from TYPE_CHECKING

    new_agent_task_input = task 
    time_greeting = get_time_based_greeting()

    if not new_agent_task_input: 
        if config.skip_reprompt:
            new_agent_task_input = f"{time_greeting}! How can I assist you today?"
            run_logger.info(f"No task provided & skip_reprompt=True. Defaulting task: '{new_agent_task_input}'")
        else:
            user_task_cli_input = ""
            default_ai_name_for_prompt = AIProfile().ai_name 
            while user_task_cli_input.strip() == "":
                # Pass logger to clean_input if it uses it, otherwise remove
                user_task_cli_input = clean_input( 
                    f"{Fore.GREEN}{time_greeting}! I am {default_ai_name_for_prompt}. " 
                    f"What can I help you with today?{Style.RESET_ALL}\nInput: "
                ).strip()
            new_agent_task_input = user_task_cli_input
    else:
        run_logger.info(f"Task for new agent from CLI argument: {new_agent_task_input}")

    ai_profile = AIProfile()
    ai_profile.set_task_description(new_agent_task_input) 
    
    additional_directives = AIDirectives()
    
    apply_overrides_to_ai_settings(
        ai_profile=ai_profile,
        directives=additional_directives,
        override_name=override_ai_name,
        override_role=override_ai_role,
        resources=resources,
        constraints=constraints,
        best_practices=best_practices,
        replace_directives=override_directives,
    )

    if not config.skip_reprompt and not any([override_ai_name, override_ai_role]):
        run_logger.info("Interactively configuring AI profile for new agent.")
        # Pass logger to interactively_revise_ai_settings if it uses it
        await interactively_revise_ai_settings(
            ai_profile=ai_profile, 
            directives=additional_directives,
            app_config=config,
            # logger=run_logger # If needed
        )
        new_agent_task_input = ai_profile.get_task_description()

    agent = create_agent(
        agent_id=agent_manager.generate_id(ai_profile.ai_name),
        task=new_agent_task_input, 
        ai_profile=ai_profile,
        directives=additional_directives,
        app_config=config, 
        file_storage=file_storage, 
        llm_provider=llm_provider,
    )
    run_logger.info(f"New agent '{agent.state.ai_profile.ai_name}' created with ID '{agent.state.agent_id}'.") # type: ignore

    _config_file_path = component_config_file or getattr(config, 'component_config_file', None)
    if _config_file_path:
        _config_file_path = Path(_config_file_path)
        if _config_file_path.exists() and _config_file_path.is_file():
            run_logger.info(f"Loading component configurations from: {_config_file_path}")
            try:
                if hasattr(agent, 'load_component_configs') and callable(agent.load_component_configs):
                    agent.load_component_configs(_config_file_path.read_text())
                else:
                    run_logger.error(f"Agent object type '{type(agent)}' does not have 'load_component_configs' method.")
            except Exception as e:
                run_logger.error(f"Failed to load component configs from '{_config_file_path}': {e}", exc_info=True)
        else:
            run_logger.warning(f"Component configuration file specified but not found or not a file: {_config_file_path}")

    try:
        # --- MODIFIED CALLS TO UTILS ---
        if not skip_news: print_motd(logger=run_logger) 
        
        # Check if config has these attributes before calling, and pass logger
        if not config.continuous_mode and hasattr(config, 'show_git_branch_info') and config.show_git_branch_info:
             print_git_branch_info(logger=run_logger)
        if hasattr(config, 'show_python_version_info') and config.show_python_version_info:
             print_python_version_info(logger=run_logger)
        
        if get_legal_warning(): # Assuming this doesn't need logger
            print(get_legal_warning())
        # --- END MODIFIED CALLS ---

        await run_interaction_loop_conversational(agent, agent_manager)
    except AgentTerminated:
        run_logger.info(f"Agent '{agent.state.ai_profile.ai_name}' session concluded from run_auto_gpt context.") # type: ignore
    except Exception as e:
        run_logger.critical(f"Critical unhandled error in agent execution: {e}", exc_info=True)

def _configure_llm_provider(config: AppConfig) -> MultiProvider:
    # ... (Implementation as provided by you, assuming it's correct)
    multi_provider = MultiProvider() 
    models_to_check: List[str] = []
    if hasattr(config, 'smart_llm') and config.smart_llm: models_to_check.append(config.smart_llm) 
    if hasattr(config, 'fast_llm') and config.fast_llm and config.fast_llm != getattr(config, 'smart_llm', None): 
        models_to_check.append(config.fast_llm) 
    unique_models_to_check = list(set(m for m in models_to_check if m))
    for model_name_str in unique_models_to_check:
        try:
            specific_provider = multi_provider.get_model_provider(model_name_str) # type: ignore
            logger.debug(f"Provider for model '{model_name_str}' ({type(specific_provider).__name__}) accessed via MultiProvider.")
        except Exception as e:
            logger.warning(f"Could not prime or access provider for model '{model_name_str}': {e}. Check LLM configurations and API keys.")
    return multi_provider

def _get_cycle_budget(continuous_mode: bool, continuous_limit: Optional[int]) -> Union[int, float]:
    # ... (Implementation as provided by you)
    if continuous_mode:
        if continuous_limit is not None and continuous_limit > 0:
            return continuous_limit
        else: 
            return math.inf
    else: 
        return 1

class UserFeedback(str, enum.Enum):
    # ... (As provided by you)
    AUTHORIZE = "GENERATE NEXT COMMAND JSON" 
    EXIT = "EXIT"
    TEXT = "TEXT" 

class UserFeedbackOptions(str, enum.Enum):
    # ... (As provided by you)
    AUTHORIZE_PROPOSED_COMMAND = "AUTHORIZE_PROPOSED_COMMAND"
    AUTHORIZE_CONTINUOUS_COMMANDS = "AUTHORIZE_CONTINUOUS_COMMANDS"
    PROVIDE_GENERAL_TEXT_INPUT_OR_NEW_TASK = "PROVIDE_GENERAL_TEXT_INPUT_OR_NEW_TASK"
    USER_REQUESTED_EXIT = "USER_REQUESTED_EXIT"

async def get_user_feedback_conversational(
    config: AppConfig, ai_profile: AIProfile, action_is_proposed: bool,
) -> Tuple[UserFeedbackOptions, str, Optional[int]]:
    # ... (Implementation as provided by you, with logger fixed)
    feedback_logger = logging.getLogger("UserFeedback") # Use specific logger
    prompt_lines: List[str] = []
    if action_is_proposed:
        prompt_lines.extend([
            f"{Fore.YELLOW}{ai_profile.ai_name} has proposed an action (details printed above).{Style.RESET_ALL}",
            f"  Enter '{config.authorise_key}' to authorise this action.",
            f"  Enter '{config.authorise_key} -N' for N continuous commands (e.g., '{config.authorise_key} -3' for 3, or '{config.authorise_key} -inf' for infinite).",
            f"  Enter '{config.exit_key}' to exit the application.",
            f"  Alternatively, type your response, new instructions, or a question for {ai_profile.ai_name}:",
        ])
    else: 
        prompt_lines.extend([
            f"{Fore.YELLOW}{ai_profile.ai_name} is awaiting your instructions.{Style.RESET_ALL}", 
            f"  Type your command, question, or task.",
            f"  Enter '{config.exit_key}' to exit the application.",
        ])
    print("\n" + "\n".join(prompt_lines))
    
    user_feedback_type: Optional[UserFeedbackOptions] = None
    user_input_text: str = ""
    num_continuous_cycles: Optional[int] = None

    while user_feedback_type is None:
        console_input = clean_input(f"{Fore.MAGENTA}Your Input:{Style.RESET_ALL} ").strip()
        if not console_input:
            feedback_logger.warning("Input cannot be empty. Please provide instructions, authorize, or exit.")
            continue

        if console_input.lower() == config.exit_key.lower():
            user_feedback_type = UserFeedbackOptions.USER_REQUESTED_EXIT
            break

        if action_is_proposed:
            if console_input.lower() == config.authorise_key.lower():
                user_feedback_type = UserFeedbackOptions.AUTHORIZE_PROPOSED_COMMAND
            elif console_input.lower().startswith(f"{config.authorise_key.lower()} -"):
                try:
                    num_str_part = console_input.split(" ", 1)[1].strip() 
                    if num_str_part.lower() == 'inf': 
                        num_continuous_cycles = int(math.inf)
                    else: 
                        parsed_num = int(num_str_part)
                        num_continuous_cycles = abs(parsed_num) if parsed_num != 0 else 1
                    user_feedback_type = UserFeedbackOptions.AUTHORIZE_CONTINUOUS_COMMANDS
                except (ValueError, IndexError):
                    feedback_logger.warning(f"Invalid continuous command format ('{console_input}'). Treating as text input.")
                    user_feedback_type = UserFeedbackOptions.PROVIDE_GENERAL_TEXT_INPUT_OR_NEW_TASK
                    user_input_text = console_input 
            else: 
                user_feedback_type = UserFeedbackOptions.PROVIDE_GENERAL_TEXT_INPUT_OR_NEW_TASK
                user_input_text = console_input
        else: 
            user_feedback_type = UserFeedbackOptions.PROVIDE_GENERAL_TEXT_INPUT_OR_NEW_TASK
            user_input_text = console_input
            
    return user_feedback_type, user_input_text, num_continuous_cycles


async def run_interaction_loop_conversational(agent: Agent, agent_manager: AgentManager) -> None:
    # ... (Implementation with fixes for async for on propose_action and do_not_execute,
    #      and execute, as provided in my response #15)
    app_config = agent.app_config
    ai_profile = agent.state.ai_profile
    loop_logger = logging.getLogger("InteractionLoop")
    continuous_cycles_approved_by_user: Union[int, float] = 0.0
    stop_reason: Optional[AgentTerminated] = None
    
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    def graceful_agent_interrupt(signum: int, frame: Optional[FrameType]) -> None:
        nonlocal stop_reason, continuous_cycles_approved_by_user
        loop_logger.info(f"{Fore.YELLOW}User interrupt (Ctrl+C) detected. Graceful shutdown initiated...{Style.RESET_ALL}")
        if stop_reason: 
            loop_logger.warning("Multiple interrupts received. Forcing exit.")
            sys.exit(1)
        continuous_cycles_approved_by_user = 0 
        stop_reason = AgentTerminated("User interrupt signal received.")
    signal.signal(signal.SIGINT, graceful_agent_interrupt)
    
    agent_needs_initial_task = not ai_profile.get_task_description()
    consecutive_proposal_failures = 0
    MAX_CONSECUTIVE_PROPOSAL_FAILURES = 3
    user_input_for_next_cycle: Optional[str] = ai_profile.get_task_description() if agent_needs_initial_task else None

    try:
        while True:
            if stop_reason: raise stop_reason

            current_task_display = ai_profile.get_task_description() or "Awaiting your instructions"
            loop_logger.info(f"\n{Fore.BLUE}--- Agent: {ai_profile.ai_name} | Current Focus: {current_task_display} ---{Style.RESET_ALL}")
            
            if continuous_cycles_approved_by_user > 0:
                 loop_logger.info(f"{Fore.GREEN}(Continuous execution: {continuous_cycles_approved_by_user if continuous_cycles_approved_by_user != math.inf else '∞'} actions remaining){Style.RESET_ALL}")
            
            action_proposal: Optional[OneShotAgentActionProposal] = None
            
            if agent_needs_initial_task and not user_input_for_next_cycle:
                pass 
            else:
                should_propose_action_now = True
                last_episode = agent.event_history.current_episode
                if last_episode: 
                    action_data = getattr(last_episode, 'action', None)
                    tool_data = getattr(action_data, 'use_tool', None) if action_data else None
                    tool_name = getattr(tool_data, 'name', None) if tool_data else None
                    last_result_obj = getattr(last_episode, 'result', None)
                    if tool_name == FINISH_COMMAND and isinstance(last_result_obj, ActionSuccessResult):
                        loop_logger.info(f"{Fore.GREEN}{ai_profile.ai_name} previously executed '{FINISH_COMMAND}' and considered its task complete. Awaiting new instructions.{Style.RESET_ALL}")
                        should_propose_action_now = False 

                if should_propose_action_now:
                    try:
                        loop_logger.info(f"{Fore.CYAN}J.A.R.V.I.S. is processing... (Input: '{user_input_for_next_cycle[:50] if user_input_for_next_cycle else 'Continuing'}'){Style.RESET_ALL}")
                        async for event_data in agent.propose_action(user_input=user_input_for_next_cycle): # Iterate here
                            if isinstance(event_data, OneShotAgentActionProposal):
                                action_proposal = event_data
                                break 
                            elif isinstance(event_data, dict):
                                event_type = event_data.get('type')
                                message = event_data.get('message', str(event_data))
                                if event_type == 'progress': loop_logger.info(f"Progress: {event_data.get('stage', '')} - {message}")
                                elif event_type == 'thought_update':
                                    thoughts_dict = event_data.get('thoughts', {})
                                    if isinstance(thoughts_dict, dict) and thoughts_dict.get('text'):
                                        loop_logger.info(f"Thoughts: {thoughts_dict.get('text')}")
                                    elif isinstance(thoughts_dict, dict) and thoughts_dict.get('speak'):
                                         loop_logger.info(f"Interim Speak: {thoughts_dict.get('speak')}")
                                else: loop_logger.debug(f"Agent event: {event_type} - {message}")
                            else: loop_logger.warning(f"CLI: Unhandled event from propose_action: {type(event_data)}")
                        
                        if not action_proposal: raise InvalidAgentResponseError("Agent did not yield a final action proposal.")
                        consecutive_proposal_failures = 0
                    except InvalidAgentResponseError as e:
                        loop_logger.error(f"{Fore.RED}Agent Response Error: {e}{Style.RESET_ALL}")
                        consecutive_proposal_failures += 1
                        if consecutive_proposal_failures >= MAX_CONSECUTIVE_PROPOSAL_FAILURES:
                            loop_logger.error(f"Agent failed {MAX_CONSECUTIVE_PROPOSAL_FAILURES} times. Terminating."); raise AgentTerminated("Too many proposal failures.")
                        print(f"{Fore.RED}J.A.R.V.I.S. is having difficulty. Please try rephrasing or provide new instructions.{Style.RESET_ALL}")
                        action_proposal = None 
                    except AgentTerminated: raise
            
            user_input_for_next_cycle = None 

            if action_proposal: 
                update_user(ai_profile, action_proposal, app_config.tts_config.speak_mode) 
            
            user_feedback_opt, user_text_from_prompt, num_cont_from_user = await get_user_feedback_conversational(
                app_config, ai_profile, bool(action_proposal)
            )
            
            if user_feedback_opt == UserFeedbackOptions.USER_REQUESTED_EXIT:
                raise AgentTerminated("User requested exit via input prompt.")
            
            agent_needs_initial_task = False 

            if user_feedback_opt == UserFeedbackOptions.PROVIDE_GENERAL_TEXT_INPUT_OR_NEW_TASK:
                loop_logger.info(f"User input: '{user_text_from_prompt}'. This will be the new focus.")
                ai_profile.set_task_description(user_text_from_prompt)
                
                if action_proposal: 
                    loop_logger.info(f"Overriding previous agent proposal due to new user input: '{user_text_from_prompt}'")
                    async for dnx_event in agent.do_not_execute(action_proposal, user_text_from_prompt): # Iterate here
                        if isinstance(dnx_event, ActionInterruptedByHuman):
                            loop_logger.info(f"Agent logged action interruption: {dnx_event.feedback}")
                            break
                        elif isinstance(dnx_event, dict) and dnx_event.get("type") == "progress":
                            loop_logger.debug(f"DNX Progress: {dnx_event.get('message')}")
                
                user_input_for_next_cycle = user_text_from_prompt 
                continuous_cycles_approved_by_user = 0 
                continue 

            elif action_proposal and (user_feedback_opt == UserFeedbackOptions.AUTHORIZE_PROPOSED_COMMAND or \
                                      user_feedback_opt == UserFeedbackOptions.AUTHORIZE_CONTINUOUS_COMMANDS):
                
                if user_feedback_opt == UserFeedbackOptions.AUTHORIZE_CONTINUOUS_COMMANDS and num_cont_from_user is not None:
                    continuous_cycles_approved_by_user = num_cont_from_user
                else: 
                    continuous_cycles_approved_by_user = 1.0 
                
                if app_config.continuous_mode:
                    global_cont_budget = _get_cycle_budget(True, app_config.continuous_limit)
                    if continuous_cycles_approved_by_user != math.inf :
                        if global_cont_budget != math.inf: 
                           continuous_cycles_approved_by_user = min(continuous_cycles_approved_by_user, global_cont_budget) if continuous_cycles_approved_by_user > 1 else global_cont_budget
                    else: 
                        continuous_cycles_approved_by_user = global_cont_budget 
                    loop_logger.info(f"Continuous mode active. Effective cycles: {continuous_cycles_approved_by_user if continuous_cycles_approved_by_user != math.inf else '∞'}")

                current_proposal_to_execute = action_proposal 
                while current_proposal_to_execute and continuous_cycles_approved_by_user > 0:
                    if stop_reason: raise stop_reason
                    
                    loop_logger.info(
                        f"Executing action. Continuous cycles remaining: "
                        f"{continuous_cycles_approved_by_user if continuous_cycles_approved_by_user != math.inf else '∞'}"
                    )

                    tool_to_exec = getattr(current_proposal_to_execute, 'use_tool', None)
                    tool_name_to_exec = getattr(tool_to_exec, 'name', None) if tool_to_exec else None

                    if tool_name_to_exec == FINISH_COMMAND:
                        tool_args_exec = getattr(tool_to_exec, 'arguments', {}) if tool_to_exec else {}
                        finish_reason_exec = tool_args_exec.get('reason', 'Task marked complete.') if isinstance(tool_args_exec, dict) else 'Task marked complete.'
                        loop_logger.info(f"{Fore.GREEN}Agent initiated '{FINISH_COMMAND}'. Reason: {finish_reason_exec}{Style.RESET_ALL}")
                        async for _ in agent.execute(proposal=current_proposal_to_execute): pass # Iterate here
                        raise AgentTerminated(f"{FINISH_COMMAND}: {finish_reason_exec}")

                    action_res: Optional[ActionResult] = None
                    async for exec_event in agent.execute(proposal=current_proposal_to_execute): # Iterate here
                        if isinstance(exec_event, ActionResult): action_res = exec_event; break
                        elif isinstance(exec_event, dict) and exec_event.get('type') == 'progress':
                            loop_logger.info(f"Exec Progress: {exec_event.get('stage','')} - {exec_event.get('message')}")
                    
                    if not action_res: loop_logger.error("Tool exec did not yield ActionResult."); continuous_cycles_approved_by_user = 0; break
                    if isinstance(action_res, ActionErrorResult): loop_logger.error(f"{Fore.RED}Action Error: {action_res.reason}{Style.RESET_ALL}")
                    
                    if continuous_cycles_approved_by_user != math.inf: continuous_cycles_approved_by_user -= 1
                    if continuous_cycles_approved_by_user == 0: loop_logger.info("Continuous budget done or single auth."); break
                    
                    loop_logger.info(f"{Fore.CYAN}Proposing next for continuous...{Style.RESET_ALL}"); current_proposal_to_execute = None
                    try:
                        async for evt_data in agent.propose_action(user_input=None): # Iterate here
                            if isinstance(evt_data, OneShotAgentActionProposal): current_proposal_to_execute = evt_data; break
                        if not current_proposal_to_execute: raise InvalidAgentResponseError("No proposal for continuous.")
                        update_user(ai_profile, current_proposal_to_execute, app_config.tts_config.speak_mode)
                        consecutive_proposal_failures = 0
                    except InvalidAgentResponseError as e:
                        loop_logger.error(f"{Fore.RED}Agent error in continuous: {e}{Style.RESET_ALL}"); continuous_cycles_approved_by_user = 0; break
                    except AgentTerminated: raise 
            
            elif not action_proposal and user_feedback_opt in [UserFeedbackOptions.AUTHORIZE_PROPOSED_COMMAND, UserFeedbackOptions.AUTHORIZE_CONTINUOUS_COMMANDS]:
                loop_logger.warning("User tried to authorize, but no valid action was proposed. Please provide new instructions.")
                user_input_for_next_cycle = None

    except AgentTerminated as e: 
        stop_reason = e
        loop_logger.info(f"Interaction loop terminated: {e.args[0] if e.args else 'Session ended.'}")
    except Exception as e: 
        loop_logger.critical(f"CRITICAL UNHANDLED EXCEPTION in interaction loop: {e}", exc_info=True)
        stop_reason = AgentTerminated(f"Critical error in agent loop: {e}")
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if agent:
            agent_name_final = getattr(ai_profile, 'ai_name', "UnknownAgent") 
            agent_id_final = getattr(agent.state, 'agent_id', "UnknownID")
            logger.info(f"Session for '{agent_name_final}' (ID: {agent_id_final}) ended.")
            
            should_skip_save = getattr(app_config, 'skip_state_save_on_exit', False)

            if not should_skip_save:
                save_as_id_input = clean_input(
                    f"Save agent state as '{agent_id_final}'? (Press Enter for this ID, 'NONE' not to save, or new ID):",
                    default=str(agent_id_final) 
                ).strip()

                if save_as_id_input.upper() == 'NONE': 
                    logger.info("Agent state not saved by user choice.")
                else:
                    final_save_id = save_as_id_input if save_as_id_input else str(agent_id_final)
                    try: 
                        await agent_manager.save_agent_state(agent, agent_id_override=final_save_id) 
                        logger.info(f"Agent state for ID '{final_save_id}' saved successfully.") 
                    except Exception as se: 
                        logger.error(f"Failed to save agent state for ID '{final_save_id}': {se}", exc_info=True)
            else:
                logger.info("Skipping agent state save as per configuration (skip_state_save_on_exit=True).")

        final_msg_to_user = stop_reason.args[0] if stop_reason and stop_reason.args else "Session concluded."
        print(f"\n{Fore.YELLOW}>>> AutoGPT J.A.R.V.I.S. session: {final_msg_to_user} <<<{Style.RESET_ALL}")

def update_user(
    ai_profile: AIProfile, action_proposal: ActionProposal, speak_mode: bool = False,
) -> None:
    # ... (Implementation as provided by you, ensuring it's robust)
    logger_update = logging.getLogger("update_user")
    thoughts_obj = getattr(action_proposal, 'thoughts', None)
    use_tool_obj = getattr(action_proposal, 'use_tool', None)
    
    spoken_text_content = ""

    if isinstance(thoughts_obj, AssistantThoughts):
        spoken_text_content = getattr(thoughts_obj, 'speak', "") or ""
    elif isinstance(thoughts_obj, str):
        spoken_text_content = thoughts_obj
    elif isinstance(thoughts_obj, ModelWithSummary):
        spoken_text_content = thoughts_obj.summary()
    
    if speak_mode and spoken_text_content:
        try: speak(remove_ansi_escape(spoken_text_content)) 
        except Exception as e: logger_update.error(f"TTS failed for main speak: {e}")

    if speak_mode and use_tool_obj:
        tool_name_to_speak = getattr(use_tool_obj, 'name', "an unspecified tool")
        try: speak(f"I will use the tool: {remove_ansi_escape(tool_name_to_speak)}")
        except Exception as e: logger_update.error(f"TTS failed for tool intention: {e}")

    if spoken_text_content:
        print(f"\n{Fore.CYAN}{ai_profile.ai_name.upper()} SAID:{Style.RESET_ALL} {markdown_to_ansi_style(remove_ansi_escape(spoken_text_content))}")

    if thoughts_obj: 
        print_assistant_thoughts(ai_name=ai_profile.ai_name, thoughts=thoughts_obj) 

    if use_tool_obj:
        tool_name_print = getattr(use_tool_obj, 'name', "N/A")
        tool_args_print = getattr(use_tool_obj, 'arguments', {})
        safe_name_print = remove_ansi_escape(tool_name_print)
        try:
            args_disp_print = json.dumps(tool_args_print, indent=2, ensure_ascii=False) if isinstance(tool_args_print, dict) else str(tool_args_print)
        except TypeError: args_disp_print = str(tool_args_print)
        print(f"{Fore.CYAN}NEXT ACTION:{Style.RESET_ALL} COMMAND = {Fore.CYAN}{safe_name_print}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{args_disp_print}{Style.RESET_ALL}")


def print_assistant_thoughts(
    ai_name: str, thoughts: Union[str, ModelWithSummary, AssistantThoughts]
) -> None:
    # ... (Implementation as provided by you)
    if isinstance(thoughts, AssistantThoughts):
        assistant_thoughts_text = getattr(thoughts, 'text', None)
        if assistant_thoughts_text:
             print(f"\n{Fore.YELLOW}{ai_name.upper()} THOUGHTS:{Style.RESET_ALL} {markdown_to_ansi_style(assistant_thoughts_text)}")
        
        reasoning = getattr(thoughts, 'reasoning', None)
        if reasoning: print(f"{Fore.YELLOW}REASONING:{Style.RESET_ALL} {markdown_to_ansi_style(reasoning)}")

        plan = getattr(thoughts, 'plan', None)
        if plan:
            plan_text = "- " + "\n- ".join(plan) if isinstance(plan, list) else str(plan)
            print(f"{Fore.YELLOW}PLAN:{Style.RESET_ALL}\n{markdown_to_ansi_style(plan_text)}")
        
        criticism = getattr(thoughts, 'criticism', None)
        if criticism: print(f"{Fore.YELLOW}CRITICISM:{Style.RESET_ALL} {markdown_to_ansi_style(criticism)}")

    elif isinstance(thoughts, str): 
        print(f"\n{Fore.YELLOW}{ai_name.upper()} THOUGHTS:{Style.RESET_ALL} {markdown_to_ansi_style(thoughts)}")
    elif isinstance(thoughts, ModelWithSummary): 
        summary = thoughts.summary()
        print(f"\n{Fore.YELLOW}{ai_name.upper()} THOUGHTS:{Style.RESET_ALL} {markdown_to_ansi_style(summary)}")


# autogpt/app/main.py

# ... (ensure all necessary imports are at the top of your main.py file)
# For example:
# import logging
# import os
# import asyncio
# from pathlib import Path
# from autogpt.app.config import AppConfig, ConfigBuilder, assert_config_has_required_llm_api_keys
# from forge.agent_protocol.database import AgentDB
# from forge.file_storage import get_storage, FileStorageBackendName
# from forge.llm.providers import MultiProvider
# from forge.logging.config import configure_logging
# from .utils import coroutine # If @coroutine decorator is used
# from autogpt.app.agent_protocol_server import AgentProtocolServer
# ...

@coroutine # If you use this decorator, ensure it's defined or imported from .utils
async def run_auto_gpt_server(
    debug: bool = False,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file_format: Optional[str] = None,
    install_plugin_deps: bool = False, # Kept for signature, not directly used here
):
    config = ConfigBuilder.build_config_from_env()
    eff_log_level = log_level or ("DEBUG" if debug else "INFO")

    # Ensure logging components of config are present or have defaults
    logging_cfg_obj = getattr(config, 'logging', None)
    tts_cfg_obj = getattr(config, 'tts_config', None) # Assuming TTS config might be relevant globally
    
    configure_logging(
        debug=debug, 
        level=eff_log_level, # type: ignore
        log_format=log_format,
        log_file_format=log_file_format,
        config=logging_cfg_obj,
        tts_config=tts_cfg_obj, 
    )

    server_logger = logging.getLogger("AutoGPTServer") # Use a specific logger for server messages

    # --- Initialize components needed by AgentProtocolServer ---
    
    # 1. File Storage for the Server
    server_data_dir = config.app_data_dir / "server_workspace" 
    server_data_dir.mkdir(parents=True, exist_ok=True)
    
    is_local_fs_srv = hasattr(config, 'file_storage_backend') and \
                  config.file_storage_backend == FileStorageBackendName.LOCAL
    restrict_to_root_srv_calc = not is_local_fs_srv or \
                           (hasattr(config, 'restrict_to_workspace') and config.restrict_to_workspace)

    fs_server = get_storage(
        config.file_storage_backend, 
        root_path=server_data_dir, 
        restrict_to_root=restrict_to_root_srv_calc
    )
    fs_server.initialize()
    server_logger.info(f"Server file storage initialized at: {fs_server.root}")

    # 2. LLM Provider
    await assert_config_has_required_llm_api_keys(config) # Check API keys
    llm_provider_server = _configure_llm_provider(config) # Assuming _configure_llm_provider is defined in this file
    server_logger.info("LLM Provider for server configured.")

    # 3. Database <--- ASSIGNMENT MOVED HERE, BEFORE USE
    db_path = config.app_data_dir / "ap_server.db" 
    database = AgentDB(database_string=f"sqlite:///{db_path.resolve()}", debug_enabled=debug)
    # Removed: await database.connect() # Assuming AgentDB __init__ handles connection setup for SQLite
    server_logger.info(f"Database initialized at: {db_path.resolve()}")

    # --- End Component Initialization ---

    host = os.getenv("AP_SERVER_HOST", "0.0.0.0") 
    port = int(os.getenv("AP_SERVER_PORT", "8000"))

    server_logger.info(f"Attempting to start Agent Protocol server on http://{host}:{port}...")

    # Import AgentProtocolServer here if not at top level, or ensure it's globally available
    from autogpt.app.agent_protocol_server import AgentProtocolServer
    
    # Instantiate AgentProtocolServer
    server = AgentProtocolServer(
        app_config=config,
        database=database, # 'database' is now defined
        file_storage=fs_server,
        llm_provider=llm_provider_server,
    )
    
    server_logger.info("AgentProtocolServer instance created. Starting Hypercorn server...")
    try:
        await server.start_server(port=port, host=host) # Call start_server on the instance
    except Exception as e_server_start:
        server_logger.critical(f"Agent Protocol Server failed to start or crashed: {e_server_start}", exc_info=True)
    finally:
        server_logger.info("Agent Protocol Server shutdown sequence initiated in finally block.")
        # Handle database closure if AgentDB has an explicit (non-async) close method
        if hasattr(database, 'close_db_connection') and callable(getattr(database, 'close_db_connection')): 
             try:
                 database.close_db_connection() 
                 server_logger.info("Database connection explicitly closed via close_db_connection().")
             except Exception as e_db_close:
                 server_logger.error(f"Error during explicit database close: {e_db_close}", exc_info=True)
        # For SQLite with SQLAlchemy, explicit close of engine often not needed, sessions are scoped.
        pass 

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if "--server" in sys.argv:
        server_kwargs_dict: Dict[str, Any] = {"debug": "--debug" in sys.argv}
        if "--log-level" in sys.argv:
            try: server_kwargs_dict["log_level"] = sys.argv[sys.argv.index("--log-level") + 1]
            except IndexError: logging.warning("--log-level flag requires a value.") # Use logging
        if "--install-plugin-deps" in sys.argv: server_kwargs_dict["install_plugin_deps"] = True 
        
        asyncio.run(run_auto_gpt_server(**server_kwargs_dict))
    else:
        agent_kwargs_dict: Dict[str, Any] = {
            "continuous": "--continuous" in sys.argv,
            "skip_reprompt": "--no-reprompt" not in sys.argv,
            "speak": "--speak" in sys.argv, 
            "debug": "--debug" in sys.argv,
            "skip_news": True, 
        }
        
        if "--continuous-limit" in sys.argv:
            try: 
                limit_val_str = sys.argv[sys.argv.index("--continuous-limit") + 1]
                agent_kwargs_dict["continuous_limit"] = int(limit_val_str) if limit_val_str.lower() != 'inf' else int(math.inf)
            except (IndexError, ValueError): logging.warning("--continuous-limit flag requires an integer value or 'inf'.")
        
        if "--log-level" in sys.argv:
            try: agent_kwargs_dict["log_level"] = sys.argv[sys.argv.index("--log-level") + 1]
            except IndexError: logging.warning("--log-level flag requires a value.")
        
        if "--task" in sys.argv:
            try:
                task_idx = sys.argv.index("--task") + 1
                if task_idx < len(sys.argv) and not sys.argv[task_idx].startswith("-"):
                     agent_kwargs_dict["task"] = sys.argv[task_idx]
                else: logging.warning("--task flag requires a value.")
            except (ValueError): logging.warning("Error parsing --task argument.")
        
        if "--override-ai-name" in sys.argv:
            try: agent_kwargs_dict["override_ai_name"] = sys.argv[sys.argv.index("--override-ai-name") + 1]
            except IndexError: logging.warning("--override-ai-name flag requires a value.")
        if "--override-ai-role" in sys.argv:
            try: agent_kwargs_dict["override_ai_role"] = sys.argv[sys.argv.index("--override-ai-role") + 1]
            except IndexError: logging.warning("--override-ai-role flag requires a value.")
        
        print(f"{Fore.GREEN}Initializing AutoGPT J.A.R.V.I.S. CLI Mode...{Style.RESET_ALL}")
        asyncio.run(run_auto_gpt(**agent_kwargs_dict))