# File: forge/agent/forge_agent.py
from __future__ import annotations

import inspect
import logging
from typing import Any, Optional, AsyncGenerator, Dict, Union # Added for streaming
from uuid import uuid4
import asyncio # For streaming sleep
import orjson # For SSE

from forge.agent.base import BaseAgent, BaseAgentSettings # BaseAgent from AGPT Forge
from forge.agent.protocols import ( # For internal component system if used directly by ForgeAgent
    AfterExecute,
    CommandProvider,
    DirectiveProvider,
    MessageProvider,
)
from forge.agent_protocol.agent import ProtocolAgent # The class providing API method stubs
from forge.agent_protocol.database.db import AgentDB
from forge.agent_protocol.models.task import (
    Step,
    StepRequestBody,
    Task,
    TaskRequestBody,
    # Imports for type hints from ProtocolAgent methods we'll call via super()
    TaskListResponse, TaskStepsListResponse, TaskArtifactsListResponse, Artifact
)
from forge.command.command import Command # If ForgeAgent itself defines commands directly
from forge.components.system.system import SystemComponent as ForgeSystemComponent # To avoid name clash
from forge.config.ai_profile import AIProfile # For default ForgeAgent state
from forge.file_storage.base import FileStorage
# LLM and Action related imports from forge if ForgeAgent.propose_action was to be fully custom
from forge.llm.prompting.schema import ChatPrompt
from forge.llm.prompting.utils import dump_prompt
from forge.llm.providers.schema import AssistantChatMessage, AssistantFunctionCall # For a SIMPLE proposal
from forge.llm.providers.utils import function_specs_from_commands # If Forge defines commands
from forge.models.action import (
    ActionErrorResult,
    ActionProposal, # Note: This is forge.models.action.ActionProposal
    ActionResult,
    ActionSuccessResult,
)
from forge.utils.exceptions import AgentException, AgentTerminated, NotFoundError

# --- AutoGPT Core Imports ---
# These are needed to instantiate and use your advanced AutoGPT agent
from autogpt.app.config import AppConfig                     # From your AutoGPT project
from autogpt.llm.providers import MultiProvider as AutoGPTMultiProvider # AutoGPT's LLM system
from autogpt.agents.agent_manager import AgentManager        # AutoGPT's state manager
from autogpt.agents.agent import Agent as CoreAutoGPTAgent   # Your advanced agent
from autogpt.agent_factory.configurators import create_agent, configure_agent_with_state # AGPT factories
from autogpt.agents.prompt_strategies.one_shot import OneShotAgentActionProposal # AGPT's proposal type

# For StreamingResponse
from fastapi.responses import StreamingResponse
from fastapi import UploadFile # For create_artifact signature match

logger = logging.getLogger(__name__)


class ForgeAgent(ProtocolAgent, BaseAgent): # Inherits from both
    app_config: AppConfig
    llm_master_provider: AutoGPTMultiProvider
    agent_manager: AgentManager
    _task_budgets: dict[str, Any] # Using Any for ModelProviderBudget to avoid import if not used directly

    def __init__(self, database: AgentDB, workspace: FileStorage, 
                 autogpt_app_config: AppConfig, autogpt_llm_provider: AutoGPTMultiProvider):
        """
        The database is used to store tasks, steps and artifact metadata.
        The workspace is used to store artifacts (files) for the ProtocolAgent scope.
        AutoGPT specific configs are passed for CoreAutoGPTAgent instantiation.
        """
        logger.info("ForgeAgent initializing...")
        
        # Initialize ProtocolAgent (handles basic DB interactions for simple endpoints)
        ProtocolAgent.__init__(self, database, workspace)
        logger.debug("ProtocolAgent part initialized.")

        # Initialize BaseAgent (handles Forge's component system if ForgeAgent uses it directly)
        # Create a default BaseAgentSettings for ForgeAgent's own minimal state
        # This state is mostly for placeholder/info if CoreAutoGPTAgent handles main logic.
        default_forge_agent_state = BaseAgentSettings(
            name="ForgeAgentHost",
            description="Host agent interfacing with AutoGPT core for task execution.",
            agent_id=str(uuid4()), # ForgeAgent's own ID, distinct from task/AutoGPT agent IDs
            ai_profile=AIProfile(
                ai_name="ForgeHost", ai_role="Protocol Interface", ai_goals=["Mediate tasks to AutoGPT"]
            ),
            task="Facilitate AutoGPT task execution via Agent Protocol.",
        )
        BaseAgent.__init__(self, default_forge_agent_state)
        logger.debug("BaseAgent part initialized.")

        # Store AutoGPT specific dependencies
        self.app_config = autogpt_app_config
        self.llm_master_provider = autogpt_llm_provider
        # AgentManager needs the workspace where it can create "agents/{task_id}/" subdirectories
        self.agent_manager = AgentManager(workspace) # `workspace` is assumed to be the root for agent data.

        # For task-scoped LLM budgets (if your MultiProvider/schema has ModelProviderBudget)
        try:
            from forge.llm.providers.schema import ModelProviderBudget # Attempt import
            self._task_budgets = defaultdict(ModelProviderBudget)
        except ImportError:
            logger.warning("ModelProviderBudget not found, task budget tracking might be limited.")
            self._task_budgets = defaultdict(dict) # Fallback to simple dict

        # ForgeAgent's own system component (e.g., for a basic 'finish' if not delegating to AutoGPT)
        self.system = ForgeSystemComponent() # Part of BaseAgent components
        logger.info("ForgeAgent fully initialized.")


    # --- Override ProtocolAgent methods to use AutoGPT core where appropriate ---

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        # This method is called via the API.
        # It should create a DB task using ProtocolAgent's capability (self.db via super).
        # Then, it initializes the *state* for the CoreAutoGPTAgent.
        logger.info(
            f"ForgeAgent: create_task received. Input: {task_request.input[:50]}{'...' if len(task_request.input or '') > 50 else ''}"
        )
        # 1. Create the task in the Agent Protocol Database using inherited method
        db_task: Task = await super().create_task(task_request)
        if not db_task or not db_task.task_id:
            logger.error("Superclass ProtocolAgent.create_task failed to return a valid task or task_id.")
            raise AgentException("Failed to create task record in database.")
        
        logger.info(f"Task record created in DB. Task ID: {db_task.task_id}")

        # 2. Initialize the CoreAutoGPTAgent for this task_id. Its state will be saved by AgentManager.
        # The agent_id for CoreAutoGPTAgent will be db_task.task_id.
        await self._get_or_create_task_scoped_autogpt_agent(db_task.task_id, db_task.input)
        
        logger.info(f"📦 CoreAutoGPTAgent initialized or confirmed for Task ID: {db_task.task_id}.")
        return db_task

    # `list_tasks`, `get_task`, `list_steps`, `get_step`, `list_artifacts` can often use
    # the default `ProtocolAgent` implementations (which just call self.db methods).
    # If customization is needed (e.g. for auth), override them.

    # --- The CRITICAL execute_step method for STREAMING ---
    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> StreamingResponse:
        logger.info(f"ForgeAgent: execute_step for task_id: {task_id}, client_input: '{step_request.input or '(none)'}'")
        # This method now returns a StreamingResponse
        
        # Validate task existence first
        try:
            task_for_step: Task = await self.db.get_task(task_id) # Fetch task details
        except NotFoundError:
            logger.error(f"Task with ID '{task_id}' not found for execute_step.")
            # Return an error as a normal JSON response for HTTP 404, not a stream
            # Or, yield a single error event and close (harder for client to distinguish 404 from stream error)
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
        
        sse_headers = {
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no', 
            'Connection': 'keep-alive'
        }
        return StreamingResponse(
            self._core_agent_streaming_logic(task=task_for_step, step_request_from_client=step_request),
            media_type="text/event-stream",
            headers=sse_headers
        )
# In forge_agent.py, within the ForgeAgent class

    async def _core_agent_streaming_logic(
            self, task: Task, step_request_from_client: StepRequestBody
    ) -> AsyncGenerator[str, None]:
        logger.info(f"ForgeAgent: ENTERING _core_agent_streaming_logic for task '{task.task_id}'")

        core_agent: CoreAutoGPTAgent
        try:
            core_agent = await self._get_or_create_task_scoped_autogpt_agent(task.task_id, task.input)
            yield self._format_sse_event('progress', {'stage': 'core_agent_retrieved', 'message': 'Core agent instance ready.'})
        except Exception as e_get_agent:
            err_msg = f"CRITICAL: Failed to get/create CoreAutoGPTAgent for task '{task.task_id}': {e_get_agent}"
            logger.critical(err_msg, exc_info=True)
            yield self._format_sse_event('error', {'message': err_msg, 'stage': 'core_agent_instantiation_failure'})
            return

        # Create an initial DB step record for this interaction
        # The 'input' for the step_request_from_client is the user's latest message/authorization
        current_db_step_record: Step = await self.db.create_step(
            task_id=task.task_id,
            input=step_request_from_client,
            is_last=False, # Assume not last initially
            additional_input=step_request_from_client.additional_input or {}
        )
        logger.info(f"DB Step record '{current_db_step_record.step_id}' created for task '{task.task_id}'")
        yield self._format_sse_event('progress', {'stage': 'db_step_created', 'step_id': current_db_step_record.step_id})


        # --- Main Agent Interaction Loop ---
        # This loop will allow the agent to propose and execute multiple times if needed
        # within a single call to /steps, until it 'FINISHES' or an error/limit is hit.
        
        MAX_INTERNAL_CYCLES = getattr(self.app_config, 'max_internal_agent_cycles_per_step', 5) # Configurable limit
        current_cycle = 0
        agent_has_finished_task = False
        last_action_result: Optional[ActionResult] = None
        # User input from this specific step request. Subsequent cycles within this stream won't have new user input
        # unless the agent explicitly asks for it and we modify this logic to pause and wait.
        # For now, the first user input kicks off potential internal cycles.
        current_user_directive_for_agent = step_request_from_client.input 

        try:
            while current_cycle < MAX_INTERNAL_CYCLES and not agent_has_finished_task:
                current_cycle += 1
                logger.info(f"Task '{task.task_id}', Stream Cycle {current_cycle}/{MAX_INTERNAL_CYCLES}")
                yield self._format_sse_event('progress', {
                    'stage': 'internal_cycle_start', 'cycle': current_cycle, 
                    'message': f'Agent starting internal cycle {current_cycle}.'
                })

                # 1. Agent Proposes Action
                # The input to propose_action depends on whether it's the first cycle of this /steps call
                # or a subsequent internal cycle based on the last action's result.
                # For CoreAutoGPTAgent, 'user_input' to propose_action might be used to set current task,
                # or if None, it continues based on history.
                
                proposal_input_arg = current_user_directive_for_agent if current_cycle == 1 else None
                if last_action_result and not proposal_input_arg: # Feed observation if available
                     # This assumes CoreAutoGPTAgent can take observations or will use its history.
                     # A more direct way would be core_agent.set_last_action_result(last_action_result) if such a method exists.
                     logger.debug(f"Task '{task.task_id}': Cycle {current_cycle}, no new user input, agent will use history/last result.")
                     # If your CoreAutoGPTAgent's propose_action needs explicit observation:
                     # proposal_input_arg = f"Observation from last action: {last_action_result.summary()}" 


                logger.info(f"Task '{task.task_id}': Cycle {current_cycle}, CoreAutoGPTAgent proposing action. Input hint: '{str(proposal_input_arg)[:50]}'")
                current_proposal: Optional[OneShotAgentActionProposal] = None # Should be this type from AutoGPT

                async for item in core_agent.propose_action(user_input=proposal_input_arg):
                    if isinstance(item, dict) and 'type' in item: # Stream out thoughts/progress from agent
                        yield self._format_sse_event(item['type'], item.get('data', item))
                    elif isinstance(item, ActionProposal): # Should be OneShotAgentActionProposal
                        current_proposal = cast(OneShotAgentActionProposal, item)
                        logger.info(f"Task '{task.task_id}': Cycle {current_cycle}, CoreAutoGPTAgent proposed: {current_proposal.use_tool.name if current_proposal.use_tool else 'No tool'}")
                        yield self._format_sse_event('agent_proposal', {'proposal': current_proposal.model_dump(exclude_none=True)})
                        break # Got the main proposal
                    else:
                        logger.warn(f"Task '{task.task_id}': Cycle {current_cycle}, Unexpected item from propose_action: {type(item)}")
                
                if not current_proposal:
                    logger.error(f"Task '{task.task_id}': Cycle {current_cycle}, CoreAutoGPTAgent did not return a proposal.")
                    yield self._format_sse_event('error', {'message': 'Agent failed to propose an action.', 'stage': 'propose_action_failed'})
                    break # Exit loop

                # For Agent Protocol, after a proposal, the client typically authorizes.
                # Here, we are auto-authorizing to continue the internal loop.
                # If current_proposal.use_tool.name == FINISH_COMMAND, we should finish.
                if current_proposal.use_tool and current_proposal.use_tool.name == getattr(self.app_config, 'finish_command_name', 'finish'): # Use configured finish command
                    logger.info(f"Task '{task.task_id}': Cycle {current_cycle}, Agent proposed FINISH command.")
                    agent_has_finished_task = True
                    last_action_result = ActionSuccessResult(outputs=current_proposal.use_tool.arguments or {}) # Simulate successful finish
                    yield self._format_sse_event('agent_action_result', {'result': last_action_result.model_dump(exclude_none=True)})
                    # No break here, let the loop condition handle it, or db_step_is_last will be set
                else:
                    # 2. Agent Executes Action (auto-authorized for this internal loop)
                    logger.info(f"Task '{task.task_id}': Cycle {current_cycle}, CoreAutoGPTAgent executing: {current_proposal.use_tool.name if current_proposal.use_tool else 'No tool'}")
                    yield self._format_sse_event('progress', {
                        'stage': 'core_agent_executing', 'cycle': current_cycle, 
                        'tool_name': current_proposal.use_tool.name if current_proposal.use_tool else 'N/A',
                        'message': f'Agent executing: {current_proposal.use_tool.name if current_proposal.use_tool else "plan"}'
                    })
                    
                    last_action_result = None
                    async for item in core_agent.execute(current_proposal):
                        if isinstance(item, dict) and 'type' in item: # Stream out thoughts/progress from agent
                            yield self._format_sse_event(item['type'], item.get('data', item))
                        elif isinstance(item, ActionResult):
                            last_action_result = item
                            logger.info(f"Task '{task.task_id}': Cycle {current_cycle}, CoreAutoGPTAgent execution result: {type(last_action_result).__name__}")
                            yield self._format_sse_event('agent_action_result', {'result': last_action_result.model_dump(exclude_none=True)})
                            break # Got the main result
                        else:
                            logger.warn(f"Task '{task.task_id}': Cycle {current_cycle}, Unexpected item from execute: {type(item)}")
                    
                    if not last_action_result:
                        logger.error(f"Task '{task.task_id}': Cycle {current_cycle}, CoreAutoGPTAgent did not return an action result.")
                        yield self._format_sse_event('error', {'message': 'Agent failed to return an action result.', 'stage': 'execute_action_failed'})
                        agent_has_finished_task = True # Treat as error, stop.
                        break
                    
                    if isinstance(last_action_result, ActionErrorResult):
                        logger.warning(f"Task '{task.task_id}': Cycle {current_cycle}, Agent execution resulted in an error: {last_action_result.reason}")
                        agent_has_finished_task = True # Stop on error
                        break
                
                # Clear current_user_directive_for_agent after the first cycle, subsequent cycles are internal
                current_user_directive_for_agent = None 

                # Brief pause to allow UI to catch up if many rapid internal cycles
                await asyncio.sleep(0.1) 

            if current_cycle >= MAX_INTERNAL_CYCLES and not agent_has_finished_task:
                logger.warning(f"Task '{task.task_id}': Reached max internal cycles ({MAX_INTERNAL_CYCLES}). Pausing agent.")
                yield self._format_sse_event('progress', {'stage': 'max_internal_cycles_reached', 'message': 'Agent reached internal cycle limit for this step.'})
                # Agent hasn't "finished" the overall task, but this /steps interaction is done.
                # The next user input will start a new /steps call.

        except (AgentFinished, AgentTerminated) as core_agent_signal:
            sig_type = 'agent_finished_task' if isinstance(core_agent_signal, AgentFinished) else 'agent_terminated'
            reason_msg = core_agent_signal.reason or ("Task completed by agent." if isinstance(core_agent_signal, AgentFinished) else "Agent process terminated.")
            yield self._format_sse_event(sig_type, {'message': reason_msg})
            logger.info(f"Task '{task.task_id}': CoreAutoGPTAgent signal '{type(core_agent_signal).__name__}': {reason_msg}")
            agent_has_finished_task = True
        except Exception as e_stream_loop:
            err_msg = f"Unhandled error in agent interaction loop for task '{task.task_id}': {e_stream_loop}"
            logger.critical(err_msg, exc_info=True)
            yield self._format_sse_event('critical_error', {'message': err_msg, 'stage': 'agent_interaction_loop_error'})
            agent_has_finished_task = True # Stop on critical error
        
        # --- Finalize DB Step and CoreAutoGPTAgent State ---
        final_db_output = "Agent interaction cycle completed."
        final_db_additional_output = {}

        if last_action_result:
            final_db_output = last_action_result.summary() if hasattr(last_action_result, 'summary') else str(last_action_result)
            final_db_additional_output['last_action_result'] = last_action_result.model_dump(exclude_none=True)
        elif current_proposal: # If it finished on a proposal (e.g. FINISH)
            final_db_output = current_proposal.thoughts.speak if current_proposal.thoughts else "Task ending."
            final_db_additional_output['last_proposal'] = current_proposal.model_dump(exclude_none=True)

        try:
            updated_db_step = await self.db.update_step(
                task_id=task.task_id,
                step_id=current_db_step_record.step_id,
                status="completed", # Or "error" if an exception occurred
                output=final_db_output[:4000], # Limit output size for DB
                additional_output=final_db_additional_output,
                is_last=agent_has_finished_task
            )
            logger.info(f"DB Step '{current_db_step_record.step_id}' updated. Is last: {agent_has_finished_task}")
            yield self._format_sse_event('db_step_updated', {'step': updated_db_step.model_dump(exclude_none=True)})
        except Exception as e_upd_db:
            logger.error(f"DB update error for step '{current_db_step_record.step_id}': {e_upd_db}", exc_info=True)
            yield self._format_sse_event('error', {'message': f'DB step update error: {e_upd_db}', 'stage': 'final_db_step_update_failure'})

        try:
            if core_agent: await self.agent_manager.save_agent_state(core_agent)
            logger.info(f"CoreAutoGPTAgent state saved for task '{task.task_id}' after stream logic.")
        except Exception as e_save_core_final:
            logger.error(f"Failed to save CoreAutoGPTAgent state for task '{task.task_id}' post_stream: {e_save_core_final}", exc_info=True)
            yield self._format_sse_event('error', {'message': f'Core agent state save failed: {e_save_core_final}', 'stage': 'core_agent_final_state_save_failure'})
            
        final_event_type = 'task_completed' if agent_has_finished_task else 'step_completed'
        yield self._format_sse_event(final_event_type, {'message': f'Agent {"task" if agent_has_finished_task else "step"} processing concluded.'})
        logger.info(f"SSE Stream: END for Task '{task.task_id}', Step '{current_db_step_record.step_id}'. Agent finished task: {agent_has_finished_task}")

    def _format_sse_event(self, event_name: str, data: dict) -> str:
        """Helper to format data as an SSE string with a specific event name."""
        json_data = orjson.dumps(data).decode("utf-8")
        return f"event: {event_name}\ndata: {json_data}\n\n"


    # --- Helper methods for ForgeAgent to manage CoreAutoGPTAgent instances ---
    async def _get_or_create_task_scoped_autogpt_agent(self, task_id_str: str, task_input_str: Optional[str]) -> CoreAutoGPTAgent:
        """Loads existing CoreAutoGPTAgent state or creates a new one for the task."""
        try:
            logger.debug(f"Loading CoreAutoGPTAgent state for task_id: {task_id_str}")
            agent_settings_from_disk = await self.agent_manager.load_agent_state(agent_id=task_id_str)
            core_agent = configure_agent_with_state(
                state=agent_settings_from_disk,
                app_config=self.app_config,
                file_storage=self._get_task_agent_file_workspace(task_id=task_id_str), # Agent's own workspace
                llm_provider=self._get_task_llm_provider(task_id=task_id_str), # Task-scoped provider
            )
            logger.info(f"CoreAutoGPTAgent state loaded for task: {task_id_str}")
        except FileNotFoundError:
            logger.info(f"No existing CoreAutoGPTAgent state found for task_id: {task_id_str}. Creating new.")
            core_agent = create_agent(
                agent_id=task_id_str, # Becomes core_agent.state.agent_id
                task=task_input_str or "Default initial task - input was missing.",
                app_config=self.app_config,
                file_storage=self._get_task_agent_file_workspace(task_id=task_id_str),
                llm_provider=self._get_task_llm_provider(task_id=task_id_str),
            )
            await self.agent_manager.save_agent_state(core_agent) # Save initial state
            logger.info(f"New CoreAutoGPTAgent created and its initial state saved for task: {task_id_str}")
        return core_agent

    def _get_task_agent_file_workspace(self, task_id: str) -> FileStorage:
        """
        AutoGPT Agent needs its own workspace. This is scoped under ProtocolAgent's main workspace.
        Example: self.workspace.root (e.g., "data/")
                 returned FileStorage root: "data/agents/{task_id}/autogpt_workspace/"
        """
        # self.workspace is the root FileStorage for ProtocolAgent from its __init__
        # AgentManager already scopes to "agents/{id}/" based on its own root.
        # We need a sub-workspace for AutoGPT's specific artifacts *within* that agent's folder.
        agent_data_dir_under_protocol_ws = os.path.join(self.agent_manager.agents_data_root_dirname, str(task_id))
        autogpt_core_workspace_subpath = os.path.join(agent_data_dir_under_protocol_ws, "core_agent_workspace")
        
        # Return new FileStorage instance scoped to this deeper path.
        # self.workspace is the FileStorage ProtocolAgent was initialized with.
        return self.workspace.clone_with_subroot(autogpt_core_workspace_subpath)


    def _get_task_llm_provider(self, task_id: str, task: Optional[Task] = None) -> AutoGPTMultiProvider:
        # Task argument is optional for flexibility if only task_id is known initially
        # This uses self.llm_master_provider (AutoGPT's MultiProvider instance)
        
        budget_for_this_task = self._task_budgets[task_id]
        config_copy = self.llm_master_provider._configuration.model_copy(deep=True)
        if config_copy.extra_request_headers is None: config_copy.extra_request_headers = {}
        config_copy.extra_request_headers["X-Task-Trace-ID"] = task_id
        if task and task.additional_input and (uid := task.additional_input.get("user_id")):
            config_copy.extra_request_headers["X-Client-UserID"] = str(uid)

        settings_copy = self.llm_master_provider._settings.model_copy(deep=True)
        settings_copy.budget = budget_for_this_task
        settings_copy.configuration = config_copy
        
        task_provider_instance = self.llm_master_provider.__class__(
            settings=settings_copy,  # type: ignore
            logger=logging.getLogger(f"{self.llm_master_provider.logger.name}.ForTaskScope.{task_id}")
        )
        # Ensure the tracked budget dict points to the budget object inside the new provider
        self._task_budgets[task_id] = task_provider_instance._budget
        return task_provider_instance


# --- Global `app` instance for Uvicorn `forge.app:app` ---
# This section is executed when `forge/app.py` is imported.

logger.info(f"TOP LEVEL: Attempting to create global FastAPI 'app' in {__file__} for Uvicorn.")
try:
    # 1. Load AutoGPT AppConfig
    # Assumes .env is in CWD from where uvicorn server is started by forge.__main__
    env_path = Path(os.getcwd()) / ".env"
    logger.debug(f"Loading .env for AppConfig from: {env_path}")
    global_autogpt_config = ConfigBuilder.build_config_from_env(dotenv_path=env_path)
    if not global_autogpt_config:
        raise RuntimeError("Failed to build AutoGPT AppConfig. Check .env file or ConfigBuilder.")
    logger.info(f"AutoGPT AppConfig loaded globally. Workspace: {global_autogpt_config.workspace_path}")

    # 2. Initialize Forge AgentDB
    # Database file path relative to AutoGPT's app_data_dir for cohesion
    db_file = Path(global_autogpt_config.app_data_dir) / "forge_agent_protocol.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_url = f"sqlite:///{db_file.resolve()}"
    global_forge_db = AgentDB(database_string=db_url, debug_enabled=global_autogpt_config.debug_mode)
    logger.info(f"Forge AgentDB (SQLite) initialized at: {db_url}")

    # 3. Initialize Root FileStorage (for ProtocolAgent, which AgentManager will use)
    # Using AutoGPT's configured storage backend for ProtocolAgent's workspace.
    # This creates a FileStorage instance pointing to a root like "data/" or ".autogpt/data/"
    # AgentManager initialized within ProtocolAgent will handle agent-specific subdirectories.
    root_file_storage = get_storage(
        backend_name=global_autogpt_config.file_storage_backend,
        root_path=global_autogpt_config.app_data_dir, # Let app_data_dir be root for this forge setup
        restrict_to_root=False # Allow creating "agents/" etc. under this root
    )
    root_file_storage.initialize() # Ensure the root directory exists
    logger.info(f"Root FileStorage for Forge initiated at '{root_file_storage.root}' (type: {type(root_file_storage)}).")

    # 4. Initialize Master LLM Provider (AutoGPT's MultiProvider)
    global_autogpt_llm_provider = AutoGPTMultiProvider() # Default init loads its config
    logger.info("AutoGPT Master LLM MultiProvider initialized globally.")

    # 5. Instantiate ProtocolAgent (which is ForgeAgent) with all dependencies
    forge_agent_instance = ForgeAgent( # ForgeAgent is the ProtocolAgent here
        database=global_forge_db,
        workspace=root_file_storage, # Pass the root file storage
        autogpt_app_config=global_autogpt_config,
        autogpt_llm_provider=global_autogpt_llm_provider
    )
    logger.info("Global ForgeAgent (as ProtocolAgent) instance created.")

    # 6. Create the FastAPI `app` instance using ForgeAgent's method
    app: FastAPI = forge_agent_instance.get_agent_app()
    logger.info(f"Global FastAPI 'app' instance obtained from ForgeAgent. Uvicorn should find 'forge.app:app'.")

except Exception as e_global_app_setup:
    logger.critical(f"FATAL ERROR DURING forge.app GLOBAL SETUP: {e_global_app_setup}", exc_info=True)
    # Fallback app to indicate server setup failure
    app = FastAPI(title="Forge App - CRITICAL SETUP FAILURE")
    @app.get("/")
    async def setup_failure_message():
        return {"error": "Server critical setup failure. See logs for details.", "exception": str(e_global_app_setup)}, 500

# The uvicorn call in forge.__main__.py "forge.app:app" will pick up the 'app' variable defined above.