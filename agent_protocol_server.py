# autogpt/app/agent_protocol_server.py

import logging
import logging.config
import pathlib
import asyncio
from typing import Any, Optional, AsyncGenerator, Dict, List
import orjson
import json
import uuid

from fastapi import APIRouter, FastAPI, UploadFile, Query, Request, HTTPException, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from forge.agent_protocol.models import (
    Artifact, Step, StepRequestBody, Task, TaskArtifactsListResponse,
    TaskListResponse, TaskRequestBody, TaskStepsListResponse, Pagination
)
from forge.agent_protocol.database import AgentDB
from forge.agent_protocol.middlewares import AgentMiddleware
from forge.file_storage.base import FileStorage as ForgeFileStorageBase
from forge.models.action import ActionResult, ActionErrorResult, ActionSuccessResult
from forge.config.ai_profile import AIProfile # User Instruction Step 2: Ensure this is at the top
from forge.utils.exceptions import AgentFinished, AgentTerminated, NotFoundError, AgentException
from forge.utils.const import ASK_COMMAND

from dotenv import load_dotenv

from autogpt.app.utils import is_port_free
from autogpt.app.config import AppConfig
from forge.llm.providers import MultiProvider as AutoGPTLLMProvider
from autogpt.agents.agent_manager import AgentManager
from autogpt.agents.agent import Agent as CoreAutoGPTAgent, AgentProgressEvent
from autogpt.agents.prompt_strategies.one_shot import OneShotAgentActionProposal
from autogpt.agent_factory.configurators import create_agent, configure_agent_with_state # User Instruction Step 2: Ensure these are at the top

from hypercorn.asyncio import serve as hypercorn_serve_async
from hypercorn.config import Config as HypercornAppConfig

logger = logging.getLogger("autogpt.app.agent_protocol_server")
load_dotenv()

# --- LOG_CFG (Standard Logging Configuration) ---
LOG_CFG = {
    "version": 1, "disable_existing_loggers": False,
    "formatters": {
        "default": {"()": "uvicorn.logging.DefaultFormatter", "fmt": "%(levelprefix)s %(asctime)s - %(name)s - %(levelname)s - %(message)s", "use_colors": None},
        "access": {"()": "uvicorn.logging.AccessFormatter", "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'},
    },
    "handlers": {
        "default": {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"},
        "access": {"formatter": "access", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"},
    },
    "loggers": {
        "autogpt": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "forge": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": True},
        "uvicorn.error": {"level": "INFO", "propagate": True},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}
logging.config.dictConfig(LOG_CFG)
# --- END LOG_CFG ---

class ChatRequestBody(BaseModel):
    message: str
    session_id: Optional[str] = None

class AgentProtocolServer:
    def __init__(
        self,
        app_config: AppConfig,
        database: AgentDB,
        file_storage: ForgeFileStorageBase,
        llm_provider: AutoGPTLLMProvider,
    ):
        self.app_config = app_config
        self.db = database
        self.file_storage_root = file_storage
        self.llm_provider_master = llm_provider
        self.agent_manager = AgentManager(file_storage)
        logger.info(f"AgentProtocolServer initialized. FileStorage root: {self.file_storage_root.root}")

    def _format_sse(self, data: Dict[str, Any], event_name: Optional[str] = None) -> str:
        try:
            json_data = orjson.dumps(data).decode("utf-8")
        except TypeError as e_orjson:
            logger.warning(f"SSE orjson dump failed: {e_orjson}. Data: {str(data)[:200]}. Trying json.dumps.")
            try:
                json_data = json.dumps(data)
            except TypeError as e_std_json:
                logger.error(f"SSE json.dumps also failed: {e_std_json}. Data: {str(data)[:200]}. Sending error event.")
                error_payload = {"type": "error", "message": "Internal SSE data serialization failure."}
                json_data = orjson.dumps(error_payload).decode("utf-8")
                event_name = "error"

        sse_message = f"data: {json_data}\n"
        if event_name:
            sse_message = f"event: {event_name}\n{sse_message}"
        sse_message += "\n"
        return sse_message

    async def start_server(
        self,
        port: int = 8000,
        host: str = "0.0.0.0",
        router_to_include: Optional[APIRouter] = None,
    ) -> None:
        logger.info(f"AP Server: Attempting to start on http://{host}:{port}")
        if not is_port_free(port):
            logger.critical(f"AP Server FATAL: Port {port} is not free. Exiting.")
            return

        app = FastAPI(
            title="AutoGPT Core - Agent Protocol Server",
            description="Serves AutoGPT core logic via the Agent Protocol.",
            version=str(getattr(self.app_config, 'version', '0.0.0')),
        )

        cors_origins_setting = getattr(self.app_config, 'cors_origins', ["*"])
        if not isinstance(cors_origins_setting, list) or not all(isinstance(o, str) for o in cors_origins_setting):
            logger.warning(f"Invalid cors_origins '{cors_origins_setting}', defaulting to ['*']")
            cors_origins_setting = ["*"]

        logger.info(f"AP Server CORS: Allowing origins: {cors_origins_setting}")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins_setting,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        if router_to_include is None:
            try:
                from forge.agent_protocol.api_router import base_router
                router_to_include = base_router
                logger.info("AP Server: Using default Agent Protocol 'base_router'.")
            except ImportError:
                logger.error("AP Server: Default 'base_router' not found and no router provided. API endpoints may be missing.")

        if router_to_include:
            app.include_router(router_to_include, prefix="/ap/v1")
            logger.info(f"AP Server: Router included with prefix '/ap/v1'.")

        app.add_middleware(AgentMiddleware, agent=self)
        logger.info("AP Server: AgentMiddleware attached.")

        ui_path_str = getattr(self.app_config, 'frontend_static_path', None)
        ui_mount = getattr(self.app_config, 'frontend_mount_path', '/frontend')
        if ui_path_str:
            project_root = pathlib.Path(__file__).resolve().parents[2]
            abs_ui_path = pathlib.Path(ui_path_str)
            if not abs_ui_path.is_absolute():
                abs_ui_path = (project_root / ui_path_str).resolve()

            if abs_ui_path.is_dir():
                logger.info(f"AP Server: Mounting static UI from '{abs_ui_path}' at '{ui_mount}'.")
                app.mount(ui_mount, StaticFiles(directory=abs_ui_path, html=True), name="static_ui")
                if ui_mount != "/" :
                    @app.get("/", include_in_schema=False)
                    async def _redirect_to_ui():
                        return RedirectResponse(url=f"{ui_mount}/index.html", status_code=307)
            else:
                logger.warning(f"AP Server: Static UI path '{abs_ui_path}' not found. Static UI disabled.")
        else:
            logger.info("AP Server: No 'frontend_static_path' configured. Static UI disabled.")
            @app.get("/", include_in_schema=False)
            async def _api_root_status():
                return {"status": "Agent Protocol Server is online", "documentation": "/docs"}

        config = HypercornAppConfig()
        config.bind = [f"{host}:{port}"]
        config.loglevel = getattr(self.app_config, 'hypercorn_log_level', "info").lower()

        logger.info(f"AP Server: Hypercorn binding to '{config.bind[0]}' with log level '{config.loglevel}'. Starting server...")
        try:
            await hypercorn_serve_async(app, config)
        except Exception as e_serve:
            logger.critical(f"AP Server: Hypercorn server CRASHED: {e_serve}", exc_info=True)
        finally:
            logger.info("AP Server: Shutdown process initiated.")

    async def _get_or_create_core_agent_instance(self, task_id: str, task_initial_input: Optional[str]) -> CoreAutoGPTAgent: # Renamed task.input to task_initial_input for clarity
        logger.debug(f"AP Server: _get_or_create_core_agent_instance for task_id: '{task_id}'")
        try:
            loaded_agent_settings = await self.agent_manager.load_agent_state(agent_id=task_id)
            core_agent = configure_agent_with_state( # configure_agent_with_state imported at top
                state=loaded_agent_settings,
                app_config=self.app_config,
                file_storage=self._get_core_agent_workspace_filestorage(task_id),
                llm_provider=self._get_task_scoped_llm_provider(task_id=task_id),
            )
            logger.info(f"Loaded existing CoreAutoGPTAgent state for task '{task_id}'.")
        except FileNotFoundError:
            logger.info(f"No existing state for task '{task_id}'. Creating new CoreAutoGPTAgent.")

            # User Instruction Step 3: Insert definition here
            # AIProfile is imported at the top of the file.
            default_server_profile = AIProfile(
                ai_name="APServerCore", # As per user instruction
                ai_role="Handle direct client requests", # As per user instruction
                ai_goals=[
                    task_initial_input or "Default: respond to the user's input directly." # Use task_initial_input from method signature
                ],
            )

            # User Instruction Step 4: Then call create_agent
            # create_agent is imported at the top of the file.
            core_agent = create_agent(
                agent_id=task_id,
                ai_profile=default_server_profile, # Using the defined profile
                task=task_initial_input or "", # Use task_initial_input from method signature, default to empty string if None
                app_config=self.app_config,
                file_storage=self._get_core_agent_workspace_filestorage(task_id), # Mapped to actual variable
                llm_provider=self._get_task_scoped_llm_provider(task_id=task_id), # Mapped to actual variable
            )
            await self.agent_manager.save_agent_state(core_agent)
            logger.info(f"New CoreAutoGPTAgent created with fallback profile and initial state saved for task '{task_id}'.")
        except Exception as e_get_create:
            logger.critical(f"Failed to get or create CoreAutoGPTAgent for task '{task_id}': {e_get_create}", exc_info=True)
            raise AgentException(f"Could not initialize agent for task {task_id}") from e_get_create
        return core_agent

    def _get_core_agent_workspace_filestorage(self, task_id: str) -> ForgeFileStorageBase:
        core_agent_files_subpath = pathlib.Path(task_id) / "core_agent_workspace_files"
        return self.agent_manager.file_manager.clone_with_subroot(core_agent_files_subpath)

    def _get_task_scoped_llm_provider(self, task_id: str, task: Optional[Task] = None) -> AutoGPTLLMProvider: # task parameter is optional and not directly used here
        return self.llm_provider_master

    async def _core_agent_streaming_interaction_logic(
        self, task: Task, client_step_request_body: StepRequestBody
    ) -> AsyncGenerator[str, None]:
        logger.info(
            f"SSE STREAM: START for task '{task.task_id}'. "
            f"Client Input: '{client_step_request_body.input[:100] if client_step_request_body.input else '(None)'}'. "
            f"Additional Input: {client_step_request_body.additional_input}"
        )

        core_agent: CoreAutoGPTAgent
        db_step: Optional[Step] = None
        agent_has_finished_overall_task = False
        stream_was_paused_for_ask_user = False
        current_action_proposal: Optional[OneShotAgentActionProposal] = None

        try:
            core_agent = await self._get_or_create_core_agent_instance(task.task_id, task.input)
            yield self._format_sse(
                {'type': 'progress', 'stage': 'agent_ready', 'message': 'Agent instance prepared.'},
                event_name='progress'
            )

            db_step = await self.db.create_step(
                task_id=task.task_id,
                input=client_step_request_body,
                is_last=False,
                additional_input=client_step_request_body.additional_input or {}
            )
            logger.info(f"SSE STREAM: DB Step '{db_step.step_id if db_step else 'N/A'}' created for task '{task.task_id}'.")
            yield self._format_sse(
                {'type': 'progress', 'stage': 'db_step_created', 'step_id': db_step.step_id if db_step else "unknown_step", 'message': 'Interaction step logged.'},
                event_name='progress'
            )

            user_input_from_client = client_step_request_body.input
            effective_agent_input: Optional[str] = None

            if user_input_from_client and user_input_from_client.strip():
                effective_agent_input = user_input_from_client.strip()
                if hasattr(core_agent.state, 'ai_profile') and isinstance(core_agent.state.ai_profile, AIProfile): # type: ignore
                    core_agent.state.ai_profile.set_task_description(effective_agent_input) # type: ignore
                    if hasattr(core_agent.state, 'task'):
                        core_agent.state.task = effective_agent_input # type: ignore
            else:
                if hasattr(core_agent.state, 'ai_profile') and core_agent.state.ai_profile.get_task_description(): # type: ignore
                    effective_agent_input = core_agent.state.ai_profile.get_task_description() # type: ignore
                else:
                    effective_agent_input = None

            logger.debug(f"SSE STREAM: Task '{task.task_id}', Agent proposing action. Effective agent input: '{str(effective_agent_input)[:70]}'")

            async for proposal_event_data in core_agent.propose_action(user_input=effective_agent_input):
                if isinstance(proposal_event_data, OneShotAgentActionProposal):
                    current_action_proposal = proposal_event_data
                    logger.info(f"SSE STREAM: Task '{task.task_id}', Agent generated OneShotAgentActionProposal.")

                    if current_action_proposal.thoughts:
                        thoughts = current_action_proposal.thoughts
                        if hasattr(thoughts, 'text') and thoughts.text and thoughts.text.strip():
                            yield self._format_sse({'type': 'agent_thought', 'content': thoughts.text}, event_name='agent_thought_event')
                            await asyncio.sleep(0.1)
                        if hasattr(thoughts, 'plan') and thoughts.plan and isinstance(thoughts.plan, list) and thoughts.plan:
                            yield self._format_sse({'type': 'agent_plan', 'content': thoughts.plan}, event_name='agent_plan_event')
                            await asyncio.sleep(0.1)
                        if hasattr(thoughts, 'self_criticism') and thoughts.self_criticism and isinstance(thoughts.self_criticism, str) and thoughts.self_criticism.strip():
                            yield self._format_sse({'type': 'agent_criticism', 'content': thoughts.self_criticism}, event_name='agent_criticism_event')
                            await asyncio.sleep(0.1)

                    if current_action_proposal.use_tool and current_action_proposal.use_tool.name:
                        tool_args_for_sse = current_action_proposal.use_tool.arguments
                        try: tool_args_for_sse = dict(tool_args_for_sse) if tool_args_for_sse else {}
                        except TypeError: tool_args_for_sse = vars(tool_args_for_sse) if tool_args_for_sse else {}
                        yield self._format_sse({'type': 'agent_tool', 'tool_name': current_action_proposal.use_tool.name, 'arguments': tool_args_for_sse or {}}, event_name='agent_tool_event')
                        await asyncio.sleep(0.1)

                    if current_action_proposal.use_tool and current_action_proposal.use_tool.name == ASK_COMMAND:
                        question_to_ask = current_action_proposal.use_tool.arguments.get('question', "Clarification needed.") if isinstance(current_action_proposal.use_tool.arguments, dict) else "Clarification needed."
                        yield self._format_sse({'type': 'ask_user', 'question': question_to_ask, 'step_id_awaiting_reply': db_step.step_id if db_step else None}, event_name='ask_user')
                        if db_step: await self.db.update_step(task_id=task.task_id, step_id=db_step.step_id, status="awaiting_user_input", output=f"Agent asked: {question_to_ask}", is_last=False)
                        await self.agent_manager.save_agent_state(core_agent)
                        stream_was_paused_for_ask_user = True
                        return
                    break
                elif isinstance(proposal_event_data, dict):
                    event_type = proposal_event_data.get('type', 'progress')
                    if event_type in ['progress', 'agent_speech']: # Only forward these types
                        yield self._format_sse(proposal_event_data, event_name=event_type)
                await asyncio.sleep(0.01)

            if not current_action_proposal:
                err_msg = "Agent failed to generate a complete action proposal."
                logger.error(f"SSE STREAM: {err_msg}")
                yield self._format_sse({'type': 'error', 'message': err_msg}, event_name='error')
                if db_step: await self.db.update_step(task_id=task.task_id, step_id=db_step.step_id, status="error", output=err_msg, is_last=True)
                agent_has_finished_overall_task = True

            action_result_obj: Optional[ActionResult] = None
            if current_action_proposal and not agent_has_finished_overall_task and not stream_was_paused_for_ask_user:
                finish_command_name = getattr(self.app_config, 'finish_command_name', 'finish')
                proposed_tool_name: Optional[str] = current_action_proposal.use_tool.name if current_action_proposal.use_tool else None

                if not proposed_tool_name or proposed_tool_name == finish_command_name or proposed_tool_name == "None":
                    speak_output = current_action_proposal.thoughts.speak if current_action_proposal.thoughts and current_action_proposal.thoughts.speak else "Task step completed."
                    if proposed_tool_name == finish_command_name:
                        if isinstance(current_action_proposal.use_tool.arguments, dict): speak_output = current_action_proposal.use_tool.arguments.get("reason", speak_output)
                        agent_has_finished_overall_task = True
                    action_result_obj = ActionSuccessResult(outputs=speak_output)
                    yield self._format_sse({'type': 'agent_speech', 'message': speak_output}, event_name='agent_speech')
                elif proposed_tool_name: # ASK_COMMAND was handled
                    yield self._format_sse({'type': 'progress', 'message': f"Executing tool: {proposed_tool_name}", 'stage': 'tool_execution_start'}, event_name='progress')
                    async for exec_event_data in core_agent.execute(proposal=current_action_proposal):
                        if isinstance(exec_event_data, ActionResult): action_result_obj = exec_event_data; break
                        elif isinstance(exec_event_data, dict): yield self._format_sse(exec_event_data, event_name=exec_event_data.get('type','progress'))
                        await asyncio.sleep(0.01)
                    if not action_result_obj: action_result_obj = ActionErrorResult(reason=f"Tool '{proposed_tool_name}' did not yield ActionResult.")
                    if isinstance(action_result_obj, ActionSuccessResult) and action_result_obj.outputs and isinstance(action_result_obj.outputs, str) and action_result_obj.outputs.strip():
                        yield self._format_sse({'type': 'agent_speech', 'message': str(action_result_obj.outputs)}, event_name='agent_speech')
                    elif isinstance(action_result_obj, ActionErrorResult):
                        yield self._format_sse({'type': 'error', 'message': f"Tool error ({proposed_tool_name}): {action_result_obj.reason}"}, event_name='error')

        except AgentTerminated as e:
            logger.info(f"SSE STREAM: AgentTerminated: {e}")
            yield self._format_sse({'type': 'error', 'message': f"Agent session terminated: {e}"}, event_name='error')
            agent_has_finished_overall_task = True
        except AgentException as e:
            logger.error(f"SSE STREAM: AgentException: {e}", exc_info=True)
            yield self._format_sse({'type': 'error', 'message': f"Agent error: {e}"}, event_name='error')
            agent_has_finished_overall_task = True
        except Exception as e_outer:
            logger.critical(f"SSE STREAM: Unexpected critical error: {e_outer}", exc_info=True)
            yield self._format_sse({'type': 'critical_error', 'message': f"Server error: {e_outer}"}, event_name='error')
            agent_has_finished_overall_task = True
        finally:
            if not stream_was_paused_for_ask_user:
                final_db_step_output = "Interaction cycle processing completed."
                final_db_additional_output = {}
                step_status = "completed"
                if 'action_result_obj' in locals() and action_result_obj:
                    if isinstance(action_result_obj, ActionErrorResult): final_db_step_output = action_result_obj.reason or "Action error."; step_status = "error"
                    elif hasattr(action_result_obj, 'outputs') and action_result_obj.outputs is not None: final_db_step_output = str(action_result_obj.outputs)
                    final_db_additional_output['action_result'] = action_result_obj.model_dump(exclude_none=True) if hasattr(action_result_obj, "model_dump") else vars(action_result_obj)
                elif 'current_action_proposal' in locals() and current_action_proposal:
                    final_db_step_output = current_action_proposal.thoughts.speak if current_action_proposal.thoughts and current_action_proposal.thoughts.speak else "Proposal generated."
                    if agent_has_finished_overall_task and not ('action_result_obj' in locals() and action_result_obj): step_status = "error" #This logic might be revisited for accuracy
                    final_db_additional_output['final_proposal'] = current_action_proposal.model_dump(exclude_none=True) if hasattr(current_action_proposal, "model_dump") else vars(current_action_proposal)
                elif agent_has_finished_overall_task : step_status = "error"; final_db_step_output = "Task ended due to error."


                if db_step:
                    try:
                        updated_step = await self.db.update_step(task_id=task.task_id, step_id=db_step.step_id, status=step_status, output=final_db_step_output[:4000], additional_output=final_db_additional_output, is_last=agent_has_finished_overall_task)
                        yield self._format_sse({"type": "sse_stream_step_completed", "final_db_step_details": updated_step.model_dump(exclude_none=True) if hasattr(updated_step, "model_dump") else vars(updated_step)}, event_name="sse_stream_step_completed")
                    except Exception as e_db:
                        logger.error(f"SSE STREAM: DB update error in finally: {e_db}", exc_info=True)
                        yield self._format_sse({'type': 'error', 'message': f'DB final update error: {e_db}'}, event_name='error')

                if 'core_agent' in locals() and isinstance(core_agent, CoreAutoGPTAgent): # Ensure core_agent is defined and of correct type
                    await self.agent_manager.save_agent_state(core_agent)
                logger.info(f"SSE STREAM: END for task '{task.task_id}', Step '{getattr(db_step, 'step_id', 'N/A')}'. Finished: {agent_has_finished_overall_task}")
                
                # --- START OF THE "EXECUTION PHASE" BLOCK (Oddly placed inside 'finally') ---
                # This block's placement here is highly unconventional and likely a logical error,
                # but it is not a Python syntax error in itself.
                # As per "only fixing syntax", this structure is preserved.
                action_result_obj: Optional[ActionResult] = None 
                if current_action_proposal and not agent_has_finished_overall_task and not stream_was_paused_for_ask_user:
                    finish_command_name = getattr(self.app_config, 'finish_command_name', 'finish')

                    proposed_tool_name: Optional[str] = None
                    if current_action_proposal.use_tool and hasattr(current_action_proposal.use_tool, 'name'):
                        proposed_tool_name = current_action_proposal.use_tool.name

                    if not proposed_tool_name or proposed_tool_name == finish_command_name or proposed_tool_name == "None":
                        speak_output = "Agent has no further action."
                        if current_action_proposal.thoughts and current_action_proposal.thoughts.speak:
                            speak_output = current_action_proposal.thoughts.speak

                        if proposed_tool_name == finish_command_name:
                            if current_action_proposal.use_tool and current_action_proposal.use_tool.arguments and isinstance(current_action_proposal.use_tool.arguments, dict):
                                 speak_output = current_action_proposal.use_tool.arguments.get("reason", speak_output)
                            # agent_has_finished_overall_task = True # Already handled in try, this might be redundant or cause issues
                            logger.info(f"SSE STREAM: Task '{task.task_id}', Agent indicates FINISH (in finally block). Reason: '{speak_output}'")
                        else:
                            logger.info(f"SSE STREAM: Task '{task.task_id}', Agent proposed NO TOOL or tool 'None' (in finally block). Spoken: '{speak_output}'")

                        action_result_obj = ActionSuccessResult(outputs=speak_output)
                        yield self._format_sse(
                            {'type': 'agent_speech', 'message': speak_output,
                             'stage': 'finish_or_no_tool_finally' if agent_has_finished_overall_task else 'no_tool_continue_finally'},
                            event_name='agent_speech'
                        )
                    elif proposed_tool_name: 
                        logger.info(f"SSE STREAM: Task '{task.task_id}', Agent executing tool (in finally block): '{proposed_tool_name}'")
                        yield self._format_sse({'type': 'progress', 'message': f"Executing tool: {proposed_tool_name}", 'stage': 'tool_execution_start_finally'}, event_name='progress')
                        
                        async for exec_event_data in core_agent.execute(proposal=current_action_proposal): # Potential re-execution
                            if isinstance(exec_event_data, ActionResult):
                                action_result_obj = exec_event_data
                                logger.info(f"SSE STREAM: Task '{task.task_id}', Tool '{proposed_tool_name}' result (in finally): {type(action_result_obj).__name__}")
                                if isinstance(action_result_obj, ActionSuccessResult) and action_result_obj.outputs:
                                    if isinstance(action_result_obj.outputs, str) and action_result_obj.outputs.strip():
                                        logger.info(f"SSE STREAM: Tool '{proposed_tool_name}' produced direct output (in finally): {action_result_obj.outputs[:100]}")
                                        yield self._format_sse(
                                            {'type': 'agent_speech', 'message': str(action_result_obj.outputs)},
                                            event_name='agent_speech'
                                        )
                                elif isinstance(action_result_obj, ActionErrorResult):
                                    logger.error(f"SSE STREAM: Tool '{proposed_tool_name}' error (in finally): {action_result_obj.reason}")
                                    yield self._format_sse(
                                        {'type': 'error', 'message': f"Error executing tool {proposed_tool_name} (in finally): {action_result_obj.reason}", 'stage': 'tool_execution_error_finally'},
                                        event_name='error'
                                    )
                                break
                            elif isinstance(exec_event_data, dict):
                                event_type = exec_event_data.get('type', 'progress')
                                yield self._format_sse(exec_event_data, event_name=event_type)
                            else:
                                logger.warning(f"SSE STREAM: Task '{task.task_id}', Unhandled event from execute (in finally): {type(exec_event_data)}")
                            await asyncio.sleep(0.01)

                        if not action_result_obj:
                            action_result_obj = ActionErrorResult(reason=f"Tool '{proposed_tool_name}' did not yield clear ActionResult after execution (in finally).")
                            logger.error(f"SSE STREAM: Task '{task.task_id}', {action_result_obj.reason}")
                # --- END OF THE "EXECUTION PHASE" BLOCK ---
            else: # This else corresponds to 'if not stream_was_paused_for_ask_user:'
                 logger.info(f"SSE STREAM: END for task '{task.task_id}', Step '{getattr(db_step, 'step_id', 'N/A')}' because stream paused for ask_user.")
        # As per comments in the prompt, any subsequent `except` or `finally` blocks that would cause a SyntaxError
        # have already been removed from this snippet. This `pass` marks the end of the method.
        pass


    async def execute_step(self, task_id: str, step_request: StepRequestBody, http_request: Request) -> StreamingResponse:
        logger.info(f"AP Server: execute_step endpoint for task='{task_id}'. Input='{step_request.input[:100] if step_request.input else '(None)'}'. Additional: {step_request.additional_input}")
        try:
            task_for_processing = await self.db.get_task(task_id)
        except NotFoundError:
            logger.warning(f"Task '{task_id}' not found for execute_step.")
            raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")

        sse_headers = {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive', 'Content-Type': 'text/event-stream'}
        return StreamingResponse(
            self._core_agent_streaming_interaction_logic(
                task=task_for_processing,
                client_step_request_body=step_request
            ),
            media_type="text/event-stream",
            headers=sse_headers
        )

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        logger.info(f"AP Server: create_task. Input: '{task_request.input[:100] if task_request.input else 'N/A'}'")
        try:
            additional_input_data = task_request.additional_input
            if not isinstance(additional_input_data, dict):
                logger.warning(f"task_request.additional_input was not a dict (type: {type(additional_input_data)}), defaulting to empty dict.")
                additional_input_data = {}
            task = await self.db.create_task(
                input=task_request.input,
                additional_input=additional_input_data,
            )
            logger.info(f"AP Server: Task '{task.task_id}' created successfully in DB.")
            return task
        except Exception as e:
            logger.error(f"AP Server: Error creating task in database: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Database error while creating task: {str(e)}")

    async def list_tasks(self, page: int = 1, page_size: int = 10) -> TaskListResponse:
        logger.debug(f"AP Server: list_tasks. Page: {page}, PageSize: {page_size}")
        tasks, pagination_details = await self.db.list_tasks(page=page, per_page=page_size)
        if not isinstance(pagination_details, Pagination):
             logger.error(f"DB list_tasks returned invalid pagination type: {type(pagination_details)}. Expected Pagination model.")
             total_items = len(tasks) if tasks else 0
             total_pages = (total_items + page_size -1) // page_size if page_size > 0 else 0
             pagination_details = Pagination(total_items=total_items, total_pages=total_pages , current_page=page, page_size=page_size)
        return TaskListResponse(tasks=tasks, pagination=pagination_details)

    async def get_task(self, task_id: str) -> Task:
        logger.debug(f"AP Server: get_task for task_id: '{task_id}'")
        try:
            return await self.db.get_task(task_id)
        except NotFoundError as e:
            logger.warning(f"Task '{task_id}' not found for get_task: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"AP Server: Error getting task '{task_id}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error retrieving task: {str(e)}")

    async def list_steps(
        self,
        task_id: str,
        request: Request,
        input: Optional[str] = None,
        page: int = Query(1, ge=1),
        page_size: int = Query(10, ge=1, le=100),
    ) -> Any:
        logger.debug(f"AP Server: list_steps for task_id: '{task_id}'. Input query: '{input[:50] if input else None}'. Page: {page}, PageSize: {page_size}")
        if input is not None:
            logger.info(f"GET /steps for task '{task_id}' with input='{input}'. Executing as a new step.")
            additional_input_from_query_str = request.query_params.get("additional_input")
            parsed_additional_input = {}
            if additional_input_from_query_str:
                try:
                    parsed_additional_input = json.loads(additional_input_from_query_str)
                    if not isinstance(parsed_additional_input, dict):
                        logger.warning(f"Parsed additional_input query param '{additional_input_from_query_str}' is not a dict. Defaulting to empty.")
                        parsed_additional_input = {}
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse additional_input query param as JSON: '{additional_input_from_query_str}'. Defaulting to empty.")
                    parsed_additional_input = {}
            step_request_body = StepRequestBody(input=input, additional_input=parsed_additional_input)
            return await self.execute_step(task_id, step_request_body, request)
        else:
            steps, pagination_details = await self.db.list_steps(task_id=task_id, page=page, per_page=page_size)
            if not isinstance(pagination_details, Pagination):
                 logger.error(f"DB list_steps returned invalid pagination type: {type(pagination_details)}. Expected Pagination model.")
                 total_items = len(steps) if steps else 0
                 total_pages = (total_items + page_size -1) // page_size if page_size > 0 else 0
                 pagination_details = Pagination(total_items=total_items, total_pages=total_pages, current_page=page, page_size=page_size)
            return TaskStepsListResponse(steps=steps, pagination=pagination_details)

    async def get_step(self, task_id: str, step_id: str) -> Step:
        logger.debug(f"AP Server: get_step for task_id: '{task_id}', step_id: '{step_id}'")
        try:
            return await self.db.get_step(task_id=task_id, step_id=step_id)
        except NotFoundError as e:
            logger.warning(f"Step '{step_id}' not found for task '{task_id}': {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"AP Server: Error getting step '{step_id}' for task '{task_id}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error retrieving step: {str(e)}")

    async def list_artifacts(self, task_id: str, page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)) -> TaskArtifactsListResponse:
        logger.debug(f"AP Server: list_artifacts for task_id: '{task_id}'. Page: {page}, PageSize: {page_size}")
        try:
            artifacts, pagination_details = await self.db.list_artifacts(task_id=task_id, page=page, per_page=page_size)
            if not isinstance(pagination_details, Pagination):
                 logger.error(f"DB list_artifacts returned invalid pagination type: {type(pagination_details)}. Expected Pagination model.")
                 total_items = len(artifacts) if artifacts else 0
                 total_pages = (total_items + page_size -1) // page_size if page_size > 0 else 0
                 pagination_details = Pagination(total_items=total_items, total_pages=total_pages, current_page=page, page_size=page_size)
            return TaskArtifactsListResponse(artifacts=artifacts, pagination=pagination_details)
        except NotFoundError:
             logger.warning(f"Task '{task_id}' not found when listing artifacts.")
             raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
        except Exception as e:
            logger.error(f"AP Server: Error listing artifacts for task '{task_id}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error retrieving artifacts: {str(e)}")

    async def create_artifact(self, task_id: str, file: UploadFile = File(...), relative_path: Optional[str] = Body(None)) -> Artifact:
        file_name = file.filename or f"upload_{uuid.uuid4().hex[:8]}.dat"
        logger.info(f"AP Server: create_artifact for task_id: '{task_id}', filename: '{file_name}', relative_path: '{relative_path}'")
        try:
            await self.db.get_task(task_id)
        except NotFoundError:
            logger.warning(f"Task '{task_id}' not found for create_artifact.")
            raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found, cannot create artifact.")

        task_artifact_storage = self._get_core_agent_workspace_filestorage(task_id)
        file_content = await file.read()
        save_path_in_task_storage: pathlib.Path
        if relative_path:
            if ".." in pathlib.Path(relative_path).parts:
                logger.error(f"Invalid relative_path for artifact: '{relative_path}' contains '..'")
                raise HTTPException(status_code=400, detail="Invalid relative_path for artifact, path traversal detected.")
            base_path = pathlib.Path(relative_path)
            if base_path.name != file_name and not base_path.suffix :
                save_path_in_task_storage = base_path / file_name
            else:
                save_path_in_task_storage = base_path
        else:
            save_path_in_task_storage = pathlib.Path(file_name)
        try:
            await task_artifact_storage.write_file(str(save_path_in_task_storage), file_content)
            artifact_uri_for_db = save_path_in_task_storage.as_posix()
            logger.info(f"Artifact '{file_name}' saved to storage. DB URI will be: '{artifact_uri_for_db}'")
        except Exception as e_save:
            logger.error(f"Error saving artifact file '{file_name}' to storage: {e_save}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to store artifact file: {str(e_save)}")
        try:
            db_artifact = await self.db.create_artifact(
                task_id=task_id, file_name=file_name, uri=artifact_uri_for_db, agent_created=False
            )
            logger.info(f"Artifact DB record created: ID '{db_artifact.artifact_id}'.")
            return db_artifact
        except Exception as e_db_create:
            logger.error(f"Error creating artifact DB record for '{file_name}': {e_db_create}", exc_info=True)
            try:
                await task_artifact_storage.delete_file(str(save_path_in_task_storage))
                logger.info(f"Cleaned up orphaned artifact file from storage: {save_path_in_task_storage}")
            except Exception as e_cleanup:
                logger.error(f"Failed to clean up orphaned file {save_path_in_task_storage} after DB error: {e_cleanup}")
            raise HTTPException(status_code=500, detail=f"Failed to create artifact DB record: {str(e_db_create)}")

    async def get_artifact(self, task_id: str, artifact_id: str) -> FileResponse:
        logger.debug(f"AP Server: get_artifact for task_id: '{task_id}', artifact_id: '{artifact_id}'")
        try:
            db_artifact = await self.db.get_artifact(artifact_id)
        except NotFoundError as e:
            logger.warning(f"Artifact '{artifact_id}' not found in DB: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        if hasattr(db_artifact, 'task_id') and db_artifact.task_id != task_id:
             logger.warning(f"Artifact '{artifact_id}' task_id mismatch. Req: '{task_id}', Found: '{db_artifact.task_id}'. Denied.")
             raise HTTPException(status_code=403, detail="Artifact does not belong to the specified task.")
        task_artifact_storage = self._get_core_agent_workspace_filestorage(task_id)
        artifact_relative_path = pathlib.Path(db_artifact.uri)
        absolute_artifact_path = (task_artifact_storage.root / artifact_relative_path).resolve()
        logger.debug(f"Attempting to serve artifact from absolute path: {absolute_artifact_path}")
        if not await task_artifact_storage.exists(str(artifact_relative_path)) or not absolute_artifact_path.is_file():
            logger.error(f"Artifact file '{db_artifact.file_name}' (uri: '{db_artifact.uri}') expected at '{absolute_artifact_path}' but not found or not a file.")
            raise HTTPException(status_code=404, detail="Artifact file not found on server storage.")
        return FileResponse(
            path=str(absolute_artifact_path), filename=db_artifact.file_name, media_type='application/octet-stream'
        )

    async def handle_chat_message(self, chat_request_body: ChatRequestBody, request: Request) -> Dict[str, str]:
        user_message = chat_request_body.message
        session_id = chat_request_body.session_id
        logger.info(f"AP Server: handle_chat_message (non-streaming). Session: '{session_id}'. Message='{user_message[:100]}'")
        is_temp_agent = False
        if session_id:
            chat_task_id = f"chat_session_{session_id}"
            task_initial_input_for_agent = f"Chat session '{session_id}' started with: {user_message}"
            try:
                task_obj = await self.db.get_task(chat_task_id)
                task_initial_input_for_agent = task_obj.input
                logger.info(f"Chat Handler: Using existing task '{chat_task_id}' for session '{session_id}'. Initial task input: '{task_initial_input_for_agent[:70]}'")
            except NotFoundError:
                logger.info(f"Chat Handler: No existing task for chat_task_id '{chat_task_id}'. Agent will be created. Initial task input: '{task_initial_input_for_agent[:70]}'")
        else:
            chat_task_id = f"temp_chat_single_turn_{uuid.uuid4().hex[:12]}"
            is_temp_agent = True
            task_initial_input_for_agent = f"Single turn chat interaction: {user_message}"
            logger.info(f"No session_id provided for chat, using temporary agent: {chat_task_id}")

        core_agent = await self._get_or_create_core_agent_instance(chat_task_id, task_initial_input_for_agent)

        if user_message and hasattr(core_agent.state, 'ai_profile') and isinstance(core_agent.state.ai_profile, AIProfile):
            if core_agent.state.ai_profile.get_task_description() != user_message:
                 logger.info(
                    f"Chat Handler: Updating agent's AIProfile task description for '{chat_task_id}' "
                    f"with current message: '{user_message[:70]}...'"
                )
                 core_agent.state.ai_profile.set_task_description(user_message)
                 if hasattr(core_agent.state, 'task'):
                    core_agent.state.task = user_message

        agent_spoken_response = "I was unable to process your message at this time."
        current_proposal: Optional[OneShotAgentActionProposal] = None
        action_result: Optional[ActionResult] = None
        try:
            async with asyncio.timeout(getattr(self.app_config, 'chat_processing_timeout', 60)):
                async for proposal_event in core_agent.propose_action(user_input=user_message):
                    if isinstance(proposal_event, OneShotAgentActionProposal):
                        current_proposal = proposal_event
                        break
                    elif isinstance(proposal_event, dict) and proposal_event.get('type') == 'thought_update': # Changed from 'thought_update' to 'progress' or similar based on other parts
                        logger.debug(f"Chat Handler - Thought Update: {proposal_event.get('thoughts')}") # Kept 'thoughts' assuming it's in the payload
                if not current_proposal:
                    raise AgentException("Agent did not decide on an action for the chat message.")
                
                proposed_tool_name: Optional[str] = None
                if current_proposal.use_tool and hasattr(current_proposal.use_tool, 'name'):
                    proposed_tool_name = current_proposal.use_tool.name
                
                finish_cmd = getattr(self.app_config, 'finish_command_name', 'finish')

                if not proposed_tool_name or proposed_tool_name == finish_cmd or proposed_tool_name == "None":
                    agent_spoken_response = current_proposal.thoughts.speak if current_proposal.thoughts and current_proposal.thoughts.speak else "No specific action taken. Is there anything else?"
                    if proposed_tool_name == finish_cmd:
                        logger.info(f"Chat Handler: Agent proposed 'finish' for chat task {chat_task_id}.")
                elif proposed_tool_name == ASK_COMMAND:
                     question = "Agent requires clarification."
                     if current_proposal.use_tool.arguments and isinstance(current_proposal.use_tool.arguments, dict):
                         question = current_proposal.use_tool.arguments.get('question', question)
                     logger.warning(f"Chat Handler: Agent proposed '{ASK_COMMAND}' ('{question}') in non-streaming chat. This mode might not be ideal for ASK_COMMAND.")
                     agent_spoken_response = (f"I have a question regarding that: '{question}'. This chat mode is not ideal for follow-up questions. Please try rephrasing or providing more details.")
                else: # Regular tool execution
                    logger.info(f"Chat Handler: Agent for '{chat_task_id}' executing tool: '{proposed_tool_name}'")
                    async for exec_result_event in core_agent.execute(proposal=current_proposal):
                        if isinstance(exec_result_event, ActionResult):
                            action_result = exec_result_event
                            break
                        elif isinstance(exec_result_event, dict) and exec_result_event.get('type') == 'progress':
                             logger.debug(f"Chat Handler - Tool Execution Progress: {exec_result_event.get('message')}")
                    
                    if not action_result:
                        raise AgentException(f"Tool '{proposed_tool_name}' execution did not yield an ActionResult.")

                    if hasattr(action_result, 'status') and action_result.status == "success": # type: ignore
                        if current_proposal.thoughts and current_proposal.thoughts.speak: # Prefer agent's summary if available
                            agent_spoken_response = current_proposal.thoughts.speak
                        elif hasattr(action_result, 'outputs') and action_result.outputs is not None: # type: ignore
                            agent_spoken_response = str(action_result.outputs) # type: ignore
                        else:
                            agent_spoken_response = f"I've completed the action: {proposed_tool_name}."
                        logger.info(f"Chat Handler: Tool '{proposed_tool_name}' success. Response: '{agent_spoken_response[:100]}'")
                    else: # Error or non-success status
                        error_reason = "Unknown error during tool execution."
                        if hasattr(action_result, 'reason') and action_result.reason: # type: ignore
                            error_reason = action_result.reason # type: ignore
                        agent_spoken_response = f"I encountered an error while trying to use {proposed_tool_name}: {error_reason}"
                        logger.error(f"Chat Handler: Tool '{proposed_tool_name}' error: {error_reason}")

                if not is_temp_agent:
                    await self.agent_manager.save_agent_state(core_agent)
                    logger.info(f"Chat Handler: State saved for persistent chat agent '{chat_task_id}'.")

        except asyncio.TimeoutError:
            logger.error(f"Chat Handler: Timeout processing message for agent '{chat_task_id}'.")
            agent_spoken_response = "Sorry, I took too long to process that. Could you try again?"
        except AgentTerminated as e:
            logger.info(f"Chat Handler: Agent '{chat_task_id}' terminated: {e}")
            agent_spoken_response = f"The chat session has ended: {e}"
        except AgentException as e:
            logger.error(f"Chat Handler: AgentException for '{chat_task_id}': {e}", exc_info=True)
            agent_spoken_response = f"I encountered an agent-related issue: {e}"
        except Exception as e:
            logger.error(f"Chat Handler: Unexpected error for '{chat_task_id}': {e}", exc_info=True)
            agent_spoken_response = "An internal error occurred. Please try again later."
        finally:
            if is_temp_agent:
                if hasattr(self.agent_manager, 'delete_agent_state'):
                    try:
                        await self.agent_manager.delete_agent_state(chat_task_id)
                        logger.info(f"Chat Handler: Temporary agent state '{chat_task_id}' deleted.")
                    except Exception as e_del:
                        logger.error(f"Chat Handler: Failed to delete temporary agent state '{chat_task_id}': {e_del}")
                else:
                     logger.warning(f"Chat Handler: AgentManager does not have 'delete_agent_state' method. Temp state for '{chat_task_id}' may persist.")

        return {"response": agent_spoken_response}