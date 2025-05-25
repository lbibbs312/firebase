# Updated api_router.py

import logging
from typing import TYPE_CHECKING, Optional, Dict # Added Dict for response model

from fastapi import APIRouter, HTTPException, Query, Request, Response, UploadFile, Body # Added Body
from fastapi.responses import StreamingResponse, JSONResponse # Added JSONResponse
from pydantic import BaseModel # For request body of chat

# Assuming models are in the same directory or a .models submodule
try:
    from .models import (
        Artifact,
        Step,
        StepRequestBody,
        Task,
        TaskArtifactsListResponse,
        TaskListResponse,
        TaskRequestBody,
        TaskStepsListResponse,
    )
except ImportError:
    # Fallback for different project structures if needed
    # from forge.agent_protocol.models import (...) # type: ignore
    raise # Or handle more gracefully depending on expected structure

if TYPE_CHECKING:
    # This should point to the actual implementation injected by the middleware
    from autogpt.app.agent_protocol_server import AgentProtocolServer as ActualAgentImplementationType

base_router = APIRouter()
logger = logging.getLogger(__name__)


# --- Pydantic model for the chat request body ---
class ChatRequestBody(BaseModel):
    message: str

# --- Pydantic model for the chat response body (optional but good practice) ---
class ChatResponseBody(BaseModel):
    response: str


@base_router.get("/", tags=["root"])
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return Response(content="Welcome to the AutoGPT Forge Agent Protocol Server (J.A.R.V.I.S. Edition)")


@base_router.get("/heartbeat", tags=["server"])
async def check_server_status():
    """
    Check if the server is running.
    """
    return Response(content="Server is running.", status_code=200)


# --- NEW CHAT ENDPOINT ---
@base_router.post("/agent/chat", tags=["agent"], response_model=ChatResponseBody)
async def agent_chat_endpoint(
    request: Request,
    chat_request: ChatRequestBody = Body(...) # Use Body for POST request payload
):
    """
    Handles a chat message from the user and returns the agent's response.
    """
    agent: "ActualAgentImplementationType" = request["agent"]
    try:
        # The handle_chat_message method in AgentProtocolServer will take the user_message
        # and return a dictionary like {"response": "agent's reply"}
        response_data = await agent.handle_chat_message(user_message=chat_request.message)
        return ChatResponseBody(**response_data) # Ensure it matches the response_model
    except HTTPException:
        # Re-raise HTTPException directly if it's already one (e.g., from DB layer)
        raise
    except Exception as e:
        logger.exception(f"Error in agent chat endpoint for message: '{chat_request.message[:100]}...'")
        # Return a structured JSON error response
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {str(e)}")


@base_router.post("/agent/tasks", tags=["agent"], response_model=Task)
async def create_agent_task(request: Request, task_request: TaskRequestBody) -> Task:
    """
    Creates a new task using the provided TaskRequestBody and returns a Task.
    """
    agent: "ActualAgentImplementationType" = request["agent"]

    try:
        task = await agent.create_task(task_request)
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error whilst trying to create a task: {task_request.input[:100] if task_request.input else 'N/A'}")
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")


@base_router.get("/agent/tasks", tags=["agent"], response_model=TaskListResponse)
async def list_agent_tasks(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, alias="pageSize"), # Use alias for consistency
) -> TaskListResponse:
    """
    Retrieves a paginated list of all tasks.
    """
    agent: "ActualAgentImplementationType" = request["agent"]
    try:
        tasks_response = await agent.list_tasks(page=page, page_size=page_size) # Use page_size
        return tasks_response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error whilst trying to list tasks")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")


@base_router.get("/agent/tasks/{task_id}", tags=["agent"], response_model=Task)
async def get_agent_task(request: Request, task_id: str) -> Task:
    """
    Gets the details of a task by ID.
    """
    agent: "ActualAgentImplementationType" = request["agent"]
    try:
        task = await agent.get_task(task_id)
        return task
    except HTTPException: # Catch specific FastAPI/custom HTTP exceptions first
        raise
    except Exception as e: # Catch any other exceptions
        logger.exception(f"Error whilst trying to get task: {task_id}")
        raise HTTPException(status_code=500, detail=f"Failed to get task {task_id}: {str(e)}")


@base_router.get(
    "/agent/tasks/{task_id}/steps",
    tags=["agent"], # Added tags for consistency
    response_model=TaskStepsListResponse, # Default response model for listing steps
    # Note: This endpoint can also return StreamingResponse if 'input' is provided.
    # FastAPI handles this by trying the response_model first, then falls back if types don't match.
    # For more explicit typing, one might separate GET with 'input' into its own endpoint.
)
async def list_agent_task_steps(
    request: Request,
    task_id: str,
    input: Optional[str] = Query(None),  # If input is provided, it becomes a step execution request
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, alias="pageSize"),
): # Return type can be TaskStepsListResponse or StreamingResponse
    """
    Retrieve a paginated list of steps for a given task.
    If ?input=foo is supplied, the agent treats it as a streaming step request (GET based execution).
    """
    agent: "ActualAgentImplementationType" = request["agent"]
    try:
        # The agent.list_steps method itself handles the conditional logic
        # for returning either TaskStepsListResponse or StreamingResponse.
        return await agent.list_steps(
            task_id=task_id,
            request=request,
            input=input, # This also needs to be handled if list_steps doesn't accept it anymore
            page=page,
            page_size=page_size, # Corrected
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error whilst trying to list or execute step for task %s", task_id)
        detail_msg = f"Failed to list/execute step(s) for task {task_id}: {e}"
        # If it was a streaming attempt that failed early, this is okay.
        # If it was listing, this is also okay.
        raise HTTPException(status_code=500, detail=detail_msg) from e


@base_router.post("/agent/tasks/{task_id}/steps", tags=["agent"], response_class=StreamingResponse)
async def execute_agent_task_step(
    request: Request, # FastAPI Request object
    task_id: str,
    step_request: Optional[StepRequestBody] = Body(None) # Use Body, make it optional
) -> StreamingResponse:
    """
    Executes the next step for a specified task and streams the output.
    If no StepRequestBody is provided, a default one might be used by the agent.
    """
    agent: "ActualAgentImplementationType" = request["agent"]
    try:
        # Use a default StepRequestBody if none is provided by the client
        current_step_request = step_request if step_request is not None else StepRequestBody(input="y") # Or handle as error if required
        # Pass the FastAPI 'request' object to agent.execute_step
        return await agent.execute_step(task_id, current_step_request, request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error whilst trying to execute a task step for task_id: {task_id}")
        raise HTTPException(status_code=500, detail=f"Failed to execute step for task {task_id}: {str(e)}")


@base_router.get(
    "/agent/tasks/{task_id}/steps/{step_id}", tags=["agent"], response_model=Step
)
async def get_agent_task_step(request: Request, task_id: str, step_id: str) -> Step:
    """
    Retrieves the details of a specific step for a given task.
    """
    agent: "ActualAgentImplementationType" = request["agent"]
    try:
        step = await agent.get_step(task_id, step_id) # Make sure agent has get_step method
        return step
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error whilst trying to get step: {step_id} for task {task_id}")
        raise HTTPException(status_code=500, detail=f"Failed to get step {step_id} for task {task_id}: {str(e)}")


@base_router.get(
    "/agent/tasks/{task_id}/artifacts",
    tags=["agent"],
    response_model=TaskArtifactsListResponse,
)
async def list_agent_task_artifacts(
    request: Request,
    task_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, alias="pageSize"),
) -> TaskArtifactsListResponse:
    """
    Retrieves a paginated list of artifacts associated with a specific task.
    """
    agent: "ActualAgentImplementationType" = request["agent"]
    try:
        artifacts_response = await agent.list_artifacts(task_id=task_id, page=page, pageSize=page_size) # Make sure agent has list_artifacts
        return artifacts_response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error whilst trying to list artifacts for task {task_id}")
        raise HTTPException(status_code=500, detail=f"Failed to list artifacts for task {task_id}: {str(e)}")


@base_router.post(
    "/agent/tasks/{task_id}/artifacts", tags=["agent"], response_model=Artifact
)
async def upload_agent_task_artifacts(
    request: Request,
    task_id: str,
    file: UploadFile, # Removed File(...) as UploadFile is sufficient
    relative_path: Optional[str] = Query(default=None)
) -> Artifact:
    """
    This endpoint is used to upload an artifact (file) associated with a specific task.
    """
    agent: "ActualAgentImplementationType" = request["agent"]

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="File with a filename must be specified in the upload.")
    try:
        artifact = await agent.create_artifact(task_id, file, relative_path)
        return artifact
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error whilst trying to upload artifact for task: {task_id}")
        raise HTTPException(status_code=500, detail=f"Failed to upload artifact for task {task_id}: {str(e)}")


@base_router.get(
    "/agent/tasks/{task_id}/artifacts/{artifact_id}",
    tags=["agent"],
    response_class=StreamingResponse, # For file downloads, StreamingResponse is appropriate
)
async def download_agent_task_artifact(
    request: Request, task_id: str, artifact_id: str
) -> StreamingResponse: # Return type is StreamingResponse
    """
    Downloads an artifact associated with a specific task.
    """
    agent: "ActualAgentImplementationType" = request["agent"]
    try:
        # The agent.get_artifact method should return a FileResponse or similar StreamingResponse
        return await agent.get_artifact(task_id, artifact_id) # Pass task_id if needed by agent.get_artifact
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error whilst trying to download artifact: {artifact_id} for task {task_id}")
        raise HTTPException(status_code=500, detail=f"Failed to download artifact {artifact_id}: {str(e)}")
