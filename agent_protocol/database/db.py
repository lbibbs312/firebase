"""
This is an example implementation of the Agent Protocol DB for development Purposes
It uses SQLite as the database and file store backend.
IT IS NOT ADVISED TO USE THIS IN PRODUCTION!
"""

import logging
import math
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    joinedload,
    mapped_column,
    relationship,
    sessionmaker,
)

from forge.utils.exceptions import NotFoundError

from ..models.artifact import Artifact
from ..models.pagination import Pagination
from ..models.task import Step, StepRequestBody, StepStatus, Task

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSON,
    }


class TaskModel(Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(primary_key=True, index=True)
    input: Mapped[str]
    additional_input: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    artifacts = relationship("ArtifactModel", back_populates="task")


class StepModel(Base):
    __tablename__ = "steps"

    step_id: Mapped[str] = mapped_column(primary_key=True, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"))
    name: Mapped[str]
    input: Mapped[str]
    status: Mapped[str]
    output: Mapped[Optional[str]]
    is_last: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    additional_input: Mapped[dict[str, Any]] = mapped_column(default=dict)
    additional_output: Mapped[Optional[dict[str, Any]]]
    artifacts = relationship("ArtifactModel", back_populates="step")


class ArtifactModel(Base):
    __tablename__ = "artifacts"

    artifact_id: Mapped[str] = mapped_column(primary_key=True, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"))
    step_id: Mapped[Optional[str]] = mapped_column(ForeignKey("steps.step_id"))
    agent_created: Mapped[bool] = mapped_column(default=False)
    file_name: Mapped[str]
    relative_path: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, onupdate=datetime.utcnow
    )

    step = relationship("StepModel", back_populates="artifacts")
    task = relationship("TaskModel", back_populates="artifacts")


def convert_to_task(task_obj: TaskModel, debug_enabled: bool = False) -> Task:
    if debug_enabled:
        logger.debug(f"Converting TaskModel to Task for task_id: {task_obj.task_id}")
    task_artifacts = [convert_to_artifact(artifact) for artifact in task_obj.artifacts]
    return Task(
        task_id=task_obj.task_id,
        created_at=task_obj.created_at,
        modified_at=task_obj.modified_at,
        input=task_obj.input,
        additional_input=task_obj.additional_input,
        artifacts=task_artifacts,
    )


def convert_to_step(step_model: StepModel, debug_enabled: bool = False) -> Step:
    if debug_enabled:
        logger.debug(f"Converting StepModel to Step for step_id: {step_model.step_id}")
    step_artifacts = [
        convert_to_artifact(artifact) for artifact in step_model.artifacts
    ]
    status = (
        StepStatus.completed if step_model.status == "completed" else StepStatus.created
    )
    return Step(
        task_id=step_model.task_id,
        step_id=step_model.step_id,
        created_at=step_model.created_at,
        modified_at=step_model.modified_at,
        name=step_model.name,
        input=step_model.input,
        status=status,
        output=step_model.output,
        artifacts=step_artifacts,
        is_last=step_model.is_last == 1,
        additional_input=step_model.additional_input,
        additional_output=step_model.additional_output,
    )


def convert_to_artifact(artifact_model: ArtifactModel) -> Artifact:
    return Artifact(
        artifact_id=artifact_model.artifact_id,
        created_at=artifact_model.created_at,
        modified_at=artifact_model.modified_at,
        agent_created=artifact_model.agent_created,
        relative_path=artifact_model.relative_path,
        file_name=artifact_model.file_name,
    )


# sqlite:///{database_name}

class AgentDB:
    def __init__(self, database_string: str, debug_enabled: bool = False) -> None:
        super().__init__()
        self.debug_enabled = debug_enabled
        if self.debug_enabled:
            logger.debug(f"Initializing AgentDB with database_string: {database_string}")

        # ← Make sure this is indented under __init__, not out at class-level!
        self.engine = create_engine(
            database_string,
            connect_args={"check_same_thread": False},  # Windows + SQLite needs this
            echo=self.debug_enabled                  # optional SQL echo
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def close(self) -> None:
        self.Session.close_all()
        self.engine.dispose()

# In AgentDB class

    async def create_task(
        self, input: Optional[str], additional_input: Optional[dict] = {}
    ) -> Task:
        task_id_to_create = str(uuid.uuid4()) # Generate ID upfront for logging
        if self.debug_enabled:
            logger.debug(f"DB: Attempting to create new task with proposed task_id: {task_id_to_create}")

        try:
            with self.Session() as session:
                new_task = TaskModel(
                    task_id=task_id_to_create, # Use pre-generated ID
                    input=input,
                    additional_input=additional_input if additional_input else {},
                )
                session.add(new_task)
                logger.debug(f"DB: TaskModel instance created and added to session for task_id: {task_id_to_create}")
                session.commit() # <<<<<<<<<<<<< CRITICAL POINT
                logger.info(f"DB: SUCCESSFULLY COMMITTED new task with task_id: {task_id_to_create}") # Explicit success log
                session.refresh(new_task)
                return convert_to_task(new_task, self.debug_enabled)
        except SQLAlchemyError as e:
            logger.error(f"DB: SQLAlchemyError during commit or refresh for task_id {task_id_to_create}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"DB: Unexpected error during create_task for task_id {task_id_to_create}: {e}", exc_info=True)
            raise

    async def get_task(self, task_id: str) -> Task:
        if self.debug_enabled:
            logger.debug(f"DB: Attempting to get task with task_id: {task_id}")
        try:
            with self.Session() as session:
                task_obj = (
                    session.query(TaskModel)
                    .options(joinedload(TaskModel.artifacts)) # Eager load artifacts
                    .filter(TaskModel.task_id == task_id) # Use explicit filter comparison
                    .first()
                )
                if task_obj:
                    logger.info(f"DB: Task FOUND with task_id: {task_id}")
                    return convert_to_task(task_obj, self.debug_enabled)
                else:
                    logger.error(f"DB: Task NOT FOUND with task_id: {task_id}. Raising NotFoundError.")
                    raise NotFoundError(f"Task with task_id {task_id} not found in database.") # More specific error message
        except SQLAlchemyError as e:
            logger.error(f"DB: SQLAlchemyError while getting task {task_id}: {e}", exc_info=True)
            raise
        except NotFoundError: # Re-raise NotFoundError specifically so it's not caught by generic Exception
            raise
        except Exception as e:
            logger.error(f"DB: Unexpected error while getting task {task_id}: {e}", exc_info=True)
            raise

    async def create_step(
        self,
        task_id: str,
        input: StepRequestBody,
        is_last: bool = False,
        additional_input: Optional[Dict[str, Any]] = {},
    ) -> Step:
        if self.debug_enabled:
            logger.debug(f"Creating new step for task_id: {task_id}")
        try:
            with self.Session() as session:
                new_step = StepModel(
                    task_id=task_id,
                    step_id=str(uuid.uuid4()),
                    name=input.input,
                    input=input.input,
                    status="created",
                    is_last=is_last,
                    additional_input=additional_input,
                )
                session.add(new_step)
                session.commit()
                session.refresh(new_step)
                if self.debug_enabled:
                    logger.debug(f"Created new step with step_id: {new_step.step_id}")
                return convert_to_step(new_step, self.debug_enabled)
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error while creating step: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while creating step: {e}")
            raise

    async def create_artifact(
        self,
        task_id: str,
        file_name: str,
        relative_path: str,
        agent_created: bool = False,
        step_id: str | None = None,
    ) -> Artifact:
        if self.debug_enabled:
            logger.debug(f"Creating new artifact for task_id: {task_id}")
        try:
            with self.Session() as session:
                if (
                    existing_artifact := session.query(ArtifactModel)
                    .filter_by(
                        task_id=task_id,
                        file_name=file_name,
                        relative_path=relative_path,
                    )
                    .first()
                ):
                    session.close()
                    if self.debug_enabled:
                        logger.debug(
                            f"Artifact {file_name} already exists at {relative_path}/"
                        )
                    return convert_to_artifact(existing_artifact)

                new_artifact = ArtifactModel(
                    artifact_id=str(uuid.uuid4()),
                    task_id=task_id,
                    step_id=step_id,
                    agent_created=agent_created,
                    file_name=file_name,
                    relative_path=relative_path,
                )
                session.add(new_artifact)
                session.commit()
                session.refresh(new_artifact)
                if self.debug_enabled:
                    logger.debug(
                        f"Created new artifact with ID: {new_artifact.artifact_id}"
                    )
                return convert_to_artifact(new_artifact)
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error while creating step: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while creating step: {e}")
            raise

    async def get_task(self, task_id: str) -> Task:
        """Get a task by its id"""
        if self.debug_enabled:
            logger.debug(f"Getting task with task_id: {task_id}")
        try:
            with self.Session() as session:
                if task_obj := (
                    session.query(TaskModel)
                    .options(joinedload(TaskModel.artifacts))
                    .filter_by(task_id=task_id)
                    .first()
                ):
                    return convert_to_task(task_obj, self.debug_enabled)
                else:
                    logger.error(f"Task not found with task_id: {task_id}")
                    raise NotFoundError("Task not found")
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error while getting task: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while getting task: {e}")
            raise

    async def get_step(self, task_id: str, step_id: str) -> Step:
        if self.debug_enabled:
            logger.debug(f"Getting step with task_id: {task_id} and step_id: {step_id}")
        try:
            with self.Session() as session:
                if step := (
                    session.query(StepModel)
                    .options(joinedload(StepModel.artifacts))
                    .filter(StepModel.step_id == step_id)
                    .first()
                ):
                    return convert_to_step(step, self.debug_enabled)

                else:
                    logger.error(
                        f"Step not found with task_id: {task_id} and step_id: {step_id}"
                    )
                    raise NotFoundError("Step not found")
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error while getting step: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while getting step: {e}")
            raise

    async def get_artifact(self, artifact_id: str) -> Artifact:
        if self.debug_enabled:
            logger.debug(f"Getting artifact with and artifact_id: {artifact_id}")
        try:
            with self.Session() as session:
                if (
                    artifact_model := session.query(ArtifactModel)
                    .filter_by(artifact_id=artifact_id)
                    .first()
                ):
                    return convert_to_artifact(artifact_model)
                else:
                    logger.error(
                        f"Artifact not found with and artifact_id: {artifact_id}"
                    )
                    raise NotFoundError("Artifact not found")
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error while getting artifact: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while getting artifact: {e}")
            raise


    async def update_step(
        self,
        task_id: str,
        step_id: str,
        status: Optional[str] = None,
        output: Optional[str] = None,
        additional_input: Optional[Dict[str, Any]] = None,
        additional_output: Optional[Dict[str, Any]] = None,
        is_last: Optional[bool] = None,  # <<< --- ADD is_last HERE ---
    ) -> Step:
        if self.debug_enabled:
            logger.debug(
                f"Updating step with task_id: {task_id} and step_id: {step_id}. "
                f"Status: {status}, is_last: {is_last}" # Log the new param
            )
        try:
            with self.Session() as session:
                if (
                    step := session.query(StepModel) # This is your SQLAlchemy model
                    .filter_by(task_id=task_id, step_id=step_id)
                    .first()
                ):
                    if status is not None:
                        step.status = status
                    if additional_input is not None:
                        step.additional_input = additional_input
                    if output is not None:
                        step.output = output
                    if additional_output is not None:
                        step.additional_output = additional_output
                    if is_last is not None:  # <<< --- ADD THIS LOGIC ---
                        step.is_last = is_last # Update the model's is_last field
                    
                    session.commit()
                    # get_step will fetch the refreshed data including the updated is_last
                    return await self.get_step(task_id, step_id)
                else:
                    logger.error(
                        "Can't update non-existent Step with "
                        f"task_id: {task_id} and step_id: {step_id}"
                    )
                    raise NotFoundError("Step not found")
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error while updating step: {e}") # Changed from "getting step"
            raise
        except Exception as e:
            logger.error(f"Unexpected error while updating step: {e}") # Changed from "getting step"
            raise

    async def update_artifact(
        self,
        artifact_id: str,
        *,
        file_name: str = "",
        relative_path: str = "",
        agent_created: Optional[Literal[True]] = None,
    ) -> Artifact:
        logger.debug(f"Updating artifact with artifact_id: {artifact_id}")
        with self.Session() as session:
            if (
                artifact := session.query(ArtifactModel)
                .filter_by(artifact_id=artifact_id)
                .first()
            ):
                if file_name:
                    artifact.file_name = file_name
                if relative_path:
                    artifact.relative_path = relative_path
                if agent_created:
                    artifact.agent_created = agent_created
                session.commit()
                return await self.get_artifact(artifact_id)
            else:
                logger.error(f"Artifact not found with artifact_id: {artifact_id}")
                raise NotFoundError("Artifact not found")

    async def list_tasks(
        self, page: int = 1, per_page: int = 10
    ) -> Tuple[List[Task], Pagination]:
        if self.debug_enabled:
            logger.debug("Listing tasks")
        try:
            with self.Session() as session:
                tasks = (
                    session.query(TaskModel)
                    .offset((page - 1) * per_page)
                    .limit(per_page)
                    .all()
                )
                total = session.query(TaskModel).count()
                pages = math.ceil(total / per_page)
                pagination = Pagination(
                    total_items=total,
                    total_pages=pages,
                    current_page=page,
                    page_size=per_page,
                )
                return [
                    convert_to_task(task, self.debug_enabled) for task in tasks
                ], pagination
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error while listing tasks: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while listing tasks: {e}")
            raise

    async def list_steps(
        self, task_id: str, page: int = 1, per_page: int = 10
    ) -> Tuple[List[Step], Pagination]:
        if self.debug_enabled:
            logger.debug(f"Listing steps for task_id: {task_id}")
        try:
            with self.Session() as session:
                steps = (
                    session.query(StepModel)
                    .filter_by(task_id=task_id)
                    .offset((page - 1) * per_page)
                    .limit(per_page)
                    .all()
                )
                total = session.query(StepModel).filter_by(task_id=task_id).count()
                pages = math.ceil(total / per_page)
                pagination = Pagination(
                    total_items=total,
                    total_pages=pages,
                    current_page=page,
                    page_size=per_page,
                )
                return [
                    convert_to_step(step, self.debug_enabled) for step in steps
                ], pagination
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error while listing steps: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while listing steps: {e}")
            raise

    async def list_artifacts(
        self, task_id: str, page: int = 1, per_page: int = 10
    ) -> Tuple[List[Artifact], Pagination]:
        if self.debug_enabled:
            logger.debug(f"Listing artifacts for task_id: {task_id}")
        try:
            with self.Session() as session:
                artifacts = (
                    session.query(ArtifactModel)
                    .filter_by(task_id=task_id)
                    .offset((page - 1) * per_page)
                    .limit(per_page)
                    .all()
                )
                total = session.query(ArtifactModel).filter_by(task_id=task_id).count()
                pages = math.ceil(total / per_page)
                pagination = Pagination(
                    total_items=total,
                    total_pages=pages,
                    current_page=page,
                    page_size=per_page,
                )
                return [
                    convert_to_artifact(artifact) for artifact in artifacts
                ], pagination
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error while listing artifacts: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while listing artifacts: {e}")
            raise
