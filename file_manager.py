import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Iterator, Optional, List, Union

from pydantic import BaseModel, ConfigDict, Field

from forge.agent import BaseAgentSettings
from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.file_storage.base import FileStorage
from forge.models.json_schema import JSONSchema
from forge.utils.file_operations import decode_textual_file

logger = logging.getLogger(__name__)

class FileError(Exception):
    """Base class for file operation errors"""
    pass

class FileNotFoundError(FileError):
    """Raised when a file is not found"""
    pass

class PermissionError(FileError):
    """Raised when permission is denied for a file operation"""
    pass

class FileSizeError(FileError):
    """Raised when a file exceeds the maximum size"""
    pass

class InvalidPathError(FileError):
    """Raised when a path is invalid or contains potentially unsafe characters"""
    pass

class FileManagerConfiguration(BaseModel):
    storage_path: str
    """Path to agent files, e.g. state"""
    workspace_path: str = Field(default="./workspace")
    """Path to files that agent has access to"""
    max_file_size: int = Field(default=10 * 1024 * 1024)  # 10MB
    """Maximum file size in bytes"""
    allowed_extensions: List[str] = Field(default=[])  # Empty list to allow all file types
    """List of allowed file extensions (empty to allow all types)"""
    backup_enabled: bool = Field(default=True)
    """Enable file backups before overwriting"""
    backup_dir: str = Field(default="backups")
    """Directory for file backups relative to workspace"""
    show_file_operations: bool = Field(default=True)
    """Show file operations in file explorer"""

    model_config = ConfigDict(
        # Prevent mutation of the configuration
        # as this wouldn't be reflected in the file storage
        frozen=False
    )


class FileManagerComponent(
    DirectiveProvider, CommandProvider, ConfigurableComponent[FileManagerConfiguration]
):
    """
    Adds general file manager (e.g. Agent state),
    workspace manager (e.g. Agent output files) support and
    commands to perform operations on files and folders.
    """

    config_class = FileManagerConfiguration

    STATE_FILE = "state.json"
    """The name of the file where the agent's state is stored."""

    def __init__(
        self,
        file_storage: FileStorage,
        agent_state: BaseAgentSettings,
        config: Optional[FileManagerConfiguration] = None,
    ):
        """Initialise the FileManagerComponent.
        Either `agent_id` or `config` must be provided.

        Args:
            file_storage (FileStorage): The file storage instance to use.
            state (BaseAgentSettings): The agent's state.
            config (FileManagerConfiguration, optional): The configuration for
            the file manager. Defaults to None.
        """
        if not agent_state.agent_id:
            raise ValueError("Agent must have an ID.")

        self.agent_state = agent_state

        if not config:
            storage_path = f"agents/{self.agent_state.agent_id}/"
            # Always use the fixed workspace path defined in the configuration
            workspace_path = "./workspace"  # Fixed: Now using a string instead of Field
            ConfigurableComponent.__init__(
                self,
                FileManagerConfiguration(
                    storage_path=storage_path, workspace_path=workspace_path
                ),
            )
        else:
            # Fixed indentation
            ConfigurableComponent.__init__(self, config)
            # Use configured workspace path
            try:
                if not os.path.exists(config.workspace_path):
                    os.makedirs(config.workspace_path, exist_ok=True)
                    logger.info(f"Created workspace directory: {config.workspace_path}")
            except PermissionError as e:
                logger.error(f"Permission denied when creating workspace directory: {config.workspace_path}")
                raise PermissionError(f"Cannot create workspace directory due to permission issues: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to create workspace directory: {config.workspace_path}", exc_info=True)
                raise RuntimeError(f"Failed to create workspace directory: {str(e)}")

        self.storage = file_storage.clone_with_subroot(self.config.storage_path)
        """Agent-related files, e.g. state, logs.
        Use `workspace` to access the agent's workspace files."""
        self.workspace = file_storage.clone_with_subroot(self.config.workspace_path)
        """Workspace that the agent has access to, e.g. for reading/writing files.
        Use `storage` to access agent-related files, e.g. state, logs."""
        self._file_storage = file_storage
        
        # Create backup directory if enabled
        if self.config.backup_enabled:
            try:
                backup_path = Path(self.config.workspace_path) / self.config.backup_dir
                self._file_storage.make_dir(backup_path)
                logger.info(f"Created backup directory: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup directory, backups may not work: {str(e)}")
            
        # Log the workspace path to make it clear where files are being stored
        logger.info(f"📂 Using fixed workspace directory: {self.config.workspace_path}")

    def _validate_path(self, path: Union[str, Path]) -> str:
        """Validate and normalize a file path for safety
        
        Args:
            path (Union[str, Path]): The path to validate
            
        Returns:
            str: The normalized path
            
        Raises:
            InvalidPathError: If the path is invalid or unsafe
        """
        path_str = str(path).replace('\\', '/')
        
        # Check for path traversal attempts
        if '..' in path_str:
            # Only flag if it's actually attempting to go outside the workspace
            normalized = os.path.normpath(path_str)
            if '..' in normalized:
                logger.warning(f"Possible path traversal attempt detected: {path_str}")
                raise InvalidPathError(f"Path '{path_str}' contains potentially unsafe '..' sequences")
        
        # Check for absolute paths that might access system files
       # if os.path.isabs(path_str) and not path_str.startswith(os.path.abspath(self.config.workspace_path)):
           ##  raise InvalidPathError(f"Cannot access files outside of workspace: {path_str}")
            
        #return path_str

    def _check_file_permissions(self, path: Union[str, Path], check_write: bool = False) -> None:
        """Check if file permissions allow requested operation
        
        Args:
            path (Union[str, Path]): The path to check
            check_write (bool): Whether to check for write permissions
            
        Raises:
            PermissionError: If permissions are insufficient
        """
        full_path = self.workspace.get_path(path)
        
        if not os.path.exists(full_path):
            # For write operations, check if the directory is writable
            if check_write:
                dir_path = os.path.dirname(full_path)
                if dir_path and os.path.exists(dir_path):
                    if not os.access(dir_path, os.W_OK):
                        logger.error(f"No write permission for directory: {dir_path}")
                        raise PermissionError(f"No permission to write to directory: {dir_path}")
            return
            
        # Check read permissions
        if not os.access(full_path, os.R_OK):
            logger.error(f"No read permission for file: {full_path}")
            raise PermissionError(f"No permission to read file: {full_path}")
            
        # Check write permissions if needed
        if check_write and not os.access(full_path, os.W_OK):
            logger.error(f"No write permission for file: {full_path}")
            raise PermissionError(f"No permission to modify file: {full_path}")

    async def save_state(self, save_as_id: Optional[str] = None) -> None:
        """Save the agent's data and state."""
        try:
            if save_as_id:
                try:
                    self._file_storage.make_dir(f"agents/{save_as_id}")
                    # Save state
                    await self._file_storage.write_file(
                        f"agents/{save_as_id}/{self.STATE_FILE}",
                        self.agent_state.model_dump_json(),
                    )
                    # Copy workspace
                    self._file_storage.copy(
                        self.config.workspace_path,
                        f"agents/{save_as_id}/workspace",
                    )
                except PermissionError as e:
                    logger.error(f"Permission denied when saving state as '{save_as_id}'", exc_info=True)
                    raise PermissionError(f"Cannot save state due to permission issues: {str(e)}")
                except Exception as e:
                    logger.error(f"Failed to save state as '{save_as_id}'", exc_info=True)
                    raise RuntimeError(f"Failed to save state: {str(e)}")
            else:
                try:
                    await self.storage.write_file(
                        self.storage.root / self.STATE_FILE, self.agent_state.model_dump_json()
                    )
                except PermissionError as e:
                    logger.error("Permission denied when saving state", exc_info=True)
                    raise PermissionError(f"Cannot save state due to permission issues: {str(e)}")
                except Exception as e:
                    logger.error("Failed to save state", exc_info=True)
                    raise RuntimeError(f"Failed to save state: {str(e)}")
        except Exception as e:
            logger.error("Unhandled exception in save_state", exc_info=True)
            raise RuntimeError(f"Failed to save state: {str(e)}")

    def get_resources(self) -> Iterator[str]:
        yield "The ability to read and write files of any type."
        yield f"Files are saved to the workspace at: {self.config.workspace_path}"

    def get_commands(self) -> Iterator[Command]:
        yield self.read_file
        yield self.write_to_file
        yield self.list_folder
        yield self.delete_file
        yield self.rename_file
        yield self.copy_file
        yield self.create_folder
        yield self.get_workspace_info

    @command(
        parameters={
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path of the file to read",
                required=True,
            )
        },
    )
    def read_file(self, filename: Union[str, Path]) -> str:
        """Read a file and return the contents

        Args:
            filename (str): The name of the file to read

        Returns:
            str: The contents of the file
        """
        # Store original filename for error messages
        original_filename = str(filename)
        
        try:
            # Validate and normalize path
            try:
                filename = self._validate_path(filename)
            except InvalidPathError as e:
                logger.warning(f"Invalid path in read_file: {original_filename}")
                raise InvalidPathError(f"Invalid file path: {str(e)}")
            
            # Try to normalize path - first attempt relative to workspace
            full_path = self.workspace.get_path(filename)
            
            # If not exists, try absolute path (if it's within workspace boundaries)
            if not os.path.exists(full_path):
                # Check if it might be an absolute path
                if os.path.exists(str(filename)) and os.path.isabs(str(filename)):
                    abs_path = os.path.abspath(str(filename))
                    # Ensure it's within workspace boundaries
                    workspace_abs = os.path.abspath(self.config.workspace_path)
                    if abs_path.startswith(workspace_abs):
                        full_path = str(filename)
                    else:
                        logger.warning(f"Attempt to read file outside workspace: {abs_path}")
                        raise FileNotFoundError(f"File '{original_filename}' not found within workspace boundaries")
                else:
                    logger.warning(f"File not found: {original_filename} (resolved to {full_path})")
                    raise FileNotFoundError(f"File '{original_filename}' does not exist")
            
            # Check permissions
            try:
                self._check_file_permissions(filename)
            except PermissionError as e:
                logger.error(f"Permission denied when reading file: {original_filename}")
                raise PermissionError(f"Cannot read file due to permission issues: {str(e)}")
                
            # Check if this is a directory
            if os.path.isdir(full_path):
                logger.warning(f"Attempted to read a directory as a file: {original_filename}")
                raise IsADirectoryError(f"'{original_filename}' is a directory, not a file")
                
            # Check file size
            file_size = os.path.getsize(full_path)
            if file_size > self.config.max_file_size:
                logger.warning(f"File exceeds maximum size: {original_filename} ({file_size} bytes)")
                raise FileSizeError(f"File '{original_filename}' exceeds maximum size of {self.config.max_file_size} bytes")
            
            # Check if this is an image file that should be processed by specialized tools
            _, ext = os.path.splitext(str(filename).lower())
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            
            if ext in image_extensions:
                # For image files, recommend using proper image analysis tools
                return f"This is an image file ({ext}, {file_size} bytes). Use 'analyze_image' command for proper analysis."
                
            # For non-image binary files
            if self._is_binary_file(full_path) and ext not in image_extensions:
                return f"[Binary file detected: {original_filename}]. File size: {file_size} bytes. Use other tools to process this file."
                    
            try:
                file = self.workspace.open_file(filename, binary=True)
                content = decode_textual_file(file, os.path.splitext(filename)[1], logger)
                return content
            except UnicodeDecodeError:
                logger.warning(f"Cannot decode file as text: {original_filename}")
                return f"Cannot read '{original_filename}' as text - it appears to be a binary file."
            except Exception as e:
                logger.error(f"Error reading file '{original_filename}': {str(e)}", exc_info=True)
                raise RuntimeError(f"Error reading file '{original_filename}': {str(e)}")
                
        except FileNotFoundError as e:
            logger.warning(f"File not found: {original_filename}")
            raise FileNotFoundError(str(e))
        except UnicodeDecodeError:
            logger.warning(f"Cannot decode file as text: {original_filename}")
            return f"Cannot read '{original_filename}' as text - it appears to be a binary file."
        except PermissionError as e:
            logger.error(f"Permission denied when reading file: {original_filename}")
            raise PermissionError(str(e))
        except IsADirectoryError as e:
            logger.warning(f"Attempted to read directory as file: {original_filename}")
            raise IsADirectoryError(str(e))
        except FileSizeError as e:
            logger.warning(f"File exceeds size limit: {original_filename}")
            raise FileSizeError(str(e))
        except InvalidPathError as e:
            logger.warning(f"Invalid file path: {original_filename}")
            raise InvalidPathError(str(e))
        except Exception as e:
            logger.error(f"Unhandled exception when reading file '{original_filename}'", exc_info=True)
            raise RuntimeError(f"Error reading file '{original_filename}': {str(e)}")

    def _is_binary_file(self, filepath: str) -> bool:
        """Check if file is binary (images, executables, etc.)"""
        # Common binary file extensions
        binary_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', 
                            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', 
                            '.tar', '.gz', '.exe', '.dll', '.so', '.bin']
        
        # Check extension first
        _, ext = os.path.splitext(filepath.lower())
        if ext in binary_extensions:
            return True
            
        # If unsure, check content
        try:
            with open(filepath, 'rb') as file:
                chunk = file.read(1024)
                # Check for null bytes which usually indicate binary content
                if b'\x00' in chunk:
                    return True
                # Count the ratio of printable ASCII characters
                text_characters = set(range(32, 127)) | {9, 10, 13}  # Tab, newline, carriage return
                if sum(c in text_characters for c in chunk) / len(chunk) < 0.7:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Error checking if file is binary: {filepath} - {str(e)}")
            # If there's any error in checking, assume text
            return False

    @command(
        ["write_file", "create_file"],
        "Write a file, creating it if necessary. "
        "If the file exists, it is overwritten.",
        {
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the file to write to",
                required=True,
            ),
            "contents": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The contents to write to the file",
                required=True,
            ),
        },
    )
    async def write_to_file(self, filename: Union[str, Path], contents: str) -> str:
        """Write contents to a file

        Args:
            filename (str): The name of the file to write to
            contents (str): The contents to write to the file

        Returns:
            str: A message indicating success or failure
        """
        # Store original filename for error messages
        original_filename = str(filename)
        
        try:
            # Validate and normalize path
            try:
                filename = self._validate_path(filename)
            except InvalidPathError as e:
                logger.warning(f"Invalid path in write_to_file: {original_filename}")
                raise InvalidPathError(f"Invalid file path: {str(e)}")
            
            # Check if file extension is allowed (if restriction is in place)
            if self.config.allowed_extensions:
                _, ext = os.path.splitext(filename.lower())
                if ext and ext[1:] not in self.config.allowed_extensions:
                    logger.warning(f"File extension not allowed: {ext} for {original_filename}")
                    raise ValueError(f"File extension '{ext}' is not allowed. Allowed extensions: {', '.join(self.config.allowed_extensions)}")
            
            # Check content size
            content_size = len(contents.encode('utf-8'))
            if content_size > self.config.max_file_size:
                logger.warning(f"Content exceeds maximum size: {content_size} bytes for {original_filename}")
                raise FileSizeError(f"File content exceeds maximum size of {self.config.max_file_size} bytes ({content_size} bytes)")
            
            # Normalize filename to prevent path issues
            filename = str(filename).replace('\\', '/')
            logger.info(f"Writing to file: {filename}")
            
            # Get full path for verification
            full_path = self.workspace.get_path(filename)
            logger.info(f"Resolved path: {full_path}")
            
            # Check permissions
            try:
                self._check_file_permissions(filename, check_write=True)
            except PermissionError as e:
                logger.error(f"Permission denied when writing to file: {original_filename}")
                raise PermissionError(f"Cannot write to file due to permission issues: {str(e)}")
            
            # Create directory if needed
            parent_dir = os.path.dirname(full_path)
            if parent_dir:
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                    logger.info(f"Ensured directory exists: {parent_dir}")
                except PermissionError as e:
                    logger.error(f"Permission denied when creating directory: {parent_dir}")
                    raise PermissionError(f"Cannot create directory '{os.path.dirname(original_filename)}': {str(e)}")
                except Exception as e:
                    logger.error(f"Error creating directory: {parent_dir}", exc_info=True)
                    raise RuntimeError(f"Error creating directory '{os.path.dirname(original_filename)}': {str(e)}")
            
            # Check if destination is a directory
            if os.path.exists(full_path) and os.path.isdir(full_path):
                logger.error(f"Cannot write to {original_filename}: It is a directory")
                raise IsADirectoryError(f"Cannot write to '{original_filename}' because it is a directory")
            
            # Check disk space (if possible)
            try:
                # Get free space on the device
                if hasattr(os, 'statvfs'):  # Unix/Linux/MacOS
                    stat = os.statvfs(parent_dir if parent_dir else '.')
                    free_space = stat.f_frsize * stat.f_bavail
                    if content_size > free_space:
                        logger.error(f"Not enough disk space to write file: {original_filename} ({content_size} bytes needed, {free_space} available)")
                        raise OSError(f"Not enough disk space to write file (need {content_size} bytes, {free_space} available)")
            except Exception as e:
                # If we can't check disk space, log a warning but continue
                logger.warning(f"Could not check disk space: {str(e)}")
            
            # Backup existing file if enabled
            if self.config.backup_enabled and self.workspace.exists(filename):
                try:
                    await self._backup_file(filename)
                except Exception as e:
                    logger.warning(f"Failed to create backup of {original_filename}: {str(e)}")
                    # Continue with the write operation anyway
            
            # Try write using workspace method first
            try:
                await self.workspace.write_file(filename, contents)
                logger.info(f"Successfully wrote file using workspace method: {full_path}")
            except Exception as workspace_error:
                # Fall back to direct file writing if workspace method fails
                logger.warning(f"Workspace write failed: {workspace_error}, trying direct file write...")
                try:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(contents)
                    logger.info(f"Successfully wrote file using direct method: {full_path}")
                except PermissionError as e:
                    logger.error(f"Permission denied when writing to file: {original_filename}")
                    raise PermissionError(f"Cannot write to file '{original_filename}': {str(e)}")
                except Exception as e:
                    logger.error(f"Error writing to file: {original_filename}", exc_info=True)
                    raise RuntimeError(f"Error writing to file '{original_filename}': {str(e)}")

            if hasattr(self, 'visual_feedback') and self.config.show_file_operations:
                # Open the file explorer to the directory containing the new file
                parent_dir = os.path.dirname(full_path)
                self.visual_feedback.open_file_explorer(parent_dir if parent_dir else self.config.workspace_path)

            return f"File {original_filename} has been written successfully to {full_path}."

        except PermissionError as e:
            logger.error(f"Permission denied when writing to file: {original_filename}")
            raise PermissionError(str(e))
        except FileSizeError as e:
            logger.warning(f"Content exceeds size limit for file: {original_filename}")
            raise FileSizeError(str(e))
        except InvalidPathError as e:
            logger.warning(f"Invalid file path: {original_filename}")
            raise InvalidPathError(str(e))
        except IsADirectoryError as e:
            logger.error(f"Cannot write to directory: {original_filename}")
            raise IsADirectoryError(str(e))
        except OSError as e:
            logger.error(f"OS error when writing to file: {original_filename}", exc_info=True)
            raise OSError(f"OS error when writing to file '{original_filename}': {str(e)}")
        except Exception as e:
            error_msg = f"Error writing to {full_path if 'full_path' in locals() else original_filename}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if hasattr(self, 'visual_feedback') and self.config.show_file_operations:
                # Try to open file explorer to help with debugging
                try:
                    parent_dir = os.path.dirname(full_path) if 'full_path' in locals() else self.config.workspace_path
                    self.visual_feedback.open_file_explorer(parent_dir)
                except:
                    pass
            
            raise RuntimeError(error_msg)

    @command(
        parameters={
            "folder": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The folder to list files in",
                required=True,
            )
        },
    )
    def list_folder(self, folder: Union[str, Path]) -> list[str]:
        """Lists files in a folder recursively

        Args:
            folder (str): The folder to search in

        Returns:
            list[str]: A list of files found in the folder
        """
        original_folder = str(folder)
        
        try:
            # Try to normalize path if necessary
            folder_path = folder
            
            # Check if folder is empty or '.' to list workspace root
            if not folder or folder == '.':
                # List the root workspace directory
                folder_path = self.config.workspace_path
                logger.info(f"Listing workspace root directory: {folder_path}")
            else:
                # Validate the path
                try:
                    folder = self._validate_path(folder)
                except InvalidPathError as e:
                    logger.warning(f"Invalid path in list_folder: {original_folder}")
                    raise InvalidPathError(f"Invalid folder path: {str(e)}")
                    
                if not os.path.isabs(str(folder)):
                    # If not absolute, resolve against workspace
                    folder_path = self.workspace.get_path(folder)
                    logger.info(f"Listing directory: {folder_path}")
            
            # Check if the directory exists
            if not os.path.exists(str(folder_path)):
                logger.warning(f"Directory not found: {original_folder}")
                return []  # Return empty list for non-existent directory
                
            # Check permissions
            if not os.access(str(folder_path), os.R_OK):
                logger.error(f"Permission denied when listing directory: {original_folder}")
                raise PermissionError(f"Cannot list directory '{original_folder}' due to permission issues")
            
            # Check if it's actually a directory
            if os.path.exists(str(folder_path)) and not os.path.isdir(str(folder_path)):
                logger.warning(f"Not a directory: {original_folder}")
                raise NotADirectoryError(f"'{original_folder}' is not a directory")
            
            # Get paths (try both absolute path and workspace-relative)
            try:
                paths = self.workspace.list_files(folder_path)
            except Exception as workspace_error:
                logger.warning(f"Workspace list_files failed: {workspace_error}, using direct file operations")
                paths = []
                # If that fails, try direct OS operations
                if os.path.exists(str(folder_path)):
                    try:
                        for root, dirs, files in os.walk(str(folder_path)):
                            for file in files:
                                paths.append(os.path.join(root, file))
                    except PermissionError as e:
                        logger.error(f"Permission denied when walking directory: {original_folder}")
                        raise PermissionError(f"Cannot list all files in '{original_folder}' due to permission issues")
                    except Exception as e:
                        logger.error(f"Error walking directory: {original_folder}", exc_info=True)
                        raise RuntimeError(f"Error listing files in directory '{original_folder}': {str(e)}")
            
            # Convert to string representation
            return [str(p) for p in paths]
        except PermissionError as e:
            logger.error(f"Permission denied when listing directory: {original_folder}")
            raise PermissionError(str(e))
        except NotADirectoryError as e:
            logger.warning(f"Not a directory: {original_folder}")
            raise NotADirectoryError(str(e))
        except InvalidPathError as e:
            logger.warning(f"Invalid folder path: {original_folder}")
            raise InvalidPathError(str(e))
        except Exception as e:
            logger.error(f"Unhandled exception when listing folder '{original_folder}'", exc_info=True)
            raise RuntimeError(f"Error listing folder '{original_folder}': {str(e)}")

    @command(
        parameters={
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the file to delete",
                required=True,
            )
        },
    )
    def delete_file(self, filename: Union[str, Path]) -> str:
        """Delete a file

        Args:
            filename (str): The name of the file to delete

        Returns:
            str: A message indicating success or failure
        """
        original_filename = str(filename)
        
        try:
            # Validate the path
            try:
                filename = self._validate_path(filename)
            except InvalidPathError as e:
                logger.warning(f"Invalid path in delete_file: {original_filename}")
                raise InvalidPathError(f"Invalid file path: {str(e)}")
                
            # Check if the file exists
            if not self.workspace.exists(filename):
                logger.warning(f"File not found for deletion: {original_filename}")
                return f"File '{original_filename}' does not exist."
            
            # Get the full path for permission checks
            full_path = self.workspace.get_path(filename)
            
            # Check permissions
            if not os.access(full_path, os.W_OK):
                logger.error(f"Permission denied when deleting file: {original_filename}")
                raise PermissionError(f"Cannot delete file '{original_filename}' due to permission issues")
            
            # Check if it's a directory
            if os.path.isdir(full_path):
                logger.warning(f"Cannot delete directory using delete_file: {original_filename}")
                raise IsADirectoryError(f"'{original_filename}' is a directory. Use appropriate directory deletion methods.")
            
            # Backup file if enabled
            if self.config.backup_enabled:
                try:
                    # Create backup in async way
                    import asyncio
                    loop = asyncio.get_event_loop()
                    loop.create_task(self._backup_file(filename))
                except Exception as e:
                    logger.warning(f"Failed to create backup before deletion: {original_filename}, {str(e)}")
                    # Continue with deletion anyway
            
            # Delete the file
            try:
                self.workspace.delete_file(filename)
                logger.info(f"Successfully deleted file: {original_filename}")
                return f"File '{original_filename}' has been deleted successfully."
            except PermissionError as e:
                logger.error(f"Permission denied when deleting file: {original_filename}")
                raise PermissionError(f"Cannot delete file '{original_filename}': {str(e)}")
            except Exception as e:
                logger.error(f"Error deleting file: {original_filename}", exc_info=True)
                raise RuntimeError(f"Error deleting file '{original_filename}': {str(e)}")
                
        except PermissionError as e:
            logger.error(f"Permission denied when deleting file: {original_filename}")
            raise PermissionError(str(e))
        except IsADirectoryError as e:
            logger.warning(f"Cannot delete directory with delete_file: {original_filename}")
            raise IsADirectoryError(str(e))
        except InvalidPathError as e:
            logger.warning(f"Invalid file path: {original_filename}")
            raise InvalidPathError(str(e))
        except Exception as e:
            logger.error(f"Unhandled exception when deleting file '{original_filename}'", exc_info=True)
            raise RuntimeError(f"Error deleting file '{original_filename}': {str(e)}")

    @command(
        parameters={
            "old_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The current path of the file or folder",
                required=True,
            ),
            "new_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The new path for the file or folder",
                required=True,
            ),
        },
    )
    def rename_file(self, old_path: Union[str, Path], new_path: Union[str, Path]) -> str:
        """Rename or move a file or folder

        Args:
            old_path (str): The current path of the file or folder
            new_path (str): The new path for the file or folder

        Returns:
            str: A message indicating success or failure
        """
        original_old_path = str(old_path)
        original_new_path = str(new_path)
        
        try:
            # Validate the paths
            try:
                old_path = self._validate_path(old_path)
                new_path = self._validate_path(new_path)
            except InvalidPathError as e:
                logger.warning(f"Invalid path in rename_file: {e}")
                raise InvalidPathError(f"Invalid file path: {str(e)}")
                
            # Check if source exists
            if not self.workspace.exists(old_path):
                logger.warning(f"Source path not found for rename: {original_old_path}")
                return f"File or folder '{original_old_path}' does not exist."
            
            # Get full paths for checks
            old_full_path = self.workspace.get_path(old_path)
            new_full_path = self.workspace.get_path(new_path)
            
            # Check permissions on source
            if not os.access(old_full_path, os.R_OK | os.W_OK):
                logger.error(f"Permission denied on source for rename: {original_old_path}")
                raise PermissionError(f"Cannot rename '{original_old_path}' due to permission issues")
            
            # Check permissions on destination directory
            new_dir = os.path.dirname(new_full_path)
            if new_dir and os.path.exists(new_dir) and not os.access(new_dir, os.W_OK):
                logger.error(f"Permission denied on destination directory for rename: {os.path.dirname(original_new_path)}")
                raise PermissionError(f"Cannot rename to '{original_new_path}' due to permission issues with destination directory")
            
            # Check if destination exists
            if os.path.exists(new_full_path):
                logger.warning(f"Destination already exists for rename: {original_new_path}")
                return f"Destination '{original_new_path}' already exists. Please delete it first or choose a different name."
            
            # Check if file extension is allowed (if restriction is in place)
            if self.config.allowed_extensions and not os.path.isdir(old_full_path):
                _, ext = os.path.splitext(new_path.lower())
                if ext and ext[1:] not in self.config.allowed_extensions:
                    logger.warning(f"File extension not allowed for rename destination: {ext}")
                    raise ValueError(f"File extension '{ext}' is not allowed for destination. Allowed extensions: {', '.join(self.config.allowed_extensions)}")
            
            # Create directory for destination if needed
            new_dir = os.path.dirname(new_full_path)
            if new_dir:
                try:
                    os.makedirs(new_dir, exist_ok=True)
                except PermissionError as e:
                    logger.error(f"Permission denied when creating directory for rename destination: {new_dir}")
                    raise PermissionError(f"Cannot create directory for destination: {str(e)}")
                except Exception as e:
                    logger.error(f"Error creating directory for rename destination: {new_dir}", exc_info=True)
                    raise RuntimeError(f"Error creating directory for destination: {str(e)}")
            
            # Perform the rename/move
            try:
                os.rename(old_full_path, new_full_path)
                logger.info(f"Successfully renamed/moved: {original_old_path} to {original_new_path}")
                return f"Successfully renamed/moved '{original_old_path}' to '{original_new_path}'."
            except PermissionError as e:
                logger.error(f"Permission denied during rename operation: {original_old_path} to {original_new_path}")
                raise PermissionError(f"Cannot rename '{original_old_path}' to '{original_new_path}': {str(e)}")
            except OSError as e:
                # This could be cross-device link error or other OS-level issues
                logger.error(f"OS error during rename: {original_old_path} to {original_new_path}", exc_info=True)
                raise OSError(f"Cannot rename '{original_old_path}' to '{original_new_path}': {str(e)}")
            except Exception as e:
                logger.error(f"Error during rename: {original_old_path} to {original_new_path}", exc_info=True)
                raise RuntimeError(f"Error renaming '{original_old_path}' to '{original_new_path}': {str(e)}")
                
        except PermissionError as e:
            logger.error(f"Permission denied in rename operation: {original_old_path} to {original_new_path}")
            raise PermissionError(str(e))
        except OSError as e:
            logger.error(f"OS error in rename operation: {original_old_path} to {original_new_path}", exc_info=True)
            raise OSError(str(e))
        except InvalidPathError as e:
            logger.warning(f"Invalid path in rename operation")
            raise InvalidPathError(str(e))
        except Exception as e:
            logger.error(f"Unhandled exception in rename operation: {original_old_path} to {original_new_path}", exc_info=True)
            raise RuntimeError(f"Error renaming '{original_old_path}' to '{original_new_path}': {str(e)}")

    @command(
        parameters={
            "source": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The source file or folder to copy",
                required=True,
            ),
            "destination": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The destination path",
                required=True,
            ),
        },
    )
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> str:
        """Copy a file or folder

        Args:
            source (str): The source file or folder to copy
            destination (str): The destination path

        Returns:
            str: A message indicating success or failure
        """
        original_source = str(source)
        original_destination = str(destination)
        
        try:
            # Validate the paths
            try:
                source = self._validate_path(source)
                destination = self._validate_path(destination)
            except InvalidPathError as e:
                logger.warning(f"Invalid path in copy_file: {e}")
                raise InvalidPathError(f"Invalid file path: {str(e)}")
                
            # Get full paths for checks
            source_path = self.workspace.get_path(source)
            destination_path = self.workspace.get_path(destination)
            
            # Check if source exists
            if not os.path.exists(source_path):
                logger.warning(f"Source not found for copy: {original_source}")
                return f"Source '{original_source}' does not exist."
            
            # Check permissions on source
            if not os.access(source_path, os.R_OK):
                logger.error(f"Permission denied on source for copy: {original_source}")
                raise PermissionError(f"Cannot read source '{original_source}' due to permission issues")
            
            # Check permissions on destination directory
            dest_dir = os.path.dirname(destination_path)
            if dest_dir:
                # Create the directory if it doesn't exist
                try:
                    os.makedirs(dest_dir, exist_ok=True)
                except PermissionError as e:
                    logger.error(f"Permission denied when creating directory for copy destination: {dest_dir}")
                    raise PermissionError(f"Cannot create directory for destination: {str(e)}")
                except Exception as e:
                    logger.error(f"Error creating directory for copy destination: {dest_dir}", exc_info=True)
                    raise RuntimeError(f"Error creating directory for destination: {str(e)}")
                    
                # Check write permission on destination directory
                if os.path.exists(dest_dir) and not os.access(dest_dir, os.W_OK):
                    logger.error(f"Permission denied on destination directory for copy: {os.path.dirname(original_destination)}")
                    raise PermissionError(f"Cannot copy to '{original_destination}' due to permission issues with destination directory")
            
            # For directories, check if destination already exists
            if os.path.isdir(source_path) and os.path.exists(destination_path):
                logger.warning(f"Destination directory already exists: {original_destination}")
                return f"Destination '{original_destination}' already exists. Please delete it first or choose a different name."
            
            # Check file extension for files (if restriction is in place)
            if self.config.allowed_extensions and not os.path.isdir(source_path):
                _, ext = os.path.splitext(destination.lower())
                if ext and ext[1:] not in self.config.allowed_extensions:
                    logger.warning(f"File extension not allowed for copy destination: {ext}")
                    raise ValueError(f"File extension '{ext}' is not allowed for destination. Allowed extensions: {', '.join(self.config.allowed_extensions)}")
            
            # Check if source is larger than maximum size
            try:
                if os.path.isfile(source_path):
                    file_size = os.path.getsize(source_path)
                    if file_size > self.config.max_file_size:
                        logger.warning(f"Source file exceeds maximum size: {original_source} ({file_size} bytes)")
                        raise FileSizeError(f"Source file '{original_source}' exceeds maximum size of {self.config.max_file_size} bytes")
            except OSError as e:
                logger.warning(f"Could not check file size: {original_source}, {str(e)}")
                # Continue anyway
            
            # Check disk space (if possible)
            try:
                if os.path.isfile(source_path):
                    # Get file size
                    file_size = os.path.getsize(source_path)
                    
                    # Get free space on the destination device
                    if hasattr(os, 'statvfs'):  # Unix/Linux/MacOS
                        stat = os.statvfs(dest_dir if dest_dir else '.')
                        free_space = stat.f_frsize * stat.f_bavail
                        if file_size > free_space:
                            logger.error(f"Not enough disk space for copy: {original_destination} (need {file_size} bytes, {free_space} available)")
                            raise OSError(f"Not enough disk space to copy file (need {file_size} bytes, {free_space} available)")
            except Exception as e:
                # If we can't check disk space, log a warning but continue
                logger.warning(f"Could not check disk space: {str(e)}")
            
            # Copy file or directory
            try:
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, destination_path)
                    logger.info(f"Successfully copied directory: {original_source} to {original_destination}")
                    return f"Successfully copied directory '{original_source}' to '{original_destination}'."
                else:
                    shutil.copy2(source_path, destination_path)
                    logger.info(f"Successfully copied file: {original_source} to {original_destination}")
                    return f"Successfully copied file '{original_source}' to '{original_destination}'."
            except PermissionError as e:
                logger.error(f"Permission denied during copy operation: {original_source} to {original_destination}")
                raise PermissionError(f"Cannot copy '{original_source}' to '{original_destination}': {str(e)}")
            except OSError as e:
                # This could be disk space, cross-device link error, or other OS-level issues
                logger.error(f"OS error during copy: {original_source} to {original_destination}", exc_info=True)
                raise OSError(f"Cannot copy '{original_source}' to '{original_destination}': {str(e)}")
            except Exception as e:
                logger.error(f"Error during copy: {original_source} to {original_destination}", exc_info=True)
                raise RuntimeError(f"Error copying '{original_source}' to '{original_destination}': {str(e)}")
                
        except PermissionError as e:
            logger.error(f"Permission denied in copy operation: {original_source} to {original_destination}")
            raise PermissionError(str(e))
        except OSError as e:
            logger.error(f"OS error in copy operation: {original_source} to {original_destination}", exc_info=True)
            raise OSError(str(e))
        except FileSizeError as e:
            logger.warning(f"File size error in copy operation: {original_source}")
            raise FileSizeError(str(e))
        except InvalidPathError as e:
            logger.warning(f"Invalid path in copy operation")
            raise InvalidPathError(str(e))
        except Exception as e:
            logger.error(f"Unhandled exception in copy operation: {original_source} to {original_destination}", exc_info=True)
            raise RuntimeError(f"Error copying '{original_source}' to '{original_destination}': {str(e)}")

    @command(
        parameters={
            "folder_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path of the folder to create",
                required=True,
            )
        },
    )
    def create_folder(self, folder_path: Union[str, Path]) -> str:
        """Create a new folder

        Args:
            folder_path (str): The path of the folder to create

        Returns:
            str: A message indicating success or failure
        """
        original_folder_path = str(folder_path)
        
        try:
            # Validate the path
            try:
                folder_path = self._validate_path(folder_path)
            except InvalidPathError as e:
                logger.warning(f"Invalid path in create_folder: {original_folder_path}")
                raise InvalidPathError(f"Invalid folder path: {str(e)}")
                
            # Check if folder already exists
            full_path = self.workspace.get_path(folder_path)
            if os.path.exists(full_path):
                if os.path.isdir(full_path):
                    logger.info(f"Folder already exists: {original_folder_path}")
                    return f"Folder '{original_folder_path}' already exists."
                else:
                    logger.warning(f"Path exists but is a file, not a folder: {original_folder_path}")
                    raise NotADirectoryError(f"Cannot create folder '{original_folder_path}' because a file with that name already exists")
            
            # Check permissions on parent directory
            parent_dir = os.path.dirname(full_path)
            if parent_dir and os.path.exists(parent_dir) and not os.access(parent_dir, os.W_OK):
                logger.error(f"Permission denied on parent directory: {os.path.dirname(original_folder_path)}")
                raise PermissionError(f"Cannot create folder '{original_folder_path}' due to permission issues with parent directory")
            
            # Create the folder
            try:
                self.workspace.make_dir(folder_path)
                logger.info(f"Successfully created folder: {original_folder_path}")
                
                # Get full path for logging
                return f"Folder '{original_folder_path}' created successfully at {full_path}."
            except PermissionError as e:
                logger.error(f"Permission denied when creating folder: {original_folder_path}")
                raise PermissionError(f"Cannot create folder '{original_folder_path}': {str(e)}")
            except OSError as e:
                logger.error(f"OS error when creating folder: {original_folder_path}", exc_info=True)
                raise OSError(f"Cannot create folder '{original_folder_path}': {str(e)}")
            except Exception as e:
                logger.error(f"Error creating folder: {original_folder_path}", exc_info=True)
                raise RuntimeError(f"Error creating folder '{original_folder_path}': {str(e)}")
                
        except PermissionError as e:
            logger.error(f"Permission denied when creating folder: {original_folder_path}")
            raise PermissionError(str(e))
        except OSError as e:
            logger.error(f"OS error when creating folder: {original_folder_path}", exc_info=True)
            raise OSError(str(e))
        except NotADirectoryError as e:
            logger.warning(f"Cannot create folder where file exists: {original_folder_path}")
            raise NotADirectoryError(str(e))
        except InvalidPathError as e:
            logger.warning(f"Invalid folder path: {original_folder_path}")
            raise InvalidPathError(str(e))
        except Exception as e:
            logger.error(f"Unhandled exception when creating folder '{original_folder_path}'", exc_info=True)
            raise RuntimeError(f"Error creating folder '{original_folder_path}': {str(e)}")

    @command(
        parameters={},
    )
    def get_workspace_info(self) -> str:
        """Get information about the workspace location and recently modified files
        
        Returns:
            str: Workspace information and recent files
        """
        try:
            workspace_path = self.config.workspace_path
            
            # Get list of recently modified files (last 5)
            recent_files = []
            try:
                files = list(Path(workspace_path).rglob("*"))
                files = [f for f in files if f.is_file()]
                # Sort by modification time, newest first
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                recent_files = [str(f) for f in files[:5]]
            except PermissionError as e:
                logger.warning(f"Permission denied when listing files in workspace: {str(e)}")
                recent_files = ["Permission denied when listing recent files"]
            except Exception as e:
                logger.warning(f"Error listing recent files: {str(e)}")
                recent_files = [f"Error listing recent files: {str(e)}"]
            
            # Get workspace size
            workspace_size = "Unknown"
            try:
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(workspace_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):  # Skip symbolic links
                            total_size += os.path.getsize(fp)
                
                # Convert to human-readable format
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if total_size < 1024 or unit == 'GB':
                        workspace_size = f"{total_size:.2f} {unit}"
                        break
                    total_size /= 1024
            except Exception as e:
                logger.warning(f"Error calculating workspace size: {str(e)}")
            
            return (f"Workspace location: {workspace_path}\n"
                    f"Workspace size: {workspace_size}\n"
                    f"Recent files:\n" + "\n".join(recent_files))
                    
        except Exception as e:
            logger.error("Error getting workspace info", exc_info=True)
            return f"Error getting workspace information: {str(e)}"

    async def _backup_file(self, filename: Union[str, Path]) -> None:
        """Create a backup of a file before modifying it"""
        if not self.workspace.exists(filename):
            return
            
        try:
            source_path = self.workspace.get_path(filename)
            if not os.path.isfile(source_path):
                return  # Only backup files, not directories
                
            # Create backup filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_str = str(filename)
            base_name = os.path.basename(filename_str)
            backup_name = f"{base_name}.{timestamp}.bak"
            
            # Create backup path
            backup_dir = Path(self.config.backup_dir)
            backup_path = backup_dir / backup_name
            
            # Ensure backup directory exists
            try:
                self.workspace.make_dir(backup_dir)
            except Exception as e:
                logger.warning(f"Failed to create backup directory: {str(e)}")
                return
            
            # Check permissions
            if not os.access(source_path, os.R_OK):
                logger.warning(f"No read permission for file to backup: {filename}")
                return
                
            # Read source file
            try:
                content = self.workspace.read_binary(filename)
            except Exception as e:
                logger.warning(f"Failed to read file for backup: {filename}, {str(e)}")
                return
            
            # Write to backup location
            try:
                await self.workspace.write_binary(backup_path, content)
                logger.debug(f"Created backup of '{filename}' at '{backup_path}'")
            except Exception as e:
                logger.warning(f"Failed to write backup file: {backup_path}, {str(e)}")
        except Exception as e:
            logger.warning(f"Failed to create backup of '{filename}': {e}")