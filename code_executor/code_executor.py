import logging
import threading
import os
import random
import shlex
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator, Literal, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError
try:
    import docker
    from docker.errors import DockerException, ImageNotFound, NotFound
    from docker.models.containers import Container as DockerContainer
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

from pydantic import BaseModel, Field
from forge.agent.components import ConfigurableComponent

from forge.agent.protocols import CommandProvider
from forge.command import Command, command
from forge.file_storage import FileStorage
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import (
    CommandExecutionError,
    InvalidArgumentError,
    OperationNotAllowedError,
)

logger = logging.getLogger(__name__)


# Import statements excluded as requested

logger = logging.getLogger(__name__)


def we_are_running_in_a_docker_container() -> bool:
    """Check if we are running in a Docker container

    Returns:
        bool: True if we are running in a Docker container, False otherwise
    """
    return os.path.exists("/.dockerenv")


def is_docker_available() -> bool:
    """Check if Docker is available and supports Linux containers

    Returns:
        bool: True if Docker is available and supports Linux containers, False otherwise
    """
    if not DOCKER_AVAILABLE:
        logger.debug("Docker package not available")
        return False
        
    try:
        client = docker.from_env()
        docker_info = client.info()
        return docker_info.get("OSType", "") == "linux"
    except Exception as e:
        logger.debug(f"Docker not available: {e}")
        return False


class ProcessTracker:
    """Tracks and manages running processes to prevent hanging"""
    
    def __init__(self):
        self.processes = {}
        self._lock = threading.Lock()
        
    def add_process(self, pid, process, description):
        """Add a process to the tracker
        
        Args:
            pid: Process ID
            process: The process object (e.g., from subprocess.Popen)
            description: Description of the process for logging
        """
        with self._lock:
            self.processes[pid] = {
                "process": process,
                "description": description,
                "start_time": time.time()
            }
            logger.debug(f"Process tracked: {pid} ({description})")
            
    def remove_process(self, pid):
        """Remove a process from tracking
        
        Args:
            pid: Process ID to remove
        """
        with self._lock:
            if pid in self.processes:
                logger.debug(f"Process untracked: {pid}")
                del self.processes[pid]
                
    def terminate_hanging_processes(self, timeout_seconds):
        """Terminate processes that have been running too long
        
        Args:
            timeout_seconds: Maximum allowed runtime in seconds
        
        Returns:
            list: List of terminated process IDs
        """
        terminated = []
        with self._lock:
            current_time = time.time()
            for pid, info in list(self.processes.items()):
                runtime = current_time - info["start_time"]
                if runtime > timeout_seconds:
                    try:
                        logger.warning(f"Terminating hanging process {pid}: {info['description']} (runtime: {runtime:.1f}s)")
                        info["process"].terminate()
                        terminated.append(pid)
                    except Exception as e:
                        logger.error(f"Failed to terminate process {pid}: {e}")
                    self.processes.pop(pid)
        return terminated
        
    def get_process_info(self):
        """Get information about all tracked processes
        
        Returns:
            list: Info about all tracked processes
        """
        with self._lock:
            current_time = time.time()
            return [{
                "pid": pid,
                "description": info["description"],
                "runtime": current_time - info["start_time"]
            } for pid, info in self.processes.items()]


class CommandTimeoutError(CommandExecutionError):
    """Raised when a command execution times out"""
    pass


class CodeExecutionError(CommandExecutionError):
    """The operation (an attempt to run arbitrary code) returned an error"""
    pass


class CodeExecutorConfiguration(BaseModel):
    execute_local_commands: bool = True
    """Enable shell command execution"""
    allow_native_execution: bool = True
    """Allow executing Python code outside of Docker when Docker is unavailable"""
    restrict_to_workspace: bool = False
    """Whether to restrict file operations to the workspace directory"""
    shell_command_control: Literal["allowlist", "denylist", "none"] = "none"
    """Controls which list is used"""
    shell_allowlist: list[str] = Field(default_factory=list)
    """List of allowed shell commands"""
    shell_denylist: list[str] = Field(default_factory=list)
    """List of prohibited shell commands"""
    docker_container_name: str = "agent_sandbox"
    """Name of the Docker container used for code execution"""
    docker_timeout: int = 120
    """Timeout for Docker commands in seconds"""
    execution_timeout: int = 60
    """Timeout for code execution in seconds (when not using Docker)"""
    ui_action_delay: float = 0.5
    """Delay between UI actions to make them observable (seconds)"""
    typing_speed: float = 0.05
    """Delay between keystrokes for typing (seconds)"""
    process_monitor_interval: int = 30
    """Interval in seconds to check for hanging processes"""
    process_grace_period: int = 30
    """Additional time beyond execution_timeout before terminating a process"""
    max_retry_attempts: int = 2
    """Maximum retry attempts for failed commands"""
    retry_delay: float = 1.0
    """Delay between retry attempts in seconds"""


class CodeExecutorComponent(
    CommandProvider, ConfigurableComponent[CodeExecutorConfiguration]
):
    """Provides commands to execute Python code and shell commands."""

    config_class = CodeExecutorConfiguration

    def __init__(
        self,
        workspace: FileStorage,
        config: Optional[CodeExecutorConfiguration] = None,
        app_config: Any = None,
    ):
        ConfigurableComponent.__init__(self, config)
        self.workspace = workspace
        self.app_config = app_config
        self._executor = ThreadPoolExecutor(max_workers=5)
        self.process_tracker = ProcessTracker()
        self._monitor_thread = None
        self._shutdown_requested = False

        # Change container name if it's empty or default to prevent different agents
        # from using the same container
        default_container_name = self.config.model_fields[
            "docker_container_name"
        ].default
        if (
            not self.config.docker_container_name
            or self.config.docker_container_name == default_container_name
        ):
            random_suffix = "".join(random.choices(string.ascii_lowercase, k=8))
            self.config.docker_container_name = (
                f"{default_container_name}_{random_suffix}"
            )

        self.docker_available = False
        self.running_in_docker = we_are_running_in_a_docker_container()
        
        if not self.running_in_docker:
            self.docker_available = is_docker_available()

        logger.info(
            f"Code executor initialized with restrict_to_workspace={self.config.restrict_to_workspace}, "
            f"execute_local_commands={self.config.execute_local_commands}, "
            f"shell_command_control={self.config.shell_command_control}"
        )
        
        # Start the process monitor
        self._start_process_monitor()

    def _start_process_monitor(self):
        """Start a background thread to monitor for hanging processes"""
        def monitor_loop():
            logger.info("Process monitor started")
            while not self._shutdown_requested:
                try:
                    # Terminate processes that have been running too long
                    timeout = self.config.execution_timeout + self.config.process_grace_period
                    terminated = self.process_tracker.terminate_hanging_processes(timeout)
                    
                    if terminated:
                        logger.warning(f"Terminated {len(terminated)} hanging processes")
                except Exception as e:
                    logger.error(f"Error in process monitor: {str(e)}", exc_info=True)
                
                # Sleep for the configured interval
                for _ in range(self.config.process_monitor_interval):
                    if self._shutdown_requested:
                        break
                    time.sleep(1)
        
        self._monitor_thread = threading.Thread(
            target=monitor_loop, 
            daemon=True,
            name="ProcessMonitor"
        )
        self._monitor_thread.start()

    def shutdown(self):
        """Shut down the component, terminating all tracked processes"""
        logger.info("Shutting down CodeExecutorComponent")
        self._shutdown_requested = True
        
        # Terminate all tracked processes
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.process_tracker.terminate_hanging_processes, 0
                )
                future.result(timeout=5)  # Allow 5 seconds for termination
        except Exception as e:
            logger.error(f"Error terminating processes during shutdown: {e}")
        
        # Shutdown thread pool
        self._executor.shutdown(wait=False)
        
        logger.info("CodeExecutorComponent shutdown complete")

    def _run_with_timeout(self, func, timeout, *args, **kwargs):
        """Run a function with a timeout
        
        Args:
            func: Function to run
            timeout: Timeout in seconds
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The function result
            
        Raises:
            CommandTimeoutError: If the function times out
        """
        try:
            future = self._executor.submit(func, *args, **kwargs)
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.warning(f"Function {func.__name__} timed out after {timeout} seconds")
            raise CommandTimeoutError(f"Operation timed out after {timeout} seconds")

    def get_commands(self) -> Iterator[Command]:
        # Always yield the python execution commands
        yield self.execute_python_code
        yield self.execute_python_file

        # Always yield the shell commands 
        yield self.execute_shell
        yield self.execute_shell_popen
        
        # Add specialized UI commands
        yield self.send_keystrokes
        yield self.send_hotkey
        yield self.click_position
        yield self.move_mouse
        yield self.wait_seconds
        
        # Add diagnostic commands
        yield self.list_processes
        yield self.kill_process

    @command(
        ["execute_python_code"],
        "Executes the given Python code with access to your entire filesystem",
        {
            "code": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The Python code to run",
                required=True,
            ),
        },
    )
    async def execute_python_code(self, code: str) -> str:
        """
        Create and execute a Python file and return the output.

        If the code generates any data that needs to be captured,
        use a print statement.

        Args:
            code (str): The Python code to run.

        Returns:
            str: The output captured from the code execution.
        """
        # Generate a random filename to prevent conflicts
        temp_path = ""
        while True:
            temp_path = f"temp{self._generate_random_string()}.py"
            if not os.path.exists(temp_path):
                break
        
        try:
            # Write to workspace temp file
            await self.workspace.write_file(temp_path, code)
            file_path = self.workspace.get_path(temp_path)

            # Open terminal window to show execution if visual feedback is available
            if hasattr(self, 'visual_feedback'):
                self.visual_feedback.open_terminal(f"{sys.executable} {file_path}")

            return self._execute_python_file_direct(file_path, [])
        except CommandTimeoutError:
            return "Error: Python code execution timed out. Your code may have an infinite loop or is taking too long to complete."
        except Exception as e:
            error_msg = f"Python code execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise CommandExecutionError(error_msg)
        finally:
            try:
                # Clean up temporary file
                if self.workspace.exists(temp_path):
                    self.workspace.delete_file(temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")

    @command(
        ["execute_python_file"],
        "Execute any Python file on your system (absolute or relative path)",
        {
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The absolute or relative path to the file to execute",
                required=True,
            ),
            "args": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                description="The (command line) arguments to pass to the script",
                required=False,
                items=JSONSchema(type=JSONSchema.Type.STRING),
            ),
        },
    )
    def execute_python_file(self, filename: Union[str, Path], args: list[str] = []) -> str:
        """Execute a Python file and return the output

        Args:
            filename (Path): The path to the file to execute (can be absolute or relative)
            args (list, optional): The arguments with which to run the python script

        Returns:
            str: The output of the file
        """
        logger.info(f"Executing python file '{filename}'")

        if not str(filename).endswith(".py"):
            raise InvalidArgumentError("Invalid file type. Only .py files are allowed.")

        # Convert to Path object
        path_obj = Path(filename)
        
        # Handle absolute path
        if path_obj.is_absolute():
            if not path_obj.exists():
                raise FileNotFoundError(
                    f"python: can't open file '{filename}': "
                    f"[Errno 2] No such file or directory"
                )
            
            for attempt in range(self.config.max_retry_attempts + 1):
                try:
                    return self._execute_python_file_direct(path_obj, args)
                except CommandTimeoutError:
                    if attempt == self.config.max_retry_attempts:
                        return "Error: Python file execution timed out after multiple attempts."
                    time.sleep(self.config.retry_delay)
                except Exception as e:
                    if attempt == self.config.max_retry_attempts:
                        logger.error(f"Python file execution failed after {attempt+1} attempts: {e}")
                        raise
                    logger.warning(f"Retrying Python file execution after error: {e}")
                    time.sleep(self.config.retry_delay)
        
        # Handle relative path - try workspace first
        try:
            file_path = self.workspace.get_path(filename)
            if self.workspace.exists(file_path):
                return self._execute_python_file_direct(file_path, args)
        except Exception as e:
            logger.warning(f"Failed to execute from workspace: {e}")
        
        # Try relative to current directory
        current_dir_path = Path.cwd() / path_obj
        if current_dir_path.exists():
            return self._execute_python_file_direct(current_dir_path, args)
            
        # If we get here, the file wasn't found
        raise FileNotFoundError(
            f"python: can't open file '{filename}': "
            f"[Errno 2] No such file or directory"
        )

    def _execute_python_file_direct(self, file_path: Path, args: list[str]) -> str:
        """Execute a Python file directly using the current Python interpreter"""
        logger.debug(f"Executing {file_path} directly with current Python interpreter...")
        
        process = None
        try:
            # Start the process
            cmd = [sys.executable, "-B", str(file_path)] + args
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf8",
                text=True,
                bufsize=1,  # Line buffered
            )
            
            # Track the process
            self.process_tracker.add_process(
                process.pid, 
                process, 
                f"Python: {file_path} {' '.join(args)}"
            )
            
            # Use non-blocking read with timeout
            stdout_data = []
            stderr_data = []
            start_time = time.time()
            
            # Set up polling for output with timeout
            while process.poll() is None:
                # Check if we've exceeded timeout
                if time.time() - start_time > self.config.execution_timeout:
                    process.terminate()
                    self.process_tracker.remove_process(process.pid)
                    raise CommandTimeoutError(
                        f"Python execution timed out after {self.config.execution_timeout} seconds"
                    )
                
                # Read from stdout and stderr without blocking
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        stdout_data.append(line)
                        
                if process.stderr:
                    line = process.stderr.readline()
                    if line:
                        stderr_data.append(line)
                        
                # Short sleep to prevent CPU spin
                time.sleep(0.01)
            
            # Read any remaining output
            if process.stdout:
                remaining_stdout, remaining_stderr = process.communicate(timeout=2)
                if remaining_stdout:
                    stdout_data.append(remaining_stdout)
                if remaining_stderr:
                    stderr_data.append(remaining_stderr)
            
            # Build the result string
            stdout_result = "".join(stdout_data)
            stderr_result = "".join(stderr_data)
            
            # Untrack the process since it's finished
            self.process_tracker.remove_process(process.pid)
            
            if process.returncode == 0:
                return stdout_result
            else:
                raise CodeExecutionError(
                    f"Python execution failed with code {process.returncode}: {stderr_result}"
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"Python execution timed out: {file_path}")
            if process:
                process.terminate()
                self.process_tracker.remove_process(process.pid)
            return "Error: Code execution exceeded the time limit."
        except CommandTimeoutError:
            raise  # Re-raise timeout errors
        except Exception as e:
            logger.error(f"Error executing Python file: {file_path}", exc_info=True)
            if process:
                process.terminate()
                self.process_tracker.remove_process(process.pid)
            raise CodeExecutionError(f"Execution failed: {str(e)}")

    def validate_command(self, command_line: str) -> tuple[bool, bool]:
        """Check whether a command is allowed and whether it may be executed in a shell.

        Args:
            command_line (str): The command line to validate

        Returns:
            bool: True if the command is allowed, False otherwise
            bool: True if the command may be executed in a shell, False otherwise
        """
        if not command_line:
            return False, False

        # If shell control is disabled (set to "none"), allow everything
        if self.config.shell_command_control == "none":
            return True, True

        command_name = shlex.split(command_line)[0]

        if self.config.shell_command_control == "allowlist":
            return command_name in self.config.shell_allowlist, True
        elif self.config.shell_command_control == "denylist":
            return command_name not in self.config.shell_denylist, True
        else:
            return True, True

    @command(
        ["execute_shell"],
        "Execute any Shell Command on your system",
        {
            "command_line": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The command line to execute",
                required=True,
            )
        },
    )
    def execute_shell(self, command_line: str) -> str:
        """Execute a shell command and return the output"""
        allow_execute, allow_shell = self.validate_command(command_line)
        if not allow_execute:
            logger.info(f"Command '{command_line}' not allowed")
            raise OperationNotAllowedError("This shell command is not allowed.")

        logger.info(f"Executing command '{command_line}' in current directory '{os.getcwd()}'")

        # Check if this is an application launch command
        app_patterns = [".exe", "notepad", "word", "excel", "paint", "explorer", "chrome", "firefox", "edge"]
        is_app_launch = any(pattern in command_line.lower() for pattern in app_patterns)
        
        if is_app_launch:
            # Use execute_shell_popen for application launches
            return self.execute_shell_popen(command_line)
        
        # For regular commands
        process = None
        try:
            # Use Popen with proper pipe setup
            process = subprocess.Popen(
                command_line,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True,
                bufsize=1,  # Line buffered
            )
            
            # Track the process
            self.process_tracker.add_process(
                process.pid, 
                process, 
                f"Shell: {command_line}"
            )
            
            # Use non-blocking read with timeout
            stdout_data = []
            stderr_data = []
            start_time = time.time()
            
            # Set up polling for output with timeout
            while process.poll() is None:
                # Check if we've exceeded timeout
                if time.time() - start_time > self.config.execution_timeout:
                    process.terminate()
                    self.process_tracker.remove_process(process.pid)
                    return f"Error: Command execution timed out after {self.config.execution_timeout} seconds"
                
                # Read from stdout and stderr without blocking
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        stdout_data.append(line)
                        
                if process.stderr:
                    line = process.stderr.readline()
                    if line:
                        stderr_data.append(line)
                        
                # Short sleep to prevent CPU spin
                time.sleep(0.01)
            
            # Read any remaining output
            if process.stdout:
                try:
                    remaining_stdout, remaining_stderr = process.communicate(timeout=2)
                    if remaining_stdout:
                        stdout_data.append(remaining_stdout)
                    if remaining_stderr:
                        stderr_data.append(remaining_stderr)
                except Exception as e:
                    logger.warning(f"Error reading remaining output: {e}")
            
            # Untrack the process since it's finished
            self.process_tracker.remove_process(process.pid)
            
            # Build the result string
            stdout_result = "".join(stdout_data)
            stderr_result = "".join(stderr_data)
            
            exit_code = process.returncode
            output = f"STDOUT:\n{stdout_result}\nSTDERR:\n{stderr_result}\nExit code: {exit_code}"
            return output
            
        except subprocess.TimeoutExpired:
            if process:
                process.terminate()
                self.process_tracker.remove_process(process.pid)
            return "Error: Command execution exceeded the time limit."
        except Exception as e:
            error_msg = f"Command execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if process and process.pid:
                try:
                    process.terminate()
                    self.process_tracker.remove_process(process.pid)
                except:
                    pass
            raise CommandExecutionError(error_msg)

    @command(
        ["execute_shell_popen"],
        "Start a Shell Command as a background process",
        {
            "command_line": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The command line to execute",
                required=True,
            )
        },
    )
    def execute_shell_popen(self, command_line: str) -> str:
        """Execute a shell command with Popen and returns an english description
        of the event and the process id

        Args:
            command_line (str): The command line to execute

        Returns:
            str: Description of the fact that the process started and its id
        """
        allow_execute, allow_shell = self.validate_command(command_line)
        if not allow_execute:
            logger.info(f"Command '{command_line}' not allowed")
            raise OperationNotAllowedError("This shell command is not allowed.")

        logger.info(f"Executing command '{command_line}' as background process")

        # Check if this is an application launch
        app_patterns = [".exe", "notepad", "start ", "open ", "explorer", "chrome", "firefox", "edge", "word", "excel"]
        is_app_launch = any(pattern in command_line.lower() for pattern in app_patterns)
        
        # Adjust command for Windows applications
        if is_app_launch and sys.platform == 'win32' and not command_line.lower().startswith('start '):
            # For Windows, ensure applications launch as maximized windows
            command_line = f'start "" /max {command_line}'

        process = None
        try:
            # Use PIPE for stdout/stderr to prevent blocking
            process = subprocess.Popen(
                command_line,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent process
            )
            
            if not process.pid:
                raise CommandExecutionError("Failed to start process (no PID returned)")
            
            # Track the process for monitoring
            self.process_tracker.add_process(
                process.pid, 
                process, 
                f"Background: {command_line}"
            )
            
            # For application launches, wait briefly to ensure it starts
            if is_app_launch:
                # Check if process exits immediately (which would indicate failure)
                for _ in range(10):  # Check for 1 second (10 x 0.1s)
                    if process.poll() is not None:
                        # Process exited quickly - likely an error
                        exit_code = process.returncode
                        stderr = process.stderr.read().decode('utf-8', errors='replace') if process.stderr else ""
                        self.process_tracker.remove_process(process.pid)
                        
                        if stderr:
                            raise CommandExecutionError(f"Application failed to start (exit code {exit_code}): {stderr}")
                        else:
                            raise CommandExecutionError(f"Application failed to start (exit code {exit_code})")
                    
                    time.sleep(0.1)
                
                # Allow more time for GUI applications to initialize
                time.sleep(1.0)
            
            # Set up a thread to collect output and automatically untrack when done
            def _monitor_process():
                try:
                    # Wait for process to complete or timeout
                    try:
                        stdout, stderr = process.communicate(timeout=self.config.execution_timeout)
                    except subprocess.TimeoutExpired:
                        # Process is still running after timeout, which is expected for background processes
                        # Keep it tracked for the ProcessTracker to handle if it hangs too long
                        pass
                except Exception as e:
                    logger.warning(f"Error monitoring background process {process.pid}: {e}")
                    self.process_tracker.remove_process(process.pid)
            
            # Start monitoring thread
            threading.Thread(
                target=_monitor_process,
                daemon=True,
                name=f"Monitor-{process.pid}"
            ).start()
            
            return f"Process started with PID: {process.pid}"
        except Exception as e:
            error_msg = f"Failed to start background process: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if process and process.pid:
                try:
                    process.terminate()
                    self.process_tracker.remove_process(process.pid)
                except:
                    pass
                    
            raise CommandExecutionError(error_msg)

    @command(
        ["send_keystrokes"],
        "Send keystrokes to the active application",
        {
            "text": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The text to type",
                required=True,
            ),
            "delay": JSONSchema(
                type=JSONSchema.Type.NUMBER,
                description="Delay between keystrokes in seconds",
                required=False,
            ),
        },
    )
    def send_keystrokes(self, text: str, delay: Optional[float] = None) -> str:
        """Send keystrokes to the active application with visible typing."""
        try:
            # Import here to avoid startup issues if not available
            import pyautogui
            
            delay = delay or self.config.typing_speed
            
            def _type_with_timeout():
                # Allow cancellation by breaking text into chunks
                for i in range(0, len(text), 20):
                    chunk = text[i:i+20]
                    pyautogui.write(chunk, interval=delay)
                    # Check if we should stop
                    if threading.current_thread().native_id in _canceled_threads:
                        return False
                return True
                
            # Keep track of thread IDs that should be canceled
            _canceled_threads = set()
            
            # Run with timeout
            try:
                result = self._run_with_timeout(
                    _type_with_timeout,
                    self.config.execution_timeout
                )
                if result:
                    return f"Successfully sent keystrokes: '{text}'"
                else:
                    return "Keystroke operation was canceled"
            except CommandTimeoutError:
                # Mark thread for cancellation and wait briefly
                if threading.current_thread().native_id:
                    _canceled_threads.add(threading.current_thread().native_id)
                time.sleep(0.5)  # Give time for the operation to notice cancellation
                return "Error: Keystroke operation timed out and was canceled"
                
        except ImportError:
            return "Error: PyAutoGUI not available. Install with pip install pyautogui"
        except Exception as e:
            error_msg = f"Error sending keystrokes: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    @command(
        ["send_hotkey"],
        "Send a keyboard shortcut combination to the active application",
        {
            "keys": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                description="List of keys to press in combination (e.g. ['ctrl', 's'] for Ctrl+S)",
                items=JSONSchema(type=JSONSchema.Type.STRING),
                required=True,
            ),
        },
    )
    def send_hotkey(self, keys: list[str]) -> str:
        """Send a keyboard shortcut to the active application."""
        try:
            # Import here to avoid startup issues if not available
            import pyautogui
            
            def _press_hotkey():
                pyautogui.hotkey(*keys)
                time.sleep(self.config.ui_action_delay)  # Wait for action to complete
                return True
            
            # Run with timeout
            try:
                self._run_with_timeout(
                    _press_hotkey,
                    self.config.execution_timeout
                )
                return f"Successfully sent hotkey: {'+'.join(keys)}"
            except CommandTimeoutError:
                return "Error: Hotkey operation timed out"
                
        except ImportError:
            return "Error: PyAutoGUI not available. Install with pip install pyautogui"
        except Exception as e:
            error_msg = f"Error sending hotkey: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    @command(
        ["click_position"],
        "Click at a specific position on the screen",
        {
            "x": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="X coordinate",
                required=True,
            ),
            "y": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Y coordinate",
                required=True,
            ),
            "button": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Mouse button to click (left, right, middle)",
                required=False,
            ),
            "clicks": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Number of clicks",
                required=False,
            ),
        },
    )
    def click_position(self, x: int, y: int, button: str = "left", clicks: int = 1) -> str:
        """Click at a specific position on the screen with a visible mouse movement."""
        try:
            # Import here to avoid startup issues if not available
            import pyautogui
            
            def _perform_click():
                # Move to position with a visible animation (duration > 0)
                pyautogui.moveTo(x, y, duration=0.5)
                time.sleep(0.2)  # Pause briefly to make movement visible
                
                # Perform click
                pyautogui.click(x=x, y=y, button=button, clicks=clicks)
                time.sleep(self.config.ui_action_delay)  # Wait for action to complete
                return True
            
            # Run with timeout
            try:
                self._run_with_timeout(
                    _perform_click,
                    self.config.execution_timeout
                )
                return f"Successfully clicked at position ({x}, {y}) with {button} button"
            except CommandTimeoutError:
                return "Error: Click operation timed out"
                
        except ImportError:
            return "Error: PyAutoGUI not available. Install with pip install pyautogui"
        except Exception as e:
            error_msg = f"Error clicking at position: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    @command(
        ["move_mouse"],
        "Move the mouse cursor to a specific position on the screen",
        {
            "x": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="X coordinate",
                required=True,
            ),
            "y": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Y coordinate",
                required=True,
            ),
            "duration": JSONSchema(
                type=JSONSchema.Type.NUMBER,
                description="Duration of movement in seconds (for visible animation)",
                required=False,
            ),
        },
    )
    def move_mouse(self, x: int, y: int, duration: float = 0.5) -> str:
        """Move the mouse cursor with a visible animation."""
        try:
            # Import here to avoid startup issues if not available
            import pyautogui
            
            def _perform_move():
                # Move with a visible animation
                pyautogui.moveTo(x, y, duration=duration)
                return True
            
            # Run with timeout
            try:
                self._run_with_timeout(
                    _perform_move,
                    self.config.execution_timeout
                )
                return f"Successfully moved mouse to position ({x}, {y})"
            except CommandTimeoutError:
                return "Error: Mouse movement operation timed out"
                
        except ImportError:
            return "Error: PyAutoGUI not available. Install with pip install pyautogui"
        except Exception as e:
            error_msg = f"Error moving mouse: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    @command(
        ["wait_seconds"],
        "Wait for a specified number of seconds",
        {
            "seconds": JSONSchema(
                type=JSONSchema.Type.NUMBER,
                description="Number of seconds to wait",
                required=True,
            ),
        },
    )
    def wait_seconds(self, seconds: float) -> str:
        """Wait for a specified number of seconds."""
        try:
            # Enforce maximum wait time
            max_wait = min(seconds, self.config.execution_timeout)
            if max_wait < seconds:
                logger.warning(f"Requested wait time {seconds}s exceeds maximum allowed {max_wait}s")
                
            # Use sleep with small increments for better responsiveness
            start_time = time.time()
            end_time = start_time + max_wait
            
            while time.time() < end_time:
                remaining = end_time - time.time()
                # Sleep in 0.1s increments for better responsiveness
                time.sleep(min(0.1, remaining))
                
            actual_wait = time.time() - start_time
            
            if max_wait < seconds:
                return f"Waited for {actual_wait:.1f} seconds (limited from requested {seconds} seconds)"
            else:
                return f"Successfully waited for {actual_wait:.1f} seconds"
        except Exception as e:
            error_msg = f"Error during wait: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    @command(
        ["list_processes"],
        "List all tracked processes",
        {},
    )
    def list_processes(self) -> str:
        """List all tracked processes with their status."""
        try:
            processes = self.process_tracker.get_process_info()
            
            if not processes:
                return "No tracked processes running."
                
            result = "Currently tracked processes:\n"
            for proc in processes:
                result += f"PID: {proc['pid']}, Runtime: {proc['runtime']:.1f}s, Command: {proc['description']}\n"
                
            return result
        except Exception as e:
            error_msg = f"Error listing processes: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    @command(
        ["kill_process"],
        "Terminate a specific process by PID",
        {
            "pid": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Process ID to terminate",
                required=True,
            ),
        },
    )
    def kill_process(self, pid: int) -> str:
        """Terminate a specific process by PID."""
        try:
            # Check if this is a tracked process
            processes = self.process_tracker.get_process_info()
            tracked = any(p["pid"] == pid for p in processes)
            
            if tracked:
                # This is one of our tracked processes
                with self.process_tracker._lock:
                    if pid in self.process_tracker.processes:
                        try:
                            self.process_tracker.processes[pid]["process"].terminate()
                            self.process_tracker.processes.pop(pid)
                            return f"Process {pid} was terminated successfully."
                        except Exception as e:
                            logger.error(f"Error terminating process {pid}: {e}")
                            return f"Failed to terminate process {pid}: {str(e)}"
                    else:
                        return f"Process {pid} is no longer being tracked."
            else:
                # This is not one of our tracked processes
                # Only allow terminating processes if running on non-Windows (safer)
                if sys.platform != "win32":
                    try:
                        os.kill(pid, signal.SIGTERM)
                        return f"Process {pid} was terminated successfully."
                    except ProcessLookupError:
                        return f"Process {pid} does not exist."
                    except PermissionError:
                        return f"Permission denied to terminate process {pid}."
                    except Exception as e:
                        logger.error(f"Error terminating process {pid}: {e}")
                        return f"Failed to terminate process {pid}: {str(e)}"
                else:
                    return f"Cannot terminate process {pid} because it is not being tracked by this component."
        except Exception as e:
            error_msg = f"Error handling process termination: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _generate_random_string(self, length: int = 8):
        """Generate a random string of fixed length"""
        characters = string.ascii_letters + string.digits
        random_string = "".join(random.choices(characters, k=length))
        return random_string