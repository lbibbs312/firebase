# Import necessary libraries
import click
import datetime
import logging
import re

# Attempt to import the speak function
# Ensure this path is correct relative to where clean_input is located
# or that the forge package is correctly installed and in the Python path.
try:
    # Assuming 'forge' package structure
    # Replace 'forge.logging.utils' with the actual path if different
    from forge.logging.utils import speak
    tts_enabled = True
except ImportError:
    # Fallback if speak function is not available
    logging.warning("Could not import 'speak' function from forge.logging.utils. Greetings will only be printed.")
    tts_enabled = False
    # Define a dummy speak function if the import fails, so calls don't error out
    def speak(text: str, **kwargs):
        """Dummy function that does nothing if TTS speak function is not available."""
        pass

# Set up logger for this module
logger = logging.getLogger(__name__)

def clean_input(prompt: str = "", default: str = "") -> str:
    """
    Handles user input prompts during Auto-GPT execution.

    This version:
    - Removes auto-answering for AI settings, customization, and command authorization
      to allow for manual review.
    - Adds spoken greetings (e.g., "Good morning") for the initial task prompt if TTS is enabled.
    - Corrects the handling of the greeting message.
    - Ensures the function always returns a string to prevent NoneType errors downstream.

    Args:
        prompt (str): The prompt message to display to the user.
        default (str): The default value to use if the user provides no input.
                         Guaranteed to be a string internally.

    Returns:
        str: The user's input or the default value. Never returns None.
    """
    # Ensure default is always a string, even if None is passed initially.
    safe_default = default if default is not None else ""

    try:
        # Convert prompt to lowercase for case-insensitive matching
        prompt_lower = prompt.lower()

        # --- GREETING FOR INITIAL TASK PROMPT ---
        if "enter the task that you want autogpt to execute" in prompt_lower:
            current_hour = datetime.datetime.now().hour
            if 5 <= current_hour < 12:
                greeting = "Good morning"
            elif 12 <= current_hour < 18:
                greeting = "Good afternoon"
            else:
                greeting = "Good evening"

            full_greeting = f"{greeting}. {prompt}"

            if tts_enabled:
                try:
                    speak(greeting)
                except Exception as e:
                    logger.error(f"Error calling speak function: {e}", exc_info=True)

            print(f"\n{full_greeting}")

            # Use click.prompt for the task input
            # Use safe_default here although it's "" for this specific path
            task_result = click.prompt(
                text="Please enter the task",
                prompt_suffix=": ",
                default="", # Explicitly no default for the main task
                show_default=False
            )
            # Ensure string return for task input
            return task_result if task_result is not None else ""


        # --- AUTO-HANDLING FOR OTHER PROMPTS ---
        elif any(x in prompt_lower for x in ["agent to run", "name of the agent"]):
            logger.info("Auto-skipping agent selection")
            return ""

        elif "[y/n]" in prompt_lower or "[Y/n]" in prompt_lower:
            if "continue with these settings" in prompt_lower or \
               "would you like to customize" in prompt_lower:
                # Fall through to the manual input section below
                pass
            else:
                logger.info(f"Auto-accepting prompt with 'y': {prompt}")
                return "y"

        elif "press enter to save as" in prompt_lower:
            logger.info("Auto-accepting default agent save name")
            return ""

        elif "follow-up question or assignment" in prompt_lower:
            logger.info("Auto-providing default follow-up instruction")
            return "Continue with the task"

        elif re.search(r"enter.*number", prompt_lower):
            logger.info("Auto-providing default number '1'")
            return "1"

        # --- MANUAL INPUT FOR CRITICAL/UNHANDLED PROMPTS ---
        else:
            logger.debug(f"Requesting manual input for: {prompt}")
            result = click.prompt(
                text=prompt,
                prompt_suffix=" ",
                default=safe_default, # Use the safe string default
                show_default=bool(safe_default)
            )

            # *** Robustness FIX: Ensure we always return a string ***
            # If click.prompt somehow returns None, return the safe_default string.
            # Otherwise, return the string result (which could be empty).
            return result if result is not None else safe_default

    # --- EXCEPTION HANDLING ---
    except KeyboardInterrupt:
        logger.info("User interrupted AutoGPT via KeyboardInterrupt (Ctrl+C).")
        logger.info("Quitting...")
        exit(0)
    except EOFError:
        logger.warning("Input stream closed unexpectedly (EOFError / Ctrl+D). Exiting.")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred in clean_input: {str(e)}", exc_info=True)
        # Fallback to the safe default string value in case of any error
        logger.warning(f"Falling back to default value '{safe_default}' due to error.")
        return safe_default # Return the guaranteed string default
