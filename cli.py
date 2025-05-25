# autogpt/app/cli.py
import logging
import os
from logging import _nameToLevel as logLevelMap
from pathlib import Path
from typing import Optional

import click
# --- Add logging config imports ---
from forge.logging.config import (
    configure_logging,
    LoggingConfig,
    LogFormatName,
)
# --- End logging config imports ---
from .telemetry import setup_telemetry


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context):
    """Main entry point group."""
    setup_telemetry()
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


@cli.command()
# --- Keep all original click options for 'run' ---
@click.option("-c", "--continuous", is_flag=True, help="Enable Continuous Mode")
@click.option(
    "-l",
    "--continuous-limit",
    type=int,
    help="Defines the number of times to run in continuous mode",
)
@click.option("--speak", is_flag=True, help="Enable Speak Mode")
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
@click.option(
    "--skip-news",
    is_flag=True,
    help="Specifies whether to suppress the output of latest news on startup.",
)
@click.option(
    "--skip-reprompt",
    "-y",
    is_flag=True,
    help="Skips the re-prompting messages at the beginning of the script",
)
@click.option(
    "--ai-name",
    type=str,
    help="AI name override",
)
@click.option(
    "--ai-role",
    type=str,
    help="AI role override",
)
@click.option(
    "--constraint",
    type=str,
    multiple=True,
    help=(
        "Add or override AI constraints to include in the prompt;"
        " may be used multiple times to pass multiple constraints"
    ),
)
@click.option(
    "--resource",
    type=str,
    multiple=True,
    help=(
        "Add or override AI resources to include in the prompt;"
        " may be used multiple times to pass multiple resources"
    ),
)
@click.option(
    "--best-practice",
    type=str,
    multiple=True,
    help=(
        "Add or override AI best practices to include in the prompt;"
        " may be used multiple times to pass multiple best practices"
    ),
)
@click.option(
    "--override-directives",
    is_flag=True,
    help=(
        "If specified, --constraint, --resource and --best-practice will override"
        " the AI's directives instead of being appended to them"
    ),
)
@click.option(
    "--debug", is_flag=True, help="Implies --log-level=DEBUG --log-format=debug"
)
@click.option("--log-level", type=click.Choice([*logLevelMap.keys()]))
@click.option(
    "--log-format",
    help=(
        "Choose a log format; defaults to 'simple'."
        " Also implies --log-file-format, unless it is specified explicitly."
        " Using the 'structured_google_cloud' format disables log file output."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
@click.option(
    "--log-file-format",
    help=(
        "Override the format used for the log file output."
        " Defaults to the application's global --log-format."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
@click.option(
    "--component-config-file",
    help="Path to a json configuration file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
)
# --- End original click options ---
def run(
    # --- Keep all original function arguments ---
    continuous: bool,
    continuous_limit: Optional[int],
    speak: bool,
    install_plugin_deps: bool,
    skip_news: bool,
    skip_reprompt: bool,
    ai_name: Optional[str],
    ai_role: Optional[str],
    resource: tuple[str],
    constraint: tuple[str],
    best_practice: tuple[str],
    override_directives: bool,
    debug: bool,
    log_level: Optional[str],         # String from CLI option
    log_format: Optional[str],        # String from CLI option
    log_file_format: Optional[str],   # String from CLI option
    component_config_file: Optional[Path],
    # --- End original function arguments ---
) -> None:
    """
    Sets up and runs an agent, based on the task specified by the user, or resumes an
    existing agent.
    """
    # === CONFIGURE LOGGING FOR 'run' COMMAND (REVISED LOGIC) ===
    log_config = LoggingConfig.from_env() # Load defaults from .env

    # 1. Determine effective log LEVEL name/int
    effective_log_level_name_or_int = log_config.level # Start with .env level (should be int)
    if debug:
        effective_log_level_name_or_int = logging.DEBUG
    elif log_level: # Command-line --log-level overrides .env (if not debug)
         effective_log_level_name_or_int = logLevelMap[log_level.upper()] # Use upper for safety

    # Ensure it's a valid integer level
    if isinstance(effective_log_level_name_or_int, str):
         effective_log_level_name_or_int = logLevelMap.get(effective_log_level_name_or_int.upper(), logging.INFO)

    # 2. Determine effective log format ENUM for console
    log_format_str = None
    if log_format: # Prioritize CLI argument string
        log_format_str = log_format
    elif debug: # Debug implies 'debug' format unless overridden by --log-format
        log_format_str = LogFormatName.DEBUG.value
    else: # Fallback to .env config value (accessing .value safely)
        if isinstance(log_config.log_format, LogFormatName):
             log_format_str = log_config.log_format.value
        else: # Handle case where log_config.log_format might be string (defensive)
             log_format_str = str(log_config.log_format)

    # Convert the determined string to an enum member
    log_format_enum = None
    if log_format_str:
        try:
            log_format_enum = LogFormatName(log_format_str)
        except ValueError:
            print(f"Warning: Invalid log format string '{log_format_str}'. Using default.")
            # Fallback to the default enum member defined in LoggingConfig
            log_format_enum = next((f.default for f in LoggingConfig.__fields__.values() if f.name == 'log_format'), LogFormatName.SIMPLE)


    # 3. Determine effective log file format ENUM
    log_file_format_str = None
    if log_file_format: # Prioritize CLI argument string
        log_file_format_str = log_file_format
    # Don't automatically use debug format for file unless console is also debug? Or do? Let's follow console format by default.
    # If debug is set, file format defaults to debug unless explicitly overridden by --log-file-format
    elif debug and not log_file_format:
         log_file_format_str = LogFormatName.DEBUG.value
    else: # Fallback to determined effective CONSOLE log format string
        log_file_format_str = log_format_str # Defaults to the console format string

    # Convert the determined string to an enum member
    log_file_format_enum = None
    if log_file_format_str:
        # Don't use file logging for structured format
        if log_file_format_str == LogFormatName.STRUCTURED:
             log_file_format_enum = None # Explicitly disable file logging
        else:
            try:
                log_file_format_enum = LogFormatName(log_file_format_str)
            except ValueError:
                 print(f"Warning: Invalid log file format string '{log_file_format_str}'. Using console format or default.")
                 # Fallback: use the console enum format determined above
                 log_file_format_enum = log_format_enum


    # 4. Configure logging *before* calling run_auto_gpt
    configure_logging(
        level=effective_log_level_name_or_int, # Pass the final INT level
        log_format=log_format_enum, # Pass the final ENUM for console
        log_file_format=log_file_format_enum, # Pass the final ENUM for file
        config=log_config # Pass the loaded config for other settings (like log_dir)
        # TTS config might need to be loaded/passed differently if needed this early
    )
    # Optional: Log that configuration happened
    logging.getLogger(__name__).info(
        f"Run Command: Logging configured to level {logging.getLevelName(effective_log_level_name_or_int)}"
        f" Console Format: {log_format_enum.value if log_format_enum else 'Default'}"
        f" File Format: {log_file_format_enum.value if log_file_format_enum else 'Disabled/Default'}"
    )
    # === END LOGGING CONFIGURATION ===


    # Put imports inside function to avoid importing everything when starting the CLI
    # Make sure run_auto_gpt does NOT call configure_logging itself anymore
    from autogpt.app.main import run_auto_gpt

    # Call the main function with all the arguments
    run_auto_gpt(
        continuous=continuous,
        continuous_limit=continuous_limit,
        skip_reprompt=skip_reprompt,
        speak=speak,
        # Pass debug flag if run_auto_gpt still needs it for other purposes
        debug=debug,
        # Pass None for logging args, as they are handled here now.
        log_level=None,
        log_format=None,
        log_file_format=None,
        skip_news=skip_news,
        install_plugin_deps=install_plugin_deps,
        override_ai_name=ai_name,
        override_ai_role=ai_role,
        resources=list(resource),
        constraints=list(constraint),
        best_practices=list(best_practice),
        override_directives=override_directives,
        component_config_file=component_config_file,
    )


@cli.command()
# --- Keep all original click options for 'serve' ---
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
@click.option(
    "--debug", is_flag=True, help="Implies --log-level=DEBUG --log-format=debug"
)
@click.option("--log-level", type=click.Choice([*logLevelMap.keys()]))
@click.option(
    "--log-format",
    help=(
        "Choose a log format; defaults to 'simple'."
        " Also implies --log-file-format, unless it is specified explicitly."
        " Using the 'structured_google_cloud' format disables log file output."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
@click.option(
    "--log-file-format",
    help=(
        "Override the format used for the log file output."
        " Defaults to the application's global --log-format."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
# --- End original click options ---
def serve(
    # --- Keep all original function arguments ---
    install_plugin_deps: bool,
    debug: bool,
    log_level: Optional[str],         # String from CLI option
    log_format: Optional[str],        # String from CLI option
    log_file_format: Optional[str],   # String from CLI option
     # --- End original function arguments ---
) -> None:
    """
    Starts an Agent Protocol compliant AutoGPT server, which creates a custom agent for
    every task.
    """
    # === CONFIGURE LOGGING FOR 'serve' COMMAND (REVISED LOGIC) ===
    # (Applying the same refined logic as in the 'run' command)
    log_config = LoggingConfig.from_env()

    # 1. Determine effective log LEVEL name/int
    effective_log_level_name_or_int = log_config.level
    if debug:
        effective_log_level_name_or_int = logging.DEBUG
    elif log_level:
         effective_log_level_name_or_int = logLevelMap[log_level.upper()]

    if isinstance(effective_log_level_name_or_int, str):
         effective_log_level_name_or_int = logLevelMap.get(effective_log_level_name_or_int.upper(), logging.INFO)

    # 2. Determine effective log format ENUM for console
    log_format_str = None
    if log_format:
        log_format_str = log_format
    elif debug:
        log_format_str = LogFormatName.DEBUG.value
    else:
        if isinstance(log_config.log_format, LogFormatName):
             log_format_str = log_config.log_format.value
        else:
             log_format_str = str(log_config.log_format)

    log_format_enum = None
    if log_format_str:
        try:
            log_format_enum = LogFormatName(log_format_str)
        except ValueError:
            print(f"Warning: Invalid log format string '{log_format_str}'. Using default.")
            log_format_enum = next((f.default for f in LoggingConfig.__fields__.values() if f.name == 'log_format'), LogFormatName.SIMPLE)

    # 3. Determine effective log file format ENUM
    log_file_format_str = None
    if log_file_format:
        log_file_format_str = log_file_format
    elif debug and not log_file_format:
         log_file_format_str = LogFormatName.DEBUG.value
    else:
        log_file_format_str = log_format_str

    log_file_format_enum = None
    if log_file_format_str:
        if log_file_format_str == LogFormatName.STRUCTURED:
             log_file_format_enum = None
        else:
            try:
                log_file_format_enum = LogFormatName(log_file_format_str)
            except ValueError:
                 print(f"Warning: Invalid log file format string '{log_file_format_str}'. Using console format or default.")
                 log_file_format_enum = log_format_enum

    # 4. Configure logging *before* calling run_auto_gpt_server
    configure_logging(
        level=effective_log_level_name_or_int,
        log_format=log_format_enum,
        log_file_format=log_file_format_enum,
        config=log_config
    )
    logging.getLogger(__name__).info(
         f"Serve Command: Logging configured to level {logging.getLevelName(effective_log_level_name_or_int)}"
         f" Console Format: {log_format_enum.value if log_format_enum else 'Default'}"
         f" File Format: {log_file_format_enum.value if log_file_format_enum else 'Disabled/Default'}"
    )
    # === END LOGGING CONFIGURATION ===


    # Put imports inside function to avoid importing everything when starting the CLI
    # Make sure run_auto_gpt_server does NOT call configure_logging itself anymore
    from autogpt.app.main import run_auto_gpt_server

    run_auto_gpt_server(
         # Pass debug flag if run_auto_gpt_server still needs it for other purposes
         debug=debug,
         # Pass None for logging args, as they are handled here now.
         log_level=None,
         log_format=None,
         log_file_format=None,
         install_plugin_deps=install_plugin_deps,
    )


if __name__ == "__main__":
    cli()
