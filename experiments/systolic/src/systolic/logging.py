import logging
import os
import sys
from typing import override

from rich.console import Console

_console = Console()


def get_log_level() -> tuple[int, bool]:
    """Get the log level and verbosity from environment variables."""
    try:
        verbose_str = os.environ["VERBOSE_LOGS"]
    except KeyError:
        verbose_str = ""

    if verbose_str == "":
        verbose = None
    elif verbose_str == "0":
        verbose = False
    else:
        verbose = True

    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    default_verbosity = {
        logging.DEBUG: True,
        logging.INFO: False,
        logging.WARN: False,
        logging.ERROR: False,
    }

    try:
        level_str = os.environ["LOG_LEVEL"]
    except KeyError:
        level_str = "info"

    try:
        level = levels[level_str.lower()]
    except KeyError:
        level = logging.INFO
        print(
            f"Warning: invalid log level `{level_str}`, expected one of: {', '.join(levels.keys())}, defaulting to info"
        )

    if verbose is None:
        return level, default_verbosity[level]
    else:
        return level, verbose


class LogFormatter(logging.Formatter):
    """A custom formatter for `setup_logging`."""

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.verbose: bool = verbose

    @override
    def format(self, record: logging.LogRecord) -> str:
        dim_color = "dim white"
        default_color = "white"

        match record.levelno:
            case logging.DEBUG:
                head_color = dim_color
                message_color = dim_color
                name = "Debug"
            case logging.INFO:
                head_color = "blue"
                message_color = default_color
                name = "Info"
            case logging.WARNING:
                head_color = "yellow"
                message_color = default_color
                name = "Warning"
            case logging.ERROR | logging.CRITICAL:
                head_color = "red"
                message_color = "red"
                name = "Error"
            case _:
                print(
                    f"Unexpected log record `levelno` `{record.levelno}`, using default format",
                    file=sys.stderr,
                )
                return logging.Formatter().format(record)

        message = f"[{head_color}]{name}[/{head_color}][{message_color}]: {record.getMessage()}[/{message_color}]"

        if self.verbose:
            message = f"{message}\n\
-> [{dim_color}]{record.pathname}:{record.funcName}:{record.lineno}[/{dim_color}]"

        with _console.capture() as capture:
            _console.print(
                message,
                end="",
            )

        cap = capture.get()
        return cap


def setup_logging(logger: logging.Logger | None = None):
    """Configure the given logger or the root logger if None."""
    if logger is None:
        logger = logging.getLogger()

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    level, verbose = get_log_level()

    handler.setFormatter(LogFormatter(verbose))
    logger.setLevel(level)

    logger.addHandler(handler)
