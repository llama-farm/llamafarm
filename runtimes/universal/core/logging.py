# src/core/logger.py
import logging
from typing import Any

import structlog
from structlog.types import EventDict, Processor


def _coerce_log_level(level: Any) -> int | str:
    """Allow level as int, numeric string, or name.

    Returns an int for numeric inputs; otherwise an upper-cased level name.
    """
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        s = level.strip()
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return s.upper()
        return s.upper()
    return level


def drop_color_message_key(_, __, event_dict: EventDict) -> EventDict:
    """
    Uvicorn logs the message a second time in the extra `color_message`, but we don't
    need it. This processor drops the key from the event dict if it exists.
    """
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging(json_logs: bool = False, log_level: str = "INFO", log_file: str = ""):
    """Setup logging with structlog, similar to server/core/logging.py."""
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ExtraAdder(),
        drop_color_message_key,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        # Format the exception only for JSON logs, as we want to pretty-print them when
        # using the ConsoleRenderer
        shared_processors.append(structlog.processors.format_exc_info)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    log_renderer: structlog.types.Processor
    if json_logs:
        log_renderer = structlog.processors.JSONRenderer()
    else:
        log_renderer = structlog.dev.ConsoleRenderer(
            exception_formatter=structlog.dev.plain_traceback
        )

    formatter = structlog.stdlib.ProcessorFormatter(
        # These run ONLY on `logging` entries that do NOT originate within
        # structlog.
        foreign_pre_chain=shared_processors,
        # These run on ALL entries after the pre_chain is done.
        processors=[
            # Remove _record & _from_structlog.
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            log_renderer,
        ],
    )

    # Clear all existing handlers from root logger to prevent duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if LOG_FILE is specified
    if log_file:
        try:
            # Ensure parent directory exists
            from pathlib import Path

            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            # Log that file logging is enabled
            root_logger.info(f"File logging enabled: {log_file}")
        except Exception as e:
            # If file logging fails, log to console but don't crash
            root_logger.error(f"Failed to set up file logging to {log_file}: {e}")

    root_logger.setLevel(_coerce_log_level(log_level))

    # Always use info level for httpcore.xxx logs
    for logger_name in ["httpcore.connection", "httpcore.http11"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

    # Configure uvicorn loggers to use our root logger setup
    for logger_name in ["uvicorn", "uvicorn.error"]:
        uvicorn_logger = logging.getLogger(logger_name)
        # Clear any existing handlers to prevent duplication
        for handler in uvicorn_logger.handlers[:]:
            uvicorn_logger.removeHandler(handler)
        # Let logs propagate to root logger (which has our structlog handler)
        uvicorn_logger.name = "uvicorn"
        uvicorn_logger.setLevel(_coerce_log_level(log_level))


class UniversalRuntimeLogger:
    """Logger wrapper for universal runtime, similar to FastAPIStructLogger."""

    def __init__(self, log_name: str = "universal-runtime"):
        self.logger = structlog.stdlib.get_logger(log_name)

    def debug(self, event: str | None = None, *args: Any, **kw: Any):
        self.logger.debug(event, *args, **kw)

    def info(self, event: str | None = None, *args: Any, **kw: Any):
        self.logger.info(event, *args, **kw)

    def warning(self, event: str | None = None, *args: Any, **kw: Any):
        self.logger.warning(event, *args, **kw)

    warn = warning

    def error(self, event: str | None = None, *args: Any, **kw: Any):
        self.logger.error(event, *args, **kw)

    def critical(self, event: str | None = None, *args: Any, **kw: Any):
        self.logger.critical(event, *args, **kw)

    def exception(self, event: str | None = None, *args: Any, **kw: Any):
        self.logger.exception(event, *args, **kw)
