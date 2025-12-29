"""
GGML logging management utilities.

Routes llama.cpp/GGML logs through Python's logging system using llama_log_set.
This replaces the default llama-cpp behavior of printing directly to stderr.
"""

import ctypes
import logging
import os
from typing import Literal

logger = logging.getLogger("ggml")

# Environment variable to control GGML output behavior
# Options: "capture" (default), "suppress", "passthrough"
GGML_LOG_MODE_ENV = "GGML_LOG_MODE"

# GGML log level mapping (from llama.cpp ggml.h)
# enum ggml_log_level {
#     GGML_LOG_LEVEL_NONE  = 0,
#     GGML_LOG_LEVEL_INFO  = 1,
#     GGML_LOG_LEVEL_WARN  = 2,
#     GGML_LOG_LEVEL_ERROR = 3,
#     GGML_LOG_LEVEL_DEBUG = 4,
#     GGML_LOG_LEVEL_CONT  = 5, // continue previous log
# };
GGML_TO_PYTHON_LOG_LEVEL = {
    0: logging.NOTSET,  # NONE
    1: logging.INFO,  # INFO
    2: logging.WARNING,  # WARN
    3: logging.ERROR,  # ERROR
    4: logging.DEBUG,  # DEBUG
    5: logging.DEBUG,  # CONT (continuation)
}

# Track state for continuation logs
_last_log_level = logging.DEBUG
_log_buffer = ""

# Store callback reference to prevent garbage collection
_callback_ref = None

# Messages that llama.cpp logs at ERROR level but are actually informational
# These get downgraded to DEBUG level
_FALSE_ERROR_PATTERNS = [
    "embeddings required but some input tokens were not marked as outputs",
    "cannot decode batches with this context",
]


def get_ggml_log_mode() -> Literal["suppress", "passthrough", "capture"]:
    """Get the GGML logging mode from environment variable.

    Returns:
        One of:
        - "capture" (default): Route GGML logs through Python's logging system
        - "suppress": Silence all GGML output
        - "passthrough": Let GGML output flow to stderr normally (llama-cpp default)
    """
    mode = os.environ.get(GGML_LOG_MODE_ENV, "capture").lower()
    if mode in ("suppress", "passthrough", "capture"):
        return mode  # type: ignore
    logger.warning(
        f"Unknown GGML_LOG_MODE '{mode}', defaulting to 'capture'. "
        "Valid options: capture, suppress, passthrough"
    )
    return "capture"


def _create_logging_callback():
    """Create a callback that routes GGML logs through Python logging."""
    from llama_cpp import llama_log_callback

    @llama_log_callback
    def logging_callback(
        level: int,
        text: bytes,
        user_data: ctypes.c_void_p,
    ):
        global _last_log_level, _log_buffer

        try:
            msg = text.decode("utf-8", errors="replace")
        except Exception:
            return

        # Handle continuation logs (level 5)
        if level == 5:
            python_level = _last_log_level
        else:
            python_level = GGML_TO_PYTHON_LOG_LEVEL.get(level, logging.DEBUG)
            _last_log_level = python_level

        # Buffer partial lines (GGML often sends without newlines)
        _log_buffer += msg

        # Only log complete lines
        while "\n" in _log_buffer:
            line, _log_buffer = _log_buffer.split("\n", 1)
            line = line.strip()
            if line:
                # Downgrade known "false error" messages to DEBUG
                effective_level = python_level
                if python_level >= logging.WARNING:
                    for pattern in _FALSE_ERROR_PATTERNS:
                        if pattern in line:
                            effective_level = logging.DEBUG
                            break
                logger.log(effective_level, line)

    return logging_callback


def _create_suppressing_callback():
    """Create a callback that suppresses all GGML logs."""
    from llama_cpp import llama_log_callback

    @llama_log_callback
    def suppressing_callback(
        level: int,
        text: bytes,
        user_data: ctypes.c_void_p,
    ):
        pass  # Silently discard all logs

    return suppressing_callback


def setup_ggml_logging():
    """Configure GGML logging based on GGML_LOG_MODE environment variable.

    This should be called once at startup to configure how GGML/llama.cpp
    logs are handled. The behavior is controlled by the GGML_LOG_MODE
    environment variable:

    - "capture" (default): Routes logs through Python's logging system
      with proper log levels. Messages appear as structured logs.
    - "suppress": Silences all GGML output completely.
    - "passthrough": Uses llama-cpp's default behavior (prints to stderr).

    Example:
        # In your server startup:
        from utils.ggml_logging import setup_ggml_logging
        setup_ggml_logging()

        # Or set environment variable:
        # GGML_LOG_MODE=suppress python -m uvicorn ...
    """
    global _callback_ref

    mode = get_ggml_log_mode()

    if mode == "passthrough":
        # Don't change anything - use llama-cpp's default
        logger.debug("GGML logging: passthrough mode (using llama-cpp default)")
        return

    try:
        from llama_cpp import llama_log_set
    except ImportError:
        logger.warning("llama-cpp not available, GGML logging not configured")
        return

    if mode == "suppress":
        _callback_ref = _create_suppressing_callback()
        llama_log_set(_callback_ref, ctypes.c_void_p(0))
        logger.debug("GGML logging: suppress mode (all output silenced)")
    elif mode == "capture":
        _callback_ref = _create_logging_callback()
        llama_log_set(_callback_ref, ctypes.c_void_p(0))
        logger.debug("GGML logging: capture mode (routing through Python logging)")


def flush_ggml_log_buffer():
    """Flush any remaining content in the GGML log buffer.

    Call this after operations that may leave partial log messages buffered.
    """
    global _log_buffer, _last_log_level

    if _log_buffer.strip():
        logger.log(_last_log_level, _log_buffer.strip())
        _log_buffer = ""
