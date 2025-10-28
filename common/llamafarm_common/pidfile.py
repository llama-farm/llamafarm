"""
PID file management utility for LlamaFarm services.

This module provides functions to write and clean up PID files in a well-known
location (~/.llamafarm/pids/) for service discovery and management.
"""

import os
import signal
import sys
from pathlib import Path
from typing import Optional

# Global variable to cache the PID file path as a string for signal handlers
_cached_pid_file_path: Optional[str] = None


def get_pid_dir() -> Path:
    """Get the directory for PID files."""
    home = Path.home()
    pid_dir = home / ".llamafarm" / "pids"
    pid_dir.mkdir(parents=True, exist_ok=True)
    return pid_dir


def get_pid_file(service_name: str) -> Path:
    """Get the path to a service's PID file."""
    return get_pid_dir() / f"{service_name}.pid"


def write_pid(service_name: str) -> None:
    """
    Write the current process ID to a PID file.

    Args:
        service_name: Name of the service (e.g., 'server', 'rag', 'universal-runtime')
    """
    global _cached_pid_file_path

    pid = os.getpid()
    pid_file = get_pid_file(service_name)

    # Cache the PID file path as a string for use in signal handlers
    # (Path objects may not work reliably during signal handling)
    _cached_pid_file_path = str(pid_file)

    # Write PID to file
    pid_file.write_text(str(pid))

    # Register signal handlers for cleanup
    _register_signal_handlers()


def cleanup_pid(service_name: str) -> None:
    """
    Remove the PID file for a service.

    Args:
        service_name: Name of the service
    """
    pid_file = get_pid_file(service_name)
    try:
        if pid_file.exists():
            pid_file.unlink()
    except Exception:
        pass  # Silently ignore errors during cleanup


def _cleanup_cached_pid() -> None:
    """Clean up the cached PID file. Safe to call from signal handlers."""
    global _cached_pid_file_path
    if _cached_pid_file_path is not None:
        try:
            # Use os.unlink directly - don't check if exists, just try to delete
            # (checking existence can cause issues during signal handling)
            os.unlink(_cached_pid_file_path)
        except (OSError, FileNotFoundError):
            pass  # Silently ignore errors during cleanup


def _register_signal_handlers() -> None:
    """Register signal handlers to clean up PID file on shutdown."""

    def signal_handler(signum, frame):
        _cleanup_cached_pid()
        # Don't call sys.exit() here - let the normal shutdown process continue
        # The application (FastAPI/Celery) will handle the signal and shutdown gracefully

    # Register handlers for common termination signals
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # On Unix systems, also handle SIGHUP
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal_handler)
