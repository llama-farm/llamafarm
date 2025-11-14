"""
PID file management utility for LlamaFarm services.

This module provides functions to write and clean up PID files in a well-known
location (~/.llamafarm/pids/) for service discovery and management.
"""

import os
from pathlib import Path

# Global variable to cache the PID file path as a string for signal handlers
_cached_pid_file_path: str = ""


def get_pid_dir() -> Path:
    """Get the directory for PID files."""
    lf_data_dir = os.getenv("LF_DATA_DIR", Path.home() / ".llamafarm")
    pid_dir = Path(lf_data_dir) / "pids"
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

    pid = os.getpid()
    pid_file = get_pid_file(service_name)

    # Write PID to file
    pid_file.write_text(str(pid))
