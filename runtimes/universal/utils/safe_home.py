"""Safe home directory resolution for embedded Python environments.

Path.home() raises RuntimeError in PyApp-embedded Python on Windows
when HOME/USERPROFILE env vars are absent during bootstrap.
"""

import os
from pathlib import Path


def safe_home() -> Path:
    """Return the user's home directory with fallback for embedded Python."""
    try:
        return Path.home()
    except RuntimeError:
        fb = (
            os.environ.get("USERPROFILE")
            or os.environ.get("APPDATA")
            or os.environ.get("LOCALAPPDATA")
        )
        if fb:
            return Path(fb)
        try:
            return Path.cwd()
        except OSError:
            return Path(".")


def get_data_dir() -> Path:
    """Return the LlamaFarm data directory (LF_DATA_DIR or ~/.llamafarm)."""
    env = os.environ.get("LF_DATA_DIR")
    if env:
        return Path(env)
    return safe_home() / ".llamafarm"
