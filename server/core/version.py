"""Version management for LlamaFarm server.

Reads the version from the .source_version file in the data directory,
which is written by the CLI during source code downloads.
When running from the repository (dev mode), always returns "dev".
"""

from pathlib import Path

from core.settings import settings


def _is_running_from_repo() -> bool:
    """Check if server is running from repository source (dev mode).

    Returns True if running from the repo, False if running from CLI-managed source.
    """
    # This file is server/core/version.py
    # If we're in the repo, the path will be something like:
    #   /path/to/llamafarm/server/core/version.py
    # If we're in CLI-managed source, it will be:
    #   ~/.llamafarm/src/server/core/version.py
    current_file = Path(__file__).resolve()
    cli_managed_path = Path.home() / ".llamafarm" / "src" / "server"

    # If the current file is not under the CLI-managed path, we're in the repo
    try:
        current_file.relative_to(cli_managed_path)
        # If we can get here, we're in CLI-managed source
        return False
    except ValueError:
        # If we can't get relative path, we're not in CLI-managed source (i.e., in repo)
        return True


def _read_source_version() -> str:
    """Read version from .source_version file in the data directory.

    Returns:
        "dev" if running from repository source, otherwise version from file.
    """
    # If running from repo (dev mode), always return "dev"
    if _is_running_from_repo():
        return "dev"

    # Otherwise, read from .source_version file (CLI-managed source)
    version_file = Path(settings.lf_data_dir) / ".source_version"

    # If file doesn't exist, we're in dev mode
    if not version_file.exists():
        return "dev"

    try:
        content = version_file.read_text(encoding="utf-8").strip()
    except OSError:
        # If we can't read the file, default to dev
        return "dev"

    # Empty file or dev/main branch means dev mode
    if not content or content.lower() in ("main", "dev"):
        return "dev"

    # Strip "v" prefix if present (e.g., "v0.0.18" -> "0.0.18")
    if content.startswith("v"):
        return content[1:]

    return content


# Read version at module import time
version = _read_source_version()
