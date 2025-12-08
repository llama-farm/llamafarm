"""Version management for LlamaFarm server.

Reads the version from the .source_version file in the data directory,
which is written by the CLI during source code downloads.
When running from the repository, uses the git tag if on one, otherwise returns "dev".
"""

import re
import subprocess
from pathlib import Path

from core.settings import settings

# Regex for valid version format
VERSION_PATTERN = re.compile(r"^v?[0-9]+(\.[0-9]+)*(-[a-zA-Z0-9]+)?$")


def _is_running_from_repo() -> bool:
    """Check if server is running from repository source (dev mode).

    Returns True if running from the repo, False if running from CLI-managed source.
    """
    # This file is server/core/version.py
    # If we're in the repo, the path will be something like:
    #   /path/to/llamafarm/server/core/version.py
    # If we're in CLI-managed source, it will be:
    #   {lf_data_dir}/src/server/core/version.py
    current_file = Path(__file__).resolve()
    cli_managed_path = Path(settings.lf_data_dir).resolve() / "src" / "server"

    # If the current file is not under the CLI-managed path, we're in the repo
    try:
        current_file.relative_to(cli_managed_path)
        # If we can get here, we're in CLI-managed source
        return False
    except ValueError:
        # If we can't get relative path, we're not in CLI-managed source (i.e., in repo)
        return True


def _get_git_ref() -> str | None:
    """Get the current git ref (tag or branch name) if in a git repository.

    Returns:
        Tag name if exactly on a tag, branch name otherwise, or None if not in git repo.
    """
    repo_dir = Path(__file__).resolve().parent.parent  # server/ directory

    try:
        # First, try to get an exact tag match
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()

        # Not on a tag, get the branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return None


def _normalize_version(content: str) -> str | None:
    """Normalize a version string, returning None if invalid.

    Args:
        content: Raw version string (e.g., "v0.1.0", "0.1.0", "main")

    Returns:
        Normalized version without "v" prefix, or None if invalid/dev.
    """
    if not content:
        return None

    # dev/main branch means dev mode
    if content.lower() in ("main", "dev"):
        return None

    # Validate version format
    if not VERSION_PATTERN.match(content):
        return None

    # Strip "v" prefix if present (e.g., "v0.0.18" -> "0.0.18")
    if content.startswith("v"):
        return content[1:]

    return content


def _read_source_version() -> str:
    """Read version from git ref or .source_version file.

    Returns:
        Version string from git tag or .source_version file, or "dev" for development.
    """
    # If running from repo, try to get version from git
    if _is_running_from_repo():
        git_ref = _get_git_ref()
        if git_ref:
            # If on a version tag, normalize it (strip 'v' prefix)
            normalized = _normalize_version(git_ref)
            if normalized:
                return normalized
            # Otherwise return the branch name as-is
            return git_ref
        return "dev"

    # Otherwise, read from .source_version file (CLI-managed source)
    # Resolve to prevent path traversal
    data_dir = Path(settings.lf_data_dir).resolve()
    version_file = data_dir / ".source_version"

    # Ensure version file is actually inside data_dir (prevent path traversal)
    try:
        version_file.resolve().relative_to(data_dir)
    except ValueError:
        # File path escapes data directory
        return "dev"

    # If file doesn't exist, we're in dev mode
    if not version_file.exists():
        return "dev"

    try:
        content = version_file.read_text(encoding="utf-8").strip()
    except OSError:
        # If we can't read the file, default to dev
        return "dev"

    version = _normalize_version(content)
    return version if version else "dev"


# Read version at module import time
version = _read_source_version()
