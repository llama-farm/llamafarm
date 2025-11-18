"""Utilities for locating and serving the designer static files."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache for the designer dist path to avoid filesystem hits on every request
_cached_designer_dist_path: Path | None = None


def get_designer_dist_path() -> Path | None:
    """Get the path to designer/dist directory.

    Checks multiple locations in order:
    0. DESIGNER_DIST_PATH environment variable (explicit configuration)
    1. ~/.llamafarm/src/designer/dist (CLI-managed source)
    2. ../designer/dist (relative to server directory when running from repo)
    3. ./designer/dist (current directory)

    Logs the selected path or a warning if none is found.
    Result is cached to avoid repeated filesystem checks.

    Returns:
        Path to designer/dist directory if found, None otherwise.
    """
    global _cached_designer_dist_path

    # Return cached value if available
    if _cached_designer_dist_path is not None:
        return _cached_designer_dist_path

    # 0. Check environment variable
    env_path = os.environ.get("DESIGNER_DIST_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists() and path.is_dir():
            logger.info(f"Using designer/dist path from DESIGNER_DIST_PATH env: {path}")
            _cached_designer_dist_path = path
            return path
        else:
            logger.warning(
                f"DESIGNER_DIST_PATH env set but path does not exist "
                f"or is not a directory: {path}"
            )

    # 1. CLI-managed source
    cli_path = Path.home() / ".llamafarm" / "src" / "designer" / "dist"
    if cli_path.exists() and cli_path.is_dir():
        logger.info(f"Using designer/dist path from CLI-managed source: {cli_path}")
        _cached_designer_dist_path = cli_path
        return cli_path

    # 2. Relative to server directory (when running from repo)
    # Path(__file__) in this file will be server/core/designer.py
    # So server_dir is server/
    server_dir = Path(__file__).parent.parent
    repo_path = server_dir.parent / "designer" / "dist"
    if repo_path.exists() and repo_path.is_dir():
        logger.info(f"Using designer/dist path from repo: {repo_path}")
        _cached_designer_dist_path = repo_path
        return repo_path

    # 3. Current directory
    cwd_path = Path.cwd() / "designer" / "dist"
    if cwd_path.exists() and cwd_path.is_dir():
        logger.info(f"Using designer/dist path from current directory: {cwd_path}")
        _cached_designer_dist_path = cwd_path
        return cwd_path

    logger.warning("designer/dist directory not found in any known location.")
    _cached_designer_dist_path = None
    return None
