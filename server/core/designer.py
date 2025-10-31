"""Utilities for locating and serving the designer static files."""

from pathlib import Path


def get_designer_dist_path() -> Path | None:
    """Get the path to designer/dist directory.

    Checks multiple locations in order:
    1. ~/.llamafarm/src/designer/dist (CLI-managed source)
    2. ../designer/dist (relative to server directory when running from repo)
    3. ./designer/dist (current directory)

    Returns:
        Path to designer/dist directory if found, None otherwise.
    """

    # Check relative to server directory (when running from repo)
    # Path(__file__) in this file will be server/core/designer.py
    # So server_dir is server/
    server_dir = Path(__file__).parent.parent
    repo_path = server_dir.parent / "designer" / "dist"
    if repo_path.exists() and (repo_path / "index.html").exists():
        return repo_path

    # Check current directory
    current_path = Path.cwd() / "designer" / "dist"
    if current_path.exists() and (current_path / "index.html").exists():
        return current_path

    # Check CLI-managed source location
    home_dir = Path.home()
    cli_path = home_dir / ".llamafarm" / "src" / "designer" / "dist"
    if cli_path.exists() and (cli_path / "index.html").exists():
        return cli_path

    return None
