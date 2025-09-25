"""Path resolution utilities for config files and data sources."""

import os
from pathlib import Path
from typing import Union, List, Optional


class PathResolver:
    """Utility class for resolving file and directory paths."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """Initialize with optional base directory.

        Args:
            base_dir: Base directory for relative path resolution.
                     If None, uses current working directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve a path, handling both absolute and relative paths.

        Args:
            path: File or directory path to resolve

        Returns:
            Resolved absolute Path object

        Raises:
            FileNotFoundError: If the resolved path doesn't exist
        """
        path = Path(path)

        # If already absolute, use as-is
        if path.is_absolute():
            resolved = path
        else:
            # Resolve relative to base directory
            resolved = self.base_dir / path

        # Resolve any .. or . components
        resolved = resolved.resolve()

        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")

        return resolved

    def resolve_config_path(self, config_path: Union[str, Path]) -> Path:
        """Resolve configuration file path with standard search locations.

        Args:
            config_path: Configuration file path

        Returns:
            Resolved configuration file path

        Raises:
            FileNotFoundError: If config file not found in any location
        """
        path = Path(config_path)

        # If absolute path provided, use directly
        if path.is_absolute():
            if path.exists():
                return path
            raise FileNotFoundError(f"Config file not found: {path}")

        # Search locations for config files (in order of preference)
        search_locations = [
            self.base_dir / path,  # Relative to base directory
            Path.cwd() / path,  # Relative to current directory
            Path.home() / ".config" / "rag" / path,  # User config directory
            Path("/etc/rag") / path,  # System config directory (Unix-like)
        ]

        for location in search_locations:
            if location.exists() and location.is_file():
                return location.resolve()

        # If not found anywhere, raise with helpful message
        searched = "\n  ".join(str(loc) for loc in search_locations)
        raise FileNotFoundError(
            f"Config file '{config_path}' not found in any of these locations:\n  {searched}"
        )

    def resolve_data_source(self, source: Union[str, Path]) -> Path:
        """Resolve data source path (file or directory).

        Args:
            source: Data source path

        Returns:
            Resolved data source path

        Raises:
            FileNotFoundError: If data source not found
        """
        path = Path(source)

        # Handle special patterns
        if str(path).startswith("~/"):
            path = path.expanduser()

        # If absolute, use directly
        if path.is_absolute():
            resolved = path
        else:
            # Try relative to base directory first, then current directory
            base_relative = self.base_dir / path
            if base_relative.exists():
                resolved = base_relative
            else:
                cwd_relative = Path.cwd() / path
                if cwd_relative.exists():
                    resolved = cwd_relative
                else:
                    # Neither exists, use base directory relative for error
                    resolved = base_relative

        resolved = resolved.resolve()

        if not resolved.exists():
            raise FileNotFoundError(f"Data source not found: {resolved}")

        return resolved

    def find_files_by_pattern(
        self, pattern: str, search_dirs: Optional[List[Union[str, Path]]] = None
    ) -> List[Path]:
        """Find files matching a glob pattern in specified directories.

        Args:
            pattern: Glob pattern (e.g., "*.csv", "**/*.json")
            search_dirs: Directories to search. If None, searches base directory.

        Returns:
            List of matching file paths
        """
        if search_dirs is None:
            search_dirs = [self.base_dir]

        matches = []
        for search_dir in search_dirs:
            dir_path = Path(search_dir)
            if not dir_path.is_absolute():
                dir_path = self.base_dir / dir_path

            if dir_path.exists() and dir_path.is_dir():
                matches.extend(dir_path.glob(pattern))

        return sorted(set(matches))  # Remove duplicates and sort

    def validate_directory_writable(self, dir_path: Union[str, Path]) -> Path:
        """Validate that a directory exists and is writable.

        Args:
            dir_path: Directory path to validate

        Returns:
            Resolved directory path

        Raises:
            FileNotFoundError: If directory doesn't exist
            PermissionError: If directory is not writable
        """
        resolved = self.resolve_path(dir_path)

        if not resolved.is_dir():
            raise FileNotFoundError(f"Directory not found: {resolved}")

        # Test write permission by trying to create a temp file
        test_file = resolved / ".rag_write_test"
        try:
            test_file.touch()
            test_file.unlink()  # Clean up
        except (PermissionError, OSError):
            raise PermissionError(f"Directory not writable: {resolved}")

        return resolved


def resolve_paths_in_config(config: dict, resolver: PathResolver) -> dict:
    """Recursively resolve any path-like values in configuration.

    This function looks for common path-related keys and resolves their values.

    Args:
        config: Configuration dictionary
        resolver: PathResolver instance

    Returns:
        Configuration with resolved paths
    """
    # Keys that typically contain file/directory paths
    path_keys = {
        "data_directory",
        "cache_directory",
        "log_directory",
        "output_directory",
        "model_path",
        "index_path",
        "db_path",
        "storage_path",
    }

    def resolve_recursive(obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key in path_keys and isinstance(value, str):
                    try:
                        # For directories, ensure they exist or can be created
                        if "directory" in key or "dir" in key:
                            path = Path(value)
                            if not path.is_absolute():
                                path = resolver.base_dir / path
                            path.mkdir(parents=True, exist_ok=True)
                            result[key] = str(path.resolve())
                        else:
                            # For files, just resolve the path
                            resolved = resolver.resolve_path(value)
                            result[key] = str(resolved)
                    except FileNotFoundError:
                        # Keep original value if path doesn't exist yet
                        # (might be created later)
                        result[key] = value
                else:
                    result[key] = resolve_recursive(value)
            return result
        elif isinstance(obj, list):
            return [resolve_recursive(item) for item in obj]
        else:
            return obj

    return resolve_recursive(config)
