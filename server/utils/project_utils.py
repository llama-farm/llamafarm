"""
Utility functions for project management operations.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile
import uuid


class ProjectUtils:
    """Utility class for project management operations."""

    @staticmethod
    def validate_project_name(name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a project name to ensure it's safe for filesystem operations.

        Args:
            name: The project name to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name:
            return False, "Project name cannot be empty"

        if len(name) > 255:
            return False, "Project name too long (max 255 characters)"

        # Check for invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            if char in name:
                return False, f"Project name contains invalid character: {char}"

        # Reject non-printable or control characters
        for ch in name:
            # str.isprintable() rejects most control characters; ensure ASCII control too
            if not ch.isprintable() or ord(ch) < 32:
                return False, "Project name contains non-printable or control characters"

        # Check for reserved names (Windows)
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        if name.upper() in reserved_names:
            return False, f"Project name '{name}' is reserved and cannot be used"

        # Check for names starting/ending with dots or spaces
        if name.startswith('.') or name.endswith('.'):
            return False, "Project name cannot start or end with a dot"

        if name.startswith(' ') or name.endswith(' '):
            return False, "Project name cannot start or end with a space"

        return True, None

    @staticmethod
    def safe_delete_directory(directory: Path, dry_run: bool = False, *, follow_symlinks: bool = False) -> List[str]:
        """
        Safely delete a directory and return a list of files that would be/were deleted.

        Args:
            directory: The directory to delete
            dry_run: If True, only return what would be deleted without actually deleting
            follow_symlinks: If True, os.walk will follow symlinked directories. Defaults to False.
                Note: Following symlinks can cause unexpected deletions outside the target tree.

        Returns:
            List of file paths that would be/were deleted
        """
        deleted_files: List[str] = []

        if not directory.exists():
            return deleted_files

        if not directory.is_dir():
            if dry_run:
                deleted_files.append(str(directory))
            else:
                directory.unlink()
                deleted_files.append(str(directory))
            return deleted_files

        # Walk through all files and directories
        # followlinks default is False to avoid traversing out of tree via symlinks
        for root, dirs, files in os.walk(directory, topdown=False, followlinks=follow_symlinks):
            root_path = Path(root)

            # Delete files
            for file in files:
                file_path = root_path / file
                deleted_files.append(str(file_path))
                if not dry_run:
                    try:
                        # If it's a symlink, unlink only
                        if file_path.is_symlink():
                            file_path.unlink()
                        else:
                            file_path.unlink()
                    except PermissionError:
                        # Skip files we cannot delete
                        pass

            # Delete directories
            for dir_name in dirs:
                dir_path = root_path / dir_name
                deleted_files.append(str(dir_path))
                if not dry_run:
                    try:
                        if dir_path.is_symlink():
                            dir_path.unlink()
                        else:
                            dir_path.rmdir()
                    except PermissionError:
                        # Skip directories we cannot delete
                        pass

        # Finally delete the root directory
        deleted_files.append(str(directory))
        if not dry_run:
            try:
                directory.rmdir()
            except PermissionError:
                pass

        return deleted_files

    @staticmethod
    def backup_directory(source: Path, backup_dir: Optional[Path] = None) -> Path:
        """
        Create a backup of a directory.

        Args:
            source: The source directory to backup
            backup_dir: Optional directory to store the backup in

        Returns:
            Path to the created backup
        """
        if backup_dir is None:
            backup_dir = source.parent

        # Generate a unique backup name and avoid collisions
        base_name = f"{source.name}_backup"
        backup_path = backup_dir / f"{base_name}_{uuid.uuid4().hex[:8]}"
        attempt = 0
        while backup_path.exists():
            attempt += 1
            backup_path = backup_dir / f"{base_name}_{attempt}_{uuid.uuid4().hex[:8]}"

        # Create the backup
        shutil.copytree(source, backup_path, symlinks=True)

        return backup_path

    @staticmethod
    def get_directory_size(directory: Path) -> int:
        """
        Get the total size of a directory in bytes.

        Args:
            directory: The directory to measure

        Returns:
            Total size in bytes
        """
        total_size = 0

        if not directory.exists() or not directory.is_dir():
            return total_size

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                except (OSError, FileNotFoundError):
                    # Skip files that can't be accessed
                    pass

        return total_size

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """
        Format a size in bytes as a human-readable string.

        Args:
            size_bytes: Size in bytes

        Returns:
            Human-readable size string
        """
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size = float(size_bytes)

        while size >= 1024.0 and i < len(size_names) - 1:
            size /= 1024.0
            i += 1

        return f"{size:.1f} {size_names[i]}"

    @staticmethod
    def count_files_in_directory(directory: Path) -> Tuple[int, int]:
        """
        Count files and directories in a directory.

        Args:
            directory: The directory to count

        Returns:
            Tuple of (file_count, directory_count)
        """
        file_count = 0
        dir_count = 0

        if not directory.exists() or not directory.is_dir():
            return file_count, dir_count

        for root, dirs, files in os.walk(directory):
            file_count += len(files)
            dir_count += len(dirs)

        return file_count, dir_count

    @staticmethod
    def create_temp_project(config_data: Optional[dict] = None) -> Path:
        """
        Create a temporary project for testing purposes.

        Args:
            config_data: Optional configuration data

        Returns:
            Path to the temporary project directory
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="llamafarm_test_"))

        # Create a basic config if none provided
        if config_data is None:
            config_data = {
                "version": "v1",
                "prompts": [],
                "rag": {
                    "parsers": {},
                    "embedders": {},
                    "vector_stores": {},
                    "retrieval_strategies": {},
                    "defaults": {
                        "parser": "",
                        "embedder": "",
                        "vector_store": "",
                        "retrieval_strategy": ""
                    }
                },
                "datasets": {},
                "models": {}
            }

        # Write a simple config file (YAML format)
        config_file = temp_dir / "llamafarm.yaml"

        # Import yaml for writing config
        try:
            import yaml  # type: ignore
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        except ImportError:
            # Fallback to JSON if YAML not available
            import json
            config_file = temp_dir / "llamafarm.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

        return temp_dir