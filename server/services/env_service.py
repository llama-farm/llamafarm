"""Service for environment variable substitution in config values.

Supports these syntaxes:
- ${VAR_NAME} - Looks up VAR_NAME in project .env, then os.environ
- ${VAR_NAME:-default} - Same as above, with fallback default value
- ${file:.env-llamafarm:VAR_NAME} - Explicitly load from specific file
- ${file:.env.local:VAR_NAME:-default} - Explicit file with default

The default .env file is automatically loaded from the project directory.
Use ${file:...} syntax to load from a different file (e.g., .env-llamafarm).
"""

import os
import re
from pathlib import Path

from dotenv import dotenv_values

from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger()

# Pattern to match ${VAR}, ${VAR:-default}, ${file:.env:VAR}, ${file:.env:VAR:-default}
# Group 1: optional "file:" prefix with filename (e.g., "file:.env.local:")
# Group 2: the filename if file: prefix is used
# Group 3: variable name
# Group 4: optional default value (after :-)
ENV_VAR_PATTERN = re.compile(r"\$\{(?:(file:([^:]+):))?([^}:]+)(?::-([^}]*))?\}")


class EnvService:
    """Service for managing environment variables and substitution in configs."""

    # Cache for loaded env files to avoid re-reading
    _file_cache: dict[str, dict[str, str]] = {}

    @classmethod
    def clear_cache(cls):
        """Clear the env file cache. Useful for testing or when files change."""
        cls._file_cache.clear()

    @classmethod
    def load_env_file(cls, file_path: Path, use_cache: bool = True) -> dict[str, str]:
        """Load environment variables from a .env file.

        Args:
            file_path: Path to the .env file
            use_cache: Whether to use cached values (default True)

        Returns:
            Dict of environment variables from the file.
            Returns empty dict if file doesn't exist.
        """
        path_str = str(file_path.resolve())

        if use_cache and path_str in cls._file_cache:
            return cls._file_cache[path_str]

        if not file_path.exists():
            logger.debug("Env file not found", path=path_str)
            return {}

        try:
            env_vars = dotenv_values(file_path)
            result = {k: v for k, v in env_vars.items() if v is not None}
            if use_cache:
                cls._file_cache[path_str] = result
            logger.debug(
                "Loaded env file",
                path=path_str,
                var_count=len(result),
            )
            return result
        except Exception as e:
            logger.warning(
                "Failed to load env file",
                path=path_str,
                error=str(e),
            )
            return {}

    @classmethod
    def substitute_env_vars(
        cls,
        value: str,
        project_dir: str | Path | None = None,
    ) -> str:
        """Substitute ${VAR} patterns in a string.

        Supports:
        - ${VAR_NAME} - Look up in .env file, then os.environ
        - ${VAR_NAME:-default} - With fallback default value
        - ${file:.env-llamafarm:VAR_NAME} - Explicitly load from specific file
        - ${file:.env.local:VAR_NAME:-default} - Explicit file with default

        Args:
            value: String potentially containing env var references
            project_dir: Project directory for resolving relative file paths

        Returns:
            String with env vars substituted
        """
        project_path = Path(project_dir) if project_dir else None

        # Load default .env file from project directory
        default_env: dict[str, str] = {}
        if project_path:
            env_file_path = project_path / ".env"
            default_env = cls.load_env_file(env_file_path)

        def replacer(match: re.Match) -> str:
            # Group 1 is full "file:.env:" prefix, group 2 is just the filename
            explicit_file = match.group(2)  # ".env.local" or None
            var_name = match.group(3)
            default = match.group(4)  # May be None

            # Determine where to look for the variable
            if explicit_file and project_path:
                # Explicit file specified: ${file:.env.local:VAR}
                file_path = project_path / explicit_file
                file_env = cls.load_env_file(file_path)
                env_value = file_env.get(var_name)
            else:
                # Standard lookup: default_env_file → os.environ
                env_value = default_env.get(var_name)
                if env_value is None:
                    env_value = os.environ.get(var_name)

            # Return value or default
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            return ""

        return ENV_VAR_PATTERN.sub(replacer, value)

    @classmethod
    def substitute_in_dict(
        cls,
        data: dict,
        project_dir: str | Path | None = None,
    ) -> dict:
        """Recursively substitute env vars in all string values of a dict.

        Args:
            data: Dictionary to process
            project_dir: Project directory for resolving relative file paths

        Returns:
            New dictionary with env vars substituted in string values
        """
        return cls._substitute_recursive(data, project_dir)

    @classmethod
    def _substitute_recursive(
        cls,
        obj,
        project_dir: str | Path | None = None,
    ):
        """Recursively substitute env vars in any data structure."""
        if isinstance(obj, str):
            return cls.substitute_env_vars(obj, project_dir)
        elif isinstance(obj, dict):
            return {
                k: cls._substitute_recursive(v, project_dir) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [cls._substitute_recursive(item, project_dir) for item in obj]
        else:
            return obj

    @classmethod
    def has_env_vars(cls, value: str) -> bool:
        """Check if a string contains env var references.

        Args:
            value: String to check

        Returns:
            True if string contains ${...} patterns
        """
        return bool(ENV_VAR_PATTERN.search(value))

    @classmethod
    def find_env_vars(cls, value: str) -> list[tuple[str | None, str]]:
        """Find all env var references in a string.

        Args:
            value: String to search

        Returns:
            List of tuples (file_name or None, var_name)
        """
        results = []
        for match in ENV_VAR_PATTERN.finditer(value):
            explicit_file = match.group(2)  # filename or None
            var_name = match.group(3)
            results.append((explicit_file, var_name))
        return results
