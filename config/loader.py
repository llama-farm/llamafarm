"""
Configuration loader for LlamaFarm that supports YAML and TOML formats
with JSON schema validation.
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
except ImportError:
    tomllib = None

try:
    import jsonschema
except ImportError:
    jsonschema = None

# Handle both relative and absolute imports
try:
    from .config_types import ConfigDict, LlamaFarmConfig
except ImportError:
    # If relative import fails, try absolute import (when run directly)
    import sys
    from pathlib import Path

    # Add current directory to path to find config_types module
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    try:
        from config_types import ConfigDict, LlamaFarmConfig
    except ImportError:
        # If all fails, define minimal types
        from typing import Any

        LlamaFarmConfig = dict[str, Any]
        ConfigDict = dict[str, Any]


class ConfigError(Exception):
    """Raised when there's an error loading or validating configuration."""

    pass


def _load_schema() -> dict:
    """Load the JSON schema from schema.yaml."""
    schema_path = Path(__file__).parent / "schema.yaml"

    if not schema_path.exists():
        raise ConfigError(f"Schema file not found: {schema_path}")

    if yaml is None:
        raise ConfigError("PyYAML is required to load the schema.")

    try:
        with open(schema_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Error loading schema: {e}") from e


def _validate_config(config: dict, schema: dict) -> None:
    """Validate configuration against JSON schema."""
    if jsonschema is None:
        # If jsonschema is not available, skip validation but warn
        print("Warning: jsonschema not installed. Skipping validation.")
        return

    try:
        jsonschema.validate(config, schema)
    except jsonschema.ValidationError as e:
        raise ConfigError(f"Configuration validation error: {e.message}") from e
    except Exception as e:
        raise ConfigError(f"Error during validation: {e}") from e


def _load_yaml_file(file_path: Path) -> dict:
    """Load configuration from a YAML file."""
    if yaml is None:
        raise ConfigError("PyYAML is required to load YAML files.")

    try:
        with open(file_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigError(f"Error loading YAML file {file_path}: {e}") from e


def _load_toml_file(file_path: Path) -> dict:
    """Load configuration from a TOML file."""
    if tomllib is None:
        if sys.version_info >= (3, 11):
            raise ConfigError("tomllib module not available")
        else:
            raise ConfigError("tomli is required to load TOML files.")

    try:
        with open(file_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        raise ConfigError(f"Error loading TOML file {file_path}: {e}") from e


def find_config_file(directory: str | Path | None = None) -> Path | None:
    """
    Find a LlamaFarm configuration file in the specified directory.

    Args:
        directory: Directory to search in. Defaults to current working directory.

    Returns:
        Path to the configuration file if found, None otherwise.

    Looks for files in this order:
    1. llamafarm.yaml
    2. llamafarm.yml
    3. llamafarm.toml
    """
    directory = Path.cwd() if directory is None else Path(directory)

    if not directory.is_dir():
        raise ConfigError(f"Directory does not exist: {directory}")

    # Check for config files in order of preference
    for filename in ["llamafarm.yaml", "llamafarm.yml", "llamafarm.toml"]:
        config_path = directory / filename
        if config_path.is_file():
            return config_path

    return None


def load_config(
    config_path: str | Path | None = None,
    directory: str | Path | None = None,
    validate: bool = True,
) -> LlamaFarmConfig:
    """
    Load and validate a LlamaFarm configuration file.

    Args:
        config_path: Explicit path to configuration file. If provided, directory is ignored.
        directory: Directory to search for configuration file. Defaults to current working dir.
        validate: Whether to validate against JSON schema. Defaults to True.

    Returns:
        Loaded and validated configuration as a typed dictionary.

    Raises:
        ConfigError: If file is not found, cannot be loaded, or validation fails.
    """
    # Determine config file path
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.is_file():
            raise ConfigError(f"Configuration file not found: {config_file}")
    else:
        config_file = find_config_file(directory)
        if config_file is None:
            search_dir = Path(directory) if directory else Path.cwd()
            raise ConfigError(f"No configuration file found in {search_dir}")

    # Load configuration based on file extension
    suffix = config_file.suffix.lower()
    if suffix in [".yaml", ".yml"]:
        config = _load_yaml_file(config_file)
    elif suffix == ".toml":
        config = _load_toml_file(config_file)
    else:
        raise ConfigError(
            f"Unsupported file format: {suffix}. Supported formats: .yaml, .yml, .toml"
        )

    # Validate against schema if requested
    if validate:
        schema = _load_schema()
        _validate_config(config, schema)

    return config  # type: ignore


def load_config_dict(
    config_path: str | Path | None = None,
    directory: str | Path | None = None,
    validate: bool = True,
) -> ConfigDict:
    """
    Load configuration as a regular dictionary
    (same as load_config but with different return type annotation).

    This is useful when you don't need strict typing or are working with dynamic configurations.
    """
    return load_config(config_path, directory, validate)
