"""
Configuration loader for LlamaFarm that supports YAML, TOML, and JSON formats
with JSON schema validation and write capabilities.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

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
    import tomli_w
except ImportError:
    tomli_w = None

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


def _load_json_file(file_path: Path) -> dict:
    """Load configuration from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        raise ConfigError(f"Error loading JSON file {file_path}: {e}") from e


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
    4. llamafarm.json
    """
    directory = Path.cwd() if directory is None else Path(directory)

    if not directory.is_dir():
        raise ConfigError(f"Directory does not exist: {directory}")

    # Check for config files in order of preference
    for filename in ["llamafarm.yaml", "llamafarm.yml", "llamafarm.toml", "llamafarm.json"]:
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
    config_file = _resolve_config_file(config_path, directory)

    # Load configuration based on file extension
    suffix = config_file.suffix.lower()
    if suffix in [".yaml", ".yml"]:
        config = _load_yaml_file(config_file)
    elif suffix == ".toml":
        config = _load_toml_file(config_file)
    elif suffix == ".json":
        config = _load_json_file(config_file)
    else:
        raise ConfigError(
            f"Unsupported file format: {suffix}. Supported formats: .yaml, .yml, .toml, .json"
        )

    # Validate against schema if requested
    if validate:
        schema = _load_schema()
        _validate_config(config, schema)

    return config  # type: ignore

def _resolve_config_file(config_path: str | Path | None = None, directory: str | Path = Path.cwd()) -> Path:
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.is_file():
            raise ConfigError(f"Configuration file not found: {config_file}")
    else:
        config_file = find_config_file(directory)
        if config_file is None:
            raise ConfigError(f"No configuration file found in {directory}")

    return config_file

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


# ============================================================================
# WRITE FUNCTIONS
# ============================================================================


def _save_yaml_file(config: dict, file_path: Path) -> None:
    """Save configuration to a YAML file."""
    if yaml is None:
        raise ConfigError("PyYAML is required to save YAML files.")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2,
            )
    except Exception as e:
        raise ConfigError(f"Error saving YAML file {file_path}: {e}") from e


def _save_toml_file(config: dict, file_path: Path) -> None:
    """Save configuration to a TOML file."""
    if tomli_w is None:
        raise ConfigError("tomli-w is required to save TOML files.")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            tomli_w.dump(config, f)
    except Exception as e:
        raise ConfigError(f"Error saving TOML file {file_path}: {e}") from e


def _save_json_file(config: dict, file_path: Path) -> None:
    """Save configuration to a JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise ConfigError(f"Error saving JSON file {file_path}: {e}") from e


def _create_backup(file_path: Path) -> Optional[Path]:
    """Create a backup of an existing file."""
    if not file_path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f".{timestamp}{file_path.suffix}")

    try:
        backup_path.write_bytes(file_path.read_bytes())
        return backup_path
    except Exception as e:
        raise ConfigError(f"Error creating backup {backup_path}: {e}") from e


def save_config(
    config: dict,
    config_path: str | Path = Path.cwd(),
    format: Optional[str] = None,
    validate: bool = True,
    create_backup: bool = True,
) -> Path:
    """
    Save a configuration to disk.

    Args:
        config: Configuration dictionary to save.
        config_path: Path where to save the configuration file.
        format: File format to use ('yaml', 'toml', 'json').
               If None, infers from file extension.
        validate: Whether to validate against JSON schema before saving.
        create_backup: Whether to create a backup of existing file.

    Returns:
        Path to the saved configuration file.

    Raises:
        ConfigError: If validation fails or file cannot be saved.
    """
    config_file = Path(config_path)

    # Validate configuration before saving
    if validate:
        schema = _load_schema()
        _validate_config(config, schema)

    # Create backup if requested and file exists
    backup_path = None
    if create_backup and config_file.exists():
        backup_path = _create_backup(config_file)

    # Determine format
    if format is None:
        suffix = config_file.suffix.lower()
        if suffix in [".yaml", ".yml"]:
            format = "yaml"
        elif suffix == ".toml":
            format = "toml"
        elif suffix == ".json":
            format = "json"
        else:
            raise ConfigError(
                f"Cannot infer format from extension '{suffix}'. "
                "Please specify format explicitly or use .yaml, .yml, .toml, or .json extension."
            )

    # Ensure parent directory exists
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Save file based on format
    try:
        if format.lower() == "yaml":
            _save_yaml_file(config, config_file)
        elif format.lower() == "toml":
            _save_toml_file(config, config_file)
        elif format.lower() == "json":
            _save_json_file(config, config_file)
        else:
            raise ConfigError(f"Unsupported format: {format}")

        return config_file

    except Exception as e:
        # If save failed and we created a backup, try to restore it
        if backup_path and backup_path.exists():
            try:
                config_file.write_bytes(backup_path.read_bytes())
                backup_path.unlink()  # Remove backup since we restored it
            except Exception:
                pass  # Don't mask the original error
        raise


def update_config(
    config_path: str | Path,
    updates: dict,
    validate: bool = True,
    create_backup: bool = True,
) -> Path:
    """
    Update an existing configuration file with new values.

    Args:
        config_path: Path to the existing configuration file.
        updates: Dictionary of updates to apply to the configuration.
        validate: Whether to validate the updated configuration.
        create_backup: Whether to create a backup before updating.

    Returns:
        Path to the updated configuration file.

    Raises:
        ConfigError: If file doesn't exist, cannot be loaded, or validation fails.
    """
    config_file = Path(config_path or ".")

    if not config_file.exists():
        raise ConfigError(f"Configuration file not found: {config_file}")

    # Load existing configuration
    config = load_config(config_path, validate=False)

    # Apply updates (deep merge)
    def deep_update(base: dict, updates: dict) -> dict:
        """Recursively update a dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                deep_update(base[key], value)
            else:
                base[key] = value
        return base

    updated_config = deep_update(config, updates)

    # Save updated configuration (preserves original format)
    return save_config(
        updated_config,
        config_file,
        validate=validate,
        create_backup=create_backup,
    )
