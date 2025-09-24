"""
Configuration loader for LlamaFarm that supports YAML, TOML, and JSON formats
with JSON schema validation and write capabilities.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
# urllib.parse imports removed - no longer needed for $ref resolution

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
except ImportError:
    tomllib = None  # type: ignore

try:
    import tomli_w  # type: ignore
except ImportError:
    tomli_w = None

try:
    import jsonschema  # type: ignore
except ImportError:
    jsonschema = None
# Removed complex referencing imports - using compile_schema.py instead

# Handle both relative and absolute imports
try:
    from config.datamodel import LlamaFarmConfig
except ImportError:
    # If relative import fails, try absolute import (when run directly)
    import sys
    from pathlib import Path

    # Add current directory to path to find config_types module
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    from ..datamodel import LlamaFarmConfig


class ConfigError(Exception):
    """Raised when there's an error loading or validating configuration."""

    pass


def _load_schema() -> dict:
    """Load the JSON schema with all $refs dereferenced using compile_schema.py."""
    try:
        # Import the dereferencing function from our compile_schema module
        import sys
        from pathlib import Path

        # Add the config directory to the path if needed
        config_dir = Path(__file__).parent.parent
        if str(config_dir) not in sys.path:
            sys.path.insert(0, str(config_dir))

        # Import and use the dereferencing function
        import importlib.util

        compile_schema_path = config_dir / "compile_schema.py"
        spec = importlib.util.spec_from_file_location("compile_schema", compile_schema_path)
        compile_schema = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(compile_schema)  # type: ignore

        return compile_schema.get_dereferenced_schema()
    except Exception as e:
        raise ConfigError(f"Error loading dereferenced schema: {e}") from e


def _validate_config(config: dict, schema: dict) -> None:
    """Validate configuration against JSON schema (schema is already dereferenced)."""
    if jsonschema is None:
        # If jsonschema is not available, skip validation but warn
        print("Warning: jsonschema not installed. Skipping validation.")
        return

    try:
        # Simple validation since schema is already fully dereferenced by compile_schema.py
        jsonschema.validate(config, schema)
    except jsonschema.ValidationError as e:
        path_str = ".".join(str(p) for p in e.path) if hasattr(e, "path") else ""
        raise ConfigError(f"Configuration validation error: {e.message}" + (f" at path {path_str}" if path_str else "")) from e
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


def load_config_dict(
    config_path: str | Path | None = None,
    directory: str | Path | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Load configuration as a regular dictionary
    (same as load_config but with different return type annotation).

    This is useful when you don't need strict typing or are working with dynamic configurations.
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

    return config


def _resolve_config_file(
    config_path: str | Path | None = None,
    directory: str | Path | None = None,
) -> Path:
    directory = directory or Path.cwd()

    config_path_resolved: Path | None = None
    if config_path is not None:
        config_path_resolved = Path(config_path)

        if config_path_resolved.suffix:
            # It's a file path
            if not config_path_resolved.is_file():
                raise ConfigError(f"Configuration file not found: {config_path_resolved}")
        else:
            # It's a directory path, look for config file within it
            config_path_resolved = find_config_file(config_path_resolved)
            if config_path_resolved is None:
                raise ConfigError(f"No configuration file found in {config_path}")
    else:
        config_path_resolved = find_config_file(directory)
        if config_path_resolved is None:
            raise ConfigError(f"No configuration file found in {directory}")

    return config_path_resolved


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
        Loaded and validated configuration as a LlamaFarmConfig object.

    Raises:
        ConfigError: If file is not found, cannot be loaded, or validation fails.
    """

    config_dict = load_config_dict(config_path, directory, validate)
    return LlamaFarmConfig(**config_dict)


# ============================================================================
# WRITE FUNCTIONS
# ============================================================================


def _save_yaml_file(config: dict, file_path: Path, force_sync: bool = False) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        file_path: Path to save the YAML file
        force_sync: If True, forces immediate write to disk (slower but safer)
    """
    if yaml is None:
        raise ConfigError("PyYAML is required to save YAML files.")

    try:
        # First dump to string to ensure complete serialization
        yaml_content = yaml.dump(
            config,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
        )
        
        # Then write the complete string to file atomically
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
            f.flush()  # Explicitly flush to ensure all data is written
            
            # Only force sync for critical files when explicitly requested
            if force_sync:
                os.fsync(f.fileno())  # Force write to disk (slower but safer)
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
    config: LlamaFarmConfig,
    config_path: str | Path | None,
    format: str | None = None,
    create_backup: bool = True,
    force_sync: bool = False,
) -> tuple[Path, LlamaFarmConfig]:
    """
    Save a configuration to disk.

    Args:
        config: Configuration dictionary to save.
        config_path: Path where to save the configuration file or directory.
                    If it's a file path, saves to that file.
                    If it's a directory, looks for existing config or defaults to llamafarm.yaml.
        format: File format to use ('yaml', 'toml', 'json').
               If None, infers from file extension.
        create_backup: Whether to create a backup of existing file.
        force_sync: If True, forces immediate write to disk for YAML files (slower but safer).
                   Use for critical configurations.

    Returns:
        Path to the saved configuration file.

    Raises:
        ConfigError: If validation fails or file cannot be saved.
    """
    config_path = Path(config_path) if config_path else Path.cwd()

    # Determine the actual config file path
    if config_path.suffix:
        # It's a file path, use it directly
        config_file = config_path
    else:
        # It's a directory path, look for existing config or use default
        try:
            existing_config = find_config_file(config_path)
            if existing_config:
                config_file = existing_config
            else:
                # No existing config, create new one with appropriate extension
                if format == "json":
                    config_file = config_path / "llamafarm.json"
                elif format == "toml":
                    config_file = config_path / "llamafarm.toml"
                else:
                    config_file = config_path / "llamafarm.yaml"
        except ConfigError:
            # Directory doesn't exist, use default filename based on format
            if format == "json":
                config_file = config_path / "llamafarm.json"
            elif format == "toml":
                config_file = config_path / "llamafarm.toml"
            else:
                config_file = config_path / "llamafarm.yaml"

    # Validate configuration before saving
    config_dict = config.model_dump(mode="json", exclude_none=True)

    # Create backup if requested and file exists
    backup_path = None
    if create_backup and config_file.exists():
        backup_path = _create_backup(config_file)

    # Determine format
    if format is None:
        suffix = config_file.suffix.lower() if config_file and config_file.suffix else ".yaml"
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

    # Save file based on format
    try:
        if format.lower() == "yaml":
            _save_yaml_file(config_dict, config_file, force_sync=force_sync)
        elif format.lower() == "toml":
            _save_toml_file(config_dict, config_file)
        elif format.lower() == "json":
            _save_json_file(config_dict, config_file)
        else:
            raise ConfigError(f"Unsupported format: {format}")

        return (config_file, LlamaFarmConfig(**config_dict))

    except Exception:
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
    create_backup: bool = True,
    force_sync: bool = False,
) -> tuple[Path, LlamaFarmConfig]:
    """
    Update an existing configuration file with new values.

    Args:
        config_path: Path to the existing configuration file or directory.
                    If it's a directory, looks for existing config file.
        updates: Dictionary of updates to apply to the configuration.
        create_backup: Whether to create a backup before updating.
        force_sync: If True, forces immediate write to disk for YAML files (slower but safer).

    Returns:
        Path to the updated configuration file.

    Raises:
        ConfigError: If file doesn't exist, cannot be loaded, or validation fails.
    """
    config_path = Path(config_path)

    # Determine the actual config file path
    config_file: Path | None = None
    if config_path.suffix:
        # It's a file path
        config_file = config_path
        if not config_file.exists():
            raise ConfigError(f"Configuration file not found: {config_file}")
    else:
        # It's a directory path, look for existing config
        try:
            config_file = find_config_file(config_path)
            if config_file is None:
                raise ConfigError(f"No configuration file found in directory: {config_path}")
        except ConfigError as e:
            raise ConfigError(
                f"Directory does not exist or contains no config file: {config_path}"
            ) from e  # noqa: E501

    # Load existing configuration
    config = load_config(config_file, validate=False)

    # Apply updates (deep merge)
    def deep_update(base: dict, updates: dict) -> dict:
        """Recursively update a dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                deep_update(base[key], value)
            else:
                base[key] = value
        return base

    config_dict = config.model_dump(mode="json", exclude_none=True)
    updated_config_dict = deep_update(config_dict, updates)

    # Save updated configuration (preserves original format)
    saved_path, cfg = save_config(
        LlamaFarmConfig(**updated_config_dict),
        config_file,
        create_backup=create_backup,
        force_sync=force_sync,
    )
    return saved_path, cfg
