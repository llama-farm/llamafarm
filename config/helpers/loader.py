"""
Configuration loader for LlamaFarm that supports YAML, TOML, and JSON formats
with JSON schema validation and write capabilities.
"""

import json
import os
import re
import socket
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import tomli_w
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import LiteralScalarString

try:
    import jsonschema  # type: ignore
except ImportError:
    jsonschema = None


# ============================================================================
# YAML UTILITIES (ruamel.yaml - single library for all YAML operations)
# ============================================================================


def _get_ruamel_yaml() -> YAML:
    """Create a configured ruamel.yaml instance for all YAML operations."""
    yaml_instance = YAML()
    yaml_instance.preserve_quotes = True
    # Configure indentation:
    # - mapping=2: 2 spaces for nested mappings
    # - sequence=4: 4 spaces for sequence items (includes the "- ")
    # - offset=2: the "-" is indented 2 spaces from the parent key
    yaml_instance.indent(mapping=2, sequence=4, offset=2)
    return yaml_instance


def _commented_map_to_dict(obj: Any) -> Any:
    """Recursively convert CommentedMap/CommentedSeq to plain dict/list."""
    if isinstance(obj, CommentedMap):
        return {k: _commented_map_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, CommentedSeq):
        return [_commented_map_to_dict(item) for item in obj]
    else:
        return obj


def _dict_to_commented_map(obj: Any) -> Any:
    """
    Recursively convert plain dict/list to CommentedMap/CommentedSeq.

    Multiline strings are converted to LiteralScalarString to use block style (|).
    """
    if isinstance(obj, dict):
        cm = CommentedMap()
        for k, v in obj.items():
            cm[k] = _dict_to_commented_map(v)
        return cm
    elif isinstance(obj, list):
        cs = CommentedSeq()
        for item in obj:
            cs.append(_dict_to_commented_map(item))
        return cs
    elif isinstance(obj, str) and "\n" in obj:
        # Use block scalar style for multiline strings
        return LiteralScalarString(obj)
    else:
        return obj


def _deep_merge(target: dict, source: dict) -> dict:
    """
    Recursively merge source dict into target dict.

    Args:
        target: Dictionary to merge into (modified in place)
        source: Dictionary with values to merge

    Returns:
        The modified target dict
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


# Removed complex referencing imports - using compile_schema.py instead

# Handle both relative and absolute imports
try:
    from config.datamodel import LlamaFarmConfig
    from .component_resolver import ComponentResolver
except ImportError:
    # If relative import fails, try absolute import (when run directly)
    import sys
    from pathlib import Path

    # Add current directory to path to find config_types module
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    from ..datamodel import LlamaFarmConfig
    from .component_resolver import ComponentResolver


class ConfigError(Exception):
    """Raised when there's an error loading or validating configuration."""

    pass


# Cache for DNS resolution result to avoid repeated lookups
_host_docker_internal_cache: bool | None = None


def _is_host_docker_internal_resolvable() -> bool:
    """
    Check if host.docker.internal is resolvable (cached).

    Returns:
        True if host.docker.internal can be resolved, False otherwise.
    """
    global _host_docker_internal_cache

    if _host_docker_internal_cache is None:
        try:
            socket.gethostbyname("host.docker.internal")
            _host_docker_internal_cache = True
        except (socket.gaierror, OSError):
            _host_docker_internal_cache = False

    return _host_docker_internal_cache


def _reset_host_docker_internal_cache() -> None:
    """
    Reset the DNS resolution cache. Primarily for testing purposes.
    """
    global _host_docker_internal_cache
    _host_docker_internal_cache = None


def _replace_localhost_url(url: str) -> str:
    """
    Replace localhost URLs with host.docker.internal if resolvable.

    Args:
        url: The URL to potentially replace

    Returns:
        The URL with localhost replaced by host.docker.internal if applicable
    """
    if not isinstance(url, str):
        return url

    # Pattern to match localhost or 127.0.0.1 URLs with optional port
    localhost_pattern = r"^(https?://)(localhost|127\.0\.0\.1)(:\d+)?(/.*)?$"

    if re.match(localhost_pattern, url) and _is_host_docker_internal_resolvable():
        # Replace localhost or 127.0.0.1 with host.docker.internal
        return re.sub(
            r"^(https?://)(localhost|127\.0\.0\.1)", r"\1host.docker.internal", url
        )

    return url


def _replace_urls_in_config(config: Any) -> Any:
    """
    Recursively traverse config and replace localhost URLs with host.docker.internal.

    Args:
        config: Configuration dictionary to process

    Returns:
        Configuration dictionary with URLs replaced
    """
    if isinstance(config, dict):
        result = {}
        for key, value in config.items():
            if isinstance(value, str):
                # Try to replace URLs in string values
                result[key] = _replace_localhost_url(value)
            elif isinstance(value, (dict, list)):
                # Recursively process nested structures
                result[key] = _replace_urls_in_config(value)
            else:
                result[key] = value
        return result
    elif isinstance(config, list):
        return [_replace_urls_in_config(item) for item in config]
    elif isinstance(config, str):
        # Handle string values in lists or other contexts
        return _replace_localhost_url(config)
    else:
        return config


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
        spec = importlib.util.spec_from_file_location(
            "compile_schema", compile_schema_path
        )
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
        raise ConfigError(
            f"Configuration validation error: {e.message}"
            + (f" at path {path_str}" if path_str else "")
        ) from e
    except Exception as e:
        raise ConfigError(f"Error during validation: {e}") from e


def _load_yaml_file(file_path: Path) -> dict:
    """Load configuration from a YAML file as a plain dict."""
    try:
        yaml_instance = _get_ruamel_yaml()
        with open(file_path, encoding="utf-8") as f:
            doc = yaml_instance.load(f)
            # Convert to plain dict for compatibility with rest of codebase
            return _commented_map_to_dict(doc) if doc else {}
    except Exception as e:
        raise ConfigError(f"Error loading YAML file {file_path}: {e}") from e


def _load_toml_file(file_path: Path) -> dict:
    """Load configuration from a TOML file."""
    try:
        with open(file_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        raise ConfigError(f"Error loading TOML file {file_path}: {e}") from e


def _load_json_file(file_path: Path) -> dict:
    """Load configuration from a JSON file."""
    try:
        with open(file_path, encoding="utf-8") as f:
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
    for filename in [
        "llamafarm.yaml",
        "llamafarm.yml",
        "llamafarm.toml",
        "llamafarm.json",
    ]:
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

    # Replace localhost URLs with host.docker.internal if resolvable
    config = _replace_urls_in_config(config)

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
                raise ConfigError(
                    f"Configuration file not found: {config_path_resolved}"
                )
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
    config_obj = LlamaFarmConfig(**config_dict)

    # Resolve reusable components (embedding/retrieval/parsers) into inline configs
    resolver = ComponentResolver(config_obj)
    return resolver.resolve_config(config_obj)


# ============================================================================
# FORMAT-PRESERVING YAML FUNCTIONS (ruamel.yaml)
# ============================================================================


def _load_yaml_preserved(file_path: Path) -> CommentedMap:
    """
    Load a YAML file preserving comments and formatting using ruamel.yaml.

    Args:
        file_path: Path to the YAML file

    Returns:
        CommentedMap that preserves comments and formatting when saved
    """
    yaml_instance = _get_ruamel_yaml()
    with open(file_path, encoding="utf-8") as f:
        return yaml_instance.load(f) or CommentedMap()


def _preserve_string_style(existing: Any, new_value: str) -> str | LiteralScalarString:
    """
    Preserve the YAML string style when replacing a value.

    If the existing value was a block scalar (LiteralScalarString) and the new value
    contains newlines, wrap it in LiteralScalarString to preserve the `|` style.

    Args:
        existing: The existing value being replaced
        new_value: The new string value

    Returns:
        The new value, possibly wrapped in LiteralScalarString
    """
    if isinstance(existing, LiteralScalarString) and isinstance(new_value, str):
        # Existing was a block scalar - preserve that style
        return LiteralScalarString(new_value)
    elif isinstance(new_value, str) and "\n" in new_value:
        # New multiline string - use block scalar style
        return LiteralScalarString(new_value)
    return new_value


def _deep_merge_preserved(
    target: CommentedMap | CommentedSeq,
    source: dict | list,
) -> CommentedMap | CommentedSeq:
    """
    Deep merge source dict/list into target CommentedMap/CommentedSeq,
    preserving comments, formatting, and string styles in the target.

    Args:
        target: ruamel.yaml CommentedMap or CommentedSeq to merge into
        source: Plain dict or list with new values

    Returns:
        The modified target with merged values
    """
    if isinstance(target, CommentedMap) and isinstance(source, dict):
        for key, value in source.items():
            if key in target:
                existing = target[key]
                if isinstance(existing, CommentedMap) and isinstance(value, dict):
                    # Recursively merge nested dicts
                    _deep_merge_preserved(existing, value)
                elif isinstance(existing, CommentedSeq) and isinstance(value, list):
                    # For lists, we need to handle more carefully
                    # Replace the list but try to preserve structure for matching items
                    _merge_list_preserved(existing, value)
                elif isinstance(value, str):
                    # Preserve string style (block scalars, etc.)
                    target[key] = _preserve_string_style(existing, value)
                else:
                    # Replace other scalar or type-changed values
                    target[key] = value
            else:
                # New key - convert multiline strings to block scalars
                if isinstance(value, str) and "\n" in value:
                    target[key] = LiteralScalarString(value)
                else:
                    target[key] = value
    elif isinstance(target, CommentedSeq) and isinstance(source, list):
        _merge_list_preserved(target, source)

    return target


def _merge_list_preserved(target: CommentedSeq, source: list) -> None:
    """
    Replace target CommentedSeq contents with source list items.

    Converts all items through _dict_to_commented_map() to ensure
    multiline strings use block scalar style.

    Args:
        target: ruamel.yaml CommentedSeq to replace contents of
        source: Plain list with new values
    """
    target.clear()
    for item in source:
        target.append(_dict_to_commented_map(item))


def _save_yaml_preserved(
    doc: CommentedMap,
    file_path: Path,
    force_sync: bool = False,
) -> None:
    """
    Save a ruamel.yaml document preserving comments and formatting.

    Args:
        doc: CommentedMap to save
        file_path: Path to save the YAML file
        force_sync: If True, forces immediate write to disk
    """
    try:
        yaml_instance = _get_ruamel_yaml()
        with open(file_path, "w", encoding="utf-8") as f:
            yaml_instance.dump(doc, f)
            f.flush()

            if force_sync:
                os.fsync(f.fileno())
    except Exception as e:
        raise ConfigError(f"Error saving YAML file {file_path}: {e}") from e


def _save_yaml(
    config_dict: dict,
    config_file: Path,
    template_path: Path | None = None,
    force_sync: bool = False,
) -> None:
    """
    Save YAML config using ruamel.yaml, preserving comments and formatting when possible.

    This function uses a single code path (ruamel.yaml) for all YAML saves:
    1. If config_file exists: load original, merge changes, save (preserves existing comments)
    2. If template_path provided: load template, merge changes, save (preserves template comments)
    3. Otherwise: convert dict to CommentedMap and save (consistent formatting)

    Args:
        config_dict: Configuration dictionary to save
        config_file: Destination path for the YAML file
        template_path: Optional template file to use for formatting new files
        force_sync: If True, forces immediate write to disk
    """
    # Determine which file to use as the formatting source
    if config_file.exists():
        # Updating existing file - preserve its formatting
        doc = _load_yaml_preserved(config_file)
        _deep_merge_preserved(doc, config_dict)
    elif template_path and template_path.exists():
        # Creating from template - preserve template formatting
        doc = _load_yaml_preserved(template_path)
        _deep_merge_preserved(doc, config_dict)
    else:
        # No source to preserve from - create fresh CommentedMap
        doc = _dict_to_commented_map(config_dict)

    # Save with ruamel.yaml (consistent behavior for all cases)
    _save_yaml_preserved(doc, config_file, force_sync=force_sync)


# ============================================================================
# WRITE FUNCTIONS (TOML and JSON)
# ============================================================================


def _save_toml_file(config: dict, file_path: Path) -> None:
    """Save configuration to a TOML file."""
    try:
        with open(file_path, "wb") as f:
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


def _create_backup(file_path: Path) -> Path | None:
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
    template_path: str | Path | None = None,
) -> tuple[Path, LlamaFarmConfig]:
    """
    Save a configuration to disk, preserving comments and formatting when possible.

    For YAML files, this function preserves comments and formatting by:
    - If updating an existing file: loads the original, merges changes, saves preserved
    - If creating from a template: loads the template, merges changes, saves preserved
    - Otherwise: uses standard YAML dump

    Args:
        config: Configuration to save.
        config_path: Path where to save the configuration file or directory.
                    If it's a file path, saves to that file.
                    If it's a directory, looks for existing config or defaults to llamafarm.yaml.
        format: File format to use ('yaml', 'toml', 'json').
               If None, infers from file extension.
        create_backup: Whether to create a backup of existing file.
        force_sync: If True, forces immediate write to disk for YAML files (slower but safer).
                   Use for critical configurations.
        template_path: Optional path to a YAML template file. When provided for new files,
                      the template's comments and formatting will be preserved.

    Returns:
        Tuple of (path to saved file, validated config).

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

    # Run custom validators for constraints beyond JSON Schema
    from config.validators import validate_llamafarm_config

    try:
        validate_llamafarm_config(config_dict)
    except ValueError as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e

    # Create backup if requested and file exists
    backup_path = None
    if create_backup and config_file.exists():
        backup_path = _create_backup(config_file)

    # Determine format
    if format is None:
        suffix = (
            config_file.suffix.lower()
            if config_file and config_file.suffix
            else ".yaml"
        )
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
            _save_yaml(
                config_dict,
                config_file,
                template_path=Path(template_path) if template_path else None,
                force_sync=force_sync,
            )
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
                raise ConfigError(
                    f"No configuration file found in directory: {config_path}"
                )
        except ConfigError as e:
            raise ConfigError(
                f"Directory does not exist or contains no config file: {config_path}"
            ) from e  # noqa: E501

    # Load existing configuration
    config = load_config(config_file, validate=False)

    # Apply updates using deep merge
    config_dict = config.model_dump(mode="json", exclude_none=True)
    _deep_merge(config_dict, updates)

    # Save updated configuration (preserves original format)
    saved_path, cfg = save_config(
        LlamaFarmConfig(**config_dict),
        config_file,
        create_backup=create_backup,
        force_sync=force_sync,
    )
    return saved_path, cfg
