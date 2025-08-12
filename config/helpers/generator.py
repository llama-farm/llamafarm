from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from config.datamodel import LlamaFarmConfig


def generate_base_config() -> dict:
    """
    Deprecated shim that now proxies to the schema-based generator using
    the built-in `config/schemas/default.yaml`. The returned config name
    is overridden to `llamafarm` for backward compatibility with callers
    that set project name after generation.
    """
    default_schema = (
        Path(__file__).parent.parent / "schemas" / "default.yaml"
    )
    cfg = generate_base_config_from_schema(str(default_schema))
    cfg.update({"name": "llamafarm"})
    return cfg


def generate_base_config_from_schema(schema_path: str) -> dict:
    """
    Generate a valid base configuration from a YAML schema/template file.

    Args:
        schema_path: Absolute or relative filesystem path to a YAML file that
                     contains a complete, valid configuration structure.

    Returns:
        Dict representation of a validated LlamaFarmConfig (model_dump JSON mode).

    Raises:
        FileNotFoundError: If the schema file cannot be found.
        ValueError: If PyYAML is not installed or if the loaded config is invalid.
    """
    if yaml is None:
        raise ValueError("PyYAML is required to load schema templates. Install 'pyyaml'.")

    path = Path(schema_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Schema file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_cfg: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Error reading schema file '{path}': {e}") from e

    # Validate against current data model to ensure correctness
    try:
        validated = LlamaFarmConfig(**raw_cfg)
    except Exception as e:
        raise ValueError(f"Schema content is not a valid LlamaFarmConfig: {e}") from e

    # Return JSON-serializable dict matching existing generate_base_config format
    return validated.model_dump(mode="json", exclude_none=True)
