import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
from config.datamodel import LlamaFarmConfig  # noqa: E402


def generate_base_config(
    namespace: str,
    name: str | None = None,
    config_template_path: str | None = None,
) -> dict:
    """
    Generate a valid base configuration from a YAML config template file.

    Args:
        config_template_path: Optional absolute or relative filesystem path to a YAML file that
                     contains a complete, valid configuration structure.
                     If not provided, uses built-in `config/templates/default.yaml`.
        name: Optional override for the resulting configuration's `name` field.

    Returns:
        Dict representation of a validated LlamaFarmConfig (model_dump JSON mode).

    Raises:
        FileNotFoundError: If the config template file cannot be found.
        ValueError: If the loaded config is invalid.
    """

    path = (
        Path(config_template_path)
        if config_template_path is not None
        else Path(__file__).parent.parent / "templates" / "default.yaml"
    )

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Config template file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_cfg: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Error reading config template file '{path}': {e}") from e

    raw_cfg.update(
        {
            "namespace": namespace,
            "name": name or raw_cfg.get("name", ""),
        }
    )

    # Validate against current data model to ensure correctness
    try:
        validated = LlamaFarmConfig(**raw_cfg)
    except Exception as e:
        raise ValueError(f"Config template content is not a valid LlamaFarmConfig: {e}") from e

    # Return JSON-serializable dict
    cfg = validated.model_dump(mode="json", exclude_none=True)
    return cfg
