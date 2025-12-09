import sys
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
from config.datamodel import LlamaFarmConfig  # noqa: E402


def _load_yaml_as_dict(file_path: Path) -> dict[str, Any]:
    """Load a YAML file as a plain dict using ruamel.yaml."""
    yaml_instance = YAML()
    with open(file_path, encoding="utf-8") as f:
        doc = yaml_instance.load(f)
        # Convert CommentedMap to plain dict recursively
        return _to_plain_dict(doc) if doc else {}


def _to_plain_dict(obj: Any) -> Any:
    """Recursively convert ruamel.yaml objects to plain Python types."""
    if hasattr(obj, "items"):  # dict-like
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_plain_dict(item) for item in obj]
    else:
        return obj


def generate_base_config(
    namespace: str,
    name: str | None = None,
    config_template_path: str | None = None,
) -> tuple[dict, Path]:
    """
    Generate a valid base configuration from a YAML config template file.

    Args:
        namespace: Namespace for the configuration.
        name: Optional override for the resulting configuration's `name` field.
        config_template_path: Optional absolute or relative filesystem path to a YAML file that
                     contains a complete, valid configuration structure.
                     If not provided, uses built-in `config/templates/default.yaml`.

    Returns:
        Tuple of (config_dict, template_path) where:
        - config_dict: Dict representation of a validated LlamaFarmConfig (model_dump JSON mode)
        - template_path: Path to the template file used (for format preservation when saving)

    Raises:
        FileNotFoundError: If the config template file cannot be found.
        ValueError: If the loaded config is invalid.
    """

    template_path = (
        Path(config_template_path)
        if config_template_path is not None
        else Path(__file__).parent.parent / "templates" / "default.yaml"
    )

    if not template_path.exists() or not template_path.is_file():
        raise FileNotFoundError(f"Config template file not found: {template_path}")

    try:
        raw_cfg = _load_yaml_as_dict(template_path)
    except Exception as e:  # pragma: no cover
        raise ValueError(
            f"Error reading config template file '{template_path}': {e}"
        ) from e

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
        raise ValueError(
            f"Config template content is not a valid LlamaFarmConfig: {e}"
        ) from e

    # Return JSON-serializable dict and the template path for format preservation
    cfg = validated.model_dump(mode="json", exclude_none=True)
    return cfg, template_path
