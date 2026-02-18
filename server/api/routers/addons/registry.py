"""Addon registry loader."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_addon_registry() -> dict[str, dict[str, Any]]:
    """
    Load addon registry from individual YAML files in addons/registry/ directory.

    Each addon has its own YAML file (e.g., stt.yaml, tts.yaml).
    Returns a dict mapping addon names to their metadata.
    """
    # Path to registry directory (relative to repo root)
    registry_dir = Path(__file__).parent.parent.parent.parent.parent / "addons" / "registry"

    if not registry_dir.exists():
        raise FileNotFoundError(f"Addon registry directory not found at {registry_dir}")

    registry = {}

    # Load all .yaml files in the registry directory
    for yaml_file in sorted(registry_dir.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                addon_data = yaml.safe_load(f)

            if not addon_data or "name" not in addon_data:
                continue  # Skip invalid files

            addon_name = addon_data["name"]
            registry[addon_name] = addon_data

        except Exception as e:
            # Log but don't fail - allow other addons to load
            logger.warning(f"Failed to load addon from {yaml_file.name}: {e}")

    return registry


# Singleton instance
_ADDON_REGISTRY: dict[str, dict[str, Any]] | None = None


def get_addon_registry() -> dict[str, dict[str, Any]]:
    """Get the addon registry (cached)."""
    global _ADDON_REGISTRY
    if _ADDON_REGISTRY is None:
        _ADDON_REGISTRY = load_addon_registry()
    return _ADDON_REGISTRY


def reload_addon_registry() -> dict[str, dict[str, Any]]:
    """Reload the addon registry from disk, replacing the cached instance."""
    global _ADDON_REGISTRY
    _ADDON_REGISTRY = load_addon_registry()
    return _ADDON_REGISTRY
