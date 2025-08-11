#!/usr/bin/env python3
"""
Schema compilation script using jsonref with proper conversion.
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import jsonref
import yaml


ROOT = Path(__file__).parent / "schema.yaml"


def load_text_from_uri(uri: str) -> str:
    """Read local file:// or plain path URIs into text (UTF-8)."""
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        path = Path(parsed.path or uri)
        content = path.read_text(encoding="utf-8")
        return content
    raise ValueError(f"Unsupported URI scheme in $ref: {uri}")


def yaml_json_loader(uri: str):
    """YAML-aware loader for jsonref: parse .yaml/.yml as YAML, else JSON."""
    text = load_text_from_uri(uri)
    if uri.endswith((".yaml", ".yml")):
        return yaml.safe_load(text)
    return json.loads(text)


def jsonref_to_dict(obj):
    """Recursively convert jsonref proxy objects to plain Python dicts/lists."""
    if isinstance(obj, dict):
        # Check if this is a jsonref proxy object
        if hasattr(obj, "__subject__"):
            # It's a jsonref proxy, get the underlying dict
            obj = dict(obj)
        return {k: jsonref_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [jsonref_to_dict(item) for item in obj]
    else:
        return obj


def load_and_deref_schema(path: Path):
    """Load YAML schema and dereference all $refs."""

    with path.open(encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    # Use jsonref to dereference
    deref = jsonref.JsonRef.replace_refs(
        schema,
        base_uri=path.as_uri(),
        loader=yaml_json_loader,
    )

    return jsonref_to_dict(deref)


def get_dereferenced_schema() -> dict:
    """Get the fully dereferenced schema (for use by other modules)."""
    schema_path = Path(__file__).parent / "schema.yaml"
    return load_and_deref_schema(schema_path)


if __name__ == "__main__":
    try:
        # Load the main schema and dereference all $refs
        deref = load_and_deref_schema(ROOT)

        # Serialize to YAML
        compiled = yaml.safe_dump(
            deref, sort_keys=False, indent=2, default_flow_style=False, width=1000
        )

        # Write to file
        output_file = Path("./schema.deref.yaml")
        output_file.write_text(compiled, encoding="utf-8")

    except Exception as e:
        print(f"Error during schema compilation: {e}")
        import traceback

        traceback.print_exc()
        raise
