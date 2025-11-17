#!/usr/bin/env python3
"""
Schema compilation script using jsonref with proper conversion.
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import jsonref  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).parent / "schema.yaml"


def load_text_from_uri(uri: str) -> str:
    """Read local file:// or plain path URIs into text (UTF-8)."""
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        # Use url2pathname to properly convert file URIs to filesystem paths
        # This handles Windows paths correctly (e.g., file:///C:/Users/... -> C:\Users\...)
        path = Path(url2pathname(parsed.path)) if parsed.scheme == "file" else Path(uri)
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported URI scheme in $ref: {uri}")


def yaml_json_loader(uri: str):
    """YAML-aware loader for jsonref: parse .yaml/.yml as YAML, else JSON."""
    text = load_text_from_uri(uri)
    if uri.endswith((".yaml", ".yml")):
        return yaml.safe_load(text)
    return json.loads(text)


def jsonref_to_dict(obj, is_root=False):
    """Recursively convert jsonref proxy objects to plain Python dicts/lists."""
    if isinstance(obj, dict):
        # Check if this is a jsonref proxy object
        if hasattr(obj, "__subject__"):
            # It's a jsonref proxy, get the underlying dict
            obj = dict(obj)

        # Strip schema metadata fields when nested (not at root level)
        # These fields should only exist at the root of the schema
        if not is_root:
            schema_metadata_fields = {"$schema", "$id"}
            obj = {k: v for k, v in obj.items() if k not in schema_metadata_fields}

        return {k: jsonref_to_dict(v, is_root=False) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [jsonref_to_dict(item, is_root=False) for item in obj]
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

    return jsonref_to_dict(deref, is_root=True)


def get_dereferenced_schema() -> dict:
    """Get the fully dereferenced schema (for use by other modules)."""
    derefed_schema = Path(__file__).parent / "schema.deref.yaml"
    if derefed_schema.exists():
        return load_and_deref_schema(derefed_schema)

    schema_path = Path(__file__).parent / "schema.yaml"
    return load_and_deref_schema(schema_path)


if __name__ == "__main__":
    try:
        # Load the main schema and dereference all $refs
        deref = load_and_deref_schema(ROOT)

        # Validate that dereferencing produced a valid schema
        if deref is None:
            raise ValueError(
                "Schema dereferencing produced None. This usually means the schema file "
                "is empty or the $ref resolution failed."
            )

        if not isinstance(deref, dict):
            raise ValueError(
                f"Schema dereferencing produced invalid type: {type(deref).__name__}. "
                "Expected a dictionary."
            )

        if not deref:
            raise ValueError(
                "Schema dereferencing produced an empty dictionary. "
                "This usually means the schema file is empty or all $ref resolutions failed."
            )

        # Ensure the schema has required top-level fields
        if "type" not in deref and "properties" not in deref:
            raise ValueError(
                f"Schema is missing required top-level fields. "
                f"A valid JSON Schema must have 'type' and/or 'properties'. "
                f"Found keys: {list(deref.keys())}"
            )

        # Serialize to YAML
        compiled = yaml.safe_dump(
            deref, sort_keys=False, indent=2, default_flow_style=False, width=1000
        )

        # Validate serialized output is not empty
        if not compiled or compiled.strip() in ("null", "null\n...", "{}"):
            raise ValueError(
                f"YAML serialization produced invalid output: {repr(compiled[:100])}"
            )

        # Write to file
        output_file = Path("./schema.deref.yaml")
        output_file.write_text(compiled, encoding="utf-8")

        # Verify the file was written correctly
        if not output_file.exists():
            raise OSError(f"Failed to write schema file: {output_file}")

        file_size = output_file.stat().st_size
        if file_size < 100:  # A valid schema should be at least 100 bytes
            raise ValueError(
                f"Schema file is suspiciously small ({file_size} bytes). "
                f"This indicates the schema may not have been dereferenced correctly."
            )

        print(f"Schema compiled to {output_file}")

        # Copy the dereferenced schema to cli/cmd/config directory
        dest_dir = Path(__file__).parent.parent / "cli" / "cmd" / "config"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "schema.json"

        # Add $id to the schema for go-jsonschema if not present
        if "$id" not in deref:
            deref["$id"] = "https://llamafarm.dev/schema.json"

        # Write the schema with $id to CLI config directory
        # Write schema as JSON to CLI config directory (schema.json)
        with dest_file.open("w", encoding="utf-8") as json_out:
            json.dump(deref, json_out, indent=2, ensure_ascii=False)
        print(f"Schema also written in JSON format to {dest_file}")

    except Exception as e:
        print(f"Error during schema compilation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
