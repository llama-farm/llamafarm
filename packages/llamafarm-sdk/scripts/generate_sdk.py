#!/usr/bin/env python3
"""Auto-generate LlamaFarm SDK types and client methods from OpenAPI spec.

Usage:
    # Generate from running server
    python scripts/generate_sdk.py

    # Generate from saved spec file
    python scripts/generate_sdk.py --spec openapi.json

    # Check if generated code is up to date (CI mode)
    python scripts/generate_sdk.py --check

    # Save spec for offline generation
    python scripts/generate_sdk.py --save-spec openapi.json

The generator:
1. Fetches OpenAPI JSON from the main LlamaFarm server (:14345)
2. Generates Pydantic models from all component schemas → _generated_types.py
3. Generates typed method stubs for all endpoints → _generated_methods.py
4. The hand-written client.py / types.py import from generated code

When new endpoints or types are added to the server, just re-run this script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Where to write generated files
SDK_SRC = Path(__file__).resolve().parent.parent / "src" / "llamafarm"

# OpenAPI type → Python type mapping
TYPE_MAP: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict[str, Any]",
}

# Schemas to skip (internal FastAPI / multipart bodies)
SKIP_SCHEMA_PREFIXES = ("Body_", "HTTPValidationError", "ValidationError")

# Endpoint prefixes that go to runtime (not main server)
RUNTIME_PREFIXES = ("/v1/finetune", "/v1/cache")


def fetch_spec(url: str = "http://localhost:14345") -> dict:
    """Fetch OpenAPI spec from a running server."""
    import httpx

    r = httpx.get(f"{url}/openapi.json", timeout=10)
    r.raise_for_status()
    return r.json()


def load_spec(path: str) -> dict:
    return json.loads(Path(path).read_text())


# =============================================================================
# Type Generation
# =============================================================================


def _python_type(schema: dict, schemas: dict, depth: int = 0) -> str:
    """Convert an OpenAPI schema to a Python type annotation."""
    if depth > 5:
        return "Any"

    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        if ref_name in SKIP_SCHEMA_PREFIXES:
            return "Any"
        return _sanitize_class_name(ref_name)

    if "allOf" in schema:
        # Take the first $ref if present
        for sub in schema["allOf"]:
            if "$ref" in sub:
                return _python_type(sub, schemas, depth + 1)
        return "Any"

    if "anyOf" in schema or "oneOf" in schema:
        variants = schema.get("anyOf") or schema.get("oneOf", [])
        types = []
        for v in variants:
            if v.get("type") == "null":
                continue
            types.append(_python_type(v, schemas, depth + 1))
        if not types:
            return "Any"
        if len(types) == 1:
            t = types[0]
            # Check if null is one of the variants
            has_null = any(v.get("type") == "null" for v in variants)
            return f"{t} | None" if has_null else t
        return " | ".join(types)

    t = schema.get("type", "object")
    fmt = schema.get("format", "")

    if t == "array":
        items = schema.get("items", {})
        item_type = _python_type(items, schemas, depth + 1)
        return f"list[{item_type}]"

    if t == "object":
        addl = schema.get("additionalProperties")
        if addl and isinstance(addl, dict):
            val_type = _python_type(addl, schemas, depth + 1)
            return f"dict[str, {val_type}]"
        return "dict[str, Any]"

    if t == "string" and fmt == "date-time":
        return "str"  # Keep as string, don't require datetime import
    if t == "string" and "enum" in schema:
        vals = schema["enum"]
        if len(vals) <= 8:
            return "Literal[" + ", ".join(repr(v) for v in vals) + "]"
        return "str"

    return TYPE_MAP.get(t, "Any")


def _sanitize_class_name(name: str) -> str:
    """Ensure valid Python class name."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if name[0].isdigit():
        name = f"_{name}"
    return name


_RESERVED = {
    "type",
    "class",
    "from",
    "import",
    "return",
    "global",
    "in",
    "is",
    "not",
    "and",
    "or",
    "schema",
    "model_fields",
    "model_config",
    "model_computed_fields",
    "model_extra",
    "model_fields_set",
}


def _sanitize_field_name(name: str) -> str:
    """Ensure valid Python field name that doesn't shadow BaseModel attrs."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if clean in _RESERVED:
        clean = f"{clean}_"
    return clean


def generate_types(spec: dict) -> str:
    """Generate Pydantic models from OpenAPI component schemas."""
    schemas = spec.get("components", {}).get("schemas", {})
    lines: list[str] = [
        '"""Auto-generated LlamaFarm SDK types from OpenAPI spec.',
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Server: {spec.get('info', {}).get('title', 'unknown')}",
        f"Schemas: {len(schemas)}",
        "",
        "DO NOT EDIT — regenerate with: python scripts/generate_sdk.py",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any, Literal",
        "",
        "from pydantic import BaseModel, Field",
        "",
        "",
    ]

    # Sort schemas for stable output, skip internal ones
    generated_count = 0
    for name in sorted(schemas.keys()):
        if any(name.startswith(p) for p in SKIP_SCHEMA_PREFIXES):
            continue

        schema = schemas[name]
        cls_name = _sanitize_class_name(name)
        desc = schema.get("description", "")
        _title = schema.get("title", name)  # noqa: F841 — reserved for future docstring use

        # Only generate for object schemas with properties
        props = schema.get("properties", {})
        if not props and schema.get("type") != "object":
            # Could be an enum or simple alias
            if "enum" in schema:
                vals = schema["enum"]
                lines.append(f"# Enum: {name}")
                lines.append(
                    f"{cls_name} = Literal[{', '.join(repr(v) for v in vals)}]"
                )
                lines.append("")
                generated_count += 1
                continue
            continue

        required = set(schema.get("required", []))

        # Class definition
        if desc:
            safe_doc = desc.replace("\n", " ").replace('"', '\\"').strip()
            safe_doc = " ".join(safe_doc.split())
            lines.append(f"class {cls_name}(BaseModel):")
            lines.append(f'    """{safe_doc}"""')
        else:
            lines.append(f"class {cls_name}(BaseModel):")

        if not props:
            lines.append("    pass")
            lines.append("")
            lines.append("")
            generated_count += 1
            continue

        for prop_name, prop_schema in props.items():
            field_name = _sanitize_field_name(prop_name)
            py_type = _python_type(prop_schema, schemas)
            prop_desc = prop_schema.get("description", "")
            default = prop_schema.get("default")
            is_required = prop_name in required

            # Build Field() args
            field_args: list[str] = []
            if is_required and default is None:
                field_args.append("...")
            elif default is not None:
                field_args.append(repr(default))
            else:
                # Optional field
                if not py_type.endswith("| None"):
                    py_type = f"{py_type} | None"
                field_args.append("None")

            if prop_desc:
                # Sanitize: single line, escape quotes
                safe_desc = (
                    prop_desc.replace("\n", " ")
                    .replace("\r", "")
                    .replace('"', '\\"')
                    .strip()
                )
                safe_desc = " ".join(safe_desc.split())  # collapse whitespace
                field_args.append(f'description="{safe_desc}"')

            # Alias if field name differs
            if field_name != prop_name:
                field_args.append(f'alias="{prop_name}"')

            field_str = ", ".join(field_args)
            lines.append(f"    {field_name}: {py_type} = Field({field_str})")

        lines.append("")
        lines.append("")
        generated_count += 1

    # Add summary comment at end
    lines.append(f"# Total generated types: {generated_count}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Method Generation
# =============================================================================


def _path_to_method_name(path: str, method: str) -> str:
    """Convert an API path to a Python method name."""
    # Strip /v1/ prefix
    p = re.sub(r"^/v1/", "", path)
    # Replace path params
    p = re.sub(r"\{([^}]+)\}", r"by_\1", p)
    # Replace slashes with underscores
    p = p.replace("/", "_").replace("-", "_")
    # Prefix non-GET methods to disambiguate (e.g. GET /buffers vs POST /buffers)
    if method != "get":
        p = f"{method}_{p}"
    return p.lower()


def _group_from_path(path: str) -> str:
    """Determine which sub-API group a path belongs to."""
    parts = path.strip("/").split("/")
    if len(parts) < 2:
        return "core"
    # /v1/ml/anomaly/... → anomaly
    # /v1/vision/... → vision
    # /v1/nlp/... → nlp
    # /v1/projects/... → projects
    prefix = parts[1] if parts[0] == "v1" else parts[0]
    if prefix == "ml" and len(parts) > 2:
        return parts[2]  # anomaly, classifier, polars, etc.
    return prefix


def _extract_params(op: dict, schemas: dict) -> list[dict]:
    """Extract path and query params from an operation."""
    params = []
    for p in op.get("parameters", []):
        params.append(
            {
                "name": _sanitize_field_name(p["name"]),
                "original_name": p["name"],
                "type": _python_type(p.get("schema", {}), schemas),
                "required": p.get("required", False),
                "in": p.get("in", "query"),
                "description": p.get("description", ""),
            }
        )
    return params


def _extract_request_body_type(op: dict, schemas: dict) -> str | None:
    """Get the request body type name if any."""
    rb = op.get("requestBody", {})
    content = rb.get("content", {})
    json_ct = content.get("application/json", {})
    schema = json_ct.get("schema", {})
    if "$ref" in schema:
        return schema["$ref"].split("/")[-1]
    if "multipart/form-data" in content:
        return None  # File upload — handle differently
    return None


def _extract_response_type(op: dict, schemas: dict) -> str:
    """Get the response type name."""
    for code in ("200", "201"):
        resp = op.get("responses", {}).get(code, {})
        content = resp.get("content", {})
        json_ct = content.get("application/json", {})
        schema = json_ct.get("schema", {})
        if "$ref" in schema:
            return _sanitize_class_name(schema["$ref"].split("/")[-1])
        if schema.get("type") == "object":
            return "dict[str, Any]"
        if schema.get("type") == "array":
            return "list[Any]"
    return "dict[str, Any]"


def generate_methods(spec: dict) -> str:
    """Generate typed method stubs grouped by sub-API."""
    schemas = spec.get("components", {}).get("schemas", {})
    paths = spec.get("paths", {})

    groups: dict[str, list[str]] = {}

    for path, path_item in sorted(paths.items()):
        for method in ("get", "post", "put", "patch", "delete"):
            if method not in path_item:
                continue
            op = path_item[method]
            group = _group_from_path(path)
            method_name = _path_to_method_name(path, method)
            summary = op.get("summary", "")
            desc = op.get("description", "")
            _tags = op.get("tags", [])  # noqa: F841 — reserved for future routing

            params = _extract_params(op, schemas)
            body_type = _extract_request_body_type(op, schemas)
            response_type = _extract_response_type(op, schemas)
            is_runtime = any(path.startswith(p) for p in RUNTIME_PREFIXES)

            # Build method signature
            sig_parts = ["self"]
            for p in params:
                if p["in"] == "path" and p["required"]:
                    sig_parts.append(f"{p['name']}: {p['type']}")
            if body_type:
                safe_body = _sanitize_class_name(body_type)
                sig_parts.append(f"request: {safe_body}")
            for p in params:
                if p["in"] == "query":
                    if p["required"]:
                        sig_parts.append(f"{p['name']}: {p['type']}")
                    else:
                        sig_parts.append(f"{p['name']}: {p['type']} | None = None")

            sig = ", ".join(sig_parts)

            # Method body
            lines: list[str] = []
            doc = summary or desc or f"{method.upper()} {path}"
            lines.append(f"    def {method_name}({sig}) -> {response_type}:")
            lines.append(f'        """{doc}"""')
            lines.append(f"        # {method.upper()} {path}")
            if is_runtime:
                lines.append(
                    "        # NOTE: runtime-direct (not proxied through main)"
                )
            lines.append("        ...")
            lines.append("")

            if group not in groups:
                groups[group] = []
            groups[group].append("\n".join(lines))

    # Build output
    out_lines = [
        '"""Auto-generated LlamaFarm SDK method stubs from OpenAPI spec.',
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Endpoints: {sum(len(v) for v in groups.values())}",
        "",
        "This file is a REFERENCE — not imported directly.",
        "Use it to verify client.py coverage and update signatures.",
        '"""',
        "",
        "from __future__ import annotations",
        "from typing import Any",
        "from ._generated_types import *  # noqa: F403",
        "",
    ]

    for group_name in sorted(groups.keys()):
        methods = groups[group_name]
        out_lines.append(f"# {'=' * 70}")
        out_lines.append(f"# {group_name.upper()} ({len(methods)} endpoints)")
        out_lines.append(f"# {'=' * 70}")
        out_lines.append("")
        out_lines.append(f"class {_sanitize_class_name(group_name).title()}API:")
        for m in methods:
            out_lines.append(m)
        out_lines.append("")

    return "\n".join(out_lines)


# =============================================================================
# Coverage Check
# =============================================================================


def check_coverage(spec: dict, client_path: Path) -> list[str]:
    """Check which server endpoints are missing from the SDK client."""
    if not client_path.exists():
        return ["client.py not found"]

    client_src = client_path.read_text()
    paths = spec.get("paths", {})
    missing = []

    for path in sorted(paths.keys()):
        if path in ("/health", "/health/liveness", "/info"):
            continue
        # Check if path fragment appears in client.py
        # Normalize path — remove {params}
        check_path = re.sub(r"\{[^}]+\}", "", path).rstrip("/")
        if check_path not in client_src and path not in client_src:
            methods = list(paths[path].keys())
            missing.append(f"  {methods[0].upper():6s} {path}")

    return missing


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate LlamaFarm SDK from OpenAPI spec"
    )
    parser.add_argument(
        "--spec", help="Path to OpenAPI JSON file (instead of fetching)"
    )
    parser.add_argument("--url", default="http://localhost:14345", help="Server URL")
    parser.add_argument("--save-spec", help="Save fetched spec to file")
    parser.add_argument(
        "--check", action="store_true", help="Check if generated code is up to date"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Check SDK endpoint coverage"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print to stdout instead of writing files",
    )
    args = parser.parse_args()

    # Load spec
    if args.spec:
        spec = load_spec(args.spec)
        print(f"Loaded spec from {args.spec}")
    else:
        try:
            spec = fetch_spec(args.url)
            print(
                f"Fetched spec from {args.url}: {len(spec.get('paths', {}))} paths, "
                f"{len(spec.get('components', {}).get('schemas', {}))} schemas"
            )
        except Exception as e:
            print(f"Error: Cannot fetch spec from {args.url}: {e}", file=sys.stderr)
            print("Start the server or use --spec <file>", file=sys.stderr)
            sys.exit(1)

    if args.save_spec:
        Path(args.save_spec).write_text(json.dumps(spec, indent=2))
        print(f"Saved spec to {args.save_spec}")

    # Coverage check
    if args.coverage:
        print("\n── SDK Coverage Check ──")
        missing = check_coverage(spec, SDK_SRC / "client.py")
        if missing:
            print(f"Missing {len(missing)} endpoints:")
            for m in missing:
                print(m)
        else:
            print("✅ All endpoints covered!")
        return

    # Generate types
    types_code = generate_types(spec)
    methods_code = generate_methods(spec)

    if args.dry_run:
        print("── _generated_types.py ──")
        print(types_code[:3000])
        print(f"\n... ({len(types_code)} chars total)\n")
        print("── _generated_methods.py ──")
        print(methods_code[:3000])
        print(f"\n... ({len(methods_code)} chars total)")
        return

    # Check mode
    types_path = SDK_SRC / "_generated_types.py"
    methods_path = SDK_SRC / "_generated_methods.py"

    if args.check:
        stale = False
        if not types_path.exists():
            print(
                "❌ _generated_types.py missing — run: python scripts/generate_sdk.py"
            )
            stale = True
        elif types_path.read_text() != types_code:
            print(
                "❌ _generated_types.py is stale — run: python scripts/generate_sdk.py"
            )
            stale = True
        if not methods_path.exists():
            print(
                "❌ _generated_methods.py missing — run: python scripts/generate_sdk.py"
            )
            stale = True
        elif methods_path.read_text() != methods_code:
            print(
                "❌ _generated_methods.py is stale — run: python scripts/generate_sdk.py"
            )
            stale = True
        if stale:
            sys.exit(1)
        print("✅ Generated SDK code is up to date")
        return

    # Write files
    types_path.write_text(types_code)
    methods_path.write_text(methods_code)
    print(f"✅ Generated {types_path.name} ({len(types_code)} chars)")
    print(f"✅ Generated {methods_path.name} ({len(methods_code)} chars)")

    # Run coverage check
    print("\n── Coverage Check ──")
    missing = check_coverage(spec, SDK_SRC / "client.py")
    if missing:
        print(f"⚠️  {len(missing)} endpoints not in client.py:")
        for m in missing[:20]:
            print(m)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    else:
        print("✅ All endpoints covered!")


if __name__ == "__main__":
    main()
