#!/usr/bin/env python3
import json
import os
from glob import glob

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    examples_root = os.path.join(repo_root, "examples")

    manifests: list[dict] = []

    # Find all example-level manifest.yaml files in deterministic order
    for manifest_path in sorted(glob(os.path.join(examples_root, "*", "manifest.yaml"))):
        try:
            manifest = load_yaml(manifest_path) or {}
            # Enrich minimal listing data for UI
            manifests.append(
                {
                    "id": manifest.get("id"),
                    "slug": manifest.get("slug"),
                    "title": manifest.get("title"),
                    "description": manifest.get("description"),
                    "tags": manifest.get("tags", []),
                    "primaryModel": manifest.get("primaryModel"),
                    # Keep the rest for server-side usage
                    "_raw": manifest,
                }
            )
        except Exception as e:
            print(f"Skipping {manifest_path}: {e}")

    # Write consolidated index for server/UI
    out_path = os.path.join(examples_root, "manifest.index.json")
    with open(out_path, "w") as f:
        json.dump({"examples": manifests}, f, indent=2)
        f.write("\n")

    print(f"Wrote {out_path} with {len(manifests)} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


