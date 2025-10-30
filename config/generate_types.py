#!/usr/bin/env python3
"""
Cross-platform script to generate config datamodel types.

This replaces generate-types.sh for better cross-platform support.
It must be run after source code is downloaded/updated but before
starting any services that depend on the datamodel.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path) -> None:
    """Run a command and handle errors."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}", file=sys.stderr)
        if e.stdout:
            print(e.stdout, end="", file=sys.stderr)
        if e.stderr:
            print(e.stderr, end="", file=sys.stderr)
        raise


def main() -> int:
    """Generate config datamodel types."""
    # Get the config directory (where this script lives)
    config_dir = Path(__file__).parent.resolve()

    print("Compiling schema...")
    run_command(["uv", "run", "python", "compile_schema.py"], cwd=config_dir)

    print("Generating types...")
    run_command(
        [
            "uv",
            "run",
            "datamodel-codegen",
            "--input",
            "schema.deref.yaml",
            "--output",
            "datamodel.py",
            "--input-file-type=jsonschema",
            "--output-model-type=pydantic_v2.BaseModel",
            "--target-python-version=3.12",
            "--use-standard-collections",
            "--formatters=ruff-format",
            "--class-name=LlamaFarmConfig",
        ],
        cwd=config_dir,
    )

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
