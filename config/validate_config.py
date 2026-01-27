#!/usr/bin/env python3
"""
CLI wrapper for LlamaFarm config validation.

Reuses existing validation infrastructure from helpers/loader.py and validators.py.
This is a thin wrapper - all validation logic lives in the existing modules.

Usage (run from config/ directory or project root):
    cd config && uv run python validate_config.py <config_path>
    cd config && uv run python validate_config.py <config_path> --verbose

    # Or from project root:
    uv run python config/validate_config.py config/tests/minimal_config.yaml

Exit codes:
    0 - Valid configuration
    1 - Invalid configuration
    2 - File not found or other error
"""

import argparse
import sys
from pathlib import Path

# Ensure we can import from the config package
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from helpers.loader import ConfigError, load_config_dict  # noqa: E402
from validators import validate_llamafarm_config  # noqa: E402


def main() -> int:
    """
    Validate a LlamaFarm configuration file.

    Returns:
        Exit code: 0 (valid), 1 (invalid), 2 (error)
    """
    parser = argparse.ArgumentParser(
        description="Validate a LlamaFarm configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python config/validate_config.py llamafarm.yaml
    uv run python config/validate_config.py /path/to/llamafarm.yaml --verbose
    uv run python config/validate_config.py . --verbose  # Find config in directory
        """,
    )
    parser.add_argument(
        "config_path",
        help="Path to config file or directory containing llamafarm.yaml",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed config info"
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)

    # Check if path exists
    if not config_path.exists():
        print(f"Error: Path not found: {config_path}")
        return 2

    try:
        # Step 1: Load and validate against JSON Schema
        # load_config_dict handles file format detection and schema validation
        config_dict = load_config_dict(config_path, validate=True)

        # Step 2: Run custom validators (uniqueness, references, naming patterns)
        validate_llamafarm_config(config_dict)

        # Success
        print(f"✓ Valid: {config_path}")

        if args.verbose:
            print(f"  Version: {config_dict.get('version', 'unknown')}")
            print(f"  Name: {config_dict.get('name', 'unnamed')}")
            print(f"  Namespace: {config_dict.get('namespace', 'default')}")

            # Show section counts
            prompts = config_dict.get("prompts", [])
            if prompts:
                print(f"  Prompts: {len(prompts)} defined")

            runtime = config_dict.get("runtime", {})
            if runtime:
                models = runtime.get("models", [])
                print(f"  Models: {len(models)} configured")

            datasets = config_dict.get("datasets", [])
            if datasets:
                print(f"  Datasets: {len(datasets)} defined")

            rag = config_dict.get("rag", {})
            if rag:
                databases = rag.get("databases", [])
                strategies = rag.get("data_processing_strategies", [])
                print(
                    f"  RAG: {len(databases)} databases, {len(strategies)} strategies"
                )

        return 0

    except ConfigError as e:
        print(f"✗ Invalid: {config_path}")
        print(f"  Schema error: {e}")
        return 1

    except ValueError as e:
        print(f"✗ Invalid: {config_path}")
        print(f"  Validation error: {e}")
        return 1

    except Exception as e:
        print(f"Error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
