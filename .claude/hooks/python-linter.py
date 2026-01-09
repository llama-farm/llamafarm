#!/usr/bin/env python3
"""
Post-tool hook to run Ruff linter on Python files after edits.
Provides feedback to Claude about linting issues.
"""

import json
import os
import subprocess
import sys


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_input = input_data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    # Only process Python files
    if not file_path.endswith(".py"):
        sys.exit(0)

    # Skip if file doesn't exist
    if not os.path.exists(file_path):
        sys.exit(0)

    # Run ruff check
    try:
        result = subprocess.run(
            ["uv", "run", "ruff", "check", file_path, "--output-format=concise"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(file_path) or ".",
        )

        if result.returncode != 0 and result.stdout.strip():
            # Report issues but don't block
            print(f"Linting issues found:\n{result.stdout}", file=sys.stderr)
            # Exit 0 to not block, just provide feedback
            sys.exit(0)

    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Silently fail if ruff not available
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
