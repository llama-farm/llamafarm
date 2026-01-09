#!/usr/bin/env python3
"""
Pre-tool hook to warn about editing sensitive files.
Blocks edits to certain protected files without explicit confirmation.
"""
import json
import sys
import os

# Files that should never be edited by Claude
BLOCKED_FILES = [
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.yaml",
    "secrets.json",
]

# Files that should trigger a warning
WARN_PATTERNS = [
    "package-lock.json",
    "yarn.lock",
    "uv.lock",
    "go.sum",
    ".git/",
]

def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_input = input_data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not file_path:
        sys.exit(0)

    basename = os.path.basename(file_path)

    # Block sensitive files
    if basename in BLOCKED_FILES:
        print(f"BLOCKED: Cannot edit sensitive file '{basename}'. "
              f"This file may contain secrets or credentials.", file=sys.stderr)
        sys.exit(2)  # Exit code 2 blocks the tool call

    # Warn about lock files and git internals
    for pattern in WARN_PATTERNS:
        if pattern in file_path:
            print(f"WARNING: Editing '{basename}' - this is a generated/lock file. "
                  f"Changes may cause dependency issues.", file=sys.stderr)
            # Don't block, just warn
            sys.exit(0)

    sys.exit(0)

if __name__ == "__main__":
    main()
