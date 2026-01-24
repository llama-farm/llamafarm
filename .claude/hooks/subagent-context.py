#!/usr/bin/env python3
"""
Subagent Context Hook - Outputs plan/progress context when subagents complete.

This hook runs when a Task subagent finishes. It outputs context about:
1. The main PLAN.md status
2. Overall progress
3. What the parent should focus on next

This helps the parent Claude stay focused on the plan after subagent work.
"""

import json
import sys
import os
import re
from pathlib import Path


def get_project_dir():
    """Get the project directory."""
    return os.environ.get('CLAUDE_PROJECT_DIR', os.getcwd())


def get_context_dir():
    """Get the context directory."""
    project_dir = Path(get_project_dir())
    context_dir = project_dir / '.claude' / 'context'

    claude_dir = project_dir / '.claude'
    if claude_dir.is_symlink():
        alt_context = project_dir / '.claude-context'
        if alt_context.exists():
            return alt_context

    return context_dir


def load_json_file(filepath):
    """Safely load a JSON file."""
    if filepath.exists():
        try:
            with open(filepath) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def check_plan_file(project_dir):
    """Check PLAN.md status."""
    plan_file = project_dir / 'PLAN.md'
    if not plan_file.exists():
        return None

    with open(plan_file) as f:
        content = f.read()

    completed = len(re.findall(r'^\s*[-*]\s*\[[xX]\]', content, re.MULTILINE))
    pending_matches = list(re.finditer(r'^\s*[-*]\s*\[\s\]\s*(.+)$', content, re.MULTILINE))
    pending = len(pending_matches)

    next_items = [m.group(1).strip() for m in pending_matches[:3]]

    return {
        'total': completed + pending,
        'completed': completed,
        'pending': pending,
        'next_items': next_items
    }


def main():
    project_dir = Path(get_project_dir())
    context_dir = get_context_dir()

    # Read subagent info from stdin
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, IOError):
        input_data = {}

    output_lines = []

    # Check PLAN.md status
    plan_state = check_plan_file(project_dir)

    if plan_state and plan_state['pending'] > 0:
        output_lines.append(f"PLAN STATUS: {plan_state['completed']}/{plan_state['total']} complete")
        output_lines.append("NEXT PLAN ITEMS:")
        for item in plan_state['next_items'][:2]:
            output_lines.append(f"  - {item}")
        output_lines.append("")
        output_lines.append("REMINDER: After this subagent completes, continue with PLAN.md")

    # Check for failing tests
    test_results = load_json_file(context_dir / 'test-results.json')
    if test_results.get('failed', 0) > 0:
        output_lines.append(f"WARNING: {test_results['failed']} tests failing - address before continuing")

    # Output context (goes to stdout, visible to parent)
    if output_lines:
        print("--- Subagent Context ---")
        print("\n".join(output_lines))
        print("------------------------")

    sys.exit(0)


if __name__ == '__main__':
    main()
