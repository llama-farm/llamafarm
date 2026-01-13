#!/usr/bin/env python3
"""
Stop Check Hook - Deterministic check before Claude can stop.

This is a COMMAND hook that returns exit codes:
  0 - Allow stopping (all work complete)
  2 - Block stopping (work remains, output next action to stderr)

Checks performed (in order of priority):
1. PLAN.md for unchecked items `- [ ]`
2. progress.json for in_progress/pending todos
3. test-results.json for failing tests
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
    """Check PLAN.md for unchecked items. Returns (count, next_items)."""
    plan_file = project_dir / 'PLAN.md'
    if not plan_file.exists():
        return 0, []

    with open(plan_file) as f:
        content = f.read()

    # Find all unchecked items: - [ ] or * [ ]
    pending_matches = list(re.finditer(r'^\s*[-*]\s*\[\s\]\s*(.+)$', content, re.MULTILINE))

    if pending_matches:
        next_items = [m.group(1).strip() for m in pending_matches[:3]]
        return len(pending_matches), next_items

    return 0, []


def check_stop_signal():
    """Check if explicit user stop signal exists."""
    stop_file = Path(get_project_dir()) / '.claude' / 'context' / 'stop-signal'
    return stop_file.exists()


def clear_stop_signal():
    """Clear the stop signal file."""
    stop_file = Path(get_project_dir()) / '.claude' / 'context' / 'stop-signal'
    if stop_file.exists():
        stop_file.unlink(missing_ok=True)


def main():
    project_dir = Path(get_project_dir())
    context_dir = get_context_dir()

    # Read input from stdin (hook context)
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, IOError):
        input_data = {}

    # Check for explicit user stop signal
    if check_stop_signal():
        clear_stop_signal()
        print("User requested stop - allowing")
        sys.exit(0)

    # === CHECK 1: PLAN.md unchecked items (highest priority) ===
    pending_plan, next_plan_items = check_plan_file(project_dir)
    if pending_plan > 0:
        next_action = next_plan_items[0] if next_plan_items else "Check PLAN.md"
        # Output to BOTH stdout (for context injection) and stderr (for blocking)
        blocking_msg = f"""
STOP BLOCKED - {pending_plan} UNCHECKED ITEMS IN PLAN.md

YOU MUST CONTINUE WORKING. DO NOT STOP.

NEXT ACTION REQUIRED: {next_action}

Your response MUST be a tool call (Read, Edit, Bash, Write, etc.) to continue work.
DO NOT output text explaining what you'll do - just DO IT with a tool call.
"""
        print(blocking_msg)  # stdout - injected into context
        print(f"PLAN.md has {pending_plan} unchecked items. Continue with: {next_action}", file=sys.stderr)
        sys.exit(2)

    # === CHECK 2: Todo list status ===
    progress = load_json_file(context_dir / 'progress.json')
    todo_list = progress.get('todo_list', [])
    tasks = progress.get('tasks', {})

    in_progress = [t for t in todo_list if t.get('status') == 'in_progress']
    pending = [t for t in todo_list if t.get('status') == 'pending']

    if in_progress:
        task = in_progress[0].get('content', 'current task')
        blocking_msg = f"""
STOP BLOCKED - TASK IN PROGRESS

YOU MUST CONTINUE WORKING. DO NOT STOP.

CURRENT TASK: {task}

Your response MUST be a tool call to continue this task.
"""
        print(blocking_msg)
        print(f"Task in progress: {task}. Complete it before stopping.", file=sys.stderr)
        sys.exit(2)

    if pending:
        task = pending[0].get('content', 'next task')
        blocking_msg = f"""
STOP BLOCKED - {len(pending)} PENDING TASKS

YOU MUST CONTINUE WORKING. DO NOT STOP.

NEXT TASK: {task}

Your response MUST be a tool call to start this task.
"""
        print(blocking_msg)
        print(f"{len(pending)} pending tasks. Continue with: {task}", file=sys.stderr)
        sys.exit(2)

    # Also check tasks counts (in case todo_list wasn't synced)
    if tasks.get('in_progress', 0) > 0 or tasks.get('pending', 0) > 0:
        print(f"Tasks remaining: {tasks.get('pending', 0)} pending, {tasks.get('in_progress', 0)} in progress", file=sys.stderr)
        sys.exit(2)

    # === CHECK 3: Failing tests ===
    test_results = load_json_file(context_dir / 'test-results.json')
    failed = test_results.get('failed', 0)

    if failed > 0:
        failures = test_results.get('failures', [])
        first_failure = failures[0] if failures else "unknown test"
        print(f"{failed} tests failing. Fix: {first_failure}", file=sys.stderr)
        sys.exit(2)

    # === ALL CHECKS PASSED ===
    print("All plan items complete, todos done, tests passing - OK to stop")
    sys.exit(0)


if __name__ == '__main__':
    main()
