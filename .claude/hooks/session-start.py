#!/usr/bin/env python3
"""
Session Start Hook - Restores context and outputs explicit directives.

This hook runs at session start/resume and outputs:
1. Saved context summary
2. PLAN.md current state (if exists)
3. Explicit next actions to take
4. Test status warnings

Output goes to stdout and is injected into Claude's context.
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
    """Get the context directory, handling symlinked installs."""
    project_dir = Path(get_project_dir())
    context_dir = project_dir / '.claude' / 'context'

    claude_dir = project_dir / '.claude'
    if claude_dir.is_symlink():
        alt_context = project_dir / '.claude-context'
        if alt_context.exists():
            return alt_context

    return context_dir


def parse_plan_file(project_dir):
    """Parse PLAN.md and extract current state."""
    plan_file = project_dir / 'PLAN.md'
    if not plan_file.exists():
        return None

    with open(plan_file) as f:
        content = f.read()

    # Count checklist items
    completed = len(re.findall(r'^\s*[-*]\s*\[[xX]\]', content, re.MULTILINE))
    pending_matches = list(re.finditer(r'^\s*[-*]\s*\[\s\]\s*(.+)$', content, re.MULTILINE))
    pending = len(pending_matches)

    # Get next items
    next_items = [m.group(1).strip() for m in pending_matches[:5]]

    # Find current phase
    current_phase = None
    phase_pattern = re.compile(r'^##\s*(?:Phase\s*\d+[:\s]*)?(.+)$', re.MULTILINE)
    phases = list(phase_pattern.finditer(content))

    if pending_matches and phases:
        first_pending_pos = pending_matches[0].start()
        for phase in phases:
            if phase.start() < first_pending_pos:
                current_phase = phase.group(1).strip()

    return {
        'total': completed + pending,
        'completed': completed,
        'pending': pending,
        'current_phase': current_phase,
        'next_items': next_items
    }


def load_json_file(filepath):
    """Safely load a JSON file."""
    if filepath.exists():
        try:
            with open(filepath) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def restore_context():
    """Restore context and output explicit directives."""
    project_dir = Path(get_project_dir())
    context_dir = get_context_dir()

    output_lines = []

    # Check for PLAN.md first (most important)
    plan_state = parse_plan_file(project_dir)

    # Load saved context
    saved_context = load_json_file(context_dir / 'saved-context.json')
    progress = load_json_file(context_dir / 'progress.json')
    test_results = load_json_file(context_dir / 'test-results.json')

    # Start output
    output_lines.append("startup hook success: === RESTORED SESSION CONTEXT ===")

    # If we have a saved markdown context, include key parts
    md_file = context_dir / 'session-context.md'
    if md_file.exists():
        try:
            with open(md_file) as f:
                output_lines.append(f.read())
        except IOError:
            pass

    # Always output fresh PLAN.md state (may have changed)
    if plan_state:
        if plan_state['pending'] > 0:
            output_lines.append("")
            output_lines.append("## ⚠️ ACTIVE PLAN DETECTED")
            output_lines.append(f"Progress: {plan_state['completed']}/{plan_state['total']} complete")
            output_lines.append(f"Phase: {plan_state['current_phase'] or 'Unknown'}")
            output_lines.append("")
            output_lines.append("NEXT UNCHECKED ITEMS:")
            for i, item in enumerate(plan_state['next_items'][:3], 1):
                output_lines.append(f"  {i}. {item}")
            output_lines.append("")
            output_lines.append("ACTION REQUIRED: Read PLAN.md and continue with next `- [ ]` item")

    # Check for failing tests
    if test_results.get('failed', 0) > 0:
        output_lines.append("")
        output_lines.append("## ⚠️ FAILING TESTS DETECTED")
        output_lines.append(f"Failed: {test_results['failed']}")
        failures = test_results.get('failures', [])
        for f in failures[:3]:
            output_lines.append(f"  - {f}")
        output_lines.append("")
        output_lines.append("ACTION REQUIRED: Fix failing tests before continuing")

    # Check for in-progress todos
    todo_list = progress.get('todo_list', [])
    in_progress = [t for t in todo_list if t.get('status') == 'in_progress']
    pending_todos = [t for t in todo_list if t.get('status') == 'pending']

    if in_progress:
        output_lines.append("")
        output_lines.append(f"## IN-PROGRESS TASK: {in_progress[0].get('content', 'Unknown')}")
        output_lines.append("ACTION REQUIRED: Complete this task before moving on")

    output_lines.append("")
    output_lines.append("=== END RESTORED CONTEXT ===")

    # Output everything
    print("\n".join(output_lines))
    return 0


def main():
    try:
        restore_context()
        sys.exit(0)
    except Exception as e:
        print(f"Note: Could not restore context: {e}", file=sys.stderr)
        # Still output something useful
        print("startup hook success: Session starting. Check PLAN.md if it exists.")
        sys.exit(0)


if __name__ == '__main__':
    main()
