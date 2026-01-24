#!/usr/bin/env python3
"""
Pre-Compact Save Hook - Saves comprehensive context before compaction.

This hook captures:
1. Current PLAN.md state (completed/pending items)
2. Progress tracking (todos, tests)
3. Next actions to take after resume
4. Session metadata

The output is designed to be actionable when restored.
"""

import json
import sys
import os
import re
from pathlib import Path
from datetime import datetime


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


def load_json_file(filepath):
    """Safely load a JSON file."""
    if filepath.exists():
        try:
            with open(filepath) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def parse_plan_file(project_dir):
    """Parse PLAN.md and extract completed/pending items."""
    plan_file = project_dir / 'PLAN.md'
    if not plan_file.exists():
        return None

    with open(plan_file) as f:
        content = f.read()

    # Extract all checklist items
    completed = []
    pending = []
    current_phase = None

    phase_pattern = re.compile(r'^##\s*(?:Phase\s*\d+[:\s]*)?(.+)$', re.MULTILINE)
    item_pattern = re.compile(r'^(\s*)[-*]\s*\[([xX\s])\]\s*(.+)$', re.MULTILINE)

    # Find current phase (first phase with incomplete items)
    phases = list(phase_pattern.finditer(content))

    for match in item_pattern.finditer(content):
        indent = len(match.group(1))
        is_complete = match.group(2).lower() == 'x'
        description = match.group(3).strip()

        # Find which phase this item belongs to
        item_pos = match.start()
        item_phase = None
        for phase in phases:
            if phase.start() < item_pos:
                item_phase = phase.group(1).strip()

        item = {
            'description': description,
            'phase': item_phase,
            'indent': indent
        }

        if is_complete:
            completed.append(item)
        else:
            pending.append(item)
            if current_phase is None and item_phase:
                current_phase = item_phase

    return {
        'exists': True,
        'total_items': len(completed) + len(pending),
        'completed_count': len(completed),
        'pending_count': len(pending),
        'current_phase': current_phase,
        'next_items': [p['description'] for p in pending[:5]],  # Next 5 items
        'completed_items': [c['description'] for c in completed[-3:]]  # Last 3 completed
    }


def save_context():
    """Save comprehensive context before compaction."""
    project_dir = Path(get_project_dir())
    context_dir = get_context_dir()
    context_dir.mkdir(parents=True, exist_ok=True)

    # Read input from stdin
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, IOError):
        input_data = {}

    # Load existing progress
    progress = load_json_file(context_dir / 'progress.json')
    test_results = load_json_file(context_dir / 'test-results.json')

    # Parse PLAN.md
    plan_state = parse_plan_file(project_dir)

    # Build comprehensive context
    context = {
        'saved_at': datetime.utcnow().isoformat() + 'Z',
        'session_id': input_data.get('session_id', progress.get('session_id', 'unknown')),
        'plan': plan_state,
        'progress': progress,
        'test_results': test_results,
        'transcript_path': input_data.get('transcript_path', ''),
    }

    # Save JSON context
    context_file = context_dir / 'saved-context.json'
    with open(context_file, 'w') as f:
        json.dump(context, f, indent=2)

    # Generate actionable markdown summary
    markdown = generate_actionable_summary(context, plan_state, progress, test_results)

    md_file = context_dir / 'session-context.md'
    with open(md_file, 'w') as f:
        f.write(markdown)

    # Also update progress.json with plan state
    if plan_state:
        progress['plan'] = {
            'total': plan_state['total_items'],
            'completed': plan_state['completed_count'],
            'pending': plan_state['pending_count'],
            'current_phase': plan_state['current_phase']
        }
        with open(context_dir / 'progress.json', 'w') as f:
            json.dump(progress, f, indent=2)

    print(f"pre-compact: saved context with {plan_state['pending_count'] if plan_state else 0} pending plan items")
    return 0


def generate_actionable_summary(context, plan_state, progress, test_results):
    """Generate actionable context summary for session restore."""
    lines = [
        "# Session Context (Auto-saved before compact)",
        "",
        f"**Saved at:** {context.get('saved_at', 'unknown')}",
        "",
    ]

    # PLAN STATUS - Most important
    if plan_state and plan_state.get('exists'):
        lines.append("## 🎯 PLAN STATUS")
        lines.append("")
        lines.append(f"**Progress:** {plan_state['completed_count']}/{plan_state['total_items']} items complete")
        lines.append(f"**Current Phase:** {plan_state.get('current_phase', 'Unknown')}")
        lines.append("")

        if plan_state['pending_count'] > 0:
            lines.append("### ⚠️ IMMEDIATE NEXT ACTIONS (from PLAN.md):")
            lines.append("")
            for i, item in enumerate(plan_state['next_items'][:3], 1):
                lines.append(f"{i}. `{item}`")
            lines.append("")
            lines.append("**DIRECTIVE:** Read PLAN.md and continue with the next unchecked item.")
        else:
            lines.append("### ✅ All plan items completed!")
        lines.append("")

    # TODO STATUS
    tasks = progress.get('tasks', {})
    todo_list = progress.get('todo_list', [])
    if tasks.get('total', 0) > 0:
        lines.append("## 📋 TODO STATUS")
        lines.append("")
        lines.append(f"**Tasks:** {tasks.get('completed', 0)}/{tasks.get('total', 0)} complete")

        pending_todos = [t for t in todo_list if t.get('status') == 'pending']
        in_progress = [t for t in todo_list if t.get('status') == 'in_progress']

        if in_progress:
            lines.append(f"**In Progress:** {in_progress[0].get('content', 'Unknown')}")
        if pending_todos:
            lines.append(f"**Next Pending:** {pending_todos[0].get('content', 'Unknown')}")
        lines.append("")

    # TEST STATUS
    if test_results:
        lines.append("## 🧪 TEST STATUS")
        lines.append("")
        passed = test_results.get('passed', 0)
        failed = test_results.get('failed', 0)
        if failed > 0:
            lines.append(f"**⚠️ FAILING:** {failed} tests failed - FIX THESE FIRST")
            failures = test_results.get('failures', [])
            for f in failures[:3]:
                lines.append(f"  - {f}")
        else:
            lines.append(f"**Passed:** {passed} tests")
        lines.append("")

    # EXPLICIT DIRECTIVES
    lines.append("---")
    lines.append("## 🚀 RESUME DIRECTIVE")
    lines.append("")

    if plan_state and plan_state.get('pending_count', 0) > 0:
        lines.append("1. **READ PLAN.md** to see full context")
        lines.append("2. **CONTINUE** with the next unchecked `- [ ]` item")
        lines.append("3. **MARK COMPLETE** in PLAN.md when done: `- [x]`")
        lines.append("4. **UPDATE TodoWrite** with current status")
    elif tasks.get('pending', 0) > 0:
        lines.append("1. **CONTINUE** with pending todo items")
        lines.append("2. **UPDATE TodoWrite** as you complete tasks")
    else:
        lines.append("1. All tracked work appears complete")
        lines.append("2. Verify with user if anything else is needed")

    lines.append("")
    return "\n".join(lines)


def main():
    try:
        save_context()
        sys.exit(0)
    except Exception as e:
        print(f"Error saving context: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
