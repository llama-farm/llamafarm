#!/usr/bin/env python3
"""
Update Progress Hook - Tracks progress after TodoWrite operations.

This hook runs after TodoWrite tool calls to:
1. Parse the todo list from the tool input
2. Update progress.json with task counts
3. Sync with plan-progress.json if a plan exists
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


def update_progress():
    """Update progress.json based on TodoWrite tool output."""
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, IOError):
        return 0

    tool_name = input_data.get('tool_name', '')

    # Only process TodoWrite calls
    if tool_name != 'TodoWrite':
        return 0

    tool_input = input_data.get('tool_input', {})
    todos = tool_input.get('todos', [])

    if not todos:
        return 0

    # Count tasks by status
    total = len(todos)
    completed = sum(1 for t in todos if t.get('status') == 'completed')
    in_progress = sum(1 for t in todos if t.get('status') == 'in_progress')
    pending = sum(1 for t in todos if t.get('status') == 'pending')

    # Update progress.json
    context_dir = get_context_dir()
    context_dir.mkdir(parents=True, exist_ok=True)

    progress_file = context_dir / 'progress.json'

    # Load existing or create new
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                progress = json.load(f)
        except (json.JSONDecodeError, IOError):
            progress = {}
    else:
        progress = {}

    # Update task counts
    progress['tasks'] = {
        'total': total,
        'completed': completed,
        'in_progress': in_progress,
        'pending': pending
    }

    # Update session info
    if not progress.get('session_id'):
        progress['session_id'] = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not progress.get('started_at'):
        progress['started_at'] = datetime.now().isoformat()

    # Determine current phase from in_progress task
    in_progress_tasks = [t for t in todos if t.get('status') == 'in_progress']
    if in_progress_tasks:
        progress['current_phase'] = in_progress_tasks[0].get('content', 'Working')
    elif pending > 0:
        pending_tasks = [t for t in todos if t.get('status') == 'pending']
        progress['current_phase'] = f"Next: {pending_tasks[0].get('content', 'Unknown')}"
    elif completed == total and total > 0:
        progress['current_phase'] = 'All tasks completed'
    else:
        progress['current_phase'] = 'In Progress'

    progress['iterations'] = progress.get('iterations', 0) + 1
    progress['last_updated'] = datetime.now().isoformat()

    # Save todos for reference
    progress['todo_list'] = [
        {'content': t.get('content', ''), 'status': t.get('status', 'pending')}
        for t in todos
    ]

    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

    print(f"Progress updated: {completed}/{total} tasks complete, {in_progress} in progress")
    return 0


def main():
    try:
        update_progress()
        sys.exit(0)
    except Exception as e:
        print(f"Error updating progress: {e}", file=sys.stderr)
        sys.exit(0)  # Don't fail the hook


if __name__ == '__main__':
    main()
