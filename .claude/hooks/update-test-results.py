#!/usr/bin/env python3
"""
Update Test Results Hook - Parses test output and updates test-results.json.

This hook is triggered after Bash commands that look like test runs.
It parses the output to extract pass/fail counts.
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


def parse_pytest_output(output):
    """Parse pytest output for pass/fail counts."""
    # Match patterns like "5 passed, 2 failed, 1 skipped"
    passed = 0
    failed = 0
    skipped = 0

    # Look for summary line
    match = re.search(r'(\d+)\s+passed', output)
    if match:
        passed = int(match.group(1))

    match = re.search(r'(\d+)\s+failed', output)
    if match:
        failed = int(match.group(1))

    match = re.search(r'(\d+)\s+skipped', output)
    if match:
        skipped = int(match.group(1))

    # Extract failure names
    failures = []
    failure_matches = re.findall(r'FAILED\s+(\S+)', output)
    failures.extend(failure_matches)

    return passed, failed, skipped, failures


def parse_jest_output(output):
    """Parse Jest/npm test output for pass/fail counts."""
    passed = 0
    failed = 0
    skipped = 0

    # Jest format: "Tests: 2 failed, 5 passed, 7 total"
    match = re.search(r'Tests:\s+(?:(\d+)\s+failed,\s+)?(?:(\d+)\s+skipped,\s+)?(\d+)\s+passed', output)
    if match:
        failed = int(match.group(1)) if match.group(1) else 0
        skipped = int(match.group(2)) if match.group(2) else 0
        passed = int(match.group(3)) if match.group(3) else 0

    failures = []
    failure_matches = re.findall(r'✕\s+(.+)', output)
    failures.extend(failure_matches)

    return passed, failed, skipped, failures


def update_test_results():
    """Update test-results.json based on tool output."""
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, IOError):
        return 0

    tool_name = input_data.get('tool_name', '')
    tool_input = input_data.get('tool_input', {})
    tool_output = input_data.get('tool_output', '')

    # Only process Bash commands that look like test runs
    if tool_name != 'Bash':
        return 0

    command = tool_input.get('command', '')
    if not any(test_cmd in command for test_cmd in ['pytest', 'npm test', 'npm run test', 'jest', 'vitest']):
        return 0

    # Parse output
    if 'pytest' in command:
        passed, failed, skipped, failures = parse_pytest_output(tool_output)
    else:
        passed, failed, skipped, failures = parse_jest_output(tool_output)

    # Skip if no test results found
    if passed == 0 and failed == 0 and skipped == 0:
        return 0

    # Update test-results.json
    context_dir = get_context_dir()
    context_dir.mkdir(parents=True, exist_ok=True)

    results_file = context_dir / 'test-results.json'

    results = {
        'last_run': datetime.utcnow().isoformat() + 'Z',
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
        'failures': failures
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Also update progress.json tests section
    progress_file = context_dir / 'progress.json'
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                progress = json.load(f)
            progress['tests'] = {
                'last_run': results['last_run'],
                'passed': passed,
                'failed': failed,
                'skipped': skipped
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except (json.JSONDecodeError, IOError):
            pass

    print(f"Updated test results: {passed} passed, {failed} failed, {skipped} skipped")
    return 0


def main():
    try:
        update_test_results()
        sys.exit(0)
    except Exception as e:
        print(f"Error updating test results: {e}", file=sys.stderr)
        sys.exit(0)  # Don't fail the hook


if __name__ == '__main__':
    main()
