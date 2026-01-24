#!/usr/bin/env python3
"""
Plan Tracker Hook

Tracks progress against the approved plan in .claude/PLAN.md.
Extracts plan steps and marks them as complete based on todo list and file changes.

This hook runs:
- On SessionStart: Load and parse the plan
- On PostToolUse: Update progress tracking
- Used by Stop hook: Check if all plan steps are done
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

PROJECT_DIR = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
CONTEXT_DIR = Path(PROJECT_DIR) / ".claude" / "context"
PLAN_FILE = Path(PROJECT_DIR) / "PLAN.md"  # In project root, NOT .claude/
PLAN_PROGRESS_FILE = CONTEXT_DIR / "plan-progress.json"
PROGRESS_FILE = CONTEXT_DIR / "progress.json"  # Main progress tracking file


def parse_plan_steps(plan_content: str) -> list[dict]:
    """Extract actionable steps from the plan markdown."""
    steps = []
    current_phase = None
    step_pattern = re.compile(r"^(\s*)[-*]\s*\[([x\s])\]\s*(.+)$", re.IGNORECASE)
    phase_pattern = re.compile(r"^#{1,3}\s*(?:Phase\s*\d+[:\s]*)?(.+)$", re.IGNORECASE)

    for line in plan_content.split("\n"):
        # Check for phase headers
        phase_match = phase_pattern.match(line)
        if phase_match:
            current_phase = phase_match.group(1).strip()
            continue

        # Check for checklist items
        step_match = step_pattern.match(line)
        if step_match:
            indent = len(step_match.group(1))
            is_complete = step_match.group(2).lower() == "x"
            description = step_match.group(3).strip()

            steps.append({
                "phase": current_phase,
                "description": description,
                "completed": is_complete,
                "indent": indent,
                "completed_at": None if not is_complete else datetime.now().isoformat()
            })

    return steps


def load_plan_progress() -> dict:
    """Load existing plan progress or initialize from plan file."""
    if PLAN_PROGRESS_FILE.exists():
        try:
            with open(PLAN_PROGRESS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Initialize from plan file
    if PLAN_FILE.exists():
        with open(PLAN_FILE) as f:
            plan_content = f.read()

        steps = parse_plan_steps(plan_content)
        progress = {
            "plan_file": str(PLAN_FILE),
            "approved_at": None,
            "started_at": None,
            "completed_at": None,
            "steps": steps,
            "total_steps": len(steps),
            "completed_steps": sum(1 for s in steps if s["completed"]),
            "current_phase": steps[0]["phase"] if steps else None
        }
        save_plan_progress(progress)
        return progress

    return {
        "plan_file": None,
        "approved_at": None,
        "started_at": None,
        "completed_at": None,
        "steps": [],
        "total_steps": 0,
        "completed_steps": 0,
        "current_phase": None
    }


def save_plan_progress(progress: dict):
    """Save plan progress to file and sync to progress.json."""
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLAN_PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

    # Also update the main progress.json
    sync_progress_json(progress)


def sync_progress_json(plan_progress: dict):
    """Sync plan progress to the main progress.json file."""
    # Load existing progress.json
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                progress = json.load(f)
        except (json.JSONDecodeError, IOError):
            progress = {}
    else:
        progress = {}

    # Update with plan progress data
    progress["session_id"] = progress.get("session_id") or f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    progress["started_at"] = plan_progress.get("started_at") or progress.get("started_at") or datetime.now().isoformat()
    progress["current_phase"] = plan_progress.get("current_phase") or "In Progress"

    # Calculate task counts from plan steps
    total_steps = plan_progress.get("total_steps", 0)
    completed_steps = plan_progress.get("completed_steps", 0)
    progress["tasks"] = {
        "total": total_steps,
        "completed": completed_steps,
        "in_progress": 1 if total_steps > completed_steps else 0,
        "pending": max(0, total_steps - completed_steps - 1)
    }

    # Load test results if available
    test_results_file = CONTEXT_DIR / "test-results.json"
    if test_results_file.exists():
        try:
            with open(test_results_file) as f:
                test_results = json.load(f)
            progress["tests"] = {
                "last_run": test_results.get("last_run", ""),
                "passed": test_results.get("passed", 0),
                "failed": test_results.get("failed", 0),
                "skipped": test_results.get("skipped", 0)
            }
        except (json.JSONDecodeError, IOError):
            pass

    # Increment iterations
    progress["iterations"] = progress.get("iterations", 0) + 1

    # Save updated progress.json
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def mark_step_complete(step_description: str) -> bool:
    """Mark a specific step as complete."""
    progress = load_plan_progress()

    for step in progress["steps"]:
        if step_description.lower() in step["description"].lower():
            if not step["completed"]:
                step["completed"] = True
                step["completed_at"] = datetime.now().isoformat()
                progress["completed_steps"] += 1

                # Update current phase if this phase is done
                phase_steps = [s for s in progress["steps"] if s["phase"] == step["phase"]]
                if all(s["completed"] for s in phase_steps):
                    # Find next incomplete phase
                    for s in progress["steps"]:
                        if not s["completed"]:
                            progress["current_phase"] = s["phase"]
                            break

                save_plan_progress(progress)
                return True

    return False


def approve_plan():
    """Mark the plan as approved and ready for execution."""
    progress = load_plan_progress()
    progress["approved_at"] = datetime.now().isoformat()
    progress["started_at"] = datetime.now().isoformat()
    save_plan_progress(progress)
    print(f"Plan approved at {progress['approved_at']}")
    print(f"Total steps: {progress['total_steps']}")


def check_plan_completion() -> dict:
    """Check if the plan is complete. Used by stop hook."""
    progress = load_plan_progress()

    if not progress["steps"]:
        return {
            "has_plan": False,
            "complete": True,
            "reason": "No plan found - proceeding without plan"
        }

    if not progress["approved_at"]:
        return {
            "has_plan": True,
            "complete": False,
            "approved": False,
            "reason": "Plan exists but not yet approved"
        }

    incomplete = [s for s in progress["steps"] if not s["completed"]]

    if incomplete:
        return {
            "has_plan": True,
            "complete": False,
            "approved": True,
            "remaining": len(incomplete),
            "total": progress["total_steps"],
            "current_phase": progress["current_phase"],
            "next_steps": [s["description"] for s in incomplete[:3]],
            "reason": f"{len(incomplete)} of {progress['total_steps']} steps remaining"
        }

    # All complete!
    if not progress["completed_at"]:
        progress["completed_at"] = datetime.now().isoformat()
        save_plan_progress(progress)

    return {
        "has_plan": True,
        "complete": True,
        "approved": True,
        "total": progress["total_steps"],
        "reason": "All plan steps completed!"
    }


def get_plan_status() -> str:
    """Get a human-readable plan status."""
    progress = load_plan_progress()

    if not progress["steps"]:
        return "No plan loaded"

    completed = progress["completed_steps"]
    total = progress["total_steps"]
    pct = (completed / total * 100) if total > 0 else 0

    status = f"Plan Progress: {completed}/{total} steps ({pct:.0f}%)"

    if progress["current_phase"]:
        status += f"\nCurrent Phase: {progress['current_phase']}"

    # Show next 3 incomplete steps
    incomplete = [s for s in progress["steps"] if not s["completed"]][:3]
    if incomplete:
        status += "\nNext steps:"
        for s in incomplete:
            status += f"\n  - {s['description']}"

    return status


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "status":
            print(get_plan_status())
        elif command == "check":
            result = check_plan_completion()
            print(json.dumps(result, indent=2))
        elif command == "approve":
            approve_plan()
        elif command == "complete" and len(sys.argv) > 2:
            step = " ".join(sys.argv[2:])
            if mark_step_complete(step):
                print(f"Marked complete: {step}")
            else:
                print(f"Step not found: {step}")
        else:
            print("Usage: plan-tracker.py [status|check|approve|complete <step>]")
    else:
        # Default: show status
        print(get_plan_status())
