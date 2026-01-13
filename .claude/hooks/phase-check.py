#!/usr/bin/env python3
"""
Phase Completion Hook

Ensures tests and demos are run before moving to the next phase.
Called by the stop hook to verify phase completion.

Usage:
    python3 .claude/hooks/phase-check.py status    # Show current phase status
    python3 .claude/hooks/phase-check.py verify    # Verify current phase is complete
    python3 .claude/hooks/phase-check.py next      # Move to next phase (if current is complete)
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
CONTEXT_DIR = Path(PROJECT_DIR) / ".claude" / "context"
PLAN_FILE = Path(PROJECT_DIR) / "PLAN.md"  # In project root, NOT .claude/
PHASE_STATUS_FILE = CONTEXT_DIR / "phase-status.json"


def parse_phases(plan_content: str) -> list[dict]:
    """Extract phases with their checkpoints from the plan."""
    phases = []
    current_phase = None
    current_section = None

    lines = plan_content.split("\n")

    for line in lines:
        # Match phase headers like "## Phase 1: Setup"
        phase_match = re.match(r"^##\s+Phase\s+(\d+)[:\s]*(.*)$", line, re.IGNORECASE)
        if phase_match:
            if current_phase:
                phases.append(current_phase)
            current_phase = {
                "number": int(phase_match.group(1)),
                "name": phase_match.group(2).strip(),
                "tests": [],  # Tests defined FIRST (TDD)
                "demo": [],   # Demo defined FIRST (TDD)
                "implementation": [],
                "verification": [],  # Run tests/demo to verify
                "checkpoint": [],
            }
            current_section = "implementation"
            continue

        if current_phase:
            # Match section headers (order matters - Tests and Demo come FIRST in TDD)
            if re.match(r"^###.*Tests?", line, re.IGNORECASE):
                current_section = "tests"
            elif re.match(r"^###.*Demo", line, re.IGNORECASE):
                current_section = "demo"
            elif re.match(r"^###.*Implementation", line, re.IGNORECASE):
                current_section = "implementation"
            elif re.match(r"^###.*Verification", line, re.IGNORECASE):
                current_section = "verification"
            elif re.match(r"^###.*Checkpoint", line, re.IGNORECASE):
                current_section = "checkpoint"
            elif re.match(r"^##\s+", line):  # New major section
                current_section = None

            # Match checkbox items
            checkbox_match = re.match(r"^\s*[-*]\s*\[([x\s])\]\s*(.+)$", line, re.IGNORECASE)
            if checkbox_match and current_section:
                is_complete = checkbox_match.group(1).lower() == "x"
                description = checkbox_match.group(2).strip()
                current_phase[current_section].append({
                    "description": description,
                    "completed": is_complete,
                })

    if current_phase:
        phases.append(current_phase)

    return phases


def load_phase_status() -> dict:
    """Load or initialize phase status."""
    if PHASE_STATUS_FILE.exists():
        try:
            with open(PHASE_STATUS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    return {
        "current_phase": 1,
        "phases_completed": [],
        "last_test_run": None,
        "last_demo_run": None,
        "test_results": {},
        "demo_results": {},
    }


def save_phase_status(status: dict):
    """Save phase status."""
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE_STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def get_current_phase_status() -> dict:
    """Get detailed status of the current phase."""
    if not PLAN_FILE.exists():
        return {"error": "No plan file found"}

    with open(PLAN_FILE) as f:
        plan_content = f.read()

    phases = parse_phases(plan_content)
    status = load_phase_status()
    current_num = status["current_phase"]

    current_phase = None
    for phase in phases:
        if phase["number"] == current_num:
            current_phase = phase
            break

    if not current_phase:
        return {"error": f"Phase {current_num} not found in plan"}

    # Calculate completion (TDD order: Tests, Demo, Implementation, Verification, Checkpoint)
    tests_total = len(current_phase["tests"])
    tests_done = sum(1 for s in current_phase["tests"] if s["completed"])

    demo_total = len(current_phase["demo"])
    demo_done = sum(1 for s in current_phase["demo"] if s["completed"])

    impl_total = len(current_phase["implementation"])
    impl_done = sum(1 for s in current_phase["implementation"] if s["completed"])

    verify_total = len(current_phase["verification"])
    verify_done = sum(1 for s in current_phase["verification"] if s["completed"])

    checkpoint_total = len(current_phase["checkpoint"])
    checkpoint_done = sum(1 for s in current_phase["checkpoint"] if s["completed"])

    return {
        "phase_number": current_num,
        "phase_name": current_phase["name"],
        "tests": {"done": tests_done, "total": tests_total},  # First (TDD)
        "demo": {"done": demo_done, "total": demo_total},     # Second (TDD)
        "implementation": {"done": impl_done, "total": impl_total},
        "verification": {"done": verify_done, "total": verify_total},
        "checkpoint": {"done": checkpoint_done, "total": checkpoint_total},
        "phase_complete": checkpoint_done == checkpoint_total and checkpoint_total > 0,
        "incomplete_items": [
            f"Test (define first): {s['description']}" for s in current_phase["tests"] if not s["completed"]
        ] + [
            f"Demo (define first): {s['description']}" for s in current_phase["demo"] if not s["completed"]
        ] + [
            s["description"] for s in current_phase["implementation"] if not s["completed"]
        ] + [
            f"Verify: {s['description']}" for s in current_phase["verification"] if not s["completed"]
        ] + [
            f"Checkpoint: {s['description']}" for s in current_phase["checkpoint"] if not s["completed"]
        ],
    }


def verify_phase_complete() -> dict:
    """Verify the current phase is complete before allowing progression."""
    phase_status = get_current_phase_status()

    if "error" in phase_status:
        return {"complete": False, "reason": phase_status["error"]}

    if not phase_status["phase_complete"]:
        incomplete = phase_status["incomplete_items"][:5]  # Show first 5
        return {
            "complete": False,
            "phase": phase_status["phase_number"],
            "reason": f"Phase {phase_status['phase_number']} not complete",
            "incomplete_items": incomplete,
            "tests": phase_status["tests"],  # TDD: Tests first
            "demo": phase_status["demo"],    # TDD: Demo second
            "implementation": phase_status["implementation"],
            "verification": phase_status.get("verification", {"done": 0, "total": 0}),
        }

    return {
        "complete": True,
        "phase": phase_status["phase_number"],
        "reason": f"Phase {phase_status['phase_number']} complete!",
    }


def advance_to_next_phase() -> dict:
    """Advance to the next phase if current is complete."""
    verification = verify_phase_complete()

    if not verification["complete"]:
        return {
            "success": False,
            "reason": verification["reason"],
            "incomplete_items": verification.get("incomplete_items", []),
        }

    status = load_phase_status()
    current = status["current_phase"]
    status["phases_completed"].append(current)
    status["current_phase"] = current + 1
    save_phase_status(status)

    return {
        "success": True,
        "previous_phase": current,
        "current_phase": current + 1,
        "reason": f"Advanced from Phase {current} to Phase {current + 1}",
    }


def print_status():
    """Print human-readable phase status."""
    phase_status = get_current_phase_status()

    if "error" in phase_status:
        print(f"Error: {phase_status['error']}")
        return

    print(f"\n{'='*60}")
    print(f"Phase {phase_status['phase_number']}: {phase_status['phase_name']}")
    print(f"{'='*60}")

    # TDD order: Tests first, Demo second, then Implementation
    tests = phase_status["tests"]
    demo = phase_status["demo"]
    impl = phase_status["implementation"]
    verify = phase_status.get("verification", {"done": 0, "total": 0})
    checkpoint = phase_status["checkpoint"]

    print(f"\n1. Tests (define first):  {tests['done']}/{tests['total']}")
    print(f"2. Demo (define first):   {demo['done']}/{demo['total']}")
    print(f"3. Implementation:        {impl['done']}/{impl['total']}")
    print(f"4. Verification:          {verify['done']}/{verify['total']}")
    print(f"5. Checkpoint:            {checkpoint['done']}/{checkpoint['total']}")

    if phase_status["phase_complete"]:
        print(f"\n✅ Phase {phase_status['phase_number']} COMPLETE - Ready to advance")
    else:
        print(f"\n❌ Phase {phase_status['phase_number']} INCOMPLETE")
        print("\nRemaining items (TDD order):")
        for item in phase_status["incomplete_items"][:10]:
            print(f"  - {item}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_status()
        sys.exit(0)

    command = sys.argv[1]

    if command == "status":
        print_status()
    elif command == "verify":
        result = verify_phase_complete()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["complete"] else 1)
    elif command == "next":
        result = advance_to_next_phase()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)
    elif command == "json":
        result = get_current_phase_status()
        print(json.dumps(result, indent=2))
    else:
        print(f"Unknown command: {command}")
        print("Usage: phase-check.py [status|verify|next|json]")
        sys.exit(1)
