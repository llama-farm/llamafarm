#!/usr/bin/env python3
"""
Stop hook that gates agent completion on passing code review.

This hook:
1. Detects uncommitted changes via git diff
2. Checks for a recent /code-review report
3. Parses the report for Critical/High severity findings
4. Blocks completion until review passes or max iterations reached

The hook does NOT perform the review itself - it instructs the agent
to run the /code-review skill, then parses the resulting report.
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("code-review-loop")

# Configuration
MAX_ITERATIONS = 5
STATE_DIR = Path(tempfile.gettempdir()) / "claude"
PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))


def get_state_dir() -> Path:
    """Get the state directory for this project."""
    sanitized = str(PROJECT_DIR).replace("/", "-").lstrip("-")
    state_dir = STATE_DIR / sanitized
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def get_state_file() -> Path:
    """Get path to state file for tracking iterations."""
    return get_state_dir() / "code-review-loop-state.json"


def get_reports_dir() -> Path:
    """Get the directory where /code-review stores reports."""
    return get_state_dir() / "reports"


def load_state() -> dict:
    """Load iteration state from file."""
    state_file = get_state_file()
    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "iterations": 0,
        "last_diff_hash": None,
        "last_review_path": None,
        "last_run": None,
    }


def save_state(state: dict) -> None:
    """Persist state to file."""
    state_file = get_state_file()
    state["last_run"] = datetime.now().isoformat()
    try:
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    except IOError as e:
        logger.warning("Failed to save state to %s: %s", state_file, e)


def clear_state() -> None:
    """Remove state file when review passes."""
    state_file = get_state_file()
    try:
        if state_file.exists():
            state_file.unlink()
    except IOError as e:
        logger.warning("Failed to clear state file %s: %s", state_file, e)


def get_uncommitted_changes() -> list[str]:
    """Get list of files with uncommitted changes."""
    try:
        # Check for staged + unstaged changes against HEAD
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

        # Fallback: check unstaged only (for new repos without commits)
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

        # Also check for untracked files that might be new
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            files = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    # Format: "XY filename" where X=staged, Y=unstaged
                    files.append(line[3:].strip())
            return files

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return []


def get_diff_hash() -> str:
    """Generate hash of current diff to detect changes since last review."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=60,
        )
        if result.returncode == 0:
            return hashlib.sha256(result.stdout.encode()).hexdigest()[:16]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()[:16]


def find_latest_review_report() -> Path | None:
    """Find the most recent code-review report."""
    reports_dir = get_reports_dir()
    if not reports_dir.exists():
        return None

    # Find all code-review-*.md files
    reports = list(reports_dir.glob("code-review-*.md"))
    if not reports:
        return None

    # Sort by modification time, newest first
    reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0]


def parse_review_report(report_path: Path) -> dict:
    """
    Parse the code review report to extract findings by severity.

    Returns:
        {
            "critical": [{"category": ..., "issue": ..., "file": ...}, ...],
            "high": [...],
            "medium": [...],
            "low": [...],
            "status": "Complete" | "In Progress",
            "timestamp": datetime or None
        }
    """
    findings = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": [],
        "status": "Unknown",
        "timestamp": None,
    }

    try:
        content = report_path.read_text()
    except IOError:
        return findings

    # Extract status
    status_match = re.search(r"\*\*Status\*\*:\s*(\w+(?:\s+\w+)?)", content)
    if status_match:
        findings["status"] = status_match.group(1)

    # Extract timestamp from filename (code-review-YYYYMMDD-HHMMSS.md)
    timestamp_match = re.search(r"code-review-(\d{8}-\d{6})\.md", report_path.name)
    if timestamp_match:
        try:
            findings["timestamp"] = datetime.strptime(
                timestamp_match.group(1), "%Y%m%d-%H%M%S"
            )
        except ValueError:
            pass

    # Parse findings sections
    # Look for patterns like:
    # ### [Category] Item Name
    # **Status**: FAIL
    # **Severity**: Critical
    # **File**: `path/to/file`
    # **Issue**: description

    finding_pattern = re.compile(
        r"###\s*\[([^\]]+)\]\s*([^\n]+)\n"  # Category and item name
        r"(?:.*?\n)*?"  # Non-greedy match of any content
        r"\*\*Status\*\*:\s*FAIL\s*\n"  # Must be FAIL status
        r"(?:.*?\n)*?"
        r"\*\*Severity\*\*:\s*(Critical|High|Medium|Low)",  # Severity
        re.IGNORECASE | re.MULTILINE,
    )

    for match in finding_pattern.finditer(content):
        category = match.group(1).strip()
        item_name = match.group(2).strip()
        severity = match.group(3).strip().lower()

        # Try to extract file and issue from the surrounding context
        context_start = match.start()
        context_end = min(match.end() + 500, len(content))
        context = content[context_start:context_end]

        file_match = re.search(r"\*\*File\*\*:\s*`([^`]+)`", context)
        issue_match = re.search(r"\*\*Issue\*\*:\s*([^\n]+)", context)

        finding = {
            "category": category,
            "item": item_name,
            "file": file_match.group(1) if file_match else "unknown",
            "issue": issue_match.group(1) if issue_match else item_name,
        }

        if severity in findings:
            findings[severity].append(finding)

    return findings


def is_review_current(report_path: Path, current_diff_hash: str, state: dict) -> bool:
    """
    Check if the review was done on the current diff (not stale).

    A review is current if the diff hash matches what was recorded
    when the review was done.
    """
    if state.get("last_diff_hash") == current_diff_hash:
        return True

    # Also check if report is recent (within last 5 minutes) as fallback
    if report_path.exists():
        report_age = datetime.now().timestamp() - report_path.stat().st_mtime
        if report_age < 300:  # 5 minutes
            return True

    return False


def decide() -> dict:
    """
    Main decision logic for whether to allow or block completion.

    Returns:
        {"decision": "approve"} or {"decision": "block", "reason": "..."}
    """
    # Check for uncommitted changes
    changes = get_uncommitted_changes()
    if not changes:
        clear_state()
        return {"decision": "approve"}

    # Load state
    state = load_state()

    # Check iteration limit (safety valve)
    if state["iterations"] >= MAX_ITERATIONS:
        clear_state()
        return {
            "decision": "approve",
            "systemMessage": f"Max review iterations ({MAX_ITERATIONS}) reached. Allowing completion.",
        }

    # Get current diff hash
    current_hash = get_diff_hash()

    # Find latest review report
    report = find_latest_review_report()

    if not report:
        # No review has been run yet
        state["iterations"] += 1
        state["last_diff_hash"] = current_hash
        save_state(state)
        return {
            "decision": "block",
            "reason": (
                "Code review required before completing.\n\n"
                f"Changed files ({len(changes)}):\n"
                + "\n".join(f"  - {f}" for f in changes[:10])
                + ("\n  ..." if len(changes) > 10 else "")
                + "\n\nRun `/code-review` on the current changes."
            ),
        }

    # Check if review is current (diff hasn't changed since review)
    if not is_review_current(report, current_hash, state):
        state["iterations"] += 1
        state["last_diff_hash"] = current_hash
        save_state(state)
        return {
            "decision": "block",
            "reason": (
                "Changes detected since last code review.\n\n"
                "Re-run `/code-review` on the updated changes."
            ),
        }

    # Parse the review report
    findings = parse_review_report(report)
    critical_high = findings["critical"] + findings["high"]

    if critical_high:
        # There are Critical/High findings that need to be addressed
        state["iterations"] += 1
        save_state(state)

        issues_list = "\n".join(
            f"  - [{f['category']}] {f['issue']}" for f in critical_high[:5]
        )
        remaining = len(critical_high) - 5 if len(critical_high) > 5 else 0

        return {
            "decision": "block",
            "reason": (
                f"Code review found {len(critical_high)} Critical/High severity issue(s):\n\n"
                f"{issues_list}"
                + (f"\n  ... and {remaining} more" if remaining else "")
                + f"\n\nFull report: {report}\n\n"
                "Address these issues, then re-run `/code-review`."
            ),
        }

    # Review passed - no Critical/High findings
    clear_state()
    return {"decision": "approve"}


def main():
    """Entry point for the hook."""
    # Read hook input from stdin (may be empty for Stop hooks)
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        input_data = {}

    # Check if we're in a git repository
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            cwd=PROJECT_DIR,
            timeout=5,
        )
        if result.returncode != 0:
            # Not a git repo, allow completion
            sys.exit(0)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # Git not available, allow completion
        sys.exit(0)

    # Make decision
    decision = decide()

    if decision.get("decision") == "block":
        # Output the decision as JSON for Claude Code to process
        print(json.dumps(decision))
        sys.exit(0)
    elif decision.get("systemMessage"):
        # Approved but with a message
        print(json.dumps(decision))
        sys.exit(0)
    else:
        # Approved, exit cleanly
        sys.exit(0)


if __name__ == "__main__":
    main()
