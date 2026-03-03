"""
TasksService for managing tasks within chat sessions.

This service provides CRUD operations for tasks with:
- Auto-incrementing IDs per session
- File-based persistence with JSON format
- Bidirectional dependency tracking (blocks/blockedBy)
- Cycle detection for dependencies
- Session-level file locking for concurrency safety
"""

import contextlib
import json
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from pydantic import BaseModel, Field


class TaskNotFoundError(Exception):
    """Raised when a task is not found."""

    pass


class CycleDetectedError(Exception):
    """Raised when adding a dependency would create a cycle."""

    pass


class InvalidStatusTransitionError(Exception):
    """Raised when a status transition is not allowed."""

    pass


class Task(BaseModel):
    """Task model for tracking work items within a session."""

    id: str
    subject: str
    description: str
    activeForm: str = ""
    status: Literal["pending", "in_progress", "completed"] = "pending"
    blocks: list[str] = Field(default_factory=list)
    blockedBy: list[str] = Field(default_factory=list)
    owner: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TasksService:
    """
    Service for managing tasks within chat sessions.

    All operations use session-level file locking for concurrency safety.
    Tasks are stored as individual JSON files in:
    {project_dir}/tasks/{session_id}/{task_id}.json
    """

    @classmethod
    def _validate_task_id(cls, task_id: str) -> None:
        """Validate that a task ID is a positive integer string.

        Prevents path traversal attacks via crafted task IDs like '../../etc/foo'.

        Raises:
            TaskNotFoundError: If the task ID is not a valid positive integer.
        """
        if not task_id.isdigit():
            raise TaskNotFoundError(f"Invalid task ID: '{task_id}'")

    @classmethod
    def _get_tasks_dir(cls, project_dir: str, session_id: str) -> Path:
        """Get the tasks directory for a session, creating it if necessary."""
        tasks_dir = Path(project_dir) / "tasks" / session_id
        tasks_dir.mkdir(parents=True, exist_ok=True)
        return tasks_dir

    @classmethod
    def _get_lock(cls, tasks_dir: Path) -> FileLock:
        """Get a file lock for the tasks directory."""
        lock_path = tasks_dir / ".lock"
        return FileLock(lock_path, timeout=10)

    @classmethod
    def _get_next_id(cls, tasks_dir: Path) -> str:
        """
        Get the next available task ID.

        Uses a counter file to track the high-water mark, ensuring IDs are never
        reused even after deletions. Must be called while holding the lock.
        """
        counter_file = tasks_dir / ".counter"

        # Read current counter value
        current_counter = 0
        if counter_file.exists():
            with contextlib.suppress(ValueError, OSError):
                current_counter = int(counter_file.read_text().strip())

        # Also scan existing files to handle legacy data or corrupted counter
        max_existing_id = 0
        for task_file in tasks_dir.glob("*.json"):
            try:
                task_id = int(task_file.stem)
                max_existing_id = max(max_existing_id, task_id)
            except ValueError:
                # Skip non-numeric file names
                continue

        # Next ID is max of counter and existing files, plus 1
        next_id = max(current_counter, max_existing_id) + 1

        # Write the new counter value
        counter_file.write_text(str(next_id))

        return str(next_id)

    @classmethod
    def _read_task(cls, tasks_dir: Path, task_id: str) -> Task:
        """
        Read a task from disk. Must be called while holding the lock.

        Raises TaskNotFoundError if the task doesn't exist.
        """
        cls._validate_task_id(task_id)
        task_path = tasks_dir / f"{task_id}.json"
        if not task_path.exists():
            raise TaskNotFoundError(f"Task '{task_id}' not found")
        return Task(**json.loads(task_path.read_text()))

    @classmethod
    def _write_task(cls, tasks_dir: Path, task: Task) -> None:
        """Write a task to disk. Must be called while holding the lock."""
        task_path = tasks_dir / f"{task.id}.json"
        task_path.write_text(json.dumps(task.model_dump(), indent=2))

    @classmethod
    def _validate_no_cycles(
        cls,
        tasks_dir: Path,
        task_id: str,
        new_blocked_by: list[str],
    ) -> None:
        """
        Validate that adding new_blocked_by to task_id won't create a cycle.

        Uses DFS to check if any task in new_blocked_by can reach task_id
        through the existing blockedBy chain. Must be called while holding the lock.

        Raises CycleDetectedError if a cycle would be created.
        """
        # Check for self-reference
        if task_id in new_blocked_by:
            raise CycleDetectedError(f"Task '{task_id}' cannot be blocked by itself")

        # Build a mapping of task_id -> blockedBy for existing tasks
        blocked_by_map: dict[str, list[str]] = {}
        for task_file in tasks_dir.glob("*.json"):
            try:
                task = Task(**json.loads(task_file.read_text()))
                blocked_by_map[task.id] = task.blockedBy
            except (ValueError, json.JSONDecodeError):
                continue

        # Add the proposed new blockedBy entries for the task being updated
        current_blocked_by = blocked_by_map.get(task_id, [])
        blocked_by_map[task_id] = list(set(current_blocked_by + new_blocked_by))

        # DFS to check if task_id can be reached from any of new_blocked_by
        # by following the blockedBy chain (which represents "blocks" in reverse)
        # If task A blockedBy B, then B blocks A, meaning A depends on B.
        # A cycle exists if: task_id -> blockedBy -> ... -> task_id

        def can_reach(start: str, target: str, visited: set[str]) -> bool:
            """Check if we can reach target from start following blockedBy."""
            if start == target:
                return True
            if start in visited:
                return False
            visited.add(start)
            for blocked_by in blocked_by_map.get(start, []):
                if can_reach(blocked_by, target, visited):
                    return True
            return False

        # Check if any of the new blockers can reach back to task_id
        for blocker_id in new_blocked_by:
            visited: set[str] = set()
            if can_reach(blocker_id, task_id, visited):
                raise CycleDetectedError(
                    f"Adding '{blocker_id}' to blockedBy would create a cycle"
                )

    @classmethod
    def create_task(
        cls,
        project_dir: str,
        session_id: str,
        subject: str,
        description: str,
        activeForm: str = "",
        blockedBy: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """
        Create a new task in the session.

        Args:
            project_dir: Path to the project directory
            session_id: The session ID
            subject: Task subject/title
            description: Task description
            activeForm: Present continuous form for spinner display
            blockedBy: List of task IDs that block this task
            metadata: Arbitrary key-value pairs to attach to the task

        Returns:
            The created Task

        Raises:
            TaskNotFoundError: If any task in blockedBy doesn't exist
            CycleDetectedError: If blockedBy would create a cycle
        """
        blockedBy = blockedBy or []
        metadata = metadata or {}
        tasks_dir = cls._get_tasks_dir(project_dir, session_id)
        lock = cls._get_lock(tasks_dir)

        with lock:
            # Validate all blockedBy task IDs
            for blocker_id in blockedBy:
                cls._validate_task_id(blocker_id)
                blocker_path = tasks_dir / f"{blocker_id}.json"
                if not blocker_path.exists():
                    raise TaskNotFoundError(f"Task '{blocker_id}' not found")

            # Get next ID
            task_id = cls._get_next_id(tasks_dir)

            # Check for cycles (new task blocked by existing tasks)
            if blockedBy:
                cls._validate_no_cycles(tasks_dir, task_id, blockedBy)

            # Create the task
            task = Task(
                id=task_id,
                subject=subject,
                description=description,
                activeForm=activeForm,
                blockedBy=blockedBy,
                metadata=metadata,
            )

            # Write the task
            cls._write_task(tasks_dir, task)

            # Update bidirectional references: add this task to blocks of blockers
            for blocker_id in blockedBy:
                blocker = cls._read_task(tasks_dir, blocker_id)
                if task_id not in blocker.blocks:
                    blocker.blocks.append(task_id)
                    cls._write_task(tasks_dir, blocker)

            return task

    @classmethod
    def get_task(cls, project_dir: str, session_id: str, task_id: str) -> Task:
        """
        Get a task by ID.

        Args:
            project_dir: Path to the project directory
            session_id: The session ID
            task_id: The task ID to retrieve

        Returns:
            The Task

        Raises:
            TaskNotFoundError: If the task doesn't exist
        """
        tasks_dir = Path(project_dir) / "tasks" / session_id

        # Check if session directory exists
        if not tasks_dir.exists():
            raise TaskNotFoundError(f"Task '{task_id}' not found")

        lock = cls._get_lock(tasks_dir)

        with lock:
            return cls._read_task(tasks_dir, task_id)

    @classmethod
    def list_tasks(cls, project_dir: str, session_id: str) -> list[Task]:
        """
        List all tasks for a session.

        Args:
            project_dir: Path to the project directory
            session_id: The session ID

        Returns:
            List of Tasks sorted by numeric ID
        """
        tasks_dir = Path(project_dir) / "tasks" / session_id

        # Return empty list if session doesn't exist
        if not tasks_dir.exists():
            return []

        lock = cls._get_lock(tasks_dir)

        with lock:
            tasks = []
            for task_file in tasks_dir.glob("*.json"):
                try:
                    task = Task(**json.loads(task_file.read_text()))
                    tasks.append(task)
                except (ValueError, json.JSONDecodeError):
                    continue

            # Sort by numeric ID
            tasks.sort(key=lambda t: int(t.id))
            return tasks

    @classmethod
    def update_task(
        cls,
        project_dir: str,
        session_id: str,
        task_id: str,
        status: Literal["pending", "in_progress", "completed"] | None = None,
        subject: str | None = None,
        description: str | None = None,
        activeForm: str | None = None,
        addBlocks: list[str] | None = None,
        addBlockedBy: list[str] | None = None,
        owner: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """
        Update a task.

        Args:
            project_dir: Path to the project directory
            session_id: The session ID
            task_id: The task ID to update
            status: New status (pending, in_progress, completed)
            subject: New subject
            description: New description
            activeForm: New activeForm
            addBlocks: Task IDs to add to blocks list
            addBlockedBy: Task IDs to add to blockedBy list
            owner: Agent/worker ID claiming this task
            metadata: Key-value pairs to merge (set key to None to delete)

        Returns:
            The updated Task

        Raises:
            TaskNotFoundError: If the task or any referenced task doesn't exist
            CycleDetectedError: If adding dependencies would create a cycle
            InvalidStatusTransitionError: If trying to reopen a completed task
        """
        tasks_dir = Path(project_dir) / "tasks" / session_id

        if not tasks_dir.exists():
            raise TaskNotFoundError(f"Task '{task_id}' not found")

        lock = cls._get_lock(tasks_dir)

        with lock:
            # Read the task
            task = cls._read_task(tasks_dir, task_id)

            # Reject reopening completed tasks
            if (
                status is not None
                and task.status == "completed"
                and status != "completed"
            ):
                raise InvalidStatusTransitionError(
                    f"Cannot reopen completed task '{task_id}'. Create a new task instead."
                )

            # Update basic fields
            if status is not None:
                task.status = status
            if subject is not None:
                task.subject = subject
            if description is not None:
                task.description = description
            if activeForm is not None:
                task.activeForm = activeForm
            if owner is not None:
                task.owner = owner

            # Handle metadata merge
            if metadata is not None:
                for key, value in metadata.items():
                    if value is None:
                        # Delete key if set to None
                        task.metadata.pop(key, None)
                    else:
                        task.metadata[key] = value

            # Handle addBlockedBy
            if addBlockedBy:
                # Validate all referenced task IDs
                for blocker_id in addBlockedBy:
                    cls._validate_task_id(blocker_id)
                    blocker_path = tasks_dir / f"{blocker_id}.json"
                    if not blocker_path.exists():
                        raise TaskNotFoundError(f"Task '{blocker_id}' not found")

                # Check for cycles
                new_blocked_by = [b for b in addBlockedBy if b not in task.blockedBy]
                if new_blocked_by:
                    cls._validate_no_cycles(tasks_dir, task_id, new_blocked_by)

                    # Add to blockedBy (avoiding duplicates)
                    for blocker_id in new_blocked_by:
                        if blocker_id not in task.blockedBy:
                            task.blockedBy.append(blocker_id)

                        # Update bidirectional reference
                        blocker = cls._read_task(tasks_dir, blocker_id)
                        if task_id not in blocker.blocks:
                            blocker.blocks.append(task_id)
                            cls._write_task(tasks_dir, blocker)

            # Handle addBlocks
            if addBlocks:
                # Validate all referenced task IDs
                for blocked_id in addBlocks:
                    cls._validate_task_id(blocked_id)
                    blocked_path = tasks_dir / f"{blocked_id}.json"
                    if not blocked_path.exists():
                        raise TaskNotFoundError(f"Task '{blocked_id}' not found")

                # For addBlocks, we're saying task_id blocks blocked_id
                # This means blocked_id is blockedBy task_id
                # Check for cycles: if blocked_id already has task_id in its path
                new_blocks = [b for b in addBlocks if b not in task.blocks]
                if new_blocks:
                    # Check cycles: adding task_id to blocked_id's blockedBy
                    for blocked_id in new_blocks:
                        cls._validate_no_cycles(tasks_dir, blocked_id, [task_id])

                    # Add to blocks and update bidirectional
                    for blocked_id in new_blocks:
                        if blocked_id not in task.blocks:
                            task.blocks.append(blocked_id)

                        # Update bidirectional reference
                        blocked = cls._read_task(tasks_dir, blocked_id)
                        if task_id not in blocked.blockedBy:
                            blocked.blockedBy.append(task_id)
                            cls._write_task(tasks_dir, blocked)

            # Handle completion: remove from blockedBy of dependent tasks
            if status == "completed":
                for blocked_id in task.blocks:
                    try:
                        blocked = cls._read_task(tasks_dir, blocked_id)
                        if task_id in blocked.blockedBy:
                            blocked.blockedBy.remove(task_id)
                            cls._write_task(tasks_dir, blocked)
                    except TaskNotFoundError:
                        # Task was deleted, skip
                        continue

            # Write the updated task
            cls._write_task(tasks_dir, task)

            return task

    @classmethod
    def delete_task(cls, project_dir: str, session_id: str, task_id: str) -> Task:
        """
        Delete a task.

        Args:
            project_dir: Path to the project directory
            session_id: The session ID
            task_id: The task ID to delete

        Returns:
            The deleted Task

        Raises:
            TaskNotFoundError: If the task doesn't exist
        """
        tasks_dir = Path(project_dir) / "tasks" / session_id

        if not tasks_dir.exists():
            raise TaskNotFoundError(f"Task '{task_id}' not found")

        lock = cls._get_lock(tasks_dir)

        with lock:
            # Read the task to return and get relationships
            task = cls._read_task(tasks_dir, task_id)

            # Remove this task from blocks of tasks it was blockedBy
            for blocker_id in task.blockedBy:
                try:
                    blocker = cls._read_task(tasks_dir, blocker_id)
                    if task_id in blocker.blocks:
                        blocker.blocks.remove(task_id)
                        cls._write_task(tasks_dir, blocker)
                except TaskNotFoundError:
                    continue

            # Remove this task from blockedBy of tasks it was blocking
            for blocked_id in task.blocks:
                try:
                    blocked = cls._read_task(tasks_dir, blocked_id)
                    if task_id in blocked.blockedBy:
                        blocked.blockedBy.remove(task_id)
                        cls._write_task(tasks_dir, blocked)
                except TaskNotFoundError:
                    continue

            # Delete the task file
            task_path = tasks_dir / f"{task_id}.json"
            task_path.unlink()

            return task
