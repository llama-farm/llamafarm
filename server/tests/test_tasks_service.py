"""
Tests for TasksService.

This module contains comprehensive TDD tests for the TasksService class,
including unit tests for task CRUD operations, dependency management,
cycle detection, and concurrency handling.

Written following TEST-DRIVEN DEVELOPMENT: tests are written before implementation.
"""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

# Import will fail until implementation exists - that's TDD!
# These imports are what we expect the implementation to provide
try:
    from services.tasks_service import (
        CycleDetectedError,
        InvalidStatusTransitionError,
        TaskNotFoundError,
        TasksService,
    )
except ImportError:
    # Define placeholder classes for type hints in tests
    # These will be replaced by actual imports once implementation exists

    class Task(BaseModel):
        """Task model - placeholder until implementation exists."""

        id: str
        subject: str
        description: str
        activeForm: str = ""
        status: Literal["pending", "in_progress", "completed"] = "pending"
        blocks: list[str] = Field(default_factory=list)
        blockedBy: list[str] = Field(default_factory=list)

    class TaskNotFoundError(Exception):
        """Raised when a task is not found."""

        pass

    class CycleDetectedError(Exception):
        """Raised when adding a dependency would create a cycle."""

        pass

    class InvalidStatusTransitionError(Exception):
        """Raised when a status transition is not allowed."""

        pass

    class TasksService:
        """Placeholder TasksService class - tests will fail until implementation exists."""

        pass


class TestTaskCreation:
    """Test cases for task creation functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        """Provide a consistent session ID for tests."""
        return "test-session-123"

    def test_create_task_with_basic_fields(self, temp_project_dir, session_id):
        """Test creating a task with subject, description, and activeForm."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Implement user authentication",
            description="Add login and registration endpoints",
            activeForm="Implementing user authentication",
        )

        assert task.subject == "Implement user authentication"
        assert task.description == "Add login and registration endpoints"
        assert task.activeForm == "Implementing user authentication"
        assert task.status == "pending"
        assert task.blocks == []
        assert task.blockedBy == []

    def test_create_task_auto_increments_ids(self, temp_project_dir, session_id):
        """Test that task IDs auto-increment: 1, 2, 3..."""
        task1 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="First task",
            description="Task 1 description",
        )
        task2 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Second task",
            description="Task 2 description",
        )
        task3 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Third task",
            description="Task 3 description",
        )

        assert task1.id == "1"
        assert task2.id == "2"
        assert task3.id == "3"

    def test_create_task_with_blocked_by_references(self, temp_project_dir, session_id):
        """Test creating a task that is blocked by other tasks."""
        # Create prerequisite tasks
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Setup database",
            description="Initialize database schema",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Configure environment",
            description="Set up environment variables",
        )

        # Create a task blocked by both
        task3 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Run migrations",
            description="Execute database migrations",
            blockedBy=["1", "2"],
        )

        assert task3.blockedBy == ["1", "2"]

    def test_create_task_updates_bidirectional_references(
        self, temp_project_dir, session_id
    ):
        """Test that blockedBy creates corresponding blocks on referenced tasks."""
        # Create prerequisite task
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Setup database",
            description="Initialize database schema",
        )

        # Create dependent task
        task2 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Run migrations",
            description="Execute database migrations",
            blockedBy=["1"],
        )

        # Re-fetch task1 to see updated blocks
        updated_task1 = TasksService.get_task(temp_project_dir, session_id, "1")

        assert "2" in updated_task1.blocks
        assert "1" in task2.blockedBy

    def test_create_task_error_on_nonexistent_blocked_by(
        self, temp_project_dir, session_id
    ):
        """Test that creating a task with non-existent blockedBy raises an error."""
        with pytest.raises(TaskNotFoundError, match="Task '999' not found"):
            TasksService.create_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                subject="Dependent task",
                description="This task depends on non-existent task",
                blockedBy=["999"],
            )

    def test_create_task_persists_to_file(self, temp_project_dir, session_id):
        """Test that created tasks are persisted to JSON files."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Persistent task",
            description="This should be saved to disk",
        )

        # Verify file exists
        task_file = Path(temp_project_dir) / "tasks" / session_id / f"{task.id}.json"
        assert task_file.exists()

        # Verify file content
        with open(task_file) as f:
            saved_data = json.load(f)
        assert saved_data["subject"] == "Persistent task"
        assert saved_data["description"] == "This should be saved to disk"

    def test_create_task_creates_directory_structure(
        self, temp_project_dir, session_id
    ):
        """Test that create_task creates necessary directories if they don't exist."""
        # Ensure tasks directory doesn't exist
        tasks_dir = Path(temp_project_dir) / "tasks" / session_id
        assert not tasks_dir.exists()

        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="First task",
            description="Should create directories",
        )

        assert tasks_dir.exists()


class TestTaskRetrieval:
    """Test cases for task retrieval functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-456"

    def test_get_task_returns_correct_task(self, temp_project_dir, session_id):
        """Test that get_task returns the correct task by ID."""
        created_task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Test task",
            description="For retrieval testing",
            activeForm="Testing task retrieval",
        )

        retrieved_task = TasksService.get_task(
            temp_project_dir, session_id, created_task.id
        )

        assert retrieved_task.id == created_task.id
        assert retrieved_task.subject == "Test task"
        assert retrieved_task.description == "For retrieval testing"
        assert retrieved_task.activeForm == "Testing task retrieval"

    def test_get_task_raises_error_for_nonexistent_task(
        self, temp_project_dir, session_id
    ):
        """Test that get_task raises TaskNotFoundError for non-existent task."""
        with pytest.raises(TaskNotFoundError, match="Task '999' not found"):
            TasksService.get_task(temp_project_dir, session_id, "999")

    def test_get_task_raises_error_for_nonexistent_session(self, temp_project_dir):
        """Test that get_task raises error when session doesn't exist."""
        with pytest.raises(TaskNotFoundError):
            TasksService.get_task(temp_project_dir, "nonexistent-session", "1")

    def test_list_tasks_returns_all_tasks(self, temp_project_dir, session_id):
        """Test that list_tasks returns all tasks for a session."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 1",
            description="First task",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 2",
            description="Second task",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 3",
            description="Third task",
        )

        tasks = TasksService.list_tasks(temp_project_dir, session_id)

        assert len(tasks) == 3
        subjects = [t.subject for t in tasks]
        assert "Task 1" in subjects
        assert "Task 2" in subjects
        assert "Task 3" in subjects

    def test_list_tasks_sorted_by_id(self, temp_project_dir, session_id):
        """Test that list_tasks returns tasks sorted by numeric ID."""
        # Create tasks in random order
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task A",
            description="First",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task B",
            description="Second",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task C",
            description="Third",
        )

        tasks = TasksService.list_tasks(temp_project_dir, session_id)

        # Should be sorted by ID
        assert tasks[0].id == "1"
        assert tasks[1].id == "2"
        assert tasks[2].id == "3"

    def test_list_tasks_returns_empty_list_for_no_tasks(
        self, temp_project_dir, session_id
    ):
        """Test that list_tasks returns empty list when no tasks exist."""
        tasks = TasksService.list_tasks(temp_project_dir, session_id)
        assert tasks == []

    def test_list_tasks_returns_empty_for_nonexistent_session(self, temp_project_dir):
        """Test that list_tasks returns empty list for non-existent session."""
        tasks = TasksService.list_tasks(temp_project_dir, "nonexistent-session")
        assert tasks == []


class TestTaskUpdate:
    """Test cases for task update functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-789"

    def test_update_task_status_pending_to_in_progress(
        self, temp_project_dir, session_id
    ):
        """Test updating task status from pending to in_progress."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Test task",
            description="Status transition test",
        )
        assert task.status == "pending"

        updated_task = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            status="in_progress",
        )

        assert updated_task.status == "in_progress"

    def test_update_task_status_in_progress_to_completed(
        self, temp_project_dir, session_id
    ):
        """Test updating task status from in_progress to completed."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Test task",
            description="Status transition test",
        )
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            status="in_progress",
        )

        updated_task = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            status="completed",
        )

        assert updated_task.status == "completed"

    def test_update_task_subject(self, temp_project_dir, session_id):
        """Test updating task subject."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Original subject",
            description="Test description",
        )

        updated_task = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            subject="Updated subject",
        )

        assert updated_task.subject == "Updated subject"

    def test_update_task_description(self, temp_project_dir, session_id):
        """Test updating task description."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Test task",
            description="Original description",
        )

        updated_task = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            description="Updated description",
        )

        assert updated_task.description == "Updated description"

    def test_update_task_active_form(self, temp_project_dir, session_id):
        """Test updating task activeForm."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Test task",
            description="Test description",
            activeForm="Original active form",
        )

        updated_task = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            activeForm="Updated active form",
        )

        assert updated_task.activeForm == "Updated active form"

    def test_update_task_add_blocks(self, temp_project_dir, session_id):
        """Test adding blocks via addBlocks parameter."""
        task1 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Blocking task",
            description="This will block others",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task to be blocked",
            description="This will be blocked",
        )

        # Add task2 to task1's blocks list
        updated_task1 = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task1.id,
            addBlocks=["2"],
        )

        assert "2" in updated_task1.blocks

        # Verify bidirectional: task2 should have task1 in blockedBy
        updated_task2 = TasksService.get_task(temp_project_dir, session_id, "2")
        assert "1" in updated_task2.blockedBy

    def test_update_task_add_blocked_by(self, temp_project_dir, session_id):
        """Test adding blockedBy via addBlockedBy parameter."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Blocking task",
            description="This will be a blocker",
        )
        task2 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Blocked task",
            description="This will be blocked",
        )

        # Add task1 to task2's blockedBy list
        updated_task2 = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task2.id,
            addBlockedBy=["1"],
        )

        assert "1" in updated_task2.blockedBy

        # Verify bidirectional: task1 should have task2 in blocks
        updated_task1 = TasksService.get_task(temp_project_dir, session_id, "1")
        assert "2" in updated_task1.blocks

    def test_update_task_completing_removes_from_blocked_by(
        self, temp_project_dir, session_id
    ):
        """Test that completing a task removes it from blockedBy of dependent tasks."""
        task1 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Blocking task",
            description="When completed, should unblock others",
        )
        task2 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Blocked task",
            description="Blocked by task 1",
            blockedBy=["1"],
        )

        # Verify initial state
        assert "1" in task2.blockedBy

        # Complete task1
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task1.id,
            status="completed",
        )

        # Task2 should no longer be blocked by task1
        updated_task2 = TasksService.get_task(temp_project_dir, session_id, "2")
        assert "1" not in updated_task2.blockedBy

    def test_update_task_persists_changes(self, temp_project_dir, session_id):
        """Test that updates are persisted to disk."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Original",
            description="Original description",
        )

        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            subject="Updated",
            status="in_progress",
        )

        # Read directly from file
        task_file = Path(temp_project_dir) / "tasks" / session_id / f"{task.id}.json"
        with open(task_file) as f:
            saved_data = json.load(f)

        assert saved_data["subject"] == "Updated"
        assert saved_data["status"] == "in_progress"

    def test_update_nonexistent_task_raises_error(self, temp_project_dir, session_id):
        """Test that updating a non-existent task raises TaskNotFoundError."""
        with pytest.raises(TaskNotFoundError, match="Task '999' not found"):
            TasksService.update_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                task_id="999",
                subject="Updated",
            )

    def test_cannot_reopen_completed_task_to_pending(
        self, temp_project_dir, session_id
    ):
        """Completed tasks cannot be set back to pending."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task to complete",
            description="Will be completed then reopened",
        )
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            status="completed",
        )

        with pytest.raises(
            InvalidStatusTransitionError, match="Cannot reopen completed task"
        ):
            TasksService.update_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                task_id=task.id,
                status="pending",
            )

    def test_cannot_reopen_completed_task_to_in_progress(
        self, temp_project_dir, session_id
    ):
        """Completed tasks cannot be set back to in_progress."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task to complete",
            description="Will be completed then reopened",
        )
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            status="completed",
        )

        with pytest.raises(
            InvalidStatusTransitionError, match="Cannot reopen completed task"
        ):
            TasksService.update_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                task_id=task.id,
                status="in_progress",
            )

    def test_can_set_completed_task_to_completed(self, temp_project_dir, session_id):
        """Setting a completed task to completed again should be allowed (no-op)."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task to complete",
            description="Will be set completed twice",
        )
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            status="completed",
        )

        # Should not raise - setting completed to completed is allowed
        updated_task = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            status="completed",
        )
        assert updated_task.status == "completed"


class TestTaskDeletion:
    """Test cases for task deletion functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-delete"

    def test_delete_task_removes_file(self, temp_project_dir, session_id):
        """Test that deleting a task removes its file."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="To be deleted",
            description="This will be deleted",
        )

        task_file = Path(temp_project_dir) / "tasks" / session_id / f"{task.id}.json"
        assert task_file.exists()

        TasksService.delete_task(temp_project_dir, session_id, task.id)

        assert not task_file.exists()

    def test_delete_task_returns_deleted_task(self, temp_project_dir, session_id):
        """Test that delete_task returns the deleted task."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="To be deleted",
            description="This will be deleted",
        )

        deleted_task = TasksService.delete_task(temp_project_dir, session_id, task.id)

        assert deleted_task.id == task.id
        assert deleted_task.subject == "To be deleted"

    def test_delete_task_removes_from_blocks_of_related_tasks(
        self, temp_project_dir, session_id
    ):
        """Test that deleting a task removes it from blocks of tasks it was blocking."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Blocking task",
            description="Will block task2",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Blocked task",
            description="Blocked by task1",
            blockedBy=["1"],
        )

        # Verify task1 blocks task2
        updated_task1 = TasksService.get_task(temp_project_dir, session_id, "1")
        assert "2" in updated_task1.blocks

        # Delete task2
        TasksService.delete_task(temp_project_dir, session_id, "2")

        # Task1 should no longer have task2 in blocks
        updated_task1 = TasksService.get_task(temp_project_dir, session_id, "1")
        assert "2" not in updated_task1.blocks

    def test_delete_task_removes_from_blocked_by_of_related_tasks(
        self, temp_project_dir, session_id
    ):
        """Test that deleting a task removes it from blockedBy of dependent tasks."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Blocking task",
            description="Will block task2",
        )
        task2 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Blocked task",
            description="Blocked by task1",
            blockedBy=["1"],
        )

        # Verify task2 is blocked by task1
        assert "1" in task2.blockedBy

        # Delete task1
        TasksService.delete_task(temp_project_dir, session_id, "1")

        # Task2 should no longer be blocked by task1
        updated_task2 = TasksService.get_task(temp_project_dir, session_id, "2")
        assert "1" not in updated_task2.blockedBy

    def test_delete_nonexistent_task_raises_error(self, temp_project_dir, session_id):
        """Test that deleting a non-existent task raises TaskNotFoundError."""
        with pytest.raises(TaskNotFoundError, match="Task '999' not found"):
            TasksService.delete_task(temp_project_dir, session_id, "999")

    def test_delete_task_with_multiple_dependencies(self, temp_project_dir, session_id):
        """Test deleting a task that has multiple blocking and blocked-by relationships."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 1",
            description="First task",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 2",
            description="Second task",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 3 (middle)",
            description="Blocked by 1, 2 and blocks others",
            blockedBy=["1", "2"],
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 4",
            description="Blocked by task 3",
            blockedBy=["3"],
        )

        # Delete task3
        TasksService.delete_task(temp_project_dir, session_id, "3")

        # Task1 and task2 should no longer have task3 in blocks
        updated_task1 = TasksService.get_task(temp_project_dir, session_id, "1")
        updated_task2 = TasksService.get_task(temp_project_dir, session_id, "2")
        assert "3" not in updated_task1.blocks
        assert "3" not in updated_task2.blocks

        # Task4 should no longer be blocked by task3
        updated_task4 = TasksService.get_task(temp_project_dir, session_id, "4")
        assert "3" not in updated_task4.blockedBy


class TestCycleDetection:
    """Test cases for cycle detection in task dependencies."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-cycles"

    def test_reject_direct_cycle_on_create(self, temp_project_dir, session_id):
        """Test that creating task A blockedBy B when B is blockedBy A is rejected."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task A",
            description="First task",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task B",
            description="Blocked by A",
            blockedBy=["1"],
        )

        # Creating task that would create cycle: A -> B -> A
        with pytest.raises(CycleDetectedError):
            TasksService.update_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                task_id="1",
                addBlockedBy=["2"],
            )

    def test_reject_indirect_cycle_on_create(self, temp_project_dir, session_id):
        """Test that indirect cycles A -> B -> C -> A are rejected."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task A",
            description="First task",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task B",
            description="Blocked by A",
            blockedBy=["1"],
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task C",
            description="Blocked by B",
            blockedBy=["2"],
        )

        # Creating dependency that would create cycle: A -> B -> C -> A
        with pytest.raises(CycleDetectedError):
            TasksService.update_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                task_id="1",
                addBlockedBy=["3"],
            )

    def test_reject_cycle_on_update_add_blocks(self, temp_project_dir, session_id):
        """Test that adding blocks that would create a cycle is rejected."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task A",
            description="First task",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task B",
            description="Blocked by A",
            blockedBy=["1"],
        )

        # Try to make A blocked by B (which would create B -> A -> B cycle)
        with pytest.raises(CycleDetectedError):
            TasksService.update_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                task_id="2",
                addBlocks=["1"],
            )

    def test_self_reference_rejected(self, temp_project_dir, session_id):
        """Test that a task cannot block itself."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task A",
            description="First task",
        )

        with pytest.raises(CycleDetectedError):
            TasksService.update_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                task_id="1",
                addBlockedBy=["1"],
            )

    def test_valid_dependency_chain_allowed(self, temp_project_dir, session_id):
        """Test that valid dependency chains without cycles are allowed."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task A",
            description="First task",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task B",
            description="Blocked by A",
            blockedBy=["1"],
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task C",
            description="Blocked by B",
            blockedBy=["2"],
        )
        task4 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task D",
            description="Blocked by C",
            blockedBy=["3"],
        )

        # This should work: A -> B -> C -> D (no cycles)
        assert task4.blockedBy == ["3"]

    def test_multiple_blockers_no_cycle(self, temp_project_dir, session_id):
        """Test that a task can be blocked by multiple tasks without creating a cycle."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task A",
            description="First task",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task B",
            description="Second task",
        )
        task3 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task C",
            description="Blocked by both A and B",
            blockedBy=["1", "2"],
        )

        # This should work: A and B both block C independently
        assert set(task3.blockedBy) == {"1", "2"}


class TestConcurrency:
    """Test cases for concurrency handling with file locking."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-concurrent"

    def test_concurrent_task_creation_unique_ids(self, temp_project_dir, session_id):
        """Test that concurrent task creation produces unique IDs."""
        results = []
        errors = []
        num_threads = 10

        def create_task(thread_num):
            try:
                task = TasksService.create_task(
                    project_dir=temp_project_dir,
                    session_id=session_id,
                    subject=f"Task from thread {thread_num}",
                    description=f"Created by thread {thread_num}",
                )
                results.append(task.id)
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=create_task, args=(i,))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All IDs should be unique
        assert len(results) == num_threads
        assert len(set(results)) == num_threads

    def test_concurrent_updates_preserve_consistency(
        self, temp_project_dir, session_id
    ):
        """Test that concurrent updates to different tasks maintain consistency."""
        # Create initial tasks
        for i in range(5):
            TasksService.create_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                subject=f"Task {i + 1}",
                description=f"Description {i + 1}",
            )

        errors = []

        def update_task(task_id, new_status):
            try:
                TasksService.update_task(
                    project_dir=temp_project_dir,
                    session_id=session_id,
                    task_id=str(task_id),
                    status=new_status,
                )
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(5):
            t = threading.Thread(target=update_task, args=(i + 1, "in_progress"))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All tasks should be in_progress
        tasks = TasksService.list_tasks(temp_project_dir, session_id)
        for task in tasks:
            assert task.status == "in_progress"

    def test_lock_prevents_race_conditions(self, temp_project_dir, session_id):
        """Test that file locking prevents race conditions during updates."""
        # Create a task
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Race condition test",
            description="Initial description",
        )

        update_count = [0]  # Use list to allow modification in nested function
        errors = []

        def update_description(thread_num):
            try:
                for _ in range(5):
                    TasksService.update_task(
                        project_dir=temp_project_dir,
                        session_id=session_id,
                        task_id=task.id,
                        description=f"Updated by thread {thread_num}",
                    )
                    update_count[0] += 1
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(3):
            t = threading.Thread(target=update_description, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # All updates should have succeeded
        assert update_count[0] == 15  # 3 threads * 5 updates each

    def test_different_sessions_do_not_block_each_other(self, temp_project_dir):
        """Test that operations on different sessions don't block each other."""
        session_a = "session-a"
        session_b = "session-b"

        completed = {"a": False, "b": False}
        time.time()

        def create_in_session_a():
            for i in range(5):
                TasksService.create_task(
                    project_dir=temp_project_dir,
                    session_id=session_a,
                    subject=f"Session A Task {i}",
                    description="Created in session A",
                )
            completed["a"] = True

        def create_in_session_b():
            for i in range(5):
                TasksService.create_task(
                    project_dir=temp_project_dir,
                    session_id=session_b,
                    subject=f"Session B Task {i}",
                    description="Created in session B",
                )
            completed["b"] = True

        thread_a = threading.Thread(target=create_in_session_a)
        thread_b = threading.Thread(target=create_in_session_b)

        thread_a.start()
        thread_b.start()

        thread_a.join()
        thread_b.join()

        assert completed["a"]
        assert completed["b"]

        # Both sessions should have their tasks
        tasks_a = TasksService.list_tasks(temp_project_dir, session_a)
        tasks_b = TasksService.list_tasks(temp_project_dir, session_b)

        assert len(tasks_a) == 5
        assert len(tasks_b) == 5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-edge"

    def test_empty_subject_allowed(self, temp_project_dir, session_id):
        """Test that empty subject is allowed (model validation may change this)."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="",
            description="Task with empty subject",
        )
        assert task.subject == ""

    def test_empty_description_allowed(self, temp_project_dir, session_id):
        """Test that empty description is allowed."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task with empty description",
            description="",
        )
        assert task.description == ""

    def test_special_characters_in_subject(self, temp_project_dir, session_id):
        """Test that special characters in subject are handled correctly."""
        special_subject = (
            "Task with 'quotes' and \"double quotes\" and <angle> & {braces}"
        )
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject=special_subject,
            description="Testing special characters",
        )
        assert task.subject == special_subject

        # Verify it persists correctly
        retrieved = TasksService.get_task(temp_project_dir, session_id, task.id)
        assert retrieved.subject == special_subject

    def test_unicode_in_task_fields(self, temp_project_dir, session_id):
        """Test that Unicode characters are handled correctly."""
        unicode_subject = "Task with émojis 🚀 and 日本語"
        unicode_description = "Description with Ελληνικά and العربية"

        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject=unicode_subject,
            description=unicode_description,
        )

        assert task.subject == unicode_subject
        assert task.description == unicode_description

        retrieved = TasksService.get_task(temp_project_dir, session_id, task.id)
        assert retrieved.subject == unicode_subject
        assert retrieved.description == unicode_description

    def test_very_long_description(self, temp_project_dir, session_id):
        """Test handling of very long description."""
        long_description = "A" * 10000  # 10KB description

        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task with long description",
            description=long_description,
        )

        assert len(task.description) == 10000

        retrieved = TasksService.get_task(temp_project_dir, session_id, task.id)
        assert retrieved.description == long_description

    def test_add_blocked_by_nonexistent_task(self, temp_project_dir, session_id):
        """Test that adding blockedBy for non-existent task raises error."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Test task",
            description="Testing",
        )

        with pytest.raises(TaskNotFoundError, match="Task '999' not found"):
            TasksService.update_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                task_id=task.id,
                addBlockedBy=["999"],
            )

    def test_add_blocks_nonexistent_task(self, temp_project_dir, session_id):
        """Test that adding blocks for non-existent task raises error."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Test task",
            description="Testing",
        )

        with pytest.raises(TaskNotFoundError, match="Task '999' not found"):
            TasksService.update_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                task_id=task.id,
                addBlocks=["999"],
            )

    def test_id_sequence_continues_after_deletion(self, temp_project_dir, session_id):
        """Test that ID sequence continues even after deletions."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 1",
            description="First",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 2",
            description="Second",
        )

        # Delete task 2
        TasksService.delete_task(temp_project_dir, session_id, "2")

        # Next task should be ID 3, not 2
        task3 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 3",
            description="Third",
        )

        assert task3.id == "3"

    def test_duplicate_blocked_by_ignored(self, temp_project_dir, session_id):
        """Test that duplicate entries in blockedBy are handled gracefully."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 1",
            description="First",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 2",
            description="Second",
            blockedBy=["1"],
        )

        # Try to add same blockedBy again
        updated_task2 = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id="2",
            addBlockedBy=["1"],
        )

        # Should only have one entry
        assert updated_task2.blockedBy.count("1") == 1


class TestConcurrencyStress:
    """Stress tests for concurrent task operations."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-stress"

    def test_many_concurrent_creates(self, temp_project_dir, session_id):
        """Test 50 concurrent task creations produce unique IDs."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        errors = []
        num_tasks = 50

        def create_task(thread_num):
            try:
                task = TasksService.create_task(
                    project_dir=temp_project_dir,
                    session_id=session_id,
                    subject=f"Task from thread {thread_num}",
                    description=f"Created by thread {thread_num}",
                )
                return task.id
            except Exception as e:
                errors.append(str(e))
                return None

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(create_task, i) for i in range(num_tasks)]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        # No errors should have occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All IDs should be unique
        assert len(results) == num_tasks
        assert len(set(results)) == num_tasks

        # IDs should be 1-50 (though not necessarily in order)
        expected_ids = {str(i) for i in range(1, num_tasks + 1)}
        assert set(results) == expected_ids

    def test_concurrent_reads_during_writes(self, temp_project_dir, session_id):
        """Test reads are consistent while writes are happening."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Create initial tasks
        for i in range(10):
            TasksService.create_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                subject=f"Task {i + 1}",
                description=f"Initial description {i + 1}",
            )

        read_errors = []
        write_errors = []
        read_results = []

        def write_task(task_id):
            try:
                for j in range(5):
                    TasksService.update_task(
                        project_dir=temp_project_dir,
                        session_id=session_id,
                        task_id=str(task_id),
                        description=f"Updated description iteration {j}",
                    )
            except Exception as e:
                write_errors.append(str(e))

        def read_task(task_id):
            try:
                for _ in range(10):
                    task = TasksService.get_task(
                        temp_project_dir, session_id, str(task_id)
                    )
                    # Verify task data is valid (not partial/corrupted)
                    assert task.id == str(task_id)
                    assert task.subject == f"Task {task_id}"
                    # Description should be a complete string, not partial
                    assert (
                        "description" in task.description.lower()
                        or "Updated" in task.description
                    )
                    read_results.append(True)
            except Exception as e:
                read_errors.append(str(e))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            # Submit write operations
            for i in range(1, 6):
                futures.append(executor.submit(write_task, i))
            # Submit read operations concurrently
            for i in range(1, 11):
                futures.append(executor.submit(read_task, i))

            for future in as_completed(futures):
                future.result()

        assert len(write_errors) == 0, f"Write errors occurred: {write_errors}"
        assert len(read_errors) == 0, f"Read errors occurred: {read_errors}"
        # All reads should have succeeded
        assert len(read_results) == 100  # 10 tasks * 10 reads each

    def test_lock_timeout_handling(self, temp_project_dir, session_id):
        """Test that lock timeout raises appropriate error."""
        from filelock import Timeout

        # Create tasks directory
        tasks_dir = Path(temp_project_dir) / "tasks" / session_id
        tasks_dir.mkdir(parents=True, exist_ok=True)

        # Create a task first so the directory exists
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Initial task",
            description="For lock testing",
        )

        # Mock FileLock to always timeout
        with patch("services.tasks_service.FileLock") as mock_file_lock:
            mock_lock = MagicMock()
            mock_lock.__enter__ = MagicMock(side_effect=Timeout("lock_file"))
            mock_lock.__exit__ = MagicMock(return_value=False)
            mock_file_lock.return_value = mock_lock

            with pytest.raises(Timeout):
                TasksService.create_task(
                    project_dir=temp_project_dir,
                    session_id=session_id,
                    subject="Timeout test",
                    description="This should timeout",
                )


class TestComplexDependencyGraphs:
    """Tests for complex blocking relationships."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-deps"

    def test_deep_dependency_chain(self, temp_project_dir, session_id):
        """Test chain A -> B -> C -> D -> E works correctly."""
        # Create task A (root)
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task A",
            description="Root task",
        )

        # Create task B blocked by A
        task_b = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task B",
            description="Blocked by A",
            blockedBy=["1"],
        )

        # Create task C blocked by B
        task_c = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task C",
            description="Blocked by B",
            blockedBy=["2"],
        )

        # Create task D blocked by C
        task_d = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task D",
            description="Blocked by C",
            blockedBy=["3"],
        )

        # Create task E blocked by D
        task_e = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task E",
            description="Blocked by D",
            blockedBy=["4"],
        )

        # Verify initial chain: E blocked by D, D by C, C by B, B by A
        assert task_e.blockedBy == ["4"]
        assert task_d.blockedBy == ["3"]
        assert task_c.blockedBy == ["2"]
        assert task_b.blockedBy == ["1"]

        # Complete A - should unblock B
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id="1",
            status="completed",
        )
        task_b = TasksService.get_task(temp_project_dir, session_id, "2")
        assert "1" not in task_b.blockedBy

        # Complete B - should unblock C
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id="2",
            status="completed",
        )
        task_c = TasksService.get_task(temp_project_dir, session_id, "3")
        assert "2" not in task_c.blockedBy

        # Complete C - should unblock D
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id="3",
            status="completed",
        )
        task_d = TasksService.get_task(temp_project_dir, session_id, "4")
        assert "3" not in task_d.blockedBy

        # Complete D - should unblock E
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id="4",
            status="completed",
        )
        task_e = TasksService.get_task(temp_project_dir, session_id, "5")
        assert "4" not in task_e.blockedBy
        assert task_e.blockedBy == []

    def test_diamond_dependency(self, temp_project_dir, session_id):
        """Test diamond: A blocks B and C, both block D."""
        # Create A (root)
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task A",
            description="Root of diamond",
        )

        # Create B blocked by A
        task_b = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task B",
            description="Left branch",
            blockedBy=["1"],
        )

        # Create C blocked by A
        task_c = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task C",
            description="Right branch",
            blockedBy=["1"],
        )

        # Create D blocked by both B and C
        task_d = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task D",
            description="Bottom of diamond",
            blockedBy=["2", "3"],
        )

        # Verify initial state
        assert set(task_d.blockedBy) == {"2", "3"}
        assert task_b.blockedBy == ["1"]
        assert task_c.blockedBy == ["1"]

        # Complete A - should unblock B and C
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id="1",
            status="completed",
        )
        task_b = TasksService.get_task(temp_project_dir, session_id, "2")
        task_c = TasksService.get_task(temp_project_dir, session_id, "3")
        assert task_b.blockedBy == []
        assert task_c.blockedBy == []

        # D should still be blocked by B and C
        task_d = TasksService.get_task(temp_project_dir, session_id, "4")
        assert set(task_d.blockedBy) == {"2", "3"}

        # Complete B - D should still be blocked by C
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id="2",
            status="completed",
        )
        task_d = TasksService.get_task(temp_project_dir, session_id, "4")
        assert task_d.blockedBy == ["3"]

        # Complete C - D should now be unblocked
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id="3",
            status="completed",
        )
        task_d = TasksService.get_task(temp_project_dir, session_id, "4")
        assert task_d.blockedBy == []

    def test_complete_task_with_many_dependents(self, temp_project_dir, session_id):
        """Test completing a task with 10 dependent tasks."""
        # Create A (blocker)
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task A",
            description="Blocks many tasks",
        )

        # Create 10 tasks blocked by A
        dependent_ids = []
        for i in range(10):
            task = TasksService.create_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                subject=f"Dependent Task {i + 1}",
                description="Blocked by A",
                blockedBy=["1"],
            )
            dependent_ids.append(task.id)

        # Verify all 10 are blocked by A
        for task_id in dependent_ids:
            task = TasksService.get_task(temp_project_dir, session_id, task_id)
            assert "1" in task.blockedBy

        # Complete A - all 10 should be unblocked
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id="1",
            status="completed",
        )

        # Verify all 10 are now unblocked
        for task_id in dependent_ids:
            task = TasksService.get_task(temp_project_dir, session_id, task_id)
            assert "1" not in task.blockedBy

    def test_complex_cycle_detection(self, temp_project_dir, session_id):
        """Test cycle detection in complex graph A->B->C->D->B."""
        # Create chain A->B->C->D
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task A",
            description="Root",
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task B",
            description="Blocked by A",
            blockedBy=["1"],
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task C",
            description="Blocked by B",
            blockedBy=["2"],
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task D",
            description="Blocked by C",
            blockedBy=["3"],
        )

        # Try to add D->B (D blocks B), which would create cycle B->C->D->B
        with pytest.raises(CycleDetectedError):
            TasksService.update_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                task_id="4",
                addBlocks=["2"],
            )

        # Try to add B blockedBy D, which would create the same cycle
        with pytest.raises(CycleDetectedError):
            TasksService.update_task(
                project_dir=temp_project_dir,
                session_id=session_id,
                task_id="2",
                addBlockedBy=["4"],
            )


class TestOwnerAndMetadata:
    """Test cases for owner and metadata fields."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-owner-metadata"

    def test_task_has_owner_field_default_empty(self, temp_project_dir, session_id):
        """Test that Task has owner field with empty string default."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task without owner",
            description="Should have empty owner by default",
        )

        assert task.owner == ""

    def test_task_has_metadata_field_default_empty_dict(
        self, temp_project_dir, session_id
    ):
        """Test that Task has metadata field with empty dict default."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task without metadata",
            description="Should have empty metadata by default",
        )

        assert task.metadata == {}

    def test_create_task_with_metadata(self, temp_project_dir, session_id):
        """Test creating a task with initial metadata."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task with metadata",
            description="Has initial metadata",
            metadata={"priority": "high", "estimate_hours": 4},
        )

        assert task.metadata == {"priority": "high", "estimate_hours": 4}

    def test_update_task_owner(self, temp_project_dir, session_id):
        """Test updating task owner."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task to claim",
            description="Will be claimed by an agent",
        )

        updated_task = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            owner="agent-123",
        )

        assert updated_task.owner == "agent-123"

    def test_update_task_owner_persists(self, temp_project_dir, session_id):
        """Test that owner update is persisted to disk."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task to claim",
            description="Owner should persist",
        )

        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            owner="worker-abc",
        )

        # Read from disk directly
        task_file = Path(temp_project_dir) / "tasks" / session_id / f"{task.id}.json"
        with open(task_file) as f:
            saved_data = json.load(f)

        assert saved_data["owner"] == "worker-abc"

    def test_update_task_metadata_merge(self, temp_project_dir, session_id):
        """Test that metadata updates merge with existing metadata."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task with metadata",
            description="Metadata will be merged",
            metadata={"key1": "value1", "key2": "value2"},
        )

        updated_task = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            metadata={"key2": "updated", "key3": "new"},
        )

        assert updated_task.metadata == {
            "key1": "value1",
            "key2": "updated",
            "key3": "new",
        }

    def test_update_task_metadata_delete_key(self, temp_project_dir, session_id):
        """Test that setting metadata key to None deletes it."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task with metadata",
            description="Metadata key will be deleted",
            metadata={"key1": "value1", "key2": "value2", "key3": "value3"},
        )

        updated_task = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            metadata={"key2": None},
        )

        assert updated_task.metadata == {"key1": "value1", "key3": "value3"}

    def test_update_task_metadata_mixed_operations(self, temp_project_dir, session_id):
        """Test adding, updating, and deleting metadata keys in one operation."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task with metadata",
            description="Multiple metadata operations",
            metadata={
                "existing": "unchanged",
                "to_update": "old",
                "to_delete": "remove_me",
            },
        )

        updated_task = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            metadata={
                "to_update": "new",
                "to_delete": None,
                "new_key": "added",
            },
        )

        assert updated_task.metadata == {
            "existing": "unchanged",
            "to_update": "new",
            "new_key": "added",
        }

    def test_metadata_persists_to_disk(self, temp_project_dir, session_id):
        """Test that metadata is persisted correctly."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task with metadata",
            description="Metadata should persist",
            metadata={"complex": {"nested": "value"}, "list": [1, 2, 3]},
        )

        # Read from disk directly
        task_file = Path(temp_project_dir) / "tasks" / session_id / f"{task.id}.json"
        with open(task_file) as f:
            saved_data = json.load(f)

        assert saved_data["metadata"] == {
            "complex": {"nested": "value"},
            "list": [1, 2, 3],
        }

    def test_get_task_includes_owner_and_metadata(self, temp_project_dir, session_id):
        """Test that get_task returns owner and metadata fields."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task with all fields",
            description="Complete task",
            metadata={"key": "value"},
        )

        # Update owner
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id="1",
            owner="test-owner",
        )

        task = TasksService.get_task(temp_project_dir, session_id, "1")

        assert task.owner == "test-owner"
        assert task.metadata == {"key": "value"}

    def test_list_tasks_includes_owner_and_metadata(self, temp_project_dir, session_id):
        """Test that list_tasks returns tasks with owner and metadata."""
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 1",
            description="First task",
            metadata={"priority": 1},
        )
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task 2",
            description="Second task",
            metadata={"priority": 2},
        )

        # Set owner on first task
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id="1",
            owner="agent-1",
        )

        tasks = TasksService.list_tasks(temp_project_dir, session_id)

        assert len(tasks) == 2
        assert tasks[0].owner == "agent-1"
        assert tasks[0].metadata == {"priority": 1}
        assert tasks[1].owner == ""
        assert tasks[1].metadata == {"priority": 2}

    def test_empty_string_owner_accepted(self, temp_project_dir, session_id):
        """Test that setting owner to empty string works (unclaim)."""
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Task to unclaim",
            description="Will be unclaimed",
        )

        # Claim the task
        TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            owner="agent-1",
        )

        # Unclaim by setting empty string
        updated_task = TasksService.update_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            task_id=task.id,
            owner="",
        )

        assert updated_task.owner == ""


class TestDataIntegrity:
    """Tests for data integrity and persistence."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-integrity"

    def test_task_survives_service_restart(self, temp_project_dir, session_id):
        """Test tasks persist across service restarts (new service instance)."""
        # Create task with service (simulating first service instance)
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Persistent task",
            description="Should survive restart",
            activeForm="Testing persistence",
        )

        # Store the ID
        task_id = task.id

        # Simulate service restart by importing the module fresh
        # In practice, TasksService is stateless and reads from disk,
        # so we just verify we can read the task using a new call
        import importlib
        import sys

        ts_module = sys.modules.get("services.tasks_service")
        if ts_module is not None:
            importlib.reload(ts_module)

        # Get task using the "restarted" service
        retrieved_task = TasksService.get_task(
            temp_project_dir, session_id, task_id
        )

        assert retrieved_task.id == task_id
        assert retrieved_task.subject == "Persistent task"
        assert retrieved_task.description == "Should survive restart"
        assert retrieved_task.activeForm == "Testing persistence"

    def test_corrupted_task_file_handled(self, temp_project_dir, session_id):
        """Test handling of corrupted JSON in task file."""
        # Create a valid task
        task = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Valid task",
            description="Will be corrupted",
        )

        # Corrupt the JSON file
        task_file = Path(temp_project_dir) / "tasks" / session_id / f"{task.id}.json"
        task_file.write_text("{ invalid json content }")

        # Try to read the corrupted task - should raise an error
        with pytest.raises((json.JSONDecodeError, ValueError)):
            TasksService.get_task(temp_project_dir, session_id, task.id)

    def test_missing_lock_file_recreated(self, temp_project_dir, session_id):
        """Test lock file is recreated if deleted."""
        # Create a task (creates lock file)
        TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="First task",
            description="Creates lock file",
        )

        # Get the lock file path and delete it
        lock_file = Path(temp_project_dir) / "tasks" / session_id / ".lock"
        if lock_file.exists():
            lock_file.unlink()

        assert not lock_file.exists()

        # Create another task - should work and recreate lock file
        task2 = TasksService.create_task(
            project_dir=temp_project_dir,
            session_id=session_id,
            subject="Second task",
            description="Recreates lock file",
        )

        assert task2.id == "2"

        # The filelock library creates the lock file when acquiring
        # Verify operations still work
        tasks = TasksService.list_tasks(temp_project_dir, session_id)
        assert len(tasks) == 2
