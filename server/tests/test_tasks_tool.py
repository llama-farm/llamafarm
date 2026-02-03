"""
Tests for TasksTool legacy compatibility and TaskToolOutput schema.

Operation dispatch, factory, registry, and integration tests are
covered comprehensively in test_builtin_tools.py. This file retains
only tests for the backwards-compatibility shims and the shared
output schema.
"""

import pytest

from tools.builtin.tasks_tool import (
    TaskCreateInput,
    TaskCreateTool,
    TasksTool,
    TasksToolInput,
    TasksToolOutput,
    TaskToolOutput,
)


class TestTaskToolOutputSchema:
    """Tests for the shared TaskToolOutput schema."""

    def test_output_schema_requires_result(self):
        """Test that result field is required."""
        with pytest.raises(ValueError):
            TaskToolOutput()

    def test_output_schema_accepts_json_string(self):
        """Test that output schema accepts a JSON string."""
        output = TaskToolOutput(result='{"success": true}')
        assert output.result == '{"success": true}'


class TestLegacyCompatibility:
    """Tests for deprecated aliases kept for backwards compatibility."""

    def test_tasks_tool_input_alias(self):
        """TasksToolInput is an alias for TaskCreateInput."""
        assert TasksToolInput is TaskCreateInput

    def test_tasks_tool_output_alias(self):
        """TasksToolOutput is an alias for TaskToolOutput."""
        assert TasksToolOutput is TaskToolOutput

    def test_tasks_tool_class_is_deprecated_shim(self):
        """TasksTool inherits from TaskCreateTool and keeps tool_name 'tasks'."""
        assert issubclass(TasksTool, TaskCreateTool)
        tool = TasksTool.__new__(TasksTool)
        assert tool.tool_name == "tasks"
