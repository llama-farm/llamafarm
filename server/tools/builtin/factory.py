"""
Builtin Tool Factory.

Creates dynamic builtin tool classes with injected context (project_dir, session_id).
Follows the same pattern as MCPToolFactory for consistency.
"""

from atomic_agents import BaseTool

from tools.builtin.tasks_tool import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TasksTool,
    TaskUpdateTool,
)


class BuiltinToolFactory:
    """Factory for creating built-in tools with injected context."""

    def __init__(self, project_dir: str, session_id: str | None):
        """Initialize the factory with context.

        Args:
            project_dir: Path to the project directory
            session_id: Session ID for task persistence. If None, stateless mode.
        """
        self._project_dir = project_dir
        self._session_id = session_id

    def _create_injected_tool(
        self, base_class: type[BaseTool]
    ) -> type[BaseTool] | None:
        """Create a tool class with context injected.

        Args:
            base_class: The base tool class to inject context into.

        Returns:
            Tool class with _project_dir and _session_id set, or None if no session_id.
        """
        if self._session_id is None:
            return None

        project_dir = self._project_dir
        session_id = self._session_id

        class InjectedTool(base_class):  # type: ignore[valid-type]
            _project_dir = project_dir
            _session_id = session_id

        InjectedTool.__name__ = base_class.__name__
        InjectedTool.__qualname__ = base_class.__qualname__

        return InjectedTool

    def create_task_create_tool(self) -> type[BaseTool] | None:
        """Create task_create tool class with context injected.

        Returns:
            Tool class with _project_dir and _session_id set, or None if no session_id.
        """
        return self._create_injected_tool(TaskCreateTool)

    def create_task_update_tool(self) -> type[BaseTool] | None:
        """Create task_update tool class with context injected.

        Returns:
            Tool class with _project_dir and _session_id set, or None if no session_id.
        """
        return self._create_injected_tool(TaskUpdateTool)

    def create_task_list_tool(self) -> type[BaseTool] | None:
        """Create task_list tool class with context injected.

        Returns:
            Tool class with _project_dir and _session_id set, or None if no session_id.
        """
        return self._create_injected_tool(TaskListTool)

    def create_task_get_tool(self) -> type[BaseTool] | None:
        """Create task_get tool class with context injected.

        Returns:
            Tool class with _project_dir and _session_id set, or None if no session_id.
        """
        return self._create_injected_tool(TaskGetTool)

    def create_tasks_tool(self) -> type[BaseTool] | None:
        """DEPRECATED: Use create_task_create_tool() instead.

        This method is kept for backwards compatibility. It returns the same
        result as create_task_create_tool() using the legacy TasksTool class.

        Returns:
            Tool class with _project_dir and _session_id set, or None if no session_id.
        """
        return self._create_injected_tool(TasksTool)

    def create_all_tools(self) -> list[type[BaseTool]]:
        """Create all built-in tools.

        Returns:
            List of tool classes with context injected.
            Tools that require session context are excluded in stateless mode.
        """
        tools: list[type[BaseTool]] = []
        for creator in [
            self.create_task_create_tool,
            self.create_task_update_tool,
            self.create_task_list_tool,
            self.create_task_get_tool,
        ]:
            tool = creator()
            if tool is not None:
                tools.append(tool)
        return tools
