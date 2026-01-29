"""
Builtin Tool Factory.

Creates dynamic builtin tool classes with injected context (project_dir, session_id).
Follows the same pattern as MCPToolFactory for consistency.
"""

from atomic_agents import BaseTool

from tools.builtin.tasks_tool import TasksTool


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

    def create_tasks_tool(self) -> type[BaseTool] | None:
        """Create tasks tool class with context injected.

        Returns:
            Tool class with _project_dir and _session_id set, or None if no session_id.
        """
        if self._session_id is None:
            return None

        # Create a new class with injected context
        # This follows atomic-agents pattern where tool classes have class-level config
        project_dir = self._project_dir
        session_id = self._session_id

        class InjectedTasksTool(TasksTool):
            """TasksTool with injected project_dir and session_id."""

            _project_dir = project_dir
            _session_id = session_id

        return InjectedTasksTool

    def create_all_tools(self) -> list[type[BaseTool]]:
        """Create all built-in tools.

        Returns:
            List of tool classes with context injected.
            Tools that require session context are excluded in stateless mode.
        """
        tools: list[type[BaseTool]] = []

        # Add tasks tool if we have a session
        tasks_tool = self.create_tasks_tool()
        if tasks_tool is not None:
            tools.append(tasks_tool)

        return tools
