import json

from atomic_agents.context import BaseDynamicContextProvider

from core.logging import FastAPIStructLogger
from services.project_service import ProjectService

logger = FastAPIStructLogger(__name__)


class ProjectContextProvider(BaseDynamicContextProvider):
    def __init__(self, title: str, namespace: str, name: str):
        super().__init__(title=title)
        self._namespace = namespace
        self._name = name

    def get_info(self) -> str:
        try:
            current_config = ProjectService.get_project(self._namespace, self._name)
            if isinstance(current_config.config, dict):
                config_json = json.dumps(current_config.config)
            else:
                config_json = current_config.config.model_dump_json()
            return f"""
        ## PROJECT CONTEXT
        You are interacting with a specific LlamaFarm project.
        The current project's namespace is provided within a <namespace></namespace> XML tag.
        The current project's name is provided within a <name></name> XML tag.
        The current project's configuration is provided as JSON content within a <project_config></project_config> XML tag.

        <namespace>{current_config.namespace}</namespace>
        <name>{current_config.name}</name>
        <project_config>{config_json}</project_config>
        """
        except Exception:
            # Gracefully degrade if project doesn't exist (e.g., deleted but still in X-Active-Project header)
            logger.warning(
                f"Failed to load project context for {self._namespace}/{self._name}. "
                "Project may have been deleted. Continuing without project context.",
                exc_info=True,
            )
            return ""
