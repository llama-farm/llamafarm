# Imports - Fixed for v2.0
import logging
import os
from pathlib import Path
from typing import Any, Literal

from atomic_agents import BaseIOSchema, BaseTool

# Import the project service
from services.project_service import ProjectService

# Use standard logging for compatibility 
logger = logging.getLogger(__name__)

# Input Schema
class ProjectsToolInput(BaseIOSchema):
    """Input schema for projects tool."""
    action: Literal["list", "create"]
    namespace: str
    project_id: str | None = None

# Output Schema
class ProjectsToolOutput(BaseIOSchema):
    """Output schema for projects tool."""
    success: bool
    message: str
    projects: list[dict[str, Any]] | None = None
    total: int | None = None

# Main Tool & Logic
class ProjectsTool(BaseTool[ProjectsToolInput, ProjectsToolOutput]):
    """Tool for managing projects - list and create operations."""
    
    def __init__(self):
        super().__init__()

    def list_projects(self, namespace: str) -> list[dict[str, Any]]:
        """List all projects in a namespace using the project service."""
        try:
            projects = ProjectService.list_projects(namespace)
            return [
                {
                    "namespace": project.namespace,
                    "project_id": project.name,
                    "path": ProjectService.get_project_dir(
                        project.namespace, project.name
                    )
                }
                for project in projects
            ]
        except Exception as e:
            # If the namespace directory doesn't exist, return empty list
            return []

    def create_project(self, namespace: str, project_id: str) -> dict[str, Any]:
        """Create a new project in the specified namespace using the project service."""
        try:
            # Check if project already exists
            project_dir = ProjectService.get_project_dir(namespace, project_id)
            if os.path.exists(project_dir):
                return {
                    "success": False,
                    "message": (
                        f"Project '{project_id}' already exists in namespace "
                        f"'{namespace}'"
                    )
                }
            
            # Create the project using the project service
            config = ProjectService.create_project(namespace, project_id)
            
            return {
                "success": True,
                "message": (
                    f"Project '{project_id}' created successfully in namespace "
                    f"'{namespace}'"
                ),
                "project": {
                    "namespace": namespace,
                    "project_id": project_id,
                    "path": project_dir
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating project: {str(e)}"
            }

    def run(self, input: ProjectsToolInput) -> ProjectsToolOutput:
        """Execute the projects tool based on the specified action."""
        try:
            logger.info(
                f"Executing projects tool - action: {input.action}, "
                f"namespace: {input.namespace}, "
                f"project_id: {getattr(input, 'project_id', None)}"
            )
            
            if input.action == "list":
                projects = self.list_projects(input.namespace)
                logger.info(
                    f"Listed projects successfully - namespace: {input.namespace}, "
                    f"project_count: {len(projects)}"
                )
                return ProjectsToolOutput(
                    success=True,
                    message=(
                        f"Found {len(projects)} projects in namespace "
                        f"'{input.namespace}'"
                    ),
                    projects=projects,
                    total=len(projects)
                )
            
            elif input.action == "create":
                if not input.project_id:
                    logger.warning(
                        f"Create action attempted without project_id - "
                        f"namespace: {input.namespace}"
                    )
                    return ProjectsToolOutput(
                        success=False,
                        message="project_id is required for create action"
                    )
                
                result = self.create_project(input.namespace, input.project_id)
                if result["success"]:
                    logger.info(
                        f"Project created successfully - namespace: {input.namespace}, "
                        f"project_id: {input.project_id}"
                    )
                    return ProjectsToolOutput(
                        success=True,
                        message=result["message"],
                        projects=[result["project"]],
                        total=1
                    )
                else:
                    logger.warning(
                        f"Project creation failed - namespace: {input.namespace}, "
                        f"project_id: {input.project_id}, error: {result['message']}"
                    )
                    return ProjectsToolOutput(
                        success=False,
                        message=result["message"]
                    )
            
            else:
                logger.error(
                    f"Unknown action attempted - action: {input.action}, "
                    f"namespace: {input.namespace}"
                )
                return ProjectsToolOutput(
                    success=False,
                    message=f"Unknown action: {input.action}"
                )
                
        except Exception as e:
            logger.exception(
                f"Unexpected error in projects tool execution - "
                f"action: {input.action}, "
                f"namespace: {input.namespace}, "
                f"project_id: {getattr(input, 'project_id', None)}, error: {str(e)}"
            )
            # Re-raise for critical errors that should be handled upstream
            # But return graceful error for user-facing responses
            return ProjectsToolOutput(
                success=False,
                message=f"Error executing projects tool: {str(e)}"
            )

# Example Usage
if __name__ == "__main__":
    # Example: List projects
    tool = ProjectsTool()
    result = tool.run(ProjectsToolInput(action="list", namespace="test"))
    logger.info(f"List result: {result.message}")
    
    # Example: Create project
    result = tool.run(
        ProjectsToolInput(action="create", namespace="test", project_id="new_project")
    )
    logger.info(f"Create result: {result.message}")