# Imports
from atomic_agents.base import BaseIOSchema
from atomic_agents.base import BaseTool, BaseToolConfig
from typing import List, Dict, Any, Optional, Literal
import os
from pathlib import Path

# Import the project service
from server.services.project_service import ProjectService

# Input Schema
class ProjectsToolInput(BaseIOSchema):
    """Input schema for projects tool."""
    action: Literal["list", "create"]
    namespace: str
    project_id: Optional[str] = None

# Output Schema
class ProjectsToolOutput(BaseIOSchema):
    """Output schema for projects tool."""
    success: bool
    message: str
    projects: Optional[List[Dict[str, Any]]] = None
    total: Optional[int] = None

# Main Tool & Logic
class ProjectsTool(BaseTool):
    input_schema = ProjectsToolInput
    output_schema = ProjectsToolOutput

    def __init__(self):
        super().__init__()

    def list_projects(self, namespace: str) -> List[Dict[str, Any]]:
        """List all projects in a namespace using the project service."""
        try:
            projects = ProjectService.list_projects(namespace)
            return [
                {
                    "namespace": project.namespace,
                    "project_id": project.name,
                    "path": ProjectService.get_project_dir(project.namespace, project.name)
                }
                for project in projects
            ]
        except Exception as e:
            # If the namespace directory doesn't exist, return empty list
            return []

    def create_project(self, namespace: str, project_id: str) -> Dict[str, Any]:
        """Create a new project in the specified namespace using the project service."""
        try:
            # Check if project already exists
            project_dir = ProjectService.get_project_dir(namespace, project_id)
            if os.path.exists(project_dir):
                return {
                    "success": False,
                    "message": f"Project '{project_id}' already exists in namespace '{namespace}'"
                }
            
            # Create the project using the project service
            config = ProjectService.create_project(namespace, project_id)
            
            return {
                "success": True,
                "message": f"Project '{project_id}' created successfully in namespace '{namespace}'",
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
            if input.action == "list":
                projects = self.list_projects(input.namespace)
                return ProjectsToolOutput(
                    success=True,
                    message=f"Found {len(projects)} projects in namespace '{input.namespace}'",
                    projects=projects,
                    total=len(projects)
                )
            
            elif input.action == "create":
                if not input.project_id:
                    return ProjectsToolOutput(
                        success=False,
                        message="project_id is required for create action"
                    )
                
                result = self.create_project(input.namespace, input.project_id)
                if result["success"]:
                    return ProjectsToolOutput(
                        success=True,
                        message=result["message"],
                        projects=[result["project"]],
                        total=1
                    )
                else:
                    return ProjectsToolOutput(
                        success=False,
                        message=result["message"]
                    )
            
            else:
                return ProjectsToolOutput(
                    success=False,
                    message=f"Unknown action: {input.action}"
                )
                
        except Exception as e:
            return ProjectsToolOutput(
                success=False,
                message=f"Error executing projects tool: {str(e)}"
            )

# Example Usage
if __name__ == "__main__":
    # Example: List projects
    tool = ProjectsTool()
    result = tool.run(ProjectsToolInput(action="list", namespace="test"))
    print(f"List result: {result.message}")
    
    # Example: Create project
    result = tool.run(ProjectsToolInput(action="create", namespace="test", project_id="new_project"))
    print(f"Create result: {result.message}") 