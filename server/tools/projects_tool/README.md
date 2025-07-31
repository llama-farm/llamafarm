# Projects Tool

A comprehensive tool for managing projects in LlamaFarm, supporting both listing and creating projects.

## Structure

This tool follows the Atomic Tool structure:

1. **Imports** - Required dependencies
2. **Input Schema** - Defines the input parameters
3. **Output Schema** - Defines the output structure
4. **Configuration** - Tool configuration
5. **Main Tool & Logic** - Core functionality
6. **Example Usage** - Usage examples

## Usage

### Basic Usage

```python
from tools.list_projects_tool.tool import ProjectsTool, ProjectsToolInput

# Create tool instance
tool = ProjectsTool()

# List projects in a namespace
result = tool.run(ProjectsToolInput(action="list", namespace="my_namespace"))
print(f"Found {result.total} projects")
for project in result.projects:
    print(f"  - {project['namespace']}/{project['project_id']}")

# Create a new project
result = tool.run(ProjectsToolInput(action="create", namespace="my_namespace", project_id="new_project"))
print(result.message)
```

### Via ToolService

```python
from services.tool_service import ToolService

# List projects
result = ToolService.execute_tool("projects", action="list", namespace="my_namespace")
print(result["message"])

# Create project
result = ToolService.execute_tool("projects", action="create", namespace="my_namespace", project_id="new_project")
print(result["message"])
```

## Input Parameters

- `action` (required): Either "list" or "create"
- `namespace` (required): The namespace to operate on
- `project_id` (optional): Required when action is "create"

## Output

- `success`: Boolean indicating if the operation was successful
- `message`: Human-readable message describing the results
- `projects`: List of project dictionaries (for list action) or created project (for create action)
- `total`: Total number of projects (for list action) or 1 (for successful create action)

## Actions

### List Projects
Lists all projects in the specified namespace.

### Create Project
Creates a new project in the specified namespace with a basic configuration file.

## Testing

Run the test script to verify functionality:

```bash
cd server/tools/projects_tool
python3 test_projects_tool.py
``` 