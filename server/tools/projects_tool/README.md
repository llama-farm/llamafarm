# Projects Tool

A comprehensive tool for managing projects in LlamaFarm, supporting both listing and creating projects across different namespaces. This tool is integrated with the chat agent and can be used both programmatically and through natural language queries.

## Overview

The Projects Tool provides:
- **List Projects**: View all projects in a specified namespace
- **Create Projects**: Create new projects with default configuration
- **Namespace Management**: Organize projects into logical groups
- **Chat Integration**: Natural language processing for project operations

## Structure

This tool follows the Atomic Tool structure:

1. **Imports** - Required dependencies (`atomic_agents`, `services`)
2. **Input Schema** - Defines the input parameters (`ProjectsToolInput`)
3. **Output Schema** - Defines the output structure (`ProjectsToolOutput`)
4. **Main Tool & Logic** - Core functionality (`ProjectsTool`)
5. **Integration** - Chat agent integration with manual execution fallback

## Usage

### Programmatic Usage

```python
from tools.projects_tool.tool import ProjectsTool, ProjectsToolInput

# Create tool instance
tool = ProjectsTool()

# List projects in a namespace
result = tool.run(ProjectsToolInput(action="list", namespace="my_namespace"))
print(f"Found {result.total} projects")
for project in result.projects:
    print(f"  - {project['project_id']} at {project['path']}")

# Create a new project
result = tool.run(ProjectsToolInput(action="create", namespace="my_namespace", project_id="new_project"))
print(result.message)
```

### Chat Agent Integration

The tool is automatically integrated with the chat agent and supports natural language queries:

```
User: "how many projects are in the rmo namespace?"
Agent: "I found 4 project(s) in the 'rmo' namespace:
• workloft
• test  
• doggo
• test123"

User: "create project myapp in test namespace"
Agent: "✅ Successfully created project 'myapp' in namespace 'test'"
```

### Supported Query Patterns

The tool recognizes various natural language patterns:

**List Projects:**
- "how many projects are in the rmo namespace?"
- "list projects in test namespace"
- "show projects from rmo"
- "projects in my_namespace"

**Create Projects:**
- "create project myapp in test namespace"
- "create a new project called demo"
- "make project test123 in rmo namespace"

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

**Example Output:**
```json
{
  "success": true,
  "message": "Found 4 projects in namespace 'rmo'",
  "projects": [
    {
      "namespace": "rmo",
      "project_id": "workloft",
      "path": "/path/to/projects/rmo/workloft"
    }
  ],
  "total": 4
}
```

### Create Project
Creates a new project in the specified namespace with a basic configuration file.

**Example Output:**
```json
{
  "success": true,
  "message": "Project 'myapp' created successfully in namespace 'test'",
  "projects": [
    {
      "namespace": "test",
      "project_id": "myapp",
      "path": "/path/to/projects/test/myapp"
    }
  ],
  "total": 1
}
```

## Integration Details

### Chat Agent Integration
The tool is integrated with the chat agent through the inference router (`server/api/routers/inference/inference.py`):

1. **Native Tool Calling**: Uses `atomic_agents` for direct tool integration
2. **Manual Execution Fallback**: Falls back to manual execution when native calling fails
3. **Namespace Extraction**: Automatically extracts namespace from natural language queries
4. **Response Formatting**: Formats tool results into user-friendly responses

### Namespace Extraction
The tool automatically extracts namespaces from various query patterns:
- `"in rmo namespace"` → extracts "rmo"
- `"from test"` → extracts "test"  
- `"rmo namespace"` → extracts "rmo"
- Defaults to "test" if no namespace is specified

### Error Handling
- **Project Already Exists**: Returns clear error message when trying to create duplicate projects
- **Missing Dependencies**: Graceful fallback when `atomic_agents` is not available
- **Invalid Namespace**: Handles edge cases and provides helpful error messages

## Testing

### Manual Testing
```bash
cd server
source .venv/bin/activate

# Test direct tool usage
python -c "
from tools.projects_tool.tool import ProjectsTool, ProjectsToolInput
tool = ProjectsTool()
result = tool.run(ProjectsToolInput(action='list', namespace='rmo'))
print(result.message)
"
```

### Chat Integration Testing
```bash
# Start the server
./start_server.sh

# Test via chat endpoint
curl -X POST http://localhost:8000/v1/inference/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: test-123" \
  -d '{"message": "how many projects are in the rmo namespace?"}'
```

### Agent Status Check
```bash
curl http://localhost:8000/v1/inference/agent-status
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the virtual environment is activated
   ```bash
   source .venv/bin/activate
   ```

2. **Tool Not Executing**: Check if `atomic_agents` is available
   ```bash
   curl http://localhost:8000/v1/inference/agent-status
   # Look for "atomic_agents_available": true
   ```

3. **Wrong Namespace**: The tool now correctly extracts namespaces from natural language queries

4. **Project Creation Fails**: Check if the project already exists or if there are permission issues

### Debug Information
The tool provides detailed logging:
- `🔧 [Manual Tool]` - Manual execution logs
- `✅ [Inference]` - Successful operations
- `❌ [Inference]` - Error conditions

## Dependencies

- `atomic_agents` - For tool integration
- `services.project_service` - For project management
- `config` - For configuration management

## File Structure

```
server/tools/projects_tool/
├── README.md
├── tool/
│   ├── __init__.py
│   └── projects_tool.py
└── test_projects_tool.py
``` 