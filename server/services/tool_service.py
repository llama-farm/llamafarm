import sys
from pathlib import Path
from typing import Dict, Any, List
from tools.projects_tool.tool import ProjectsTool

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

class ToolService:
    """
    Service for managing and executing tools.
    """
    
    _tools = {}
    
    @classmethod
    def register_tool(cls, tool_name: str, tool_class):
        """Register a tool with the service."""
        cls._tools[tool_name] = tool_class
    
    @classmethod
    def execute_tool(cls, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with given parameters."""
        if tool_name not in cls._tools:
            return {
                "success": False,
                "message": f"Tool '{tool_name}' not found"
            }
        
        try:
            tool_class = cls._tools[tool_name]
            tool_instance = tool_class()
            
            # Create input schema instance with kwargs
            input_schema = tool_class.input_schema(**kwargs)
            result = tool_instance.run(input_schema)
            
            return {
                "success": result.success,
                "message": result.message,
                "data": {
                    "total": result.total,
                    "projects": result.projects
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error executing tool '{tool_name}': {str(e)}"
            }
    
    @classmethod
    def get_available_tools(cls) -> List[Dict[str, Any]]:
        """Get information about available tools."""
        tools_info = []
        
        for tool_name, tool_class in cls._tools.items():
            tools_info.append({
                "name": tool_name,
                "description": tool_class.__doc__ or f"Tool: {tool_name}",
                "input_schema": str(tool_class.input_schema),
                "output_schema": str(tool_class.output_schema)
            })
        
        return tools_info

# Register the projects tool
ToolService.register_tool("projects", ProjectsTool) 