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
        print(f"🔧 [ToolService] Registering tool: {tool_name}")
        cls._tools[tool_name] = tool_class
    
    @classmethod
    def execute_tool(cls, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with given parameters."""
        print(f"🚀 [ToolService] execute_tool called with tool_name='{tool_name}', kwargs={kwargs}")
        
        if tool_name not in cls._tools:
            print(f"❌ [ToolService] Tool '{tool_name}' not found. Available tools: {list(cls._tools.keys())}")
            return {
                "success": False,
                "message": f"Tool '{tool_name}' not found"
            }
        
        print(f"✅ [ToolService] Found tool '{tool_name}', executing...")
        
        try:
            tool_class = cls._tools[tool_name]
            print(f"🔧 [ToolService] Creating instance of {tool_class.__name__}")
            tool_instance = tool_class()
            
            # Create input schema instance with kwargs
            print(f"📝 [ToolService] Creating input schema with kwargs: {kwargs}")
            input_schema = tool_class.input_schema(**kwargs)
            print(f"📋 [ToolService] Input schema created: {input_schema}")
            
            print(f"⚡ [ToolService] Running tool...")
            result = tool_instance.run(input_schema)
            print(f"📤 [ToolService] Tool execution completed. Result: {result}")
            
            response = {
                "success": result.success,
                "message": result.message,
                "data": {
                    "total": result.total,
                    "projects": result.projects
                }
            }
            print(f"🎯 [ToolService] Returning response: {response}")
            return response
            
        except Exception as e:
            print(f"💥 [ToolService] Error executing tool '{tool_name}': {str(e)}")
            import traceback
            print(f"📚 [ToolService] Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "message": f"Error executing tool '{tool_name}': {str(e)}"
            }
    
    @classmethod
    def get_available_tools(cls) -> List[Dict[str, Any]]:
        """Get information about available tools."""
        print(f"📋 [ToolService] get_available_tools called")
        tools_info = []
        
        for tool_name, tool_class in cls._tools.items():
            tool_info = {
                "name": tool_name,
                "description": tool_class.__doc__ or f"Tool: {tool_name}",
                "input_schema": str(tool_class.input_schema),
                "output_schema": str(tool_class.output_schema)
            }
            tools_info.append(tool_info)
            print(f"🔧 [ToolService] Available tool: {tool_name} - {tool_info['description']}")
        
        print(f"📊 [ToolService] Returning {len(tools_info)} available tools")
        return tools_info

# Register the projects tool
print(f"🔧 [ToolService] Initializing and registering tools...")
ToolService.register_tool("projects", ProjectsTool)
print(f"✅ [ToolService] Tool registration completed. Registered tools: {list(ToolService._tools.keys())}") 