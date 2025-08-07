"""Tool service for the new atomic tool architecture"""

import logging
from typing import Any

from tools.core import get_tool, list_tools
from tools.core.errors import ToolExecutionError, ToolNotFoundError

from .analyzers import MessageAnalyzer
from .models import IntegrationType, ToolResult

logger = logging.getLogger(__name__)


class AtomicToolExecutor:
    """Handles tool execution using the new atomic tool architecture"""
    
    @staticmethod
    def execute_tool(tool_name: str, message: str, request_context: 
        dict[str, Any] | None = None) -> ToolResult:
        """
        Execute a tool by name using message analysis with optional request context.
        
        Args:
            tool_name: Name of the tool to execute
            message: User message to analyze for parameters
            request_context: Optional dict of additional context from request fields
            
        Returns:
            ToolResult with execution details
        """
        try:
            # Ensure tools are initialized before use
            if not ensure_tools_initialized():
                raise ToolExecutionError(
                    tool_name, "Tool registry not properly initialized"
                )
            
            # Get tool from registry
            tool = get_tool(tool_name)
            
            # Analyze message for parameters - each tool handles its own analysis
            if tool_name == "projects":
                input_dict = AtomicToolExecutor._extract_projects_input(
                    message, request_context or {})
                # Convert dict to ProjectsToolInput
                from tools.projects_tool.tool.projects_tool import ProjectsToolInput
                input_data = ProjectsToolInput(**input_dict)
            else:
                raise ToolExecutionError(
                    tool_name, 
                    f"Message analysis not implemented for tool: {tool_name}"
                    )
            
            # Execute tool
            result = tool.run(input_data)
            
            # Convert atomic_agents tool output to legacy ToolResult format
            return ToolResult(
                success=result.success,
                action=getattr(input_data, "action", "unknown"),
                namespace=getattr(input_data, "namespace", "unknown"),
                message=(
                    result.message 
                    if hasattr(result, 'message') 
                    else ("Success" if result.success else "Tool execution failed")
                    ),
                result=result,
                integration_type=IntegrationType.MANUAL
            )
            
        except ToolNotFoundError as e:
            logger.error(f"Tool not found: {e}")
            return ToolResult(
                success=False,
                action="unknown",
                namespace="unknown",
                message=f"Tool '{tool_name}' not found in registry",
                integration_type=IntegrationType.MANUAL_FAILED
            )
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                success=False,
                action="unknown",
                namespace="unknown",
                message=f"Tool execution failed: {str(e)}",
                integration_type=IntegrationType.MANUAL_FAILED
            )
    
    @staticmethod
    def _extract_projects_input(
        message: str, request_context: dict[str, Any]
        ) -> dict[str, Any]:
        """Extract input parameters for projects tool from message and request context"""
        # Extract projects-specific fields from generic request context
        request_namespace = request_context.get("namespace")
        request_project_id = request_context.get("project_id")
        
        # Use enhanced LLM-based analysis with request field support
        analysis = MessageAnalyzer.analyze_with_llm(
            message, request_namespace, request_project_id
        )
        
        input_data = {
            "action": analysis.action,
            "namespace": analysis.namespace
        }
        
        if analysis.action.lower() == "create" and analysis.project_id:
            input_data["project_id"] = analysis.project_id
        
        return input_data
    
    @staticmethod
    def get_available_tools() -> list[dict[str, Any]]:
        """Get information about all available tools from registry"""
        # Ensure tools are initialized before listing
        if not ensure_tools_initialized():
            return []
        
        available_tools = []
        
        for tool_name in list_tools():
            try:
                tool = get_tool(tool_name)
                info = tool.get_schema_info()
                
                # Format for API response
                tool_info = {
                    "name": tool_name,
                    "description": info["description"],
                    "version": info["metadata"]["version"],
                    "enabled": tool.health_check(),
                    "input_schema": info["input_schema"],
                    "output_schema": info["output_schema"],
                    "metadata": info["metadata"]
                }
                
                # Add projects-specific information for compatibility
                if tool_name == "projects":
                    tool_info.update({
                        "actions": ["list", "create"],
                        "parameters": {
                            "action": "Required: 'list' or 'create'",
                            "namespace": "Required: namespace string",
                            "project_id": (
                                "Required for create action: project identifier"
                                )
                        },
                        "examples": [
                            "List my projects",
                            "Show projects in <name> namespace",
                            "List how many projects I have in <name>",
                            "Create a new project called my_app",
                            "Create project demo in test namespace"
                        ]
                    })
                
                available_tools.append(tool_info)
                
            except Exception as e:
                logger.error(f"Failed to get info for tool '{tool_name}': {e}")
                available_tools.append({
                    "name": tool_name,
                    "description": f"Error loading tool: {e}",
                    "enabled": False,
                    "error": str(e)
                })
        
        return available_tools
    
    @staticmethod
    def health_check_all() -> dict[str, bool]:
        """Perform health checks on all registered tools"""
        # Ensure tools are initialized before health checking
        if not ensure_tools_initialized():
            return {}
        
        health_status = {}
        
        for tool_name in list_tools():
            try:
                tool = get_tool(tool_name)
                health_status[tool_name] = tool.health_check()
            except Exception as e:
                logger.error(f"Health check failed for tool '{tool_name}': {e}")
                health_status[tool_name] = False
        
        return health_status


class ToolRegistryManager:
    """Manages tool registration and initialization"""
    
    @staticmethod
    def initialize_tools():
        """Initialize and register all available tools"""
        try:
            logger.info("Starting tool initialization...")
            
            # Force import and instantiate tools to ensure @tool decorator runs
            from tools.projects_tool.tool.projects_tool import ProjectsTool
            
            # Create instance to trigger registration if needed
            projects_tool = ProjectsTool()
            logger.info(f"ProjectsTool instantiated: {projects_tool}")
            
            # Verify registration worked
            registered_tools = list_tools()
            logger.info(f"Tools registered after import: {registered_tools}")
            
            if "projects" not in registered_tools:
                logger.warning("Projects tool not found in registry after import")
                # Try manual registration if auto-registration failed
                from tools.core import register_tool
                register_tool("projects", projects_tool)
                logger.info("Manually registered projects tool")
                
                # Verify again
                registered_tools = list_tools()
                logger.info(f"Tools registered after manual registration: {registered_tools}")
            
            # Add future tool imports here
            # from tools.documents_tool.tool import DocumentsTool
            # documents_tool = DocumentsTool()
            # from tools.analytics_tool.tool import AnalyticsTool
            # analytics_tool = AnalyticsTool()
            
            # Final verification
            final_tools = list_tools()
            logger.info(
                f"Successfully initialized {len(final_tools)} tools: {final_tools}"
            )
            
            return len(final_tools) > 0
            
        except ImportError as e:
            logger.error(f"Failed to import tools: {e}")
            logger.error(
                "Check that tools.projects_tool.tool.projects_tool module exists"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            logger.exception("Full traceback:")
            return False
    
    @staticmethod
    def get_registry_status() -> dict[str, Any]:
        """Get status information about the tool registry"""
        try:
            # Ensure tools are initialized before getting status
            if not ensure_tools_initialized():
                return {
                    "total_tools": 0,
                    "registered_tools": [],
                    "health_status": {},
                    "healthy_tools": 0,
                    "registry_available": False,
                    "error": "Tool registry not initialized"
                }
            
            tools = list_tools()
            health_status = AtomicToolExecutor.health_check_all()
            
            return {
                "total_tools": len(tools),
                "registered_tools": tools,
                "health_status": health_status,
                "healthy_tools": sum(bool(status) for status in health_status.values()),
                "registry_available": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get registry status: {e}")
            return {
                "total_tools": 0,
                "registered_tools": [],
                "health_status": {},
                "healthy_tools": 0,
                "registry_available": False,
                "error": str(e)
            }


# Module initialization state
_tools_initialized = False

def ensure_tools_initialized():
    """Ensure tools are initialized exactly once. Call this explicitly when needed."""
    global _tools_initialized
    if not _tools_initialized:
        try:
            logger.info("Initializing tools on first use...")
            success = ToolRegistryManager.initialize_tools()
            if success:
                logger.info("Tool initialization completed successfully")
                _tools_initialized = True
            else:
                logger.warning("Tool initialization completed with issues")
                _tools_initialized = False
        except Exception as e:
            logger.error(f"Tool initialization failed: {e}")
            _tools_initialized = False
    return _tools_initialized