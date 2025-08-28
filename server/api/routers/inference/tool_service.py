"""Tool service for the new atomic tool architecture"""

import logging
from typing import Any

from tools.core import get_tool, list_tools
from tools.core.errors import ToolExecutionError, ToolNotFoundError

from .analyzers import MessageAnalyzer, IntentClassifier
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
            elif tool_name == "project_config":
                input_dict = AtomicToolExecutor._extract_project_config_input(
                    message, request_context or {})
                # Convert dict to ProjectConfigToolInput
                from tools.project_config_tool.tool.project_config_tool import (
                    ProjectConfigToolInput
                )
                input_data = ProjectConfigToolInput(**input_dict)
            elif tool_name == "prompt_engineering":
                input_dict = AtomicToolExecutor._extract_prompt_engineering_input(
                    message, request_context or {})
                # Convert dict to PromptEngineeringToolInput
                from tools.prompt_engineering_tool.tool.prompt_engineering_tool import (
                    PromptEngineeringToolInput
                )
                input_data = PromptEngineeringToolInput(**input_dict)
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
    def execute_with_llm(message: str, request_context: dict[str, Any] | None = None) -> ToolResult:
        """LLM-first execution: classify intent → route to tool → run.

        Falls back to manual failure with an explanatory message on errors.
        """
        try:
            if not ensure_tools_initialized():
                return ToolResult(
                    success=False,
                    action="unknown",
                    namespace="unknown",
                    message="Tool system not available",
                    integration_type=IntegrationType.MANUAL_FAILED,
                )

            available = list_tools()
            classifier = IntentClassifier()

            namespace = (request_context or {}).get("namespace") if request_context else None
            project_id = (request_context or {}).get("project_id") if request_context else None

            intent = classifier.classify(
                message=message,
                available_tools=available,
                request_namespace=namespace,
                request_project_id=project_id,
            )

            # Build tool input dict from intent
            input_dict: dict[str, Any] = {
                "action": intent.action,
            }
            if intent.namespace is not None:
                input_dict["namespace"] = intent.namespace
            if intent.project_id is not None:
                input_dict["project_id"] = intent.project_id
            if intent.parameters:
                input_dict.update(intent.parameters)

            # Instantiate correct input model and tool instance
            tool = get_tool(intent.tool_name)
            if intent.tool_name == "projects":
                from tools.projects_tool.tool.projects_tool import ProjectsToolInput
                input_data = ProjectsToolInput(**input_dict)
            elif intent.tool_name == "project_config":
                from tools.project_config_tool.tool.project_config_tool import ProjectConfigToolInput
                input_data = ProjectConfigToolInput(**input_dict)
            elif intent.tool_name == "prompt_engineering":
                from tools.prompt_engineering_tool.tool.prompt_engineering_tool import PromptEngineeringToolInput
                input_data = PromptEngineeringToolInput(**input_dict)
            else:
                return ToolResult(
                    success=False,
                    action="unknown",
                    namespace=input_dict.get("namespace", "unknown"),
                    message=f"Unsupported tool selected by classifier: {intent.tool_name}",
                    integration_type=IntegrationType.MANUAL_FAILED,
                )

            result = tool.run(input_data)
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
                integration_type=IntegrationType.MANUAL,
            )
        except Exception as e:
            logger.error("LLM-first execution failed: %s", str(e))
            return ToolResult(
                success=False,
                action="unknown",
                namespace=(request_context or {}).get("namespace", "unknown") if request_context else "unknown",
                message=f"LLM-based routing failed: {str(e)}",
                integration_type=IntegrationType.MANUAL_FAILED,
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
    def _extract_project_config_input(
        message: str, request_context: dict[str, Any]
        ) -> dict[str, Any]:
        """Extract input parameters for project config tool from message and request context"""
        # Extract projects-specific fields from generic request context
        request_namespace = request_context.get("namespace")
        request_project_id = request_context.get("project_id")

        # Analyze message to determine the action and intent
        message_lower = message.lower()

        # Determine action based on message keywords
        action = "analyze_config"  # default action

        schema_keywords = ["schema", "structure", "format", "documentation", "what can", "options"]
        analyze_keywords = ["analyze", "status", "current", "what's configured", "overview"]
        suggest_keywords = ["suggest", "recommend", "how to", "want to", "should i"]
        modify_keywords = ["change", "modify", "update", "set", "switch", "configure"]

        if any(word in message_lower for word in schema_keywords):
            action = "get_schema"
        elif any(word in message_lower for word in analyze_keywords):
            action = "analyze_config"
        elif any(word in message_lower for word in suggest_keywords):
            action = "suggest_changes"
        elif any(word in message_lower for word in modify_keywords):
            action = "modify_config"

        input_data = {
            "action": action,
            "namespace": request_namespace or "unknown",
            "project_id": request_project_id or "unknown",
        }

        # Add user intent for actions that need it
        if action in ["suggest_changes", "modify_config"]:
            input_data["user_intent"] = message

        # For modify_config, we'd need to parse specific changes from the message
        # For now, we'll let the tool handle suggestion generation
        if action == "modify_config":
            # This is complex - for now we'll convert to suggest_changes
            # and let the user apply suggestions manually
            input_data["action"] = "suggest_changes"

        return input_data

    @staticmethod
    def _extract_prompt_engineering_input(
        message: str, request_context: dict[str, Any]
        ) -> dict[str, Any]:
        """Extract input parameters for prompt engineering tool from message and request context"""
        # Extract project fields from generic request context
        request_namespace = request_context.get("namespace")
        request_project_id = request_context.get("project_id")

        # Analyze message to determine the action and intent
        message_lower = message.lower()

        # Determine action based on message keywords
        action = "start_engineering"  # default action

        # Action detection keywords
        analyze_keywords = ["analyze", "context", "current setup", "what's configured"]
        start_keywords = ["create", "generate", "help", "need", "want", "build", "make"]
        continue_keywords = ["continue", "next", "yes", "no", "answer"]

        if any(word in message_lower for word in analyze_keywords):
            action = "analyze_context"
        elif any(word in message_lower for word in start_keywords) or not request_context.get("session_active"):
            action = "start_engineering"
        elif any(word in message_lower for word in continue_keywords) and request_context.get("session_active"):
            action = "continue_conversation"
        elif "generate" in message_lower and "prompt" in message_lower:
            action = "generate_prompts"
        elif "save" in message_lower and "prompt" in message_lower:
            action = "save_prompts"

        input_data = {
            "action": action,
            "namespace": request_namespace or "unknown",
            "project_id": request_project_id or "unknown",
        }

        # Add user input for starting engineering
        if action == "start_engineering":
            input_data["user_input"] = message

        # For continue_conversation, we'd need additional context about the question
        # This is simplified - in practice, session management would handle this
        if action == "continue_conversation":
            input_data["user_response"] = message
            input_data["question_answered"] = request_context.get("last_question", "unknown")

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

                # Add tool-specific information for compatibility
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
                elif tool_name == "project_config":
                    tool_info.update({
                        "actions": [
                            "get_schema", "analyze_config", "suggest_changes", "modify_config"
                        ],
                        "parameters": {
                            "action": "Required: action type",
                            "namespace": "Required: namespace string",
                            "project_id": "Required: project identifier",
                            "user_intent": "Optional: description of desired changes",
                            "changes": "Optional: specific configuration changes"
                        },
                        "examples": [
                            "What configuration options are available?",
                            "Show me the current configuration",
                            "I want to switch to OpenAI instead of Ollama",
                            "How can I add custom prompts?",
                            "What RAG settings should I use?",
                            "Configure this project for customer support"
                        ]
                    })
                elif tool_name == "prompt_engineering":
                    tool_info.update({
                        "actions": [
                            "analyze_context", "start_engineering", "continue_conversation",
                            "generate_prompts", "save_prompts"
                        ],
                        "parameters": {
                            "action": "Required: action type",
                            "namespace": "Required: namespace string",
                            "project_id": "Required: project identifier",
                            "user_input": "Optional: initial prompt requirements",
                            "user_response": "Optional: response to follow-up question",
                            "question_answered": "Optional: which question was answered"
                        },
                        "examples": [
                            "I need help creating prompts for customer support",
                            "Create prompts for technical documentation",
                            "Generate prompts for content creation",
                            "Help me optimize prompts for my use case",
                            "I want prompts that work well with OpenAI",
                            "Create professional prompts for data analysis"
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
            from tools.project_config_tool.tool.project_config_tool import ProjectConfigTool
            from tools.prompt_engineering_tool.tool.prompt_engineering_tool import PromptEngineeringTool

            # Create instances to trigger registration if needed
            projects_tool = ProjectsTool()
            project_config_tool = ProjectConfigTool()
            prompt_engineering_tool = PromptEngineeringTool()
            logger.info(f"ProjectsTool instantiated: {projects_tool}")
            logger.info(f"ProjectConfigTool instantiated: {project_config_tool}")
            logger.info(f"PromptEngineeringTool instantiated: {prompt_engineering_tool}")

            # Verify registration worked
            registered_tools = list_tools()
            logger.info(f"Tools registered after import: {registered_tools}")

            # Check and manually register tools if needed
            if "projects" not in registered_tools:
                logger.warning("Projects tool not found in registry after import")
                from tools.core import register_tool
                register_tool("projects", projects_tool)
                logger.info("Manually registered projects tool")

            if "project_config" not in registered_tools:
                logger.warning("Project config tool not found in registry after import")
                from tools.core import register_tool
                register_tool("project_config", project_config_tool)
                logger.info("Manually registered project config tool")

            if "prompt_engineering" not in registered_tools:
                logger.warning("Prompt engineering tool not found in registry after import")
                from tools.core import register_tool
                register_tool("prompt_engineering", prompt_engineering_tool)
                logger.info("Manually registered prompt engineering tool")

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