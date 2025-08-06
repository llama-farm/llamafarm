from typing import Any

from atomic_agents import BasicChatInputSchema

from .analyzers import MessageAnalyzer, ResponseAnalyzer
from .factories import AgentFactory
from .models import IntegrationType, ProjectAction, ToolResult

# Store agent instances to maintain conversation context
agent_sessions: dict[str, Any] = {}

class ToolExecutor:
    """Handles tool execution (both native and manual)"""
    
    @staticmethod
    def execute_manual(
        message: str, request_context: dict[str, Any] | None = None) -> ToolResult:
        """Manually execute tool based on enhanced message analysis"""
        try:
            from tools.projects_tool.tool import ProjectsTool, ProjectsToolInput
            projects_tool = ProjectsTool()
            
            # Extract request fields
            context = request_context or {}
            request_namespace = context.get("namespace")
            request_project_id = context.get("project_id")
            
            # Use enhanced LLM-based analysis
            analysis = MessageAnalyzer.analyze_with_llm(
                message, request_namespace, request_project_id
                )
            action = (
                ProjectAction.CREATE 
                if analysis.action.lower() == "create" 
                else ProjectAction.LIST
                )
            
            if action == ProjectAction.CREATE:
                if not analysis.project_id:
                    return ToolResult(
                        success=False,
                        action=action.value,
                        namespace=analysis.namespace,
                        message=(
                            "Please specify a project name to create. "
                            "For example: 'Create project my_app'"
                            )
                    )
                
                tool_input = ProjectsToolInput(
                    action=action.value,
                    namespace=analysis.namespace,
                    project_id=analysis.project_id
                )
            else:
                tool_input = ProjectsToolInput(
                    action=action.value, namespace=analysis.namespace)
            
            print(
                f"🔧 [Manual Tool] Executing {action.value} "
                f"in namespace '{analysis.namespace}'" + 
                (f" with project_id '{tool_input.project_id}'" 
                if hasattr(tool_input, 'project_id') and tool_input.project_id 
                else "")
                )
            print(
                f"🧠 [LLM Analysis] Confidence: {analysis.confidence:.2f}, "
                f"Reasoning: {analysis.reasoning}")
            
            result = projects_tool.run(tool_input)
            
            return ToolResult(
                success=result.success,
                action=action.value,
                namespace=analysis.namespace,
                result=result,
                integration_type=IntegrationType.MANUAL
            )
            
        except Exception as e:
            print(f"❌ [Manual Tool] Error: {str(e)}")
            return ToolResult(
                success=False,
                action="unknown",
                namespace="unknown",
                message=f"Tool execution failed: {str(e)}",
                integration_type=IntegrationType.MANUAL_FAILED
            )

class ResponseFormatter:
    """Handles response formatting"""
    
    @staticmethod
    def format_tool_response(tool_result: ToolResult) -> str:
        """Format tool execution results into a natural response"""
        if not tool_result.success:
            return f"I encountered an issue: {tool_result.message}"
        
        result = tool_result.result
        action = tool_result.action
        namespace = tool_result.namespace
        
        if action == ProjectAction.LIST.value:
            if result.total == 0:
                return f"I found no projects in the '{namespace}' namespace."
            
            response = (
                f"I found {result.total} project(s) in the '{namespace}' "
                "namespace:\n\n"
                )
            if result.projects:
                for project in result.projects:
                    response += f"• **{project['project_id']}**\n"
                    response += f"  Path: `{project['path']}`\n"
                    if project.get('description'):
                        response += f"  Description: {project['description']}\n"
                    response += "\n"
            
            return response.strip()
        
        elif action == ProjectAction.CREATE.value:
            if result.success:
                return (
                    f"✅ Successfully created project '{result.project_id}' "
                    f"in namespace '{namespace}'"
                    )
            else:
                return f"❌ Failed to create project: {result.message}"
        
        return str(result)

    @staticmethod
    def create_tool_info(tool_result: ToolResult) -> list[dict]:
        """Create tool result information for response"""
        return [{
            "tool_used": "projects",
            "integration_type": tool_result.integration_type.value,
            "action": tool_result.action,
            "namespace": tool_result.namespace,
            "message": (
                f"{tool_result.integration_type.value.replace('_', ' ').title()} "
                f"{'successful' if tool_result.success else 'failed'}"
                )
        }]

class ChatProcessor:
    """Main chat processing logic"""
    
    @staticmethod
    def process_chat(request: Any, session_id: str) -> tuple[str, list[dict] | None]:
        """Process chat request and return response with tool info"""
        # Get or create agent
        if session_id not in agent_sessions:
            agent = AgentFactory.create_agent()
            agent_sessions[session_id] = agent
            print(f"🆕 [Inference] Created new agent session: {session_id}")
        else:
            agent = agent_sessions[session_id]
            print(f"🔄 [Inference] Using existing agent session: {session_id}")

        # Run agent
        print(
            f"🤖 [Inference] Running agent with message: "
            f"'{request.message[:100]}...'"
            )
        input_schema = BasicChatInputSchema(chat_message=request.message)
        response = agent.run(input_schema)
        
        response_message = response.chat_message 
        if hasattr(response, 'chat_message'):
            response_message = response.chat_message
        else:
            response_message = str(response)
        print(f"📤 [Inference] Initial agent response: {response_message[:200]}...")
        
        # Check if manual execution is needed
        if ResponseAnalyzer.needs_manual_execution(response_message, request.message):
            print(
                f"🔧 [Inference] Template/incomplete response detected: "
                f"'{response_message[:100]}...'"
                )
            
            # Pass request fields to enhanced analysis via generic context
            request_context = {
                "namespace": getattr(request, 'namespace', None),
                "project_id": getattr(request, 'project_id', None)
            }
            tool_result = ToolExecutor.execute_manual(request.message, request_context)
            
            if tool_result.success:
                response_message = ResponseFormatter.format_tool_response(tool_result)
                tool_info = ResponseFormatter.create_tool_info(tool_result)
                print("✅ [Inference] Manual execution successful")
            else:
                response_message = tool_result.message
                tool_info = ResponseFormatter.create_tool_info(tool_result)
                print("❌ [Inference] Manual execution failed")
        
        elif MessageAnalyzer.is_project_related(request.message):
            tool_info = [{
                "tool_used": "projects",
                "integration_type": IntegrationType.NATIVE.value,
                "message": "Native tool integration used"
            }]
        else:
            tool_info = None
        
        return response_message, tool_info

class AgentSessionManager:
    """Manages agent sessions"""
    
    @staticmethod
    def get_session(session_id: str) -> Any:
        """Get existing session or create new one"""
        if session_id not in agent_sessions:
            agent = AgentFactory.create_agent()
            agent_sessions[session_id] = agent
            print(f"🆕 [Inference] Created new agent session: {session_id}")
        return agent_sessions[session_id]
    
    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Delete a chat session"""
        if session_id in agent_sessions:
            agent_sessions[session_id].reset_history()
            del agent_sessions[session_id]
            print(f"🗑️ [Inference] Deleted session: {session_id}")
            return True
        return False
    
    @staticmethod
    def get_session_count() -> int:
        """Get number of active sessions"""
        return len(agent_sessions)
    
    @staticmethod
    def get_session_ids() -> list[str]:
        """Get list of active session IDs"""
        return list(agent_sessions.keys()) 