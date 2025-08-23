import asyncio
import contextlib
import threading
from typing import Any

from atomic_agents import BasicChatInputSchema  # type: ignore
from openai import AsyncOpenAI

from core.logging import FastAPIStructLogger

from .analyzers import MessageAnalyzer, ResponseAnalyzer
from .factories import AgentFactory
from .models import ChatMessage, ChatRequest, IntegrationType, ProjectAction, ToolResult

# Initialize logger
logger = FastAPIStructLogger()

# Store agent instances to maintain conversation context
# Protected by a re-entrant lock for thread-safety across workers
agent_sessions: dict[str, Any] = {}
_agent_sessions_lock = threading.RLock()

class ToolExecutor:
    """Handles tool execution (both native and manual)"""

    @staticmethod
    def execute_manual(
        message: str, request_context: dict[str, Any] | None = None) -> ToolResult:
        """Manually execute tool based on enhanced message analysis"""
        try:
            # Ensure tools are initialized before manual execution
            from .tool_service import ensure_tools_initialized
            if not ensure_tools_initialized():
                logger.error("Failed to initialize tool registry for manual execution")
                return ToolResult(
                    success=False,
                    action="unknown",
                    namespace="unknown",
                    message="Tool system not available",
                    integration_type=IntegrationType.MANUAL_FAILED
                )

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
                        namespace=analysis.namespace or "unknown",
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

            logger.info(
                "Executing manual tool action",
                action=action.value,
                namespace=analysis.namespace,
                project_id=(
                    getattr(tool_input, 'project_id', None)
                    if hasattr(tool_input, 'project_id') else None
                ),
                confidence=analysis.confidence,
                reasoning=analysis.reasoning
            )

            result = projects_tool.run(tool_input)

            return ToolResult(
                success=result.success,
                action=action.value,
                namespace=analysis.namespace or "unknown",
                result=result,
                integration_type=IntegrationType.MANUAL
            )

        except Exception as e:
            logger.error("Manual tool execution failed", error=str(e))
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
    def process_chat(
        request: ChatRequest, session_id: str
    ) -> tuple[str, list[dict] | None]:
        """Process chat request and return response with tool info.

        Expects an OpenAI-style ChatRequest with messages and optional metadata.
        """
        try:
            logger.info("Starting chat processing", session_id=session_id)

            # Get or create agent
            with _agent_sessions_lock:
                if session_id not in agent_sessions:
                    agent = AgentFactory.create_agent()
                    agent_sessions[session_id] = agent
                    logger.info("Created new agent session", session_id=session_id)
                else:
                    agent = agent_sessions[session_id]
                    logger.info("Using existing agent session", session_id=session_id)

            # Extract latest user message
            latest_user_message: str | None = None
            for message in reversed(request.messages):
                if isinstance(message, ChatMessage) and message.role == "user" and message.content:
                    latest_user_message = message.content
                    break
            if latest_user_message is None:
                raise ValueError("No user message found in request.messages")

            # Run agent
            logger.info(
                "Running agent with message",
                message_preview=f"{latest_user_message[:100]}...",
            )
            input_schema = BasicChatInputSchema(chat_message=latest_user_message)
            response = agent.run(input_schema)

            response_message = response.chat_message
            if hasattr(response, 'chat_message'):
                response_message = response.chat_message
            else:
                response_message = str(response)
            logger.info(
                "Initial agent response",
                response_preview=response_message[:200] + "...",
            )

            # Initialize tool_info to avoid UnboundLocalError
            tool_info = None

            # Check if manual execution is needed
            try:
                needs_manual = ResponseAnalyzer.needs_manual_execution(
                    response_message,
                    latest_user_message,
                )
                logger.info("Response analysis completed", needs_manual=needs_manual)
            except Exception as e:
                logger.error(
                    "Error in ResponseAnalyzer.needs_manual_execution", error=str(e)
                )
                needs_manual = False

            if needs_manual:
                logger.info(
                    "Template/incomplete response detected",
                    response_preview=response_message[:100] + "..."
                )

                try:
                    # Pass request fields to enhanced analysis via generic context
                    request_context = {
                        "namespace": request.metadata.get("namespace") or "unknown",
                        "project_id": request.metadata.get("project_id"),
                    }
                    tool_result = ToolExecutor.execute_manual(
                        latest_user_message, request_context
                    )

                    if tool_result.success:
                        response_message = ResponseFormatter.format_tool_response(
                            tool_result,
                        )
                        tool_info = ResponseFormatter.create_tool_info(tool_result)
                        logger.info("Manual execution successful")
                    else:
                        response_message = tool_result.message
                        tool_info = ResponseFormatter.create_tool_info(tool_result)
                        logger.error("Manual execution failed")
                except Exception as e:
                    logger.error("Error in manual tool execution", error=str(e))
                    response_message = (
                        "I encountered an error while processing your request: "
                        f"{str(e)}"
                    )
                    tool_info = None

            elif MessageAnalyzer.is_project_related(latest_user_message):
                try:
                    tool_info = [{
                        "tool_used": "projects",
                        "integration_type": IntegrationType.NATIVE.value,
                        "message": "Native tool integration used"
                    }]
                    logger.info("Set tool_info for project-related message")
                except Exception as e:
                    logger.error(
                        "Error setting tool_info for project-related message",
                        error=str(e)
                    )
                    tool_info = None
            else:
                tool_info = None
                logger.info("No special tool handling needed")

            logger.info("Chat processing completed successfully")
            return response_message, tool_info

        except Exception as e:
            logger.error("Fatal error in chat processing", error=str(e), exc_info=True)
            error_message = f"I'm sorry, I encountered an unexpected error: {str(e)}"
            return error_message, None

    @staticmethod
    async def stream_chat(request: ChatRequest, session_id: str):
        """Return an iterator/async-iterator of assistant content chunks.

        Uses AtomicAgent streaming if available; otherwise falls back to chunking the full response.
        """
        logger.info("Starting chat streaming", session_id=session_id)

        with _agent_sessions_lock:
            if session_id not in agent_sessions:
                agent = AgentFactory.create_agent()
                agent_sessions[session_id] = agent
                logger.info("Created new agent session", session_id=session_id)
            else:
                agent = agent_sessions[session_id]

        latest_user_message: str | None = None
        for message in reversed(request.messages):
            if isinstance(message, ChatMessage) and message.role == "user" and message.content:
                latest_user_message = message.content
                break
        if latest_user_message is None:
            # yield an error and end
            yield "No user message found in request.messages"
            return

        input_schema = BasicChatInputSchema(chat_message=latest_user_message)

        try:
            # Preflight once to detect tool need
            pre = await asyncio.to_thread(agent.run, input_schema)
            pre_msg = getattr(pre, "chat_message", str(pre))
            try:
                needs_manual = ResponseAnalyzer.needs_manual_execution(pre_msg, latest_user_message)
            except Exception:
                needs_manual = False

            if needs_manual:
                # Run tools, format final narration prompt
                request_context = {
                    "namespace": request.metadata.get("namespace") or "unknown",
                    "project_id": request.metadata.get("project_id"),
                }
                tool_result = ToolExecutor.execute_manual(latest_user_message, request_context)
                narration = ResponseFormatter.format_tool_response(tool_result)

                # Stream narrated response from JSON-mode agent (no tools)
                # Stream narrated response directly from OpenAI-compatible API for fine-grained deltas
                from core.settings import settings
                aoai = AsyncOpenAI(
                    base_url=settings.ollama_host,
                    api_key=settings.ollama_api_key,
                )
                stream = await aoai.chat.completions.create(
                    model=settings.ollama_model,
                    messages=[{"role": "user", "content": narration}],
                    temperature=0.1,
                    top_p=0.9,
                    stream=True,
                )
                async for event in stream:
                    piece = getattr(getattr(event.choices[0], "delta", object()), "content", None)
                    if not piece:
                        continue
                    yield piece
                return

            # Otherwise stream directly from the main agent (sync client)
            from core.settings import settings
            aoai = AsyncOpenAI(
                base_url=settings.ollama_host,
                api_key=settings.ollama_api_key,
            )
            stream = await aoai.chat.completions.create(
                model=settings.ollama_model,
                messages=[{"role": "user", "content": latest_user_message}],
                temperature=0.1,
                top_p=0.9,
                stream=True,
            )
            async for event in stream:
                piece = getattr(getattr(event.choices[0], "delta", object()), "content", None)
                if not piece:
                    continue
                yield piece
            return
        except Exception:
            logger.error("Agent streaming failed; falling back to single response", exc_info=True)
            yield "I encountered an error while processing your request"

class AgentSessionManager:
    """Manages agent sessions"""

    @staticmethod
    def get_session(session_id: str) -> Any:
        """Get existing session or create new one"""
        with _agent_sessions_lock:
            if session_id not in agent_sessions:
                agent = AgentFactory.create_agent()
                agent_sessions[session_id] = agent
                logger.info("Created new agent session", session_id=session_id)
            return agent_sessions[session_id]

    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Delete a chat session"""
        with _agent_sessions_lock:
            if session_id in agent_sessions:
                with contextlib.suppress(Exception):
                    agent_sessions[session_id].reset_history()
                del agent_sessions[session_id]
                logger.info("Deleted session", session_id=session_id)
                return True
            return False

    @staticmethod
    def get_session_count() -> int:
        """Get number of active sessions"""
        with _agent_sessions_lock:
            return len(agent_sessions)

    @staticmethod
    def get_session_ids() -> list[str]:
        """Get list of active session IDs"""
        with _agent_sessions_lock:
            return list(agent_sessions.keys())