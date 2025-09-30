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
        message: str, request_context: dict[str, Any] | None = None
    ) -> ToolResult:
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
                    integration_type=IntegrationType.MANUAL_FAILED,
                )

            # Route using the LLM-first executor (classifier with heuristic fallback)
            from .tool_service import AtomicToolExecutor

            result = AtomicToolExecutor.execute_with_llm(message, request_context or {})
            return result

        except Exception as e:
            logger.error("Manual tool execution failed", error=str(e))
            return ToolResult(
                success=False,
                action="unknown",
                namespace="unknown",
                message=f"Tool execution failed: {str(e)}",
                integration_type=IntegrationType.MANUAL_FAILED,
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
                f"I found {result.total} project(s) in the '{namespace}' namespace:\n\n"
            )
            if result.projects:
                for project in result.projects:
                    response += f"• **{project['project_id']}**\n"
                    response += f"  Path: `{project['path']}`\n"
                    if project.get("description"):
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
        return [
            {
                "tool_used": "projects",
                "integration_type": tool_result.integration_type.value,
                "action": tool_result.action,
                "namespace": tool_result.namespace,
                "message": (
                    f"{tool_result.integration_type.value.replace('_', ' ').title()} "
                    f"{'successful' if tool_result.success else 'failed'}"
                ),
            }
        ]


class ChatProcessor:
    """Main chat processing logic"""

    @staticmethod
    async def process_chat(request: ChatRequest, session_id: str | None):
        """Return an iterator/async-iterator of assistant content chunks.

        Uses AtomicAgent streaming if available; otherwise falls back to chunking the full response.
        """
        logger.info("Starting chat streaming", session_id=session_id)

        if session_id is None:
            # Stateless: create a throwaway agent
            agent = AgentFactory.create_agent()
        else:
            with _agent_sessions_lock:
                if session_id not in agent_sessions:
                    agent = AgentFactory.create_agent()
                    agent_sessions[session_id] = agent
                    logger.info("Created new agent session", session_id=session_id)
                else:
                    agent = agent_sessions[session_id]

        logger.info("Incoming chat message", messages=request.messages)

        latest_user_message: str | None = None
        for message in reversed(request.messages):
            if (
                isinstance(message, ChatMessage)
                and message.role == "user"
                and message.content
            ):
                latest_user_message = message.content
                break
        if latest_user_message is None:
            # yield an error and end
            yield "No user message found in request.messages"
            return

        logger.info("Latest user message", latest_user_message=latest_user_message)
        input_schema = BasicChatInputSchema(chat_message=latest_user_message)

        try:
            # Stream narrated response from JSON-mode agent (no tools)
            # Stream narrated response directly from OpenAI-compatible API for fine-grained deltas
            async for partial_response in agent.run_async_stream(input_schema):
                logger.info("Processing partial response", message=partial_response)
                if (
                    hasattr(partial_response, "chat_message")
                    and partial_response.chat_message
                ):
                    yield partial_response.chat_message
            return
        except Exception:
            logger.error(
                "Agent streaming failed; falling back to single response", exc_info=True
            )
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
