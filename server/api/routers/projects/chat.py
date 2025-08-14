import builtins
import threading
import time
import uuid

from atomic_agents import AtomicAgent
from fastapi import Header, HTTPException, Response
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from agents.project_chat_orchestrator import (
    ProjectChatOrchestratorAgentFactory,
    ProjectChatOrchestratorAgentInputSchema,
)
from api.routers.inference.models import ChatRequest, ChatResponse
from api.routers.shared.response_utils import set_session_header
from services.project_service import ProjectService

from .projects import router

"""
Project-scoped chat endpoint that uses the same OpenAI-style models as inference.
"""


# Store agent instances to maintain conversation context
# In production, use Redis, database, or other persistent storage
agent_sessions: builtins.dict[str, AtomicAgent] = {}
_agent_sessions_lock = threading.RLock()


@router.post("/{namespace}/{project_id}/chat/completions", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    namespace: str,
    project_id: str,
    response: Response,
    session_id: str | None = Header(None, alias="X-Session-ID"),
):
    """Send a message to the chat agent"""
    try:
        project_config = ProjectService.load_config(namespace, project_id)

        # If no session ID provided, create a new one and ensure thread-safe session map access
        with _agent_sessions_lock:
            if not session_id or session_id not in agent_sessions:
                if not session_id:
                    session_id = str(uuid.uuid4())
                agent = ProjectChatOrchestratorAgentFactory.create_agent(project_config)
                agent_sessions[session_id] = agent
            else:
                # Use existing agent to maintain conversation context
                agent = agent_sessions[session_id]

        # Attach routing to metadata for downstream consistency
        request.metadata["namespace"] = namespace
        request.metadata["project_id"] = project_id

        # Extract the latest user message
        latest_user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                latest_user_message = msg.content
                break
        if latest_user_message is None:
            raise HTTPException(status_code=400, detail="No user message provided")

        # Process the user's input through the agent and get the response
        input_schema = ProjectChatOrchestratorAgentInputSchema(
            chat_message=latest_user_message
        )
        agent_response = agent.run(input_schema)

        # Extract the message from the response
        if hasattr(agent_response, "chat_message"):
            response_message = agent_response.chat_message
        else:
            # Fallback if the response structure is different
            response_message = str(agent_response)

        # Set session header
        set_session_header(response, session_id)

        # model_name should come from the project config
        # model_name = request.model or settings.ollama_model

        completion: ChatCompletion = ChatCompletion(
            id=f"chat-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model="todo: use model from project config",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=response_message,
                    ),
                    finish_reason="stop",
                )
            ],
        )
        return completion

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat: {str(e)}"
        ) from e


@router.delete("/{namespace}/{project_id}/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    with _agent_sessions_lock:
        if session_id in agent_sessions:
            del agent_sessions[session_id]
            return {"message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
