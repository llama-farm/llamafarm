import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Header
from atomic_agents import BasicChatInputSchema

# Import business logic from separate modules
from .models import ChatRequest, ChatResponse, IntegrationType, ProjectAction
from .services import ChatProcessor, AgentSessionManager, ToolExecutor, ResponseFormatter
from .analyzers import MessageAnalyzer, ResponseAnalyzer
from .factories import ModelManager
from core.config import settings

try:
    from tools.projects_tool.tool import ProjectsTool, ProjectsToolInput
    print("✅ [Inference] Successfully imported ProjectsTool")
except ImportError as e:
    print(f"❌ [Inference] Failed to import ProjectsTool: {e}")
    print("💡 [Inference] Make sure you're running in the virtual environment")
    raise

router = APIRouter(
    prefix="/inference",
    tags=["inference"],
)

# Route handlers
@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, 
    session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Send a message to the chat agent with enhanced tool integration"""
    print(f"💬 [Inference] /chat endpoint called with message: '{request.message[:100]}...'")
    print(f"🆔 [Inference] Session ID: {session_id}")
    
    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Process chat
        response_message, tool_info = ChatProcessor.process_chat(request, session_id)
        
        return ChatResponse(
            message=response_message, 
            session_id=session_id,
            tool_results=tool_info
        )
        
    except Exception as e:
        print(f"💥 [Inference] Error in chat endpoint: {str(e)}")
        import traceback
        print(f"📚 [Inference] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    if AgentSessionManager.delete_session(session_id):
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@router.get("/tools")
async def get_available_tools():
    """Get information about available tools"""
    capabilities = ModelManager.get_capabilities(settings.ollama_model)
    
    return [
        {
            "name": "projects",
            "description": "Tool for managing projects - list and create operations",
            "integration": f"{'native' if capabilities.supports_tools else 'manual'}_with_fallback",
            "model_support": capabilities.supports_tools,
            "current_model": settings.ollama_model,
            "actions": [action.value for action in ProjectAction],
            "parameters": {
                "action": "Required: 'list' or 'create'",
                "namespace": "Required: namespace string", 
                "project_id": "Required for create action: project identifier"
            },
            "examples": [
                "List my projects",
                "Show projects in <name> namespace", 
                "List how many projects I have in <name>",
                "Create a new project called my_app",
                "Create project demo in test namespace"
            ]
        }
    ]

@router.post("/test-native-tools")
async def test_native_tools(request: ChatRequest):
    """Test the enhanced tool integration"""
    print(f"🧪 [Inference] Testing enhanced tool integration with: '{request.message}'")
    
    try:
        from .factories import AgentFactory
        
        agent = AgentFactory.create_agent()
        capabilities = ModelManager.get_capabilities(settings.ollama_model)
        
        # Test with the user's message
        input_schema = BasicChatInputSchema(chat_message=request.message)
        response = agent.run(input_schema)
        
        response_message = response.chat_message if hasattr(response, 'chat_message') else str(response)
        
        # Check if manual execution is needed
        manual_result = None
        if MessageAnalyzer.is_project_related(request.message):
            if ResponseAnalyzer.needs_manual_execution(response_message, request.message):
                print(f"🔧 [Test] Testing manual execution...")
                manual_result = ToolExecutor.execute_manual(request.message)
        
        print(f"✅ [Inference] Test completed successfully")
        
        return {
            "user_message": request.message,
            "agent_response": response_message,
            "model_supports_tools": capabilities.supports_tools,
            "current_model": settings.ollama_model,
            "template_detected": ResponseAnalyzer.is_template_response(response_message),
            "manual_execution_test": manual_result.__dict__ if manual_result else None,
            "integration_type": "enhanced_with_detection",
            "status": "success"
        }
        
    except Exception as e:
        print(f"💥 [Inference] Error in test_native_tools: {str(e)}")
        import traceback
        print(f"📚 [Inference] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error testing tools: {str(e)}")

@router.get("/agent-status")
async def get_agent_status():
    """Get status of all active agent sessions"""
    capabilities = ModelManager.get_capabilities(settings.ollama_model)
    
    # Check if atomic_agents is available
    atomic_agents_available = False
    try:
        import atomic_agents
        atomic_agents_available = True
    except ImportError:
        pass
    
    return {
        "active_sessions": AgentSessionManager.get_session_count(),
        "session_ids": AgentSessionManager.get_session_ids(),
        "current_model": settings.ollama_model,
        "model_supports_tools": capabilities.supports_tools,
        "integration_type": "enhanced_detection_with_fallback",
        "tools_enabled": True,
        "manual_fallback_available": True,
        "atomic_agents_available": atomic_agents_available,
        "environment_status": "healthy" if atomic_agents_available else "missing_dependencies"
    }