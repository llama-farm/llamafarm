import uuid
from typing import Dict, Optional, List, Any, Tuple
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
import json
import re
from dataclasses import dataclass
from enum import Enum

# Correct v2.0 imports
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory, SystemPromptGenerator
import instructor
from openai import OpenAI
from core.config import settings

# Import your fixed tool
from tools.projects_tool.tool import ProjectsTool, ProjectsToolInput

router = APIRouter(
    prefix="/inference",
    tags=["inference"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
    session_id: str
    tool_results: Optional[List[Dict]] = None

class IntegrationType(Enum):
    NATIVE = "native_atomic_agents"
    MANUAL = "manual_execution"
    MANUAL_FAILED = "manual_execution_failed"

class ProjectAction(Enum):
    LIST = "list"
    CREATE = "create"

@dataclass
class ToolResult:
    success: bool
    action: str
    namespace: str
    message: str = ""
    result: Any = None
    integration_type: IntegrationType = IntegrationType.MANUAL

@dataclass
class ModelCapabilities:
    supports_tools: bool
    instructor_mode: instructor.Mode
    
# Store agent instances to maintain conversation context
agent_sessions: Dict[str, AtomicAgent] = {}

# Constants
TOOL_CALLING_MODELS = [
    "llama3.1", "llama3.1:8b", "llama3.1:70b", "llama3.1:405b",
    "mistral-nemo", "firefunction-v2", "hermes3"
]

TEMPLATE_INDICATORS = [
    "[number of projects]", "[project list]", "[namespace]", "[projects]",
    "{{", "}}", "${", "[total]", "[count]", "**[list of projects",
    "**[projects", "[list of projects", "I will use the project tool",
    "To view the list, I will", "**[namespace", "currently **[",
]

PROJECT_KEYWORDS = ["project", "list", "create", "show", "namespace"]

NAMESPACE_PATTERNS = [
    r"in\s+(\w+)\s+namespace",
    r"namespace\s+(\w+)",
    r"in\s+(\w+)(?:\s|$)",
    r"from\s+(\w+)(?:\s|$)"
]

PROJECT_ID_PATTERNS = [
    r"create\s+(?:project\s+)?(?:called\s+)?['\"]?(\w+)['\"]?",
    r"new\s+project\s+['\"]?(\w+)['\"]?",
    r"project\s+['\"]?(\w+)['\"]?"
]

CREATE_KEYWORDS = ["create", "new", "add", "make"]
EXCLUDED_NAMESPACES = ["the", "a", "an", "my", "projects", "project"]

class MessageAnalyzer:
    """Handles message analysis and parameter extraction"""
    
    @staticmethod
    def extract_namespace(message: str) -> str:
        """Extract namespace from user message or return default"""
        message_lower = message.lower()
        
        # Look for explicit namespace mentions
        for pattern in NAMESPACE_PATTERNS:
            match = re.search(pattern, message_lower)
            if match:
                namespace = match.group(1)
                if namespace not in EXCLUDED_NAMESPACES:
                    return namespace
        
        return "test"  # default

    @staticmethod
    def extract_project_id(message: str) -> Optional[str]:
        """Extract project ID from create project messages"""
        message_lower = message.lower()
        
        for pattern in PROJECT_ID_PATTERNS:
            match = re.search(pattern, message_lower)
            if match:
                return match.group(1)
        
        return None

    @staticmethod
    def determine_action(message: str) -> ProjectAction:
        """Determine if user wants to create or list projects"""
        message_lower = message.lower()
        return ProjectAction.CREATE if any(word in message_lower for word in CREATE_KEYWORDS) else ProjectAction.LIST

    @staticmethod
    def is_project_related(message: str) -> bool:
        """Check if message is project-related"""
        return any(word in message.lower() for word in PROJECT_KEYWORDS)

class ResponseAnalyzer:
    """Handles response analysis and validation"""
    
    @staticmethod
    def is_template_response(response: str) -> bool:
        """Detect if response contains template placeholders"""
        response_lower = response.lower()
        return any(indicator.lower() in response_lower for indicator in TEMPLATE_INDICATORS)

    @staticmethod
    def needs_manual_execution(response: str, message: str) -> bool:
        """Determine if manual tool execution is needed"""
        if not MessageAnalyzer.is_project_related(message):
            return False
            
        return (
            ResponseAnalyzer.is_template_response(response) or
            len(response) < 50 or
            "I don't have access" in response or
            "cannot directly" in response or
            "[list of projects" in response or
            "**[" in response or
            "I will use the project tool" in response
        )

class ModelManager:
    """Handles model capabilities and configuration"""
    
    @staticmethod
    def get_capabilities(model_name: str) -> ModelCapabilities:
        """Get model capabilities based on model name"""
        model_lower = model_name.lower()
        supports_tools = any(supported in model_lower for supported in TOOL_CALLING_MODELS)
        
        return ModelCapabilities(
            supports_tools=supports_tools,
            instructor_mode=instructor.Mode.TOOLS if supports_tools else instructor.Mode.JSON
        )

    @staticmethod
    def create_client(capabilities: ModelCapabilities) -> Any:
        """Create instructor client with appropriate mode"""
        ollama_client = OpenAI(
            base_url=settings.ollama_host,
            api_key=settings.ollama_api_key,
        )
        
        return instructor.from_openai(ollama_client, mode=capabilities.instructor_mode)

class ToolExecutor:
    """Handles tool execution (both native and manual)"""
    
    @staticmethod
    def execute_manual(message: str) -> ToolResult:
        """Manually execute tool based on message analysis"""
        try:
            projects_tool = ProjectsTool()
            action = MessageAnalyzer.determine_action(message)
            namespace = MessageAnalyzer.extract_namespace(message)
            
            if action == ProjectAction.CREATE:
                project_id = MessageAnalyzer.extract_project_id(message)
                if not project_id:
                    return ToolResult(
                        success=False,
                        action=action.value,
                        namespace=namespace,
                        message="Please specify a project name to create. For example: 'Create project my_app'"
                    )
                
                tool_input = ProjectsToolInput(
                    action=action.value,
                    namespace=namespace,
                    project_id=project_id
                )
            else:
                tool_input = ProjectsToolInput(action=action.value, namespace=namespace)
            
            print(f"🔧 [Manual Tool] Executing {action.value} in namespace '{namespace}'" + 
                  (f" with project_id '{tool_input.project_id}'" if hasattr(tool_input, 'project_id') and tool_input.project_id else ""))
            
            result = projects_tool.run(tool_input)
            
            return ToolResult(
                success=result.success,
                action=action.value,
                namespace=namespace,
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
            
            response = f"I found {result.total} project(s) in the '{namespace}' namespace:\n\n"
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
                return f"✅ Successfully created project '{result.project_id}' in namespace '{namespace}'"
            else:
                return f"❌ Failed to create project: {result.message}"
        
        return str(result)

    @staticmethod
    def create_tool_info(tool_result: ToolResult) -> List[Dict]:
        """Create tool result information for response"""
        return [{
            "tool_used": "projects",
            "integration_type": tool_result.integration_type.value,
            "action": tool_result.action,
            "namespace": tool_result.namespace,
            "message": f"{tool_result.integration_type.value.replace('_', ' ').title()} {'successful' if tool_result.success else 'failed'}"
        }]

class AgentFactory:
    """Factory for creating agents with proper configuration"""
    
    @staticmethod
    def create_system_prompt(capabilities: ModelCapabilities) -> SystemPromptGenerator:
        """Create system prompt based on model capabilities"""
        return SystemPromptGenerator(
            background=[
                "You are a helpful assistant for LlamaFarm project management.",
                "You have access to a projects tool that can list and create projects in different namespaces.",
                f"Tool calling support: {'NATIVE' if capabilities.supports_tools else 'FALLBACK'}",
            ],
            steps=[
                "Analyze the user's request to determine if they need project management assistance",
                "For listing projects: use action='list' with the appropriate namespace",
                "For creating projects: use action='create' with namespace and project_id",
                "Always provide clear, helpful responses based on the tool results",
                "If tool calling fails, indicate that manual processing will be attempted",
            ],
            output_instructions=[
                "Be helpful and friendly in your responses",
                "When using tools, briefly explain what you're doing",
                "Provide clear summaries of project operations",
                "Use the exact namespace mentioned by the user",
                "Format project lists in a readable way with bullet points",
                "If you cannot use tools natively, still provide a helpful response",
            ]
        )

    @staticmethod
    def create_agent_config(capabilities: ModelCapabilities, system_prompt: SystemPromptGenerator, tools: List = None) -> AgentConfig:
        """Create agent configuration"""
        client = ModelManager.create_client(capabilities)
        history = ChatHistory()
        
        config_params = {
            "client": client,
            "model": settings.ollama_model,
            "history": history,
            "system_role": "system",
            "system_prompt_generator": system_prompt,
            "model_api_parameters": {
                "temperature": 0.1,
                "top_p": 0.9,
            }
        }
        
        if capabilities.supports_tools and tools:
            config_params["tools"] = tools
        
        return AgentConfig(**config_params)

    @staticmethod
    def create_agent() -> AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema]:
        """Create a new agent instance with enhanced tool integration"""
        print(f"🔧 [Inference] Creating new agent...")
        
        # Get model capabilities
        capabilities = ModelManager.get_capabilities(settings.ollama_model)
        print(f"🔍 [Inference] Model {settings.ollama_model} tool calling support: {capabilities.supports_tools}")
        print(f"✅ [Inference] Using {capabilities.instructor_mode.value} mode for instructor")

        # Create tool instances
        projects_tool = ProjectsTool()
        print(f"✅ [Inference] Created ProjectsTool instance")
        
        # Create system prompt and agent config
        system_prompt = AgentFactory.create_system_prompt(capabilities)
        tools = [projects_tool] if capabilities.supports_tools else None
        agent_config = AgentFactory.create_agent_config(capabilities, system_prompt, tools)
        
        if capabilities.supports_tools:
            print(f"✅ [Inference] Added native tool support")
        else:
            print(f"⚠️ [Inference] No native tools added - will use manual execution")
        
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](agent_config)
        print(f"✅ [Inference] Agent created successfully")
        return agent

class ChatProcessor:
    """Main chat processing logic"""
    
    @staticmethod
    def process_chat(request: ChatRequest, session_id: str) -> Tuple[str, Optional[List[Dict]]]:
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
        print(f"🤖 [Inference] Running agent with message: '{request.message[:100]}...'")
        input_schema = BasicChatInputSchema(chat_message=request.message)
        response = agent.run(input_schema)
        
        response_message = response.chat_message if hasattr(response, 'chat_message') else str(response)
        print(f"📤 [Inference] Initial agent response: {response_message[:200]}...")
        
        # Check if manual execution is needed
        if ResponseAnalyzer.needs_manual_execution(response_message, request.message):
            print(f"🔧 [Inference] Template/incomplete response detected: '{response_message[:100]}...'")
            
            tool_result = ToolExecutor.execute_manual(request.message)
            
            if tool_result.success:
                response_message = ResponseFormatter.format_tool_response(tool_result)
                tool_info = ResponseFormatter.create_tool_info(tool_result)
                print(f"✅ [Inference] Manual execution successful")
            else:
                response_message = tool_result.message
                tool_info = ResponseFormatter.create_tool_info(tool_result)
                print(f"❌ [Inference] Manual execution failed")
        
        elif MessageAnalyzer.is_project_related(request.message):
            tool_info = [{
                "tool_used": "projects",
                "integration_type": IntegrationType.NATIVE.value,
                "message": "Native tool integration used"
            }]
        else:
            tool_info = None
        
        return response_message, tool_info

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
    if session_id in agent_sessions:
        agent_sessions[session_id].reset_history()
        del agent_sessions[session_id]
        print(f"🗑️ [Inference] Deleted session: {session_id}")
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
    
    return {
        "active_sessions": len(agent_sessions),
        "session_ids": list(agent_sessions.keys()),
        "current_model": settings.ollama_model,
        "model_supports_tools": capabilities.supports_tools,
        "integration_type": "enhanced_detection_with_fallback",
        "tools_enabled": True,
        "manual_fallback_available": True
    }