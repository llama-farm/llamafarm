from typing import Any

import instructor
from atomic_agents import (
    AgentConfig,
    AtomicAgent,
    BasicChatInputSchema,
    BasicChatOutputSchema,
)
from openai import OpenAI

from core.settings import settings

from .models import ModelCapabilities

# Constants
TOOL_CALLING_MODELS = [
    "llama3.1", "llama3.1:8b", "llama3.1:70b", "llama3.1:405b",
    "mistral-nemo", "firefunction-v2", "hermes3"
]

class ModelManager:
    """Handles model capabilities and configuration"""
    
    @staticmethod
    def get_capabilities(model_name: str) -> ModelCapabilities:
        """Get model capabilities based on model name"""
        model_lower = model_name.lower()
        supports_tools = any
        (supported in model_lower for supported in TOOL_CALLING_MODELS)
        
        return ModelCapabilities(
            supports_tools=supports_tools,
            instructor_mode=instructor.Mode.TOOLS 
            if supports_tools else instructor.Mode.JSON
        )

    @staticmethod
    def create_client(capabilities: ModelCapabilities) -> Any:
        """Create instructor client with appropriate mode"""
        ollama_client = OpenAI(
            base_url=settings.ollama_host,
            api_key=settings.ollama_api_key,
        )
        
        return instructor.from_openai(ollama_client, mode=capabilities.instructor_mode)

class AgentFactory:
    """Factory for creating agents with proper configuration"""
    
    @staticmethod
    def create_system_prompt(capabilities: ModelCapabilities) -> str:
        """Create system prompt based on model capabilities"""
        return f"""You are a helpful assistant for LlamaFarm project management.
You have access to a projects tool that can list and create projects in different namespaces.
Tool calling support: {'NATIVE' if capabilities.supports_tools else 'FALLBACK'}

Analyze the user's request to determine if they need project management assistance.
For listing projects: use action='list' with the appropriate namespace
For creating projects: use action='create' with namespace and project_id
Always provide clear, helpful responses based on the tool results.

Be helpful and friendly in your responses. When using tools, briefly explain what you're doing.
Provide clear summaries of project operations. Use the exact namespace mentioned by the user.
Format project lists in a readable way with bullet points."""

    @staticmethod
    def create_agent_config(
        capabilities: ModelCapabilities,
        system_prompt: str,
        tools: list[Any] | None = None,
        ) -> AgentConfig:
        """Create agent configuration"""
        client = ModelManager.create_client(capabilities)
        
        config_params = {
            "client": client,
            "model": settings.ollama_model,
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
        print("🔧 [Inference] Creating new agent...")
        
        # Get model capabilities
        capabilities = ModelManager.get_capabilities(settings.ollama_model)
        print(
            f"🔍 [Inference] Model {settings.ollama_model} tool calling support: "
            f"{capabilities.supports_tools}"
            )
        print(
            f"✅ [Inference] Using {capabilities.instructor_mode.value} "
            "mode for instructor"
            )

        # Create tool instances
        try:
            from tools.projects_tool.tool import ProjectsTool
            projects_tool = ProjectsTool()
            print("✅ [Inference] Created ProjectsTool instance")
        except ImportError:
            print("❌ [Inference] Failed to import ProjectsTool: {e}")
            projects_tool = None
        
        # Create system prompt and agent config
        system_prompt = AgentFactory.create_system_prompt(capabilities)
        tools = (
            [projects_tool]
            if capabilities.supports_tools and projects_tool
            else None
        )
        agent_config = AgentFactory.create_agent_config(
            capabilities, system_prompt, tools
        )
        
        if capabilities.supports_tools and projects_tool:
            print("✅ [Inference] Added native tool support")
        else:
            print("⚠️ [Inference] No native tools added - will use manual execution")
        
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](agent_config)
        print("✅ [Inference] Agent created successfully")
        return agent 