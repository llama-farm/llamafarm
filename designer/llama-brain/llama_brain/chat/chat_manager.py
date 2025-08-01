"""Chat manager for handling conversations."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import requests

from llama_brain.config.hardcoded import get_settings
from llama_brain.rag.simple_knowledge_base import SimpleKnowledgeBase
from .models import ChatSession, ChatMessage, MessageRole, ChatResponse, AgentAction


class ChatManager:
    """Manages chat sessions and interactions."""
    
    def __init__(self):
        self.settings = get_settings()
        self.knowledge_base = SimpleKnowledgeBase()
        self.sessions: Dict[str, ChatSession] = {}
        self._load_sessions()
    
    def _load_sessions(self):
        """Load existing chat sessions from disk."""
        sessions_dir = self.settings.chat_history_dir
        for session_file in sessions_dir.glob("*.json"):
            try:
                with open(session_file) as f:
                    session_data = json.load(f)
                session = ChatSession(**session_data)
                self.sessions[session.session_id] = session
            except Exception as e:
                print(f"Failed to load session {session_file}: {e}")
    
    def _save_session(self, session: ChatSession):
        """Save a chat session to disk."""
        session_file = self.settings.chat_history_dir / f"{session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session.dict(), f, indent=2, default=str)
    
    def create_session(self) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id=session_id)
        
        # Add system message
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content=self._get_system_prompt()
        )
        session.add_message(system_msg)
        
        self.sessions[session_id] = session
        self._save_session(session)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        return self.sessions.get(session_id)
    
    async def process_message(self, session_id: str, user_message: str) -> ChatResponse:
        """Process a user message and generate a response."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Add user message
        user_msg = ChatMessage(role=MessageRole.USER, content=user_message)
        session.add_message(user_msg)
        
        # Get relevant context from knowledge base
        context = await self._get_context(user_message)
        
        # Generate response using local model
        response = await self._generate_response(session, context)
        
        # Add assistant response
        assistant_msg = ChatMessage(role=MessageRole.ASSISTANT, content=response.message)
        session.add_message(assistant_msg)
        
        # Save session
        self._save_session(session)
        
        return response
    
    async def _get_context(self, user_message: str) -> List[Dict]:
        """Get relevant context from the knowledge base."""
        # Determine which component the user is asking about
        component = self._detect_component(user_message)
        print(f"🔍 [DEBUG] Detected component: {component}")
        
        # Search knowledge base
        print(f"🔍 [DEBUG] Searching RAG with query: '{user_message}'")
        results = await self.knowledge_base.search(user_message, component)
        print(f"🔍 [DEBUG] RAG returned {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"🔍 [DEBUG] Result {i+1}: {result.get('content', '')[:100]}...")
        
        return results[:3]  # Top 3 most relevant results
    
    def _detect_component(self, message: str) -> Optional[str]:
        """Detect which component the user is asking about."""
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ['model', 'llm', 'openai', 'ollama', 'anthropic']):
            return 'models'
        elif any(keyword in message_lower for keyword in ['rag', 'retrieval', 'document', 'search', 'embed']):
            return 'rag'
        elif any(keyword in message_lower for keyword in ['prompt', 'template', 'strategy']):
            return 'prompts'
        
        return None
    
    async def _generate_response(self, session: ChatSession, context: List[Dict]) -> ChatResponse:
        """Generate a response using the local model WITH FULL PIPELINE."""
        try:
            # 1. GET APPROPRIATE PROMPT TEMPLATE
            prompt_template = await self._get_prompt_template(session.get_last_user_message())
            
            # 2. INJECT RAG CONTEXT INTO PROMPT
            context_text = self._format_context(context)
            messages = self._build_messages_with_prompt_and_context(
                session, prompt_template, context_text
            )
            
            # 3. SEND TO LLM (with any custom params from user request)
            llm_params = self._extract_llm_params(session.get_last_user_message())
            response = await self._call_ollama(messages, llm_params)
            
            # 4. AUTO-TRIGGER AGENTS IF NEEDED
            actions = await self._auto_trigger_agents(response, session)
            
            # 5. EXECUTE AGENT ACTIONS AUTOMATICALLY  
            executed_actions = []
            for action in actions:
                if self._should_auto_execute(action):
                    result = await self._execute_agent_action(action)
                    executed_actions.append(result)
                    # Update response with agent results
                    response = await self._merge_agent_results(response, result)
            
            return ChatResponse(
                message=response,
                actions=actions,
                context_used=context,
                confidence=0.8,
                suggestions=self._generate_suggestions(response),
                executed_actions=executed_actions
            )
            
        except Exception as e:
            return ChatResponse(
                message=f"I encountered an error: {str(e)}. Please try again or rephrase your question.",
                confidence=0.1
            )
    
    def _format_context(self, context: List[Dict]) -> str:
        """Format context information for the model."""
        if not context:
            return "No specific documentation found."
        
        formatted = "Relevant documentation:\n"
        for i, doc in enumerate(context, 1):
            content = doc.get('content', '')[:500]  # Limit content length
            component = doc.get('metadata', {}).get('component', 'unknown')
            formatted += f"{i}. [{component}] {content}\n\n"
        
        return formatted
    
    async def _call_ollama(self, messages: List[Dict], custom_params: Dict = None) -> str:
        """Call the Ollama API for chat completion with custom parameters."""
        try:
            url = f"{self.settings.ollama_host}/api/chat"
            
            # Default parameters
            options = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
            
            # Override with custom params if provided
            if custom_params:
                options.update(custom_params)
            
            payload = {
                "model": custom_params.get('model', self.settings.chat_model),
                "messages": messages,
                "stream": False,
                "options": options
            }
            
            response = requests.post(url, json=payload, timeout=60)  # Longer timeout
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', 'No response generated.')
            
        except requests.exceptions.RequestException as e:
            return f"Failed to connect to Ollama: {e}"
        except Exception as e:
            return f"Error generating response: {e}"
    
    def _parse_actions(self, response: str) -> List[AgentAction]:
        """Parse the response for agent actions."""
        actions = []
        
        # Simple parsing - look for action keywords
        response_lower = response.lower()
        
        if 'create config' in response_lower or 'generate config' in response_lower:
            # Determine component type
            if 'model' in response_lower:
                actions.append(AgentAction(
                    agent_type="model",
                    action="create",
                    parameters={"type": "basic_config"}
                ))
            elif 'rag' in response_lower:
                actions.append(AgentAction(
                    agent_type="rag",
                    action="create",
                    parameters={"type": "basic_config"}
                ))
            elif 'prompt' in response_lower:
                actions.append(AgentAction(
                    agent_type="prompt",
                    action="create",
                    parameters={"type": "basic_config"}
                ))
        
        return actions
    
    def _generate_suggestions(self, response: str) -> List[str]:
        """Generate helpful suggestions based on the response."""
        suggestions = []
        
        # Add component-specific suggestions
        if 'model' in response.lower():
            suggestions.extend([
                "Try: 'Show me a production model config'",
                "Try: 'How do I set up multiple providers?'"
            ])
        elif 'rag' in response.lower():
            suggestions.extend([
                "Try: 'Create a RAG config for PDF documents'",
                "Try: 'How do I improve retrieval accuracy?'"
            ])
        elif 'prompt' in response.lower():
            suggestions.extend([
                "Try: 'Show me domain-specific prompt templates'",
                "Try: 'How do I set up A/B testing for prompts?'"
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the assistant."""
        return """You are Llama Brain, an AI assistant specialized in helping users create and configure LlamaFarm components.

You have access to comprehensive documentation about:
- Models: Multi-provider LLM management (OpenAI, Anthropic, Ollama, etc.)
- RAG: Document processing and retrieval systems
- Prompts: Template management and prompt engineering

When users ask for help:
1. Provide clear, actionable guidance
2. Reference specific configuration examples when available
3. Suggest next steps or alternatives
4. If asked to create configs, explain what you would generate

You are helpful, concise, and focused on practical solutions. Always ask for clarification if the user's request is ambiguous.

Available actions you can suggest:
- create config: Generate a new configuration file
- edit config: Modify an existing configuration
- validate config: Check configuration validity

Be conversational but professional. Use emojis sparingly and only when they add clarity."""
    
    # === STREAMLINED PIPELINE METHODS ===
    
    async def _get_prompt_template(self, user_message: str) -> str:
        """Get appropriate prompt template based on user intent."""
        from llama_brain.integrations import LlamaFarmClient
        
        client = LlamaFarmClient()
        message_lower = user_message.lower()
        
        # Detect intent and select prompt
        if any(word in message_lower for word in ['create', 'generate', 'make', 'build']):
            if 'model' in message_lower:
                return "model_creation_prompt"
            elif 'rag' in message_lower:
                return "rag_creation_prompt" 
            elif 'prompt' in message_lower:
                return "prompt_creation_prompt"
        elif any(word in message_lower for word in ['help', 'how', 'what', 'explain']):
            return "explanation_prompt"
        
        return "general_assistant_prompt"
    
    def _build_messages_with_prompt_and_context(self, session: ChatSession, prompt_template: str, context: str) -> List[Dict]:
        """Build messages with proper prompt template and RAG context injection."""
        messages = []
        
        # System message with specialized prompt
        system_prompt = self._get_specialized_system_prompt(prompt_template)
        messages.append({"role": "system", "content": system_prompt})
        
        # Get conversation history (excluding system messages)
        history = session.get_context(max_messages=8)
        for msg in history:
            if msg['role'] != 'system':
                messages.append(msg)
        
        # Inject context into the last user message
        if messages and messages[-1]['role'] == 'user':
            original_content = messages[-1]['content']
            
            # Smart context injection based on prompt template
            if context and context.strip() != "No specific documentation found.":
                if prompt_template.endswith('_creation_prompt'):
                    # For creation prompts, provide examples and best practices
                    messages[-1]['content'] = f"""Context from LlamaFarm documentation:
{context}

Based on the above documentation and examples, please address: {original_content}

Provide specific configuration examples and step-by-step guidance."""
                else:
                    # For general questions, provide relevant info
                    messages[-1]['content'] = f"""Relevant documentation:
{context}

User question: {original_content}"""
            
        return messages
    
    def _get_specialized_system_prompt(self, prompt_template: str) -> str:
        """Get specialized system prompt based on template type."""
        base_prompt = """You are Llama Brain, an expert AI assistant for LlamaFarm configuration management."""
        
        if prompt_template == "model_creation_prompt":
            return f"""{base_prompt}

SPECIALIZATION: Model Configuration Expert
- You help users create model configurations for various providers (OpenAI, Anthropic, Ollama, etc.)
- You provide working configuration examples with proper validation
- You explain provider-specific settings and best practices
- When users want configs created, you provide the exact YAML/JSON structure
- You suggest appropriate models for different use cases (development, production, cost-optimization)

ALWAYS provide actionable, specific configuration examples."""
            
        elif prompt_template == "rag_creation_prompt":
            return f"""{base_prompt}

SPECIALIZATION: RAG System Expert  
- You help users set up document processing and retrieval systems
- You recommend appropriate parsers for different document types
- You configure embedding models and vector stores
- You optimize retrieval strategies for different use cases
- When users want RAG configs, you provide complete JSON configurations

ALWAYS include parser, embedder, and vector store configurations."""

        elif prompt_template == "prompt_creation_prompt":
            return f"""{base_prompt}

SPECIALIZATION: Prompt Engineering Expert
- You help users create prompt templates and strategies
- You design domain-specific prompts for different industries
- You set up A/B testing and evaluation frameworks
- You create context-aware prompt systems
- When users want prompt configs, you provide complete YAML structures

ALWAYS include template definitions and strategy configurations."""

        else:
            return f"""{base_prompt}

You provide comprehensive help with LlamaFarm components. Use the provided documentation context to give accurate, specific guidance. When users want to create configurations, offer to help generate them."""
    
    def _extract_llm_params(self, user_message: str) -> Dict:
        """Extract custom LLM parameters from user message."""
        params = {}
        message_lower = user_message.lower()
        
        # Extract temperature
        import re
        temp_match = re.search(r'temp(?:erature)?\s*[:=]?\s*(0?\.\d+|\d+\.?\d*)', message_lower)
        if temp_match:
            try:
                params['temperature'] = float(temp_match.group(1))
            except ValueError:
                pass
        
        # Extract model preference
        if 'llama' in message_lower and '3.2' in message_lower:
            params['model'] = 'llama3.2:3b'
        elif 'llama3.1' in message_lower:
            params['model'] = 'llama3.1:8b'
            
        # Extract max tokens
        token_match = re.search(r'(?:max.?tokens?|tokens?)\s*[:=]?\s*(\d+)', message_lower)
        if token_match:
            try:
                params['max_tokens'] = int(token_match.group(1))
            except ValueError:
                pass
        
        return params
    
    async def _auto_trigger_agents(self, response: str, session: ChatSession) -> List[AgentAction]:
        """Auto-trigger agents based on response analysis."""
        actions = []
        response_lower = response.lower()
        user_message = session.get_last_user_message()
        
        # Enhanced intent detection
        if any(phrase in response_lower for phrase in [
            'create a config', 'generate config', 'here\'s the config', 
            'configuration file', 'i\'ll create', 'let me generate'
        ]) or any(phrase in user_message.lower() for phrase in [
            'create config', 'generate config', 'make a config', 'build config'
        ]):
            
            # Determine component and requirements
            component = self._detect_component(user_message)
            requirements = self._extract_requirements(user_message)
            
            if component:
                actions.append(AgentAction(
                    agent_type=component,
                    action="create",
                    parameters=requirements
                ))
        
        return actions
    
    def _extract_requirements(self, user_message: str) -> Dict:
        """Extract requirements from user message for agent actions."""
        requirements = {}
        message_lower = user_message.lower()
        
        # Use case detection
        if any(word in message_lower for word in ['dev', 'development', 'local']):
            requirements['use_case'] = 'development'
        elif any(word in message_lower for word in ['prod', 'production', 'deploy']):
            requirements['use_case'] = 'production'
        else:
            requirements['use_case'] = 'general'
        
        # Provider preferences
        providers = []
        if 'ollama' in message_lower:
            providers.append('ollama')
        if 'openai' in message_lower:
            providers.append('openai')
        if 'anthropic' in message_lower:
            providers.append('anthropic')
        
        if providers:
            requirements['providers'] = providers
        
        # Document types for RAG
        doc_types = []
        if 'pdf' in message_lower:
            doc_types.append('pdf')
        if 'markdown' in message_lower or 'md' in message_lower:
            doc_types.append('markdown')
        if 'text' in message_lower or 'txt' in message_lower:
            doc_types.append('text')
        
        if doc_types:
            requirements['document_types'] = doc_types
        
        # Domain for prompts
        if 'legal' in message_lower:
            requirements['domain'] = 'legal'
        elif 'medical' in message_lower:
            requirements['domain'] = 'medical'
        elif 'technical' in message_lower:
            requirements['domain'] = 'technical'
        
        return requirements
    
    def _should_auto_execute(self, action: AgentAction) -> bool:
        """Determine if an agent action should be executed automatically."""
        # Auto-execute config creation for now
        # In production, you might want user confirmation for some actions
        return action.action in ['create', 'validate']
    
    async def _execute_agent_action(self, action: AgentAction) -> Dict:
        """Execute an agent action and return results."""
        from llama_brain.agents import ModelAgent, RAGAgent, PromptAgent
        
        try:
            # Get the appropriate agent
            if action.agent_type == "model":
                agent = ModelAgent()
            elif action.agent_type == "rag":
                agent = RAGAgent()
            elif action.agent_type == "prompt":
                agent = PromptAgent()
            else:
                return {"success": False, "error": f"Unknown agent type: {action.agent_type}"}
            
            # Execute the action
            if action.action == "create":
                config = await agent.create_config(action.parameters)
                filename = f"{action.agent_type}_{action.parameters.get('use_case', 'custom')}"
                file_path = await agent.save_config(config, filename)
                
                return {
                    "success": True,
                    "action": action.action,
                    "agent_type": action.agent_type,
                    "config": config,
                    "file_path": str(file_path),
                    "message": f"✅ Created {action.agent_type} configuration at {file_path}"
                }
            
            elif action.action == "validate":
                validation = await agent.validate_config(action.parameters.get("config", {}))
                return {
                    "success": validation.get("valid", False),
                    "action": action.action, 
                    "agent_type": action.agent_type,
                    "validation": validation,
                    "message": f"Validation {'passed' if validation.get('valid') else 'failed'}"
                }
            
        except Exception as e:
            return {
                "success": False,
                "action": action.action,
                "agent_type": action.agent_type, 
                "error": str(e),
                "message": f"❌ Failed to execute {action.action} for {action.agent_type}: {e}"
            }
    
    async def _merge_agent_results(self, original_response: str, agent_result: Dict) -> str:
        """Merge agent execution results into the response."""
        if agent_result.get("success"):
            # Add success message to response
            return f"{original_response}\n\n{agent_result['message']}"
        else:
            # Add error message
            return f"{original_response}\n\n⚠️ {agent_result.get('message', 'Agent action failed')}"