"""
OpenAI-Compatible API integration.

This module provides integration with OpenAI-compatible APIs including:
- OpenAI (GPT models)
- Anthropic Claude (via API)
- Google Gemini
- Together.ai
- Anyscale
- Perplexity
- Groq
- Mistral AI
- Deepseek
- xAI Grok
- And other OpenAI-compatible endpoints
"""

import os
from typing import Dict, Any, Optional, List, Union, Generator
import logging
import tiktoken

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from ...base import BaseCloudAPI

logger = logging.getLogger(__name__)


class OpenAICompatibleAPI(BaseCloudAPI):
    """OpenAI-compatible API implementation for multiple providers."""
    
    # Provider configurations
    PROVIDER_CONFIGS = {
        "openai": {
            "base_url": None,  # Uses default OpenAI URL
            "env_var": "OPENAI_API_KEY",
            "default_model": "gpt-4o-mini"
        },
        "together": {
            "base_url": "https://api.together.xyz/v1",
            "env_var": "TOGETHER_API_KEY",
            "default_model": "meta-llama/Llama-3-70b-chat-hf"
        },
        "anyscale": {
            "base_url": "https://api.endpoints.anyscale.com/v1",
            "env_var": "ANYSCALE_API_KEY",
            "default_model": "meta-llama/Llama-2-70b-chat-hf"
        },
        "perplexity": {
            "base_url": "https://api.perplexity.ai",
            "env_var": "PERPLEXITY_API_KEY",
            "default_model": "llama-3.1-sonar-large-128k-online"
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "env_var": "GROQ_API_KEY",
            "default_model": "llama-3.1-70b-versatile"
        },
        "mistral": {
            "base_url": "https://api.mistral.ai/v1",
            "env_var": "MISTRAL_API_KEY",
            "default_model": "mistral-large-latest"
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "env_var": "DEEPSEEK_API_KEY",
            "default_model": "deepseek-chat"
        },
        "grok": {
            "base_url": "https://api.x.ai/v1",
            "env_var": "XAI_API_KEY",
            "default_model": "grok-beta"
        },
        "fireworks": {
            "base_url": "https://api.fireworks.ai/inference/v1",
            "env_var": "FIREWORKS_API_KEY",
            "default_model": "accounts/fireworks/models/llama-v3p1-70b-instruct"
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "env_var": "OPENROUTER_API_KEY",
            "default_model": "meta-llama/llama-3.1-70b-instruct"
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI-compatible API."""
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        # Get provider-specific configuration
        self.provider = config.get("provider", "openai").lower()
        
        # Get provider config or use custom settings
        if self.provider in self.PROVIDER_CONFIGS:
            provider_config = self.PROVIDER_CONFIGS[self.provider]
            self.base_url = config.get("base_url", provider_config["base_url"])
            default_env_var = provider_config["env_var"]
            self.default_model = config.get("default_model", provider_config["default_model"])
        else:
            # Custom provider
            self.base_url = config.get("base_url")
            default_env_var = "API_KEY"
            self.default_model = config.get("default_model", "gpt-3.5-turbo")
        
        # Get API key from config or environment
        api_key = config.get("api_key") or os.getenv(default_env_var)
        
        if not api_key:
            raise ValueError(f"API key not provided for {self.provider}. Set {default_env_var} environment variable or provide 'api_key' in config.")
        
        # Create client with optional base URL for different providers
        client_kwargs = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        # For OpenAI, clear organization env vars that might interfere
        if self.provider == "openai":
            os.environ.pop('OPENAI_ORG_ID', None)
            os.environ.pop('OPENAI_ORGANIZATION', None)
            client_kwargs["organization"] = None
        
        self.client = OpenAI(**client_kwargs)
        
        # Debug: Log provider and configuration
        logger.debug(f"{self.provider} client created")
        if self.base_url:
            logger.debug(f"Using base URL: {self.base_url}")
    
    def _get_provider_api_key(self) -> Optional[str]:
        """Get API key for the configured provider from environment."""
        if self.provider in self.PROVIDER_CONFIGS:
            env_var = self.PROVIDER_CONFIGS[self.provider]["env_var"]
            return os.getenv(env_var)
        return None
    
    def validate_credentials(self) -> bool:
        """Validate API credentials."""
        try:
            # Try to list models to validate credentials
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Failed to validate {self.provider} credentials: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            models = self.client.models.list()
            return [
                {
                    "id": model.id,
                    "object": model.object,
                    "created": model.created,
                    "owned_by": model.owned_by
                }
                for model in models.data
            ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            # Return provider-specific models if listing fails
            if self.provider in self.PROVIDER_CONFIGS:
                return [{"id": self.default_model, "provider": self.provider}]
            return []
    
    def generate(self, prompt: str, model: Optional[str] = None, 
                 stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text from prompt."""
        model = model or self.default_model
        
        # Most modern models use chat format
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, model, stream, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Chat with the model."""
        model = model or self.default_model
        
        try:
            # Debug: Log request details
            logger.debug(f"Making {self.provider} API call with model: {model}")
            
            params = {
                "model": model,
                "messages": messages,
                "stream": stream
            }
            
            # Add optional parameters
            if "max_tokens" in kwargs:
                params["max_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                params["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                params["top_p"] = kwargs["top_p"]
            if "frequency_penalty" in kwargs:
                params["frequency_penalty"] = kwargs["frequency_penalty"]
            if "presence_penalty" in kwargs:
                params["presence_penalty"] = kwargs["presence_penalty"]
            if "stop" in kwargs:
                params["stop"] = kwargs["stop"]
            if "tools" in kwargs:
                params["tools"] = kwargs["tools"]
            if "tool_choice" in kwargs:
                params["tool_choice"] = kwargs["tool_choice"]
            
            response = self.client.chat.completions.create(**params)
            
            if stream:
                return self._stream_chat_response(response)
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            # Log additional details for debugging
            if hasattr(e, 'response'):
                logger.debug(f"Error response: {getattr(e.response, 'text', 'No text')}")
            logger.debug(f"Provider: {self.provider}")
            raise e  # Re-raise to allow fallback handling
    
    def _stream_completion_response(self, response) -> Generator[str, None, None]:
        """Stream completion response."""
        for chunk in response:
            if chunk.choices[0].text:
                yield chunk.choices[0].text
    
    def _stream_chat_response(self, response) -> Generator[str, None, None]:
        """Stream chat response."""
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text for the specified model."""
        model = model or self.default_model
        
        try:
            # Get the appropriate encoding for the model
            if "gpt-4" in model:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to cl100k_base encoding for other models
                encoding = tiktoken.get_encoding("cl100k_base")
            
            tokens = encoding.encode(text)
            return len(tokens)
            
        except Exception as e:
            logger.error(f"Failed to count tokens: {e}")
            # Rough estimate if tiktoken fails
            return len(text) // 4
    
    def get_usage(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        # Most providers don't provide direct usage API
        provider_dashboards = {
            "openai": "https://platform.openai.com/usage",
            "together": "https://api.together.xyz/dashboard",
            "anyscale": "https://console.anyscale.com",
            "perplexity": "https://www.perplexity.ai/settings/api",
            "groq": "https://console.groq.com",
            "mistral": "https://console.mistral.ai",
            "deepseek": "https://platform.deepseek.com",
            "grok": "https://console.x.ai",
            "fireworks": "https://app.fireworks.ai",
            "openrouter": "https://openrouter.ai/dashboard"
        }
        
        dashboard_url = provider_dashboards.get(self.provider, "Check provider dashboard")
        
        return {
            "provider": self.provider,
            "note": f"Usage statistics available at {dashboard_url}",
            "recommendation": f"Use {self.provider} dashboard for detailed usage tracking"
        }
    
    def create_embedding(self, text: Union[str, List[str]], 
                        model: Optional[str] = None) -> List[float]:
        """Create embeddings for text."""
        # Default embedding models per provider
        embedding_models = {
            "openai": "text-embedding-ada-002",
            "together": "togethercomputer/m2-bert-80M-8k-retrieval",
            "mistral": "mistral-embed",
            "groq": "nomic-embed-text-v1.5"
        }
        
        if model is None:
            model = embedding_models.get(self.provider, "text-embedding-ada-002")
        
        try:
            if isinstance(text, str):
                text = [text]
            
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            
            # Return first embedding if single text, otherwise all embeddings
            if len(text) == 1:
                return response.data[0].embedding
            else:
                return [item.embedding for item in response.data]
                
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return []
    
    def moderate_content(self, text: str) -> Dict[str, Any]:
        """Check content for policy violations (OpenAI only)."""
        if self.provider != "openai":
            return {
                "note": f"Content moderation not available for {self.provider}",
                "flagged": False
            }
        
        try:
            response = self.client.moderations.create(input=text)
            result = response.results[0]
            
            return {
                "flagged": result.flagged,
                "categories": result.categories.model_dump(),
                "category_scores": result.category_scores.model_dump()
            }
            
        except Exception as e:
            logger.error(f"Failed to moderate content: {e}")
            return {"error": str(e)}