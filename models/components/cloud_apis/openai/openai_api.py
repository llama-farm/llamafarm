"""
OpenAI API integration.

This module provides integration with OpenAI's API for GPT models.
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


class OpenAIAPI(BaseCloudAPI):
    """OpenAI API implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI API."""
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        # Get API key from config or environment
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        # Create client without organization to avoid mismatched organization errors
        # Explicitly set organization=None to prevent auto-reading from OPENAI_ORG_ID env var
        self.client = OpenAI(api_key=api_key, organization=None)
        
        # Debug: Log what organization is set
        logger.debug(f"OpenAI client created with organization: {self.client.organization}")
            
        self.default_model = config.get("default_model", "gpt-4o-mini")
    
    def validate_credentials(self) -> bool:
        """Validate API credentials."""
        try:
            # Try to list models to validate credentials
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Failed to validate OpenAI credentials: {e}")
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
                if model.id.startswith(("gpt", "text-"))  # Filter to text models
            ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def generate(self, prompt: str, model: Optional[str] = None, 
                 stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text from prompt."""
        model = model or self.default_model
        
        # Convert to chat format for chat models
        if model.startswith("gpt"):
            messages = [{"role": "user", "content": prompt}]
            return self.chat(messages, model, stream, **kwargs)
        
        # For completion models (legacy)
        try:
            params = {
                "model": model,
                "prompt": prompt,
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
            
            response = self.client.completions.create(**params)
            
            if stream:
                return self._stream_completion_response(response)
            else:
                return response.choices[0].text
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise e  # Re-raise to allow fallback handling
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Chat with the model."""
        model = model or self.default_model
        
        try:
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
            logger.debug(f"Client organization: {self.client.organization}")
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
            if model.startswith("gpt-4"):
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif model.startswith("gpt-3.5"):
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to cl100k_base encoding
                encoding = tiktoken.get_encoding("cl100k_base")
            
            tokens = encoding.encode(text)
            return len(tokens)
            
        except Exception as e:
            logger.error(f"Failed to count tokens: {e}")
            # Rough estimate if tiktoken fails
            return len(text) // 4
    
    def get_usage(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        # OpenAI doesn't provide direct usage API
        # This would need to be tracked separately or through their dashboard
        return {
            "note": "Usage statistics available at https://platform.openai.com/usage",
            "recommendation": "Use OpenAI dashboard for detailed usage tracking"
        }
    
    def create_embedding(self, text: Union[str, List[str]], 
                        model: str = "text-embedding-ada-002") -> List[float]:
        """Create embeddings for text."""
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
        """Check content for policy violations."""
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