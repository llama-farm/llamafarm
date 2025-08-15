"""
Ollama model application integration.

This module provides integration with Ollama for running models locally.
"""

import os
import json
import subprocess
import requests
from typing import Dict, Any, Optional, List, Generator, Union
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


class OllamaApp:
    """Ollama application for running models locally."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama app."""
        self.config = config
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 300)
        self._process = None
        
    def is_running(self) -> bool:
        """Check if Ollama service is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_service(self) -> bool:
        """Start Ollama service if not running."""
        if self.is_running():
            logger.info("Ollama service already running")
            return True
        
        try:
            logger.info("Starting Ollama service...")
            self._process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for service to start
            for _ in range(30):  # 30 second timeout
                if self.is_running():
                    logger.info("Ollama service started successfully")
                    return True
                time.sleep(1)
            
            logger.error("Ollama service failed to start within timeout")
            return False
            
        except FileNotFoundError:
            logger.error("Ollama not found. Please install from https://ollama.ai")
            return False
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return False
    
    def stop_service(self) -> None:
        """Stop Ollama service."""
        if self._process:
            logger.info("Stopping Ollama service...")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            logger.info(f"Pulling model: {model_name}")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=None  # No timeout for model downloads
            )
            response.raise_for_status()
            
            # Stream progress
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        logger.info(f"Pull progress: {data['status']}")
                    if "error" in data:
                        logger.error(f"Pull error: {data['error']}")
                        return False
            
            logger.info(f"Successfully pulled model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model."""
        try:
            response = requests.delete(
                f"{self.base_url}/api/delete",
                json={"name": model_name},
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"Deleted model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False
    
    def generate(self, prompt: str, model: Optional[str] = None, 
                 stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text using Ollama model."""
        # If a specific model is requested, check if it's available in Ollama
        # If not, fall back to the configured default model
        if model:
            available_models = [m["name"] for m in self.list_models()]
            if model not in available_models:
                logger.info(f"Model {model} not available in Ollama, using default: {self.config.get('default_model')}")
                model = self.config.get("default_model", "llama3.2:3b")
        else:
            model = self.config.get("default_model", "llama3.2:3b")
        
        # Prepare request
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        # Add generation parameters
        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            options["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            options["top_k"] = kwargs["top_k"]
        if "seed" in kwargs:
            options["seed"] = kwargs["seed"]
        
        if options:
            data["options"] = options
        
        try:
            url = f"{self.base_url}/api/generate"
            logger.info(f"Making request to {url} with model {model}")
            logger.debug(f"Request data: {data}")
            
            response = requests.post(
                url,
                json=data,
                stream=stream,
                timeout=None if stream else self.timeout
            )
            response.raise_for_status()
            
            if stream:
                return self._stream_response(response)
            else:
                result = response.json()
                return result.get("response", "")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if stream:
                return iter(["Error: Generation failed"])
            else:
                return "Error: Generation failed"
    
    def _stream_response(self, response) -> Generator[str, None, None]:
        """Stream response from Ollama."""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Chat with Ollama model."""
        model = model or self.config.get("default_model", "llama3.2")
        
        # Prepare request
        data = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        # Add generation parameters
        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs["max_tokens"]
        
        if options:
            data["options"] = options
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=data,
                stream=stream,
                timeout=None if stream else self.timeout
            )
            response.raise_for_status()
            
            if stream:
                return self._stream_chat_response(response)
            else:
                result = response.json()
                return result.get("message", {}).get("content", "")
                
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            if stream:
                return iter(["Error: Chat failed"])
            else:
                return "Error: Chat failed"
    
    def _stream_chat_response(self, response) -> Generator[str, None, None]:
        """Stream chat response from Ollama."""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
    
    def create_model(self, name: str, modelfile: str) -> bool:
        """Create a custom model from a Modelfile."""
        try:
            response = requests.post(
                f"{self.base_url}/api/create",
                json={"name": name, "modelfile": modelfile},
                stream=True,
                timeout=None
            )
            response.raise_for_status()
            
            # Stream progress
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        logger.info(f"Create progress: {data['status']}")
                    if "error" in data:
                        logger.error(f"Create error: {data['error']}")
                        return False
            
            logger.info(f"Successfully created model: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None
    
    def check_gpu_support(self) -> Dict[str, Any]:
        """Check GPU support and availability."""
        # This would typically check nvidia-smi or similar
        # For now, return basic info
        return {
            "gpu_available": False,  # Would check actual GPU
            "gpu_memory": 0,
            "recommendation": "CPU mode - consider GPU for better performance"
        }