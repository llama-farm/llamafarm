"""Base classes for model components."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Generator
from pathlib import Path


class BaseFineTuner(ABC):
    """Base class for all fine-tuning implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize fine-tuner with configuration."""
        self.config = config
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        pass
    
    @abstractmethod
    def prepare_model(self) -> Any:
        """Prepare the model for fine-tuning."""
        pass
    
    @abstractmethod
    def prepare_dataset(self) -> Any:
        """Prepare the dataset for training."""
        pass
    
    @abstractmethod
    def start_training(self, job_id: Optional[str] = None) -> Any:
        """Start the fine-tuning process."""
        pass
    
    @abstractmethod
    def stop_training(self) -> None:
        """Stop the current training process."""
        pass
    
    @abstractmethod
    def get_training_status(self) -> Optional[Any]:
        """Get the current training status."""
        pass
    
    @abstractmethod
    def export_model(self, output_path: Path, format: str = "pytorch") -> None:
        """Export the fine-tuned model."""
        pass
    
    @abstractmethod
    def get_supported_methods(self) -> List[str]:
        """Get list of supported fine-tuning methods."""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported model architectures."""
        pass


class BaseModelApp(ABC):
    """Base class for model application runners (Ollama, vLLM, etc)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model app with configuration."""
        self.config = config
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the service is running."""
        pass
    
    @abstractmethod
    def start_service(self) -> bool:
        """Start the model service."""
        pass
    
    @abstractmethod
    def stop_service(self) -> None:
        """Stop the model service."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, model: Optional[str] = None, 
                 stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Chat with the model."""
        pass


class BaseModelRepository(ABC):
    """Base class for model repository integrations (HuggingFace, etc)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize repository with configuration."""
        self.config = config
    
    @abstractmethod
    def search_models(self, query: str, **filters) -> List[Dict[str, Any]]:
        """Search for models in the repository."""
        pass
    
    @abstractmethod
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        pass
    
    @abstractmethod
    def download_model(self, model_id: str, output_path: Path) -> bool:
        """Download a model from the repository."""
        pass
    
    @abstractmethod
    def upload_model(self, model_path: Path, model_id: str, **metadata) -> bool:
        """Upload a model to the repository."""
        pass
    
    @abstractmethod
    def list_user_models(self, username: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models for a specific user."""
        pass


class BaseCloudAPI(ABC):
    """Base class for cloud API integrations (OpenAI, Claude, etc)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize cloud API with configuration."""
        self.config = config
        self.api_key = config.get("api_key")
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate API credentials."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, model: Optional[str] = None, 
                 stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Chat with the model."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text for the specified model."""
        pass
    
    @abstractmethod
    def get_usage(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        pass