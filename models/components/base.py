"""Base classes for model components."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Generator, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np


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


class FineTuningConfig(BaseModel):
    """Configuration model for fine-tuning operations."""
    
    # Base model configuration
    base_model: Dict[str, Any] = Field(..., description="Base model configuration")
    
    # Fine-tuning method configuration
    method: Dict[str, Any] = Field(..., description="Fine-tuning method configuration")
    
    # Framework configuration
    framework: Dict[str, Any] = Field(..., description="Framework configuration")
    
    # Training arguments
    training_args: Dict[str, Any] = Field(..., description="Training arguments")
    
    # Dataset configuration
    dataset: Dict[str, Any] = Field(..., description="Dataset configuration")
    
    # Environment configuration
    environment: Dict[str, Any] = Field(default_factory=dict, description="Environment settings")
    
    # Output configuration
    output: Dict[str, Any] = Field(default_factory=dict, description="Output settings")
    
    # Hardware optimizations
    hardware_optimizations: Dict[str, Any] = Field(default_factory=dict, description="Hardware optimizations")
    
    # Advanced configuration
    advanced: Dict[str, Any] = Field(default_factory=dict, description="Advanced settings")
    
    class Config:
        extra = "allow"  # Allow additional fields for flexibility


class TrainingJob(BaseModel):
    """Represents a fine-tuning job."""
    
    job_id: str = Field(..., description="Unique job identifier")
    name: Optional[str] = Field(None, description="Human-readable job name")
    status: str = Field("pending", description="Job status")
    config: FineTuningConfig = Field(..., description="Job configuration")
    
    # Timing information
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    # Progress information
    current_epoch: int = Field(default=0, description="Current training epoch")
    total_epochs: int = Field(default=1, description="Total training epochs")
    current_step: int = Field(default=0, description="Current training step")
    total_steps: int = Field(default=0, description="Total training steps")
    
    # Results and metrics
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Training metrics")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Output paths
    output_dir: Optional[Path] = Field(None, description="Output directory")
    checkpoint_dir: Optional[Path] = Field(None, description="Checkpoint directory")
    
    def is_complete(self) -> bool:
        """Check if the job is complete."""
        return self.status in ["completed", "failed", "cancelled"]
    
    def get_progress(self) -> float:
        """Get training progress as a percentage."""
        if self.total_steps > 0:
            return (self.current_step / self.total_steps) * 100
        elif self.total_epochs > 0:
            return (self.current_epoch / self.total_epochs) * 100
        return 0.0


class Detection(BaseModel):
    """Represents a detection result."""
    
    label: str = Field(..., description="Class label")
    confidence: float = Field(..., description="Confidence score")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Classification(BaseModel):
    """Represents a classification result."""
    
    label: str = Field(..., description="Class label")
    confidence: float = Field(..., description="Confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TrainingExample(BaseModel):
    """Represents a training example for few-shot learning."""
    
    image_path: str = Field(..., description="Path to image")
    annotations: Optional[Dict[str, Any]] = Field(None, description="Annotations (bbox, labels, etc)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseImageRecognizer(ABC):
    """Base class for image recognition models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize image recognizer with configuration."""
        self.config = config
        self.device = self._detect_device()
        self.model = None
    
    @abstractmethod
    def _detect_device(self) -> str:
        """Detect and return the best available device."""
        pass
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the model weights."""
        pass
    
    @abstractmethod
    def detect(self, image_path: Union[Path, str], 
               confidence: float = 0.5,
               nms_threshold: float = 0.45) -> List[Detection]:
        """Perform object detection on an image."""
        pass
    
    @abstractmethod
    def classify(self, image_path: Union[Path, str],
                 top_k: int = 5) -> List[Classification]:
        """Perform image classification."""
        pass
    
    @abstractmethod
    def segment(self, image_path: Union[Path, str]) -> np.ndarray:
        """Perform image segmentation."""
        pass
    
    @abstractmethod
    def batch_process(self, image_paths: List[Union[Path, str]],
                     task: str = "detect",
                     batch_size: int = 32) -> List[Any]:
        """Process multiple images in batch."""
        pass
    
    @abstractmethod
    def export_model(self, output_path: Path, format: str = "onnx") -> None:
        """Export model to different formats."""
        pass
    
    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported tasks."""
        pass
    
    @abstractmethod
    def visualize(self, image_path: Union[Path, str],
                  results: Union[List[Detection], List[Classification], np.ndarray],
                  output_path: Optional[Path] = None) -> Optional[np.ndarray]:
        """Visualize results on image."""
        pass


class BaseImageTrainer(ABC):
    """Base class for few-shot image model training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration."""
        self.config = config
        self.examples: List[TrainingExample] = []
        self.model = None
    
    @abstractmethod
    def add_examples(self, examples: List[TrainingExample]) -> None:
        """Add training examples."""
        pass
    
    @abstractmethod
    def prepare_dataset(self) -> Any:
        """Prepare dataset from examples."""
        pass
    
    @abstractmethod
    def train_few_shot(self, base_model: str,
                      epochs: int = 10,
                      batch_size: int = 4,
                      learning_rate: float = 0.001) -> Any:
        """Train model with few-shot learning."""
        pass
    
    @abstractmethod
    def evaluate(self, test_data: List[Union[Path, str]]) -> Dict[str, float]:
        """Evaluate model performance."""
        pass
    
    @abstractmethod
    def save_model(self, output_path: Path) -> None:
        """Save trained model."""
        pass
    
    @abstractmethod
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        pass
    
    @abstractmethod
    def apply_augmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply data augmentation to increase training data."""
        pass