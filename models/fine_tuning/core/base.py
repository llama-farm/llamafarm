"""
Base classes for fine-tuning system.

This module defines the abstract base classes and interfaces for the fine-tuning
system, following the same patterns as the RAG system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


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
    
    # Progress tracking
    current_epoch: int = Field(0, description="Current training epoch")
    total_epochs: int = Field(0, description="Total training epochs")
    current_step: int = Field(0, description="Current training step")
    total_steps: int = Field(0, description="Total training steps")
    
    # Metrics
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Training metrics")
    
    # Output paths
    output_dir: Optional[Path] = Field(None, description="Output directory")
    checkpoint_dir: Optional[Path] = Field(None, description="Checkpoint directory")
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ["pending", "running", "completed", "failed", "cancelled"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v


class BaseFineTuner(ABC):
    """Abstract base class for fine-tuning implementations."""
    
    def __init__(self, config: FineTuningConfig):
        """Initialize the fine-tuner with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._current_job: Optional[TrainingJob] = None
        
    @abstractmethod
    def validate_config(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    @abstractmethod
    def prepare_model(self) -> Any:
        """
        Prepare the base model for fine-tuning.
        
        Returns:
            Prepared model object
        """
        pass
    
    @abstractmethod
    def prepare_dataset(self) -> Any:
        """
        Prepare the dataset for training.
        
        Returns:
            Prepared dataset object
        """
        pass
    
    @abstractmethod
    def start_training(self, job_id: Optional[str] = None) -> TrainingJob:
        """
        Start the fine-tuning process.
        
        Args:
            job_id: Optional job identifier
            
        Returns:
            Training job object
        """
        pass
    
    @abstractmethod
    def stop_training(self) -> None:
        """Stop the current training process."""
        pass
    
    @abstractmethod
    def get_training_status(self) -> Optional[TrainingJob]:
        """
        Get the current training status.
        
        Returns:
            Current training job or None if not training
        """
        pass
    
    @abstractmethod
    def resume_training(self, checkpoint_path: Path) -> TrainingJob:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Training job object
        """
        pass
    
    @abstractmethod
    def evaluate_model(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Evaluate the fine-tuned model.
        
        Args:
            checkpoint_path: Optional path to specific checkpoint
            
        Returns:
            Evaluation metrics
        """
        pass
    
    @abstractmethod
    def export_model(self, output_path: Path, format: str = "pytorch") -> None:
        """
        Export the fine-tuned model.
        
        Args:
            output_path: Output path for exported model
            format: Export format ("pytorch", "onnx", "gguf", etc.)
        """
        pass
    
    def get_supported_methods(self) -> List[str]:
        """
        Get list of supported fine-tuning methods.
        
        Returns:
            List of supported method names
        """
        return ["lora", "qlora", "full_finetune"]
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported base models.
        
        Returns:
            List of supported model names
        """
        return ["llama", "mistral", "gpt2", "t5"]
    
    def estimate_resources(self) -> Dict[str, Any]:
        """
        Estimate resource requirements for the current configuration.
        
        Returns:
            Resource estimates (memory, time, storage, etc.)
        """
        # Default implementation with basic estimates
        method_type = self.config.method.get("type", "lora")
        base_model = self.config.base_model.get("name", "unknown")
        
        # Basic estimates based on method and model
        estimates = {
            "memory_gb": 8,
            "training_time_hours": 2,
            "storage_gb": 10,
            "gpu_required": False
        }
        
        # Adjust based on method
        if method_type == "full_finetune":
            estimates["memory_gb"] *= 4
            estimates["training_time_hours"] *= 3
            estimates["gpu_required"] = True
        elif method_type == "qlora":
            estimates["memory_gb"] *= 0.5
            estimates["gpu_required"] = True
            
        # Adjust based on model size (rough estimates)
        if "3b" in base_model.lower():
            estimates["memory_gb"] *= 0.5
        elif "8b" in base_model.lower():
            estimates["memory_gb"] *= 1.0
        elif "13b" in base_model.lower():
            estimates["memory_gb"] *= 1.5
        elif "70b" in base_model.lower():
            estimates["memory_gb"] *= 8
            estimates["training_time_hours"] *= 4
            
        return estimates
    
    def get_hardware_recommendations(self) -> Dict[str, Any]:
        """
        Get hardware recommendations for the current configuration.
        
        Returns:
            Hardware recommendations
        """
        estimates = self.estimate_resources()
        
        recommendations = {
            "min_memory_gb": estimates["memory_gb"],
            "recommended_memory_gb": estimates["memory_gb"] * 1.5,
            "gpu_required": estimates["gpu_required"],
            "estimated_time": f"{estimates['training_time_hours']:.1f} hours",
            "storage_required_gb": estimates["storage_gb"]
        }
        
        # Add specific hardware recommendations
        if estimates["memory_gb"] <= 8:
            recommendations["suitable_hardware"] = ["Mac M1/M2", "RTX 3060", "RTX 4060"]
        elif estimates["memory_gb"] <= 16:
            recommendations["suitable_hardware"] = ["Mac Studio", "RTX 3070", "RTX 4070"]
        elif estimates["memory_gb"] <= 24:
            recommendations["suitable_hardware"] = ["RTX 3090", "RTX 4090", "RTX A5000"]
        else:
            recommendations["suitable_hardware"] = ["A100", "H100", "Multi-GPU setup"]
            
        return recommendations


class BaseDataProcessor(ABC):
    """Abstract base class for dataset processors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processor with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process_dataset(self, dataset_path: Union[str, Path]) -> Any:
        """
        Process the dataset for fine-tuning.
        
        Args:
            dataset_path: Path to dataset file or directory
            
        Returns:
            Processed dataset
        """
        pass
    
    @abstractmethod
    def validate_dataset(self, dataset_path: Union[str, Path]) -> List[str]:
        """
        Validate the dataset format and content.
        
        Args:
            dataset_path: Path to dataset file or directory
            
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    def get_dataset_info(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Args:
            dataset_path: Path to dataset file or directory
            
        Returns:
            Dataset information
        """
        path = Path(dataset_path)
        
        info = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "format": path.suffix.lower() if path.is_file() else "directory"
        }
        
        return info


class BaseStrategy(ABC):
    """Abstract base class for fine-tuning strategies."""
    
    def __init__(self, strategy_config: Dict[str, Any]):
        """Initialize the strategy with configuration."""
        self.strategy_config = strategy_config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def to_config(self, overrides: Optional[Dict[str, Any]] = None) -> FineTuningConfig:
        """
        Convert strategy to fine-tuning configuration.
        
        Args:
            overrides: Optional configuration overrides
            
        Returns:
            Fine-tuning configuration
        """
        pass
    
    @abstractmethod
    def validate_hardware(self) -> List[str]:
        """
        Validate hardware requirements for this strategy.
        
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this strategy.
        
        Returns:
            Strategy information
        """
        return {
            "name": self.strategy_config.get("name", "Unknown Strategy"),
            "description": self.strategy_config.get("description", ""),
            "use_cases": self.strategy_config.get("use_cases", []),
            "hardware_requirements": self.strategy_config.get("hardware_requirements", {}),
            "performance_priority": self.strategy_config.get("performance_priority", "balanced"),
            "resource_usage": self.strategy_config.get("resource_usage", "medium"),
            "complexity": self.strategy_config.get("complexity", "moderate"),
            "training_time_estimate": self.strategy_config.get("training_time_estimate", "Unknown")
        }