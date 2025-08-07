"""
Model Manager for coordinating all model operations.

This module provides a unified interface for model operations including
fine-tuning, inference, and model management.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from components import (
    FineTunerFactory,
    ModelAppFactory,
    ModelRepositoryFactory,
    CloudAPIFactory
)
from core.strategy_manager import StrategyManager
from core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ModelManager:
    """Central manager for all model operations."""
    
    def __init__(self, config: Optional[Union[str, Path, Dict[str, Any]]] = None):
        """Initialize Model Manager.
        
        Args:
            config: Configuration file path, dict, or None for defaults
        """
        self.config_loader = ConfigLoader()
        
        if config:
            if isinstance(config, (str, Path)):
                self.config = self.config_loader.load_config(config)
            else:
                self.config = config
        else:
            self.config = self.config_loader.get_default_config()
        
        self.strategy_manager = StrategyManager()
        
        # Component instances
        self._fine_tuner = None
        self._model_app = None
        self._repository = None
        self._cloud_api = None
    
    @classmethod
    def from_strategy(cls, strategy_name: str, overrides: Optional[Dict[str, Any]] = None):
        """Create ModelManager from a predefined strategy.
        
        Args:
            strategy_name: Name of the strategy to use
            overrides: Optional configuration overrides
        """
        strategy_manager = StrategyManager()
        config = strategy_manager.load_strategy(strategy_name, overrides)
        return cls(config)
    
    def get_fine_tuner(self) -> Optional[Any]:
        """Get or create fine-tuner instance."""
        if not self._fine_tuner and "fine_tuner" in self.config:
            self._fine_tuner = FineTunerFactory.create(self.config["fine_tuner"])
        return self._fine_tuner
    
    def get_model_app(self) -> Optional[Any]:
        """Get or create model app instance."""
        if not self._model_app and "model_app" in self.config:
            self._model_app = ModelAppFactory.create(self.config["model_app"])
        return self._model_app
    
    def get_repository(self) -> Optional[Any]:
        """Get or create repository instance."""
        if not self._repository and "repository" in self.config:
            self._repository = ModelRepositoryFactory.create(self.config["repository"])
        return self._repository
    
    def get_cloud_api(self) -> Optional[Any]:
        """Get or create cloud API instance."""
        if not self._cloud_api and "cloud_api" in self.config:
            self._cloud_api = CloudAPIFactory.create(self.config["cloud_api"])
        return self._cloud_api
    
    # Fine-tuning operations
    def fine_tune(self, dataset_path: str, output_dir: str, **kwargs) -> Dict[str, Any]:
        """Start fine-tuning process."""
        fine_tuner = self.get_fine_tuner()
        if not fine_tuner:
            raise ValueError("No fine-tuner configured")
        
        # Update configuration
        if "dataset" not in fine_tuner.config:
            fine_tuner.config["dataset"] = {}
        fine_tuner.config["dataset"]["path"] = dataset_path
        
        if "training_args" not in fine_tuner.config:
            fine_tuner.config["training_args"] = {}
        fine_tuner.config["training_args"]["output_dir"] = output_dir
        
        # Apply any additional kwargs
        for key, value in kwargs.items():
            if key in ["method", "training_args", "base_model"]:
                fine_tuner.config[key].update(value)
        
        # Validate configuration
        errors = fine_tuner.validate_config()
        if errors:
            raise ValueError(f"Configuration errors: {errors}")
        
        # Start training
        job = fine_tuner.start_training()
        return {
            "job_id": job.job_id,
            "status": job.status,
            "started_at": str(job.started_at)
        }
    
    def get_training_status(self) -> Optional[Dict[str, Any]]:
        """Get current training status."""
        fine_tuner = self.get_fine_tuner()
        if not fine_tuner:
            return None
        
        status = fine_tuner.get_training_status()
        if status:
            return {
                "job_id": status.job_id,
                "status": status.status,
                "current_epoch": status.current_epoch,
                "total_epochs": status.total_epochs,
                "metrics": status.metrics
            }
        return None
    
    # Inference operations
    def generate(self, prompt: str, model: Optional[str] = None, 
                 stream: bool = False, **kwargs) -> Union[str, Any]:
        """Generate text using configured model."""
        # Try cloud API first if configured
        cloud_api = self.get_cloud_api()
        if cloud_api:
            try:
                return cloud_api.generate(prompt, model, stream, **kwargs)
            except Exception as e:
                logger.warning(f"Cloud API failed: {e}, trying local model")
        
        # Fall back to local model app
        model_app = self.get_model_app()
        if model_app:
            # Ensure service is running
            if not model_app.is_running():
                model_app.start_service()
            
            return model_app.generate(prompt, model, stream, **kwargs)
        
        raise ValueError("No model configured for generation")
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             stream: bool = False, **kwargs) -> Union[str, Any]:
        """Chat with configured model."""
        # Try cloud API first if configured
        cloud_api = self.get_cloud_api()
        if cloud_api:
            try:
                return cloud_api.chat(messages, model, stream, **kwargs)
            except Exception as e:
                logger.warning(f"Cloud API failed: {e}, trying local model")
        
        # Fall back to local model app
        model_app = self.get_model_app()
        if model_app:
            # Ensure service is running
            if not model_app.is_running():
                model_app.start_service()
            
            return model_app.chat(messages, model, stream, **kwargs)
        
        raise ValueError("No model configured for chat")
    
    # Repository operations
    def search_models(self, query: str, **filters) -> List[Dict[str, Any]]:
        """Search for models in repository."""
        repository = self.get_repository()
        if not repository:
            raise ValueError("No repository configured")
        
        return repository.search_models(query, **filters)
    
    def download_model(self, model_id: str, output_path: str) -> bool:
        """Download model from repository."""
        repository = self.get_repository()
        if not repository:
            raise ValueError("No repository configured")
        
        return repository.download_model(model_id, Path(output_path))
    
    def upload_model(self, model_path: str, model_id: str, **metadata) -> bool:
        """Upload model to repository."""
        repository = self.get_repository()
        if not repository:
            raise ValueError("No repository configured")
        
        return repository.upload_model(Path(model_path), model_id, **metadata)
    
    # Utility methods
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models from all sources."""
        models = {}
        
        # Local models
        model_app = self.get_model_app()
        if model_app:
            try:
                models["local"] = [m["name"] for m in model_app.list_models()]
            except:
                models["local"] = []
        
        # Cloud models
        cloud_api = self.get_cloud_api()
        if cloud_api:
            try:
                models["cloud"] = [m["id"] for m in cloud_api.list_models()]
            except:
                models["cloud"] = []
        
        return models
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate all configured components."""
        validation_results = {}
        
        # Validate fine-tuner
        fine_tuner = self.get_fine_tuner()
        if fine_tuner:
            validation_results["fine_tuner"] = fine_tuner.validate_config()
        
        # Validate model app
        model_app = self.get_model_app()
        if model_app:
            validation_results["model_app"] = []
            if not model_app.is_running():
                validation_results["model_app"].append("Service not running")
        
        # Validate cloud API
        cloud_api = self.get_cloud_api()
        if cloud_api:
            validation_results["cloud_api"] = []
            if not cloud_api.validate_credentials():
                validation_results["cloud_api"].append("Invalid credentials")
        
        return validation_results
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        summary = {
            "components": {
                "fine_tuner": self.config.get("fine_tuner", {}).get("type"),
                "model_app": self.config.get("model_app", {}).get("type"),
                "repository": self.config.get("repository", {}).get("type"),
                "cloud_api": self.config.get("cloud_api", {}).get("type")
            }
        }
        
        # Add strategy info if available
        if "strategy" in self.config:
            summary["strategy"] = self.config["strategy"]
        
        return summary