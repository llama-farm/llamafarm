"""
Factory classes for creating fine-tuning components.

This module provides factory classes for creating fine-tuners, data processors,
and other components based on configuration, following the same patterns as the RAG system.
"""

from typing import Dict, Any, Type, Optional, List
from pathlib import Path
import logging
import importlib

from .base import BaseFineTuner, BaseDataProcessor, FineTuningConfig

logger = logging.getLogger(__name__)


class FineTunerFactory:
    """Factory for creating fine-tuning implementations."""
    
    _registry: Dict[str, Type[BaseFineTuner]] = {}
    
    @classmethod
    def register(cls, name: str, tuner_class: Type[BaseFineTuner]) -> None:
        """
        Register a fine-tuner implementation.
        
        Args:
            name: Name to register the tuner under
            tuner_class: Fine-tuner class to register
        """
        cls._registry[name] = tuner_class
        logger.debug(f"Registered fine-tuner: {name}")
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a fine-tuner implementation.
        
        Args:
            name: Name of tuner to unregister
        """
        if name in cls._registry:
            del cls._registry[name]
            logger.debug(f"Unregistered fine-tuner: {name}")
    
    @classmethod
    def create(cls, config: FineTuningConfig) -> BaseFineTuner:
        """
        Create a fine-tuner from configuration.
        
        Args:
            config: Fine-tuning configuration
            
        Returns:
            Fine-tuner instance
            
        Raises:
            ValueError: If framework type is not supported
            ImportError: If required dependencies are missing
        """
        framework_type = config.framework.get("type", "pytorch")
        
        if framework_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unsupported framework type: {framework_type}. "
                f"Available types: {available}"
            )
        
        tuner_class = cls._registry[framework_type]
        
        try:
            return tuner_class(config)
        except ImportError as e:
            raise ImportError(
                f"Failed to create {framework_type} fine-tuner. "
                f"Missing dependencies: {e}"
            )
    
    @classmethod
    def get_available_tuners(cls) -> List[str]:
        """
        Get list of available fine-tuner types.
        
        Returns:
            List of available tuner names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_tuner_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tuner.
        
        Args:
            name: Tuner name
            
        Returns:
            Tuner information or None if not found
        """
        if name not in cls._registry:
            return None
            
        tuner_class = cls._registry[name]
        
        # Try to get instance to check supported methods/models
        try:
            # Create minimal config for inspection
            dummy_config = FineTuningConfig(
                base_model={"name": "dummy"},
                method={"type": "lora"},
                framework={"type": name},
                training_args={},
                dataset={}
            )
            instance = tuner_class(dummy_config)
            
            return {
                "name": name,
                "class": tuner_class.__name__,
                "module": tuner_class.__module__,
                "supported_methods": instance.get_supported_methods(),
                "supported_models": instance.get_supported_models(),
                "description": tuner_class.__doc__ or "No description available"
            }
        except Exception as e:
            logger.warning(f"Could not get info for tuner {name}: {e}")
            return {
                "name": name,
                "class": tuner_class.__name__,
                "module": tuner_class.__module__,
                "error": str(e)
            }


class DataProcessorFactory:
    """Factory for creating dataset processors."""
    
    _registry: Dict[str, Type[BaseDataProcessor]] = {}
    
    @classmethod
    def register(cls, name: str, processor_class: Type[BaseDataProcessor]) -> None:
        """
        Register a data processor implementation.
        
        Args:
            name: Name to register the processor under
            processor_class: Processor class to register
        """
        cls._registry[name] = processor_class
        logger.debug(f"Registered data processor: {name}")
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a data processor implementation.
        
        Args:
            name: Name of processor to unregister
        """
        if name in cls._registry:
            del cls._registry[name]
            logger.debug(f"Unregistered data processor: {name}")
    
    @classmethod
    def create(cls, processor_type: str, config: Dict[str, Any]) -> BaseDataProcessor:
        """
        Create a data processor from configuration.
        
        Args:
            processor_type: Type of processor to create
            config: Processor configuration
            
        Returns:
            Data processor instance
            
        Raises:
            ValueError: If processor type is not supported
        """
        if processor_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unsupported processor type: {processor_type}. "
                f"Available types: {available}"
            )
        
        processor_class = cls._registry[processor_type]
        return processor_class(config)
    
    @classmethod
    def get_available_processors(cls) -> List[str]:
        """
        Get list of available processor types.
        
        Returns:
            List of available processor names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create_from_format(cls, data_format: str, config: Dict[str, Any]) -> BaseDataProcessor:
        """
        Create a processor based on data format.
        
        Args:
            data_format: Data format (jsonl, json, csv, etc.)
            config: Processor configuration
            
        Returns:
            Data processor instance
        """
        # Map data formats to processor types
        format_mapping = {
            "jsonl": "jsonl_processor",
            "json": "json_processor", 
            "csv": "csv_processor",
            "alpaca": "alpaca_processor",
            "sharegpt": "sharegpt_processor"
        }
        
        processor_type = format_mapping.get(data_format.lower())
        if not processor_type:
            # Default to jsonl processor
            processor_type = "jsonl_processor"
            logger.warning(f"Unknown data format {data_format}, using jsonl processor")
        
        return cls.create(processor_type, config)


def load_default_components():
    """Load and register default fine-tuning components."""
    
    # Register PyTorch fine-tuner
    try:
        from ..trainers.pytorch_trainer import PyTorchFineTuner
        FineTunerFactory.register("pytorch", PyTorchFineTuner)
        logger.debug("Registered PyTorch fine-tuner")
    except ImportError as e:
        logger.warning(f"Could not register PyTorch fine-tuner: {e}")
    
    # Register LlamaFactory fine-tuner
    try:
        from ..trainers.llamafactory_trainer import LlamaFactoryFineTuner
        FineTunerFactory.register("llamafactory", LlamaFactoryFineTuner)
        logger.debug("Registered LlamaFactory fine-tuner")
    except ImportError as e:
        logger.warning(f"Could not register LlamaFactory fine-tuner: {e}")
    
    # Register data processors
    try:
        from ..data.processors import JSONLProcessor, AlpacaProcessor
        DataProcessorFactory.register("jsonl_processor", JSONLProcessor)
        DataProcessorFactory.register("json_processor", JSONLProcessor)  # Alias
        DataProcessorFactory.register("alpaca_processor", AlpacaProcessor)
        logger.debug("Registered data processors")
    except ImportError as e:
        logger.warning(f"Could not register data processors: {e}")
    except Exception as e:
        logger.warning(f"Error registering data processors: {e}")


def auto_detect_framework(config: FineTuningConfig) -> str:
    """
    Auto-detect the best framework based on configuration and available dependencies.
    
    Args:
        config: Fine-tuning configuration
        
    Returns:
        Recommended framework name
    """
    method_type = config.method.get("type", "lora")
    hardware_type = config.environment.get("device", "auto")
    
    # Check available frameworks
    available_frameworks = FineTunerFactory.get_available_tuners()
    
    # Prefer LlamaFactory for LoRA/QLoRA on GPU
    if ("llamafactory" in available_frameworks and 
        method_type in ["lora", "qlora"] and 
        hardware_type in ["cuda", "auto"]):
        return "llamafactory"
    
    # Default to PyTorch if available
    if "pytorch" in available_frameworks:
        return "pytorch"
    
    # Return first available
    if available_frameworks:
        return available_frameworks[0]
    
    raise RuntimeError("No fine-tuning frameworks available")


def validate_config_compatibility(config: FineTuningConfig) -> List[str]:
    """
    Validate configuration compatibility across components.
    
    Args:
        config: Fine-tuning configuration
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check framework and method compatibility
    framework_type = config.framework.get("type", "pytorch")
    method_type = config.method.get("type", "lora")
    
    try:
        tuner_class = FineTunerFactory._registry.get(framework_type)
        if tuner_class:
            # Create dummy instance to check supported methods
            dummy_config = FineTuningConfig(
                base_model={"name": "dummy"},
                method={"type": method_type},
                framework={"type": framework_type},
                training_args={},
                dataset={}
            )
            instance = tuner_class(dummy_config)
            
            if method_type not in instance.get_supported_methods():
                errors.append(
                    f"Method '{method_type}' not supported by framework '{framework_type}'"
                )
    except Exception as e:
        errors.append(f"Could not validate framework compatibility: {e}")
    
    # Check hardware and method compatibility
    device = config.environment.get("device", "auto")
    if method_type == "full_finetune" and device == "cpu":
        errors.append("Full fine-tuning is not recommended on CPU")
    
    # Check memory requirements
    batch_size = config.training_args.get("per_device_train_batch_size", 1)
    seq_length = config.training_args.get("max_seq_length", 512)
    
    if method_type == "full_finetune" and batch_size > 2 and seq_length > 1024:
        errors.append("Large batch size + long sequences may cause OOM with full fine-tuning")
    
    return errors


# Auto-load default components when module is imported
try:
    load_default_components()
except Exception as e:
    logger.warning(f"Could not load default components: {e}")