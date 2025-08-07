"""
LlamaFarm Fine-Tuning Module

This module provides a comprehensive fine-tuning system for language models
following the same patterns as the RAG system, with strategy-based configuration
and factory patterns for extensibility.
"""

from .core.base import BaseFineTuner, FineTuningConfig
from .core.factory import FineTunerFactory
from .core.strategies import StrategyManager
from .trainers.pytorch_trainer import PyTorchFineTuner
from .trainers.llamafactory_trainer import LlamaFactoryFineTuner
from .data.processors import DatasetProcessor, JSONLProcessor

__version__ = "1.0.0"

__all__ = [
    "BaseFineTuner",
    "FineTuningConfig", 
    "FineTunerFactory",
    "StrategyManager",
    "PyTorchFineTuner",
    "LlamaFactoryFineTuner",
    "DatasetProcessor",
    "JSONLProcessor",
]