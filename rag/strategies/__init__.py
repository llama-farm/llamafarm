"""
RAG Strategies Module

This module provides a high-level strategy system for configuring RAG pipelines.
Instead of manually configuring individual parsers, extractors, embedders, and retrieval strategies,
users can select from predefined strategies optimized for specific use cases.
"""

from .manager import StrategyManager
from .config import StrategyConfig
from .loader import StrategyLoader

__all__ = ["StrategyManager", "StrategyConfig", "StrategyLoader"]