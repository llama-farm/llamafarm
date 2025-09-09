"""
RAG Strategies Module - New Schema Only

This module provides strategy handling for the new RAG schema format.
NO BACKWARD COMPATIBILITY - everything uses new schema directly.
"""

from .loader import StrategyLoader
from .handler import SchemaHandler

# Keep DirectSchemaHandler as alias for backward compatibility
DirectSchemaHandler = SchemaHandler

# Legacy imports commented out - DO NOT USE
# from .manager import StrategyManager  # DEPRECATED
# from .config import StrategyConfig  # DEPRECATED

__all__ = ["StrategyLoader", "SchemaHandler", "DirectSchemaHandler"]