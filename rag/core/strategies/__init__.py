"""
RAG Strategies Module - New Schema Only

This module provides strategy handling for the new RAG schema format.
NO BACKWARD COMPATIBILITY - everything uses new schema directly.
"""

from .handler import SchemaHandler

__all__ = ["SchemaHandler"]
