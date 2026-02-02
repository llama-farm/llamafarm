"""
Semantic routing for Atmosphere.

Routes intents to the best available capability using
embedding-based similarity matching.
"""

from .semantic import SemanticRouter, RouteResult
from .gradient import GradientTable, GradientEntry
from .embeddings import EmbeddingEngine
from .executor import Executor, ExecutionResult

__all__ = [
    "SemanticRouter",
    "RouteResult",
    "GradientTable",
    "GradientEntry",
    "EmbeddingEngine",
    "Executor",
    "ExecutionResult",
]
