"""Universal retrieval strategies that work across all vector databases."""

# Import all universal strategies for easy access
from .basic_similarity import BasicSimilarityStrategy
from .metadata_filtered import MetadataFilteredStrategy
from .multi_query import MultiQueryStrategy
from .reranked import RerankedStrategy
from .hybrid_universal import HybridUniversalStrategy

# Export all strategies
__all__ = [
    "BasicSimilarityStrategy",
    "MetadataFilteredStrategy", 
    "MultiQueryStrategy",
    "RerankedStrategy",
    "HybridUniversalStrategy"
]

# Strategy metadata for auto-discovery
UNIVERSAL_STRATEGIES = {
    "BasicSimilarityStrategy": {
        "class": BasicSimilarityStrategy,
        "version": "1.0.0",
        "description": "Simple vector similarity search that works with any database",
        "aliases": ["basic"],
        "use_cases": ["getting_started", "simple_search", "baseline_testing"],
        "performance": "fast",
        "complexity": "low"
    },
    "MetadataFilteredStrategy": {
        "class": MetadataFilteredStrategy,
        "version": "1.0.0", 
        "description": "Vector search with intelligent metadata filtering",
        "aliases": ["filtered"],
        "use_cases": ["domain_specific", "multi_tenant", "categorized_content"],
        "performance": "medium",
        "complexity": "medium"
    },
    "MultiQueryStrategy": {
        "class": MultiQueryStrategy,
        "version": "1.0.0",
        "description": "Uses multiple query variations to improve recall",
        "aliases": ["multi_query"],
        "use_cases": ["ambiguous_queries", "query_expansion", "complex_questions"],
        "performance": "medium",
        "complexity": "medium"
    },
    "RerankedStrategy": {
        "class": RerankedStrategy,
        "version": "1.0.0",
        "description": "Multi-factor re-ranking for sophisticated relevance scoring",
        "aliases": ["reranked"],
        "use_cases": ["production_systems", "time_sensitive", "multi_factor_relevance"],
        "performance": "slower",
        "complexity": "high"
    },
    "HybridUniversalStrategy": {
        "class": HybridUniversalStrategy,
        "version": "1.0.0",
        "description": "Combines multiple strategies with configurable weights", 
        "aliases": ["hybrid"],
        "use_cases": ["production", "balanced_results", "complex_requirements"],
        "performance": "variable",
        "complexity": "high"
    }
}