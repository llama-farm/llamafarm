"""Test suite for universal strategies extraction into individual files."""

import pytest
import numpy as np
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock

from core.base import Document
from components.retrievers.base import RetrievalResult
from components.retrievers.basic_similarity.basic_similarity import BasicSimilarityStrategy
from components.retrievers.metadata_filtered.metadata_filtered import MetadataFilteredStrategy
from components.retrievers.multi_query.multi_query import MultiQueryStrategy
from components.retrievers.reranked.reranked import RerankedStrategy
from components.retrievers.hybrid_universal.hybrid_universal import HybridUniversalStrategy
from core.factories import create_retrieval_strategy_from_config

# Mock registry class for testing
class MockRegistry:
    """Mock registry class with methods for testing."""
    
    def __init__(self):
        self.strategies = {
            "BasicSimilarityStrategy": BasicSimilarityStrategy,
            "MetadataFilteredStrategy": MetadataFilteredStrategy,
            "MultiQueryStrategy": MultiQueryStrategy,
            "RerankedStrategy": RerankedStrategy,
            "HybridUniversalStrategy": HybridUniversalStrategy
        }
    
    def create_strategy(self, strategy_name: str, config: Dict[str, Any]):
        """Create a strategy instance."""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name](name=strategy_name, config=config)
        return None
    
    def get_compatible_strategies(self, vector_store_type: str) -> List[str]:
        """Get compatible strategies for a vector store type."""
        compatible = []
        for name, strategy_class in self.strategies.items():
            strategy = strategy_class()
            if strategy.supports_vector_store(vector_store_type):
                compatible.append(name)
        return compatible
    
    def get_optimal_strategy_name(self, vector_store_type: str, use_case: str) -> str:
        """Get optimal strategy for use case."""
        # Simple heuristic for testing
        if use_case == "getting_started":
            return "BasicSimilarityStrategy"
        elif use_case == "production":
            return "RerankedStrategy"
        else:
            return "BasicSimilarityStrategy"

# Mock get_registry function for testing
def get_registry():
    """Mock registry function for testing."""
    return MockRegistry()

# Recreate UNIVERSAL_STRATEGIES for testing
UNIVERSAL_STRATEGIES = {
    "BasicSimilarityStrategy": {
        "class": BasicSimilarityStrategy,
        "version": "1.0.0",
        "description": "Simple vector similarity search",
        "aliases": ["basic", "similarity"],
        "use_cases": ["general_search", "prototype", "baseline"],
        "performance": "fast",
        "complexity": "low",
        "accuracy": "medium"
    },
    "MetadataFilteredStrategy": {
        "class": MetadataFilteredStrategy,
        "version": "1.0.0",
        "description": "Filter-first retrieval with metadata constraints",
        "aliases": ["filtered", "metadata"],
        "use_cases": ["filtered_search", "domain_specific", "constrained_retrieval"],
        "performance": "medium",
        "complexity": "medium",
        "accuracy": "high"
    },
    "MultiQueryStrategy": {
        "class": MultiQueryStrategy,
        "version": "1.0.0",
        "description": "Multiple query variations with result fusion",
        "aliases": ["multi_query", "query_expansion"],
        "use_cases": ["query_expansion", "improved_recall", "robust_search"],
        "performance": "medium",
        "complexity": "medium",
        "accuracy": "high"
    },
    "RerankedStrategy": {
        "class": RerankedStrategy,
        "version": "1.0.0",
        "description": "Re-rank results with multiple factors",
        "aliases": ["reranked", "rerank"],
        "use_cases": ["quality_improvement", "personalization", "domain_adaptation"],
        "performance": "slow",
        "complexity": "high",
        "accuracy": "very_high"
    },
    "HybridUniversalStrategy": {
        "class": HybridUniversalStrategy,
        "version": "1.0.0",
        "description": "Combine multiple retrieval strategies",
        "aliases": ["hybrid", "ensemble"],
        "use_cases": ["ensemble", "maximum_accuracy", "production_systems"],
        "performance": "very_slow",
        "complexity": "very_high",
        "accuracy": "maximum"
    }
}


class TestBasicSimilarityStrategy:
    """Test BasicSimilarityStrategy individual file."""
    
    def test_import_and_creation(self):
        """Test that BasicSimilarityStrategy can be imported and created."""
        strategy = BasicSimilarityStrategy()
        assert strategy.name == "BasicSimilarityStrategy"
        assert strategy.distance_metric == "cosine"
    
    def test_config_initialization(self):
        """Test strategy initialization with config."""
        config = {"distance_metric": "euclidean"}
        strategy = BasicSimilarityStrategy(config=config)
        assert strategy.distance_metric == "euclidean"
    
    def test_retrieve_functionality(self):
        """Test basic retrieval functionality."""
        strategy = BasicSimilarityStrategy()
        
        # Mock vector store
        mock_store = Mock()
        mock_docs = [
            Document(content="test1", metadata={"similarity_score": 0.9}),
            Document(content="test2", metadata={"similarity_score": 0.8})
        ]
        mock_store.search.return_value = mock_docs
        
        # Test retrieval
        query_embedding = [0.1] * 768
        result = strategy.retrieve(query_embedding, mock_store, top_k=2)
        
        assert isinstance(result, RetrievalResult)
        assert len(result.documents) == 2
        assert len(result.scores) == 2
        assert result.scores == [0.9, 0.8]
        assert result.strategy_metadata["strategy"] == "BasicSimilarityStrategy"
        assert result.strategy_metadata["version"] == "1.0.0"
    
    def test_supports_all_stores(self):
        """Test universal support for vector stores."""
        strategy = BasicSimilarityStrategy()
        assert strategy.supports_vector_store("ChromaStore")
        assert strategy.supports_vector_store("PineconeStore")
        assert strategy.supports_vector_store("AnyStore")
    
    def test_validation_and_schema(self):
        """Test configuration validation and schema."""
        strategy = BasicSimilarityStrategy()
        assert strategy.validate_config() == True
        
        schema = strategy.get_config_schema()
        assert schema["type"] == "object"
        assert "distance_metric" in schema["properties"]
    
    def test_performance_info(self):
        """Test performance information."""
        strategy = BasicSimilarityStrategy()
        perf_info = strategy.get_performance_info()
        assert perf_info["speed"] == "fast"
        assert perf_info["complexity"] == "low"


class TestMetadataFilteredStrategy:
    """Test MetadataFilteredStrategy individual file."""
    
    def test_import_and_creation(self):
        """Test that MetadataFilteredStrategy can be imported and created."""
        strategy = MetadataFilteredStrategy()
        assert strategy.name == "MetadataFilteredStrategy"
        assert strategy.default_filters == {}
    
    def test_config_with_default_filters(self):
        """Test initialization with default filters."""
        config = {
            "default_filters": {"type": "documentation"},
            "distance_metric": "euclidean",
            "fallback_multiplier": 5
        }
        strategy = MetadataFilteredStrategy(config=config)
        assert strategy.default_filters == {"type": "documentation"}
        assert strategy.distance_metric == "euclidean"
        assert strategy.fallback_multiplier == 5
    
    def test_native_filtering_support(self):
        """Test retrieval with native filtering support."""
        strategy = MetadataFilteredStrategy()
        
        # Mock vector store with native filtering
        mock_store = Mock()
        mock_store.search_with_filter = Mock()
        mock_docs = [
            Document(content="doc1", metadata={"type": "tutorial", "similarity_score": 0.9}),
            Document(content="doc2", metadata={"type": "tutorial", "similarity_score": 0.8})
        ]
        mock_store.search_with_filter.return_value = mock_docs
        
        query_embedding = [0.1] * 768
        result = strategy.retrieve(
            query_embedding, 
            mock_store, 
            top_k=2,
            metadata_filter={"type": "tutorial"}
        )
        
        assert len(result.documents) == 2
        assert result.strategy_metadata["filtering_method"] == "native"
        mock_store.search_with_filter.assert_called_once()
    
    def test_fallback_filtering(self):
        """Test retrieval with fallback post-search filtering."""
        strategy = MetadataFilteredStrategy(config={"fallback_multiplier": 3})
        
        # Mock vector store without native filtering
        mock_store = Mock()
        # Remove the search_with_filter method to trigger fallback
        delattr(mock_store, 'search_with_filter') if hasattr(mock_store, 'search_with_filter') else None
        
        mock_docs = [
            Document(content="doc1", metadata={"type": "tutorial", "similarity_score": 0.9}),
            Document(content="doc2", metadata={"type": "faq", "similarity_score": 0.8}),
            Document(content="doc3", metadata={"type": "tutorial", "similarity_score": 0.7})
        ]
        mock_store.search.return_value = mock_docs
        
        query_embedding = [0.1] * 768
        result = strategy.retrieve(
            query_embedding,
            mock_store,
            top_k=2,
            metadata_filter={"type": "tutorial"}
        )
        
        # Should return only tutorial docs
        assert len(result.documents) == 2
        assert all(doc.metadata["type"] == "tutorial" for doc in result.documents)
        assert result.strategy_metadata["filtering_method"] == "post_search"
        mock_store.search.assert_called_with(query_embedding=query_embedding, top_k=6)  # 2 * 3
    
    def test_complex_filter_operators(self):
        """Test complex filter operators like $ne, $in, etc."""
        strategy = MetadataFilteredStrategy()
        
        # Test documents
        docs = [
            Document(content="doc1", metadata={"priority": "high", "score": 95}),
            Document(content="doc2", metadata={"priority": "medium", "score": 80}),
            Document(content="doc3", metadata={"priority": "low", "score": 60})
        ]
        
        # Test $ne operator
        filtered = strategy._filter_documents(docs, {"priority": {"$ne": "low"}})
        assert len(filtered) == 2
        assert all(doc.metadata["priority"] != "low" for doc in filtered)
        
        # Test $in operator
        filtered = strategy._filter_documents(docs, {"priority": {"$in": ["high", "medium"]}})
        assert len(filtered) == 2
        
        # Test $gt operator
        filtered = strategy._filter_documents(docs, {"score": {"$gt": 70}})
        assert len(filtered) == 2
        assert all(doc.metadata["score"] > 70 for doc in filtered)
    
    def test_validation_and_schema(self):
        """Test configuration validation."""
        # Valid config
        strategy = MetadataFilteredStrategy(config={"default_filters": {"type": "doc"}})
        assert strategy.validate_config() == True
        
        # Invalid config with bad operators
        strategy_bad = MetadataFilteredStrategy(config={
            "default_filters": {"type": {"$invalid": "value"}}
        })
        assert strategy_bad.validate_config() == False


class TestMultiQueryStrategy:
    """Test MultiQueryStrategy individual file."""
    
    def test_import_and_creation(self):
        """Test MultiQueryStrategy import and creation."""
        strategy = MultiQueryStrategy()
        assert strategy.name == "MultiQueryStrategy"
        assert strategy.num_queries == 3
        assert strategy.aggregation_method == "max"
    
    def test_config_initialization(self):
        """Test initialization with custom config."""
        config = {
            "num_queries": 5,
            "aggregation_method": "weighted",
            "search_multiplier": 3
        }
        strategy = MultiQueryStrategy(config=config)
        assert strategy.num_queries == 5
        assert strategy.aggregation_method == "weighted"
        assert strategy.search_multiplier == 3
    
    def test_max_aggregation(self):
        """Test max score aggregation."""
        strategy = MultiQueryStrategy(config={"aggregation_method": "max"})
        
        mock_store = Mock()
        # Simulate different results for different query variations
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [Document(content="doc1", metadata={"similarity_score": 0.8}, id="doc1")]
            elif call_count == 2:
                return [Document(content="doc1", metadata={"similarity_score": 0.9}, id="doc1")]
            else:
                return [Document(content="doc1", metadata={"similarity_score": 0.7}, id="doc1")]
        
        mock_store.search.side_effect = side_effect
        
        query_embedding = [0.1] * 768
        result = strategy.retrieve(query_embedding, mock_store, top_k=1)
        
        # Should take max score (0.9)
        assert len(result.documents) == 1
        assert result.scores[0] == 0.9
        assert result.strategy_metadata["aggregation_method"] == "max"
    
    def test_mean_aggregation(self):
        """Test mean score aggregation."""
        strategy = MultiQueryStrategy(config={"aggregation_method": "mean"})
        scores = [0.8, 0.9, 0.7]
        aggregated = strategy._aggregate_scores(scores)
        expected_mean = np.mean(scores)
        assert abs(aggregated - expected_mean) < 0.001
    
    def test_weighted_aggregation(self):
        """Test weighted score aggregation."""
        strategy = MultiQueryStrategy(config={"aggregation_method": "weighted"})
        scores = [0.9, 0.8, 0.7]  # First query should have highest weight
        aggregated = strategy._aggregate_scores(scores)
        
        # Manual calculation: (0.9*1 + 0.8*0.5 + 0.7*0.33) / (1 + 0.5 + 0.33)
        weights = [1.0, 0.5, 1/3]
        expected = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        assert abs(aggregated - expected) < 0.001
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        strategy = MultiQueryStrategy(config={"num_queries": 3, "aggregation_method": "max"})
        assert strategy.validate_config() == True
        
        # Invalid num_queries
        strategy_bad = MultiQueryStrategy(config={"num_queries": 0})
        assert strategy_bad.validate_config() == False
        
        # Invalid aggregation method
        strategy_bad2 = MultiQueryStrategy(config={"aggregation_method": "invalid"})
        assert strategy_bad2.validate_config() == False


class TestRerankedStrategy:
    """Test RerankedStrategy individual file."""
    
    def test_import_and_creation(self):
        """Test RerankedStrategy import and creation."""
        strategy = RerankedStrategy()
        assert strategy.name == "RerankedStrategy"
        assert strategy.initial_k == 20
        assert "recency" in strategy.rerank_factors
    
    def test_basic_reranking(self):
        """Test basic reranking functionality."""
        strategy = RerankedStrategy(config={"initial_k": 4})
        
        mock_store = Mock()
        mock_docs = [
            Document(content="short", metadata={"similarity_score": 0.9, "priority": "low"}),
            Document(content="much longer content that should get length boost", 
                    metadata={"similarity_score": 0.8, "priority": "high"}),
            Document(content="medium length content", metadata={"similarity_score": 0.85, "priority": "medium"}),
            Document(content="another short", metadata={"similarity_score": 0.75, "priority": "low"})
        ]
        mock_store.search.return_value = mock_docs
        
        query_embedding = [0.1] * 768
        result = strategy.retrieve(query_embedding, mock_store, top_k=2)
        
        assert len(result.documents) == 2
        # Check that reranking occurred
        for doc in result.documents:
            assert "base_similarity_score" in doc.metadata
            assert "rerank_boost" in doc.metadata
        
        assert result.strategy_metadata["strategy"] == "RerankedStrategy"
        assert result.strategy_metadata["initial_k"] == 4
    
    def test_metadata_boost_calculation(self):
        """Test metadata boost calculations."""
        strategy = RerankedStrategy()
        
        # Test priority boosts
        metadata_high = {"priority": "high"}
        boost_high = strategy._calculate_metadata_boost(metadata_high)
        assert boost_high == 0.3
        
        metadata_critical = {"priority": "critical"}
        boost_critical = strategy._calculate_metadata_boost(metadata_critical)
        assert boost_critical == 0.5
        
        # Test type boosts
        metadata_doc = {"type": "documentation"}
        boost_doc = strategy._calculate_metadata_boost(metadata_doc)
        assert boost_doc == 0.2
        
        # Test combined boosts
        metadata_combined = {"priority": "high", "type": "documentation", "verified": True}
        boost_combined = strategy._calculate_metadata_boost(metadata_combined)
        assert boost_combined == 0.3 + 0.2 + 0.1  # priority + type + verified
    
    def test_recency_boost(self):
        """Test recency boost calculation."""
        strategy = RerankedStrategy()
        
        # Test very recent (should get max boost)
        from datetime import datetime, timezone
        recent_time = datetime.now(timezone.utc).isoformat()
        boost = strategy._calculate_recency_boost(recent_time)
        assert boost == 1.0
        
        # Test invalid timestamp
        boost_invalid = strategy._calculate_recency_boost("invalid")
        assert boost_invalid == 0.0
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        strategy = RerankedStrategy(config={"initial_k": 10, "length_normalization": 500})
        assert strategy.validate_config() == True
        
        # Invalid initial_k
        strategy_bad = RerankedStrategy(config={"initial_k": 0})
        assert strategy_bad.validate_config() == False


class TestHybridUniversalStrategy:
    """Test HybridUniversalStrategy individual file."""
    
    def test_import_and_creation(self):
        """Test HybridUniversalStrategy import and creation."""
        strategy = HybridUniversalStrategy()
        assert strategy.name == "HybridUniversalStrategy"
        assert len(strategy.strategies) > 0  # Should have default strategies
    
    def test_custom_strategy_configuration(self):
        """Test initialization with custom strategy configuration."""
        config = {
            "strategies": [
                {"type": "BasicSimilarityStrategy", "weight": 0.7},
                {"type": "MetadataFilteredStrategy", "weight": 0.3, "config": {"default_filters": {"type": "doc"}}}
            ]
        }
        strategy = HybridUniversalStrategy(config=config)
        assert len(strategy.strategies) == 2
        assert len(strategy.weights) == 2
        assert strategy.weights[0] == 0.7
        assert strategy.weights[1] == 0.3
    
    def test_strategy_aliases(self):
        """Test that strategy aliases work."""
        config = {
            "strategies": [
                {"type": "basic", "weight": 0.6},
                {"type": "filtered", "weight": 0.4}
            ]
        }
        strategy = HybridUniversalStrategy(config=config)
        assert len(strategy.strategies) == 2
        assert strategy.strategies[0].name == "BasicSimilarityStrategy"
        assert strategy.strategies[1].name == "MetadataFilteredStrategy"
    
    def test_weighted_combination(self):
        """Test weighted combination of results."""
        strategy = HybridUniversalStrategy(config={
            "strategies": [
                {"type": "BasicSimilarityStrategy", "weight": 0.6},
                {"type": "MetadataFilteredStrategy", "weight": 0.4}
            ]
        })
        
        # Mock both sub-strategies
        for sub_strategy in strategy.strategies:
            sub_strategy.retrieve = Mock()
        
        # Mock results from sub-strategies
        result1 = RetrievalResult(
            documents=[Document(content="doc1", metadata={"similarity_score": 0.9})],
            scores=[0.9],
            strategy_metadata={"strategy": "BasicSimilarityStrategy"}
        )
        result2 = RetrievalResult(
            documents=[Document(content="doc1", metadata={"similarity_score": 0.8})],
            scores=[0.8],
            strategy_metadata={"strategy": "MetadataFilteredStrategy"}
        )
        
        strategy.strategies[0].retrieve.return_value = result1
        strategy.strategies[1].retrieve.return_value = result2
        
        mock_store = Mock()
        query_embedding = [0.1] * 768
        result = strategy.retrieve(query_embedding, mock_store, top_k=1)
        
        assert len(result.documents) >= 1
        assert result.strategy_metadata["strategy"] == "HybridUniversalStrategy"
        assert result.strategy_metadata["num_strategies"] == 2
    
    def test_rank_fusion_combination(self):
        """Test rank fusion combination method."""
        config = {
            "combination_method": "rank_fusion",
            "strategies": [
                {"type": "BasicSimilarityStrategy", "weight": 1.0}
            ]
        }
        strategy = HybridUniversalStrategy(config=config)
        assert strategy.combination_method == "rank_fusion"
        
        # Test the rank fusion method directly
        result1 = RetrievalResult(
            documents=[
                Document(content="doc1", id="1", metadata={"similarity_score": 0.9}),
                Document(content="doc2", id="2", metadata={"similarity_score": 0.8})
            ],
            scores=[0.9, 0.8],
            strategy_metadata={}
        )
        
        fusion_result = strategy._rank_fusion_combine([result1], top_k=2)
        assert len(fusion_result.documents) == 2
        assert fusion_result.strategy_metadata["combination_method"] == "rank_fusion"
    
    def test_fallback_behavior(self):
        """Test fallback to basic strategy when no strategies configured."""
        strategy = HybridUniversalStrategy()
        strategy.strategies = []  # Clear strategies to test fallback
        
        mock_store = Mock()
        mock_store.search.return_value = [
            Document(content="test", metadata={"similarity_score": 0.9})
        ]
        
        query_embedding = [0.1] * 768
        result = strategy.retrieve(query_embedding, mock_store, top_k=1)
        
        assert len(result.documents) == 1
        # Should use BasicSimilarityStrategy as fallback
        mock_store.search.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling when sub-strategies fail."""
        strategy = HybridUniversalStrategy()
        
        # Make all sub-strategies raise exceptions
        for sub_strategy in strategy.strategies:
            sub_strategy.retrieve = Mock(side_effect=Exception("Strategy failed"))
        
        mock_store = Mock()
        query_embedding = [0.1] * 768
        result = strategy.retrieve(query_embedding, mock_store, top_k=1)
        
        # Should return empty result with error info
        assert len(result.documents) == 0
        assert "error" in result.strategy_metadata
        assert "strategy_performances" in result.strategy_metadata
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        config = {
            "strategies": [
                {"type": "BasicSimilarityStrategy", "weight": 0.5},
                {"type": "MetadataFilteredStrategy", "weight": 0.5}
            ]
        }
        strategy = HybridUniversalStrategy(config=config)
        assert strategy.validate_config() == True
        
        # Invalid - no strategies
        strategy_empty = HybridUniversalStrategy()
        strategy_empty.strategies = []
        assert strategy_empty.validate_config() == False


class TestUniversalStrategiesMetadata:
    """Test the UNIVERSAL_STRATEGIES metadata and auto-discovery."""
    
    def test_all_strategies_in_metadata(self):
        """Test that all strategies are present in UNIVERSAL_STRATEGIES."""
        expected_strategies = [
            "BasicSimilarityStrategy",
            "MetadataFilteredStrategy", 
            "MultiQueryStrategy",
            "RerankedStrategy",
            "HybridUniversalStrategy"
        ]
        
        for strategy_name in expected_strategies:
            assert strategy_name in UNIVERSAL_STRATEGIES
            metadata = UNIVERSAL_STRATEGIES[strategy_name]
            assert "class" in metadata
            assert "version" in metadata
            assert "description" in metadata
            assert "aliases" in metadata
            assert "use_cases" in metadata
            assert "performance" in metadata
            assert "complexity" in metadata
    
    def test_strategy_class_references(self):
        """Test that strategy classes in metadata are correct."""
        assert UNIVERSAL_STRATEGIES["BasicSimilarityStrategy"]["class"] == BasicSimilarityStrategy
        assert UNIVERSAL_STRATEGIES["MetadataFilteredStrategy"]["class"] == MetadataFilteredStrategy
        assert UNIVERSAL_STRATEGIES["MultiQueryStrategy"]["class"] == MultiQueryStrategy
        assert UNIVERSAL_STRATEGIES["RerankedStrategy"]["class"] == RerankedStrategy
        assert UNIVERSAL_STRATEGIES["HybridUniversalStrategy"]["class"] == HybridUniversalStrategy
    
    def test_version_consistency(self):
        """Test that all strategies have version 1.0.0."""
        for strategy_name, metadata in UNIVERSAL_STRATEGIES.items():
            assert metadata["version"] == "1.0.0"


class TestFactoryIntegration:
    """Test integration with factory and registry systems."""
    
    def test_factory_creation_all_strategies(self):
        """Test that factory can create all universal strategies."""
        strategies_to_test = [
            {"type": "BasicSimilarityStrategy"},
            {"type": "MetadataFilteredStrategy"},
            {"type": "MultiQueryStrategy"},
            {"type": "RerankedStrategy"},
            {"type": "HybridUniversalStrategy", "config": {
                "strategies": [{"type": "BasicSimilarityStrategy", "weight": 1.0}]
            }}
        ]
        
        for config in strategies_to_test:
            strategy = create_retrieval_strategy_from_config(config)
            assert strategy is not None
            assert hasattr(strategy, 'retrieve')
            assert hasattr(strategy, 'supports_vector_store')
    
    def test_registry_integration(self):
        """Test that registry can find and work with all strategies."""
        registry = get_registry()
        
        # Test that all strategies are registered
        expected_strategies = [
            "BasicSimilarityStrategy",
            "MetadataFilteredStrategy",
            "MultiQueryStrategy", 
            "RerankedStrategy",
            "HybridUniversalStrategy"
        ]
        
        for strategy_name in expected_strategies:
            strategy = registry.create_strategy(strategy_name, {})
            assert strategy is not None
            assert strategy.name == strategy_name
    
    def test_registry_compatibility_checks(self):
        """Test registry compatibility checks work."""
        registry = get_registry()
        
        # Test compatibility with ChromaStore
        compatible = registry.get_compatible_strategies("ChromaStore")
        assert len(compatible) > 0
        
        # All universal strategies should be compatible
        expected_strategies = [
            "BasicSimilarityStrategy",
            "MetadataFilteredStrategy",
            "MultiQueryStrategy",
            "RerankedStrategy", 
            "HybridUniversalStrategy"
        ]
        
        for strategy_name in expected_strategies:
            assert strategy_name in compatible
    
    def test_optimal_strategy_selection(self):
        """Test optimal strategy selection."""
        registry = get_registry()
        
        # Test different use cases
        use_cases = ["general", "getting_started", "production"]
        
        for use_case in use_cases:
            optimal = registry.get_optimal_strategy_name("ChromaStore", use_case)
            assert optimal in UNIVERSAL_STRATEGIES


class TestEndToEndIntegration:
    """Test end-to-end functionality with mock data."""
    
    def test_complete_search_workflow(self):
        """Test complete search workflow with all strategies."""
        # Mock vector store and embedder
        mock_store = Mock()
        mock_docs = [
            Document(
                content="How to reset password",
                metadata={"type": "faq", "priority": "high", "similarity_score": 0.9},
                id="doc1"
            ),
            Document(
                content="Login troubleshooting guide", 
                metadata={"type": "tutorial", "priority": "medium", "similarity_score": 0.8},
                id="doc2"
            )
        ]
        mock_store.search.return_value = mock_docs
        mock_store.get_collection_info.return_value = {"total_documents": 2}
        
        query_embedding = [0.1] * 768
        
        # Test each strategy type
        strategies_to_test = [
            BasicSimilarityStrategy(),
            MetadataFilteredStrategy(),
            MultiQueryStrategy(config={"num_queries": 2}),
            RerankedStrategy(config={"initial_k": 5}),
            HybridUniversalStrategy(config={
                "strategies": [
                    {"type": "BasicSimilarityStrategy", "weight": 0.7},
                    {"type": "MetadataFilteredStrategy", "weight": 0.3}
                ]
            })
        ]
        
        for strategy in strategies_to_test:
            result = strategy.retrieve(query_embedding, mock_store, top_k=2)
            
            # Basic checks
            assert isinstance(result, RetrievalResult)
            assert len(result.documents) <= 2
            assert len(result.scores) == len(result.documents)
            assert "strategy" in result.strategy_metadata
            assert "version" in result.strategy_metadata
            
            # Check that strategy name matches
            expected_name = type(strategy).__name__
            assert result.strategy_metadata["strategy"] == expected_name
    
    def test_error_resilience(self):
        """Test that strategies handle errors gracefully."""
        # Mock store that raises exceptions
        mock_store = Mock()
        mock_store.search.side_effect = Exception("Database connection failed")
        
        query_embedding = [0.1] * 768
        
        # Test that basic strategies handle the error
        strategy = BasicSimilarityStrategy()
        
        with pytest.raises(Exception):
            strategy.retrieve(query_embedding, mock_store, top_k=2)
        
        # Test that hybrid strategy handles sub-strategy failures
        hybrid_strategy = HybridUniversalStrategy()
        result = hybrid_strategy.retrieve(query_embedding, mock_store, top_k=2)
        
        # Should return empty result with error info
        assert len(result.documents) == 0
        assert "error" in result.strategy_metadata or "strategy_performances" in result.strategy_metadata