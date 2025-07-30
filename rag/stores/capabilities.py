"""Database capability mapping for retrieval strategies."""

from typing import Dict, List, Set, Any


class DatabaseCapabilities:
    """Defines what capabilities each vector database supports."""
    
    # Define universal capabilities
    CAPABILITIES = {
        "basic_similarity": "Basic vector similarity search",
        "metadata_filtering": "Filter by metadata fields",
        "batch_operations": "Batch insert/search operations",
        "distance_metrics": "Support for different distance metrics",
        "hybrid_search": "Native sparse + dense search",
        "faceted_search": "Faceted/grouped search results",
        "geo_search": "Geographical/spatial search",
        "full_text_search": "Traditional text search integration",
        "real_time_updates": "Real-time document updates",
        "transactions": "ACID transaction support",
        "clustering": "Document clustering capabilities",
        "recommendations": "Similar document recommendations"
    }
    
    # Database-specific capabilities
    DATABASE_CAPABILITIES = {
        "ChromaStore": {
            "supported": [
                "basic_similarity",
                "metadata_filtering", 
                "batch_operations",
                "distance_metrics"
            ],
            "distance_metrics": ["cosine", "euclidean", "manhattan"],
            "max_batch_size": 1000,
            "native_filtering": True,
            "filter_operators": ["$eq", "$ne", "$in", "$nin", "$gt", "$lt", "$gte", "$lte"],
            "notes": "Excellent for development and small-medium datasets"
        },
        
        "PineconeStore": {
            "supported": [
                "basic_similarity",
                "metadata_filtering",
                "batch_operations",
                "distance_metrics",
                "real_time_updates"
            ],
            "distance_metrics": ["cosine", "euclidean", "dotproduct"],
            "max_batch_size": 100,
            "native_filtering": True,
            "filter_operators": ["$eq", "$ne", "$in", "$nin", "$gt", "$lt", "$gte", "$lte"],
            "notes": "Managed service, excellent for production"
        },
        
        "WeaviateStore": {
            "supported": [
                "basic_similarity",
                "metadata_filtering",
                "batch_operations",
                "distance_metrics",
                "hybrid_search",
                "full_text_search",
                "geo_search"
            ],
            "distance_metrics": ["cosine", "dot", "l2-squared", "manhattan", "hamming"],
            "max_batch_size": 100,
            "native_filtering": True,
            "hybrid_search": True,
            "graphql_api": True,
            "notes": "GraphQL API, excellent hybrid search capabilities"
        },
        
        "QdrantStore": {
            "supported": [
                "basic_similarity",
                "metadata_filtering",
                "batch_operations", 
                "distance_metrics",
                "geo_search",
                "clustering",
                "recommendations"
            ],
            "distance_metrics": ["cosine", "euclidean", "dot", "manhattan"],
            "max_batch_size": 1000,
            "native_filtering": True,
            "filter_operators": ["$eq", "$ne", "$in", "$nin", "$gt", "$lt", "$gte", "$lte", "$range"],
            "payload_indexing": True,
            "notes": "High performance, advanced filtering, clustering support"
        },
        
        "MilvusStore": {
            "supported": [
                "basic_similarity",
                "metadata_filtering",
                "batch_operations",
                "distance_metrics",
                "clustering"
            ],
            "distance_metrics": ["L2", "IP", "COSINE", "HAMMING", "JACCARD"],
            "max_batch_size": 1000,
            "native_filtering": True,
            "distributed": True,
            "notes": "Highly scalable, distributed architecture"
        },
        
        "FAISSStore": {
            "supported": [
                "basic_similarity",
                "batch_operations",
                "distance_metrics"
            ],
            "distance_metrics": ["L2", "IP", "COSINE"],
            "max_batch_size": 10000,
            "native_filtering": False,  # Requires post-processing
            "approximate_search": True,
            "gpu_support": True,
            "notes": "Fastest pure vector search, minimal metadata support"
        }
    }
    
    # Strategy compatibility matrix
    STRATEGY_REQUIREMENTS = {
        "BasicSimilarityStrategy": {
            "required_capabilities": ["basic_similarity"],
            "optional_capabilities": ["distance_metrics"],
            "description": "Universal basic vector similarity search"
        },
        
        "MetadataFilteredStrategy": {
            "required_capabilities": ["basic_similarity"],
            "preferred_capabilities": ["metadata_filtering", "native_filtering"],
            "fallback_behavior": "Post-search filtering if native filtering unavailable",
            "description": "Vector search with metadata filtering"
        },
        
        "MultiQueryStrategy": {
            "required_capabilities": ["basic_similarity"],
            "preferred_capabilities": ["batch_operations"],
            "description": "Multiple query variations with score aggregation"
        },
        
        "RerankedStrategy": {
            "required_capabilities": ["basic_similarity"],
            "optional_capabilities": ["batch_operations"],
            "description": "Multi-factor re-ranking of search results"
        },
        
        "HybridUniversalStrategy": {
            "required_capabilities": ["basic_similarity"],
            "preferred_capabilities": ["metadata_filtering", "batch_operations"],
            "description": "Combines multiple retrieval strategies"
        },
        
        # Database-specific optimized strategies (when available)
        "ChromaHybridStrategy": {
            "required_capabilities": ["basic_similarity", "metadata_filtering"],
            "database_specific": "ChromaStore",
            "description": "ChromaDB-optimized hybrid strategy"
        },
        
        "WeaviateHybridStrategy": {
            "required_capabilities": ["basic_similarity", "hybrid_search"],
            "database_specific": "WeaviateStore", 
            "description": "Weaviate native hybrid search strategy"
        }
    }

    @classmethod
    def get_supported_strategies(cls, database_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all strategies supported by a database type."""
        if database_type not in cls.DATABASE_CAPABILITIES:
            return {}
        
        db_caps = cls.DATABASE_CAPABILITIES[database_type]
        supported_capabilities = set(db_caps["supported"])
        
        compatible_strategies = {}
        
        for strategy_name, strategy_info in cls.STRATEGY_REQUIREMENTS.items():
            # Skip database-specific strategies for other databases
            if "database_specific" in strategy_info:
                if strategy_info["database_specific"] != database_type:
                    continue
            
            # Check required capabilities
            required = set(strategy_info.get("required_capabilities", []))
            if not required.issubset(supported_capabilities):
                continue
            
            # Calculate compatibility score
            preferred = set(strategy_info.get("preferred_capabilities", []))
            optional = set(strategy_info.get("optional_capabilities", []))
            
            preferred_supported = preferred.intersection(supported_capabilities)
            optional_supported = optional.intersection(supported_capabilities)
            
            compatibility_score = (
                len(preferred_supported) / len(preferred) if preferred else 1.0
            ) * 0.7 + (
                len(optional_supported) / len(optional) if optional else 0.0
            ) * 0.3
            
            compatible_strategies[strategy_name] = {
                "compatibility_score": compatibility_score,
                "required_met": True,
                "preferred_met": list(preferred_supported),
                "preferred_missing": list(preferred - preferred_supported),
                "optional_met": list(optional_supported),
                "fallback_behavior": strategy_info.get("fallback_behavior"),
                "description": strategy_info.get("description", "")
            }
        
        return compatible_strategies
    
    @classmethod
    def get_optimal_strategy(cls, database_type: str, use_case: str = "general") -> str:
        """Get the optimal strategy for a database and use case."""
        strategies = cls.get_supported_strategies(database_type)
        
        if not strategies:
            return "BasicSimilarityStrategy"  # Universal fallback
        
        # Use case specific recommendations
        use_case_preferences = {
            "general": ["HybridUniversalStrategy", "RerankedStrategy", "BasicSimilarityStrategy"],
            "simple": ["BasicSimilarityStrategy"],
            "filtered": ["MetadataFilteredStrategy", "HybridUniversalStrategy"],
            "complex": ["HybridUniversalStrategy", "MultiQueryStrategy", "RerankedStrategy"],
            "performance": ["BasicSimilarityStrategy", "MultiQueryStrategy"]
        }
        
        preferred_order = use_case_preferences.get(use_case, use_case_preferences["general"])
        
        # Find the best available strategy
        for preferred in preferred_order:
            if preferred in strategies:
                return preferred
        
        # Return highest compatibility score as fallback
        best_strategy = max(strategies.items(), key=lambda x: x[1]["compatibility_score"])
        return best_strategy[0]
    
    @classmethod
    def validate_configuration(
        cls, 
        database_type: str, 
        strategy_name: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a strategy configuration for a specific database."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check if database type is supported
        if database_type not in cls.DATABASE_CAPABILITIES:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Unsupported database type: {database_type}")
            return validation_result
        
        # Check if strategy is compatible
        supported_strategies = cls.get_supported_strategies(database_type)
        if strategy_name not in supported_strategies:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Strategy {strategy_name} is not compatible with {database_type}"
            )
            return validation_result
        
        # Database-specific validation
        db_caps = cls.DATABASE_CAPABILITIES[database_type]
        strategy_info = supported_strategies[strategy_name]
        
        # Check distance metrics
        if "distance_metric" in config:
            metric = config["distance_metric"]
            supported_metrics = db_caps.get("distance_metrics", [])
            if supported_metrics and metric not in supported_metrics:
                validation_result["warnings"].append(
                    f"Distance metric '{metric}' may not be supported. "
                    f"Supported metrics: {supported_metrics}"
                )
        
        # Check batch sizes
        if "batch_size" in config:
            batch_size = config["batch_size"]
            max_batch = db_caps.get("max_batch_size")
            if max_batch and batch_size > max_batch:
                validation_result["warnings"].append(
                    f"Batch size {batch_size} exceeds recommended maximum {max_batch}"
                )
        
        # Add recommendations based on missing preferred capabilities
        missing_preferred = strategy_info.get("preferred_missing", [])
        if missing_preferred:
            validation_result["recommendations"].append(
                f"For optimal performance, consider using a database that supports: {missing_preferred}"
            )
        
        return validation_result
    
    @classmethod
    def get_database_info(cls, database_type: str) -> Dict[str, Any]:
        """Get comprehensive information about a database's capabilities."""
        if database_type not in cls.DATABASE_CAPABILITIES:
            return {}
        
        db_info = cls.DATABASE_CAPABILITIES[database_type].copy()
        db_info["supported_strategies"] = list(cls.get_supported_strategies(database_type).keys())
        db_info["optimal_strategies"] = {
            use_case: cls.get_optimal_strategy(database_type, use_case)
            for use_case in ["general", "simple", "filtered", "complex", "performance"]
        }
        
        return db_info