# Universal Retrieval Strategies

This directory contains a **database-agnostic retrieval system** that automatically optimizes strategies based on your vector database capabilities. The system is designed to be plug-and-play with easy contribution paths.

## 🏗️ Architecture Overview

```
retrieval/
├── base.py                    # Base classes and interfaces
├── registry.py                # Plugin registration system  
├── factory.py                 # Strategy creation with database optimization
├── strategies/
│   ├── universal/             # Database-agnostic strategies (individual files)
│   │   ├── __init__.py        # Auto-discovery and metadata
│   │   ├── basic_similarity.py
│   │   ├── metadata_filtered.py
│   │   ├── multi_query.py
│   │   ├── reranked.py
│   │   └── hybrid_universal.py
│   └── database_specific.py   # Database-optimized implementations
└── README.md                  # This file

stores/
└── capabilities.py            # Database capability definitions
```

## 🔄 How It Works

1. **Universal Strategies** - Work with any vector database
2. **Capability Mapping** - Each database declares its supported features
3. **Automatic Optimization** - System chooses the best strategy for your database
4. **Plugin System** - Easy to add new strategies and database support

### Example Flow:
```python
# You specify what you want
strategy_config = {"type": "HybridUniversalStrategy"}

# System automatically optimizes for your database
factory.create_retrieval_strategy_from_config(
    strategy_config, 
    database_type="ChromaStore"  # Auto-selects ChromaDB-optimized version if available
)
```

## 🎯 Universal Strategies

### **BasicSimilarityStrategy**
Simple vector similarity search that works with any database.

```json
{
  "retrieval_strategy": {
    "type": "BasicSimilarityStrategy",
    "config": {
      "distance_metric": "cosine"
    }
  }
}
```

**Use Cases:**
- Getting started with RAG
- Simple semantic search
- Baseline performance testing

---

### **MetadataFilteredStrategy**
Vector search with intelligent metadata filtering.

```json
{
  "retrieval_strategy": {
    "type": "MetadataFilteredStrategy", 
    "config": {
      "distance_metric": "cosine",
      "default_filters": {
        "priority": ["high", "medium"],
        "type": "documentation"
      }
    }
  }
}
```

**Features:**
- **Native filtering** when database supports it
- **Fallback filtering** for databases without native support
- **Complex filter operators** ($ne, $in, $gt, etc.)

**Use Cases:**
- Domain-specific searches
- Multi-tenant applications
- Content categorization

---

### **MultiQueryStrategy**
Uses multiple query variations to improve recall.

```json
{
  "retrieval_strategy": {
    "type": "MultiQueryStrategy",
    "config": {
      "num_queries": 3,
      "aggregation_method": "max",
      "distance_metric": "cosine"
    }
  }
}
```

**Aggregation Methods:**
- `max` - Best score across all queries
- `mean` - Average score across queries  
- `weighted` - Weighted average (later queries weighted less)

**Use Cases:**
- Ambiguous queries
- Query expansion scenarios
- Improving recall for complex questions

---

### **RerankedStrategy**
Multi-factor re-ranking for sophisticated relevance scoring.

```json
{
  "retrieval_strategy": {
    "type": "RerankedStrategy",
    "config": {
      "initial_k": 20,
      "rerank_factors": {
        "recency": 0.1,
        "length": 0.05,
        "metadata_boost": 0.2
      }
    }
  }
}
```

**Reranking Factors:**
- `recency` - Boost newer documents
- `length` - Preference for document length
- `metadata_boost` - Priority/type-based boosts

**Use Cases:**
- Production systems requiring nuanced ranking
- Time-sensitive content
- Multi-factor relevance

---

### **HybridUniversalStrategy**
Combines multiple strategies with configurable weights.

```json
{
  "retrieval_strategy": {
    "type": "HybridUniversalStrategy",
    "config": {
      "strategies": [
        {"type": "BasicSimilarityStrategy", "weight": 0.5},
        {"type": "MetadataFilteredStrategy", "weight": 0.3},
        {"type": "RerankedStrategy", "weight": 0.2}
      ]
    }
  }
}
```

**Use Cases:**
- Production systems
- Balanced precision and recall
- Complex retrieval requirements

## 🗄️ Database Capability System

The system automatically detects your database capabilities and optimizes strategies accordingly.

### Supported Databases

| Database | Basic Search | Metadata Filtering | Batch Ops | Hybrid Search | Notes |
|----------|-------------|-------------------|-----------|---------------|-------|
| **ChromaDB** | ✅ | ✅ | ✅ | ❌ | Great for development |
| **Pinecone** | ✅ | ✅ | ✅ | ❌ | Managed service |
| **Weaviate** | ✅ | ✅ | ✅ | ✅ | GraphQL API |
| **Qdrant** | ✅ | ✅ | ✅ | ❌ | High performance |
| **Milvus** | ✅ | ✅ | ✅ | ❌ | Distributed |
| **FAISS** | ✅ | ❌* | ✅ | ❌ | Fastest, limited metadata |

*\* Requires post-processing for metadata filtering*

### Usage Examples

```python
from retrieval.factory import create_retrieval_strategy_from_config
from retrieval.registry import get_registry

# Get optimal strategy for your database
registry = get_registry()
optimal_strategy = registry.get_optimal_strategy_name("ChromaStore", "general")
print(f"Optimal strategy: {optimal_strategy}")

# Get compatibility information
compatible = registry.get_compatible_strategies("ChromaStore")
for name, info in compatible.items():
    print(f"{name}: {info['compatibility_score']:.2f}")

# Validate configuration
validation = registry.validate_strategy_config(
    "ChromaStore", 
    "HybridUniversalStrategy",
    {"distance_metric": "cosine"}
)
print(f"Valid: {validation['valid']}")
```

## 🔧 Configuration Examples

### Quick Start (Universal)
```bash
# Use universal strategies with any database
uv run python cli.py --config config_examples/universal_retrieval_config.json \
  ingest samples/csv/small_sample.csv

uv run python cli.py --config config_examples/universal_retrieval_config.json \
  search "password reset"
```

### Advanced Hybrid
```bash
# Multi-strategy approach
uv run python cli.py --config config_examples/hybrid_universal_config.json \
  ingest samples/csv/small_sample.csv

uv run python cli.py --config config_examples/hybrid_universal_config.json \
  search "login authentication"
```

### Database-Optimized (Automatic)
```python
from api import SearchAPI

# System automatically chooses ChromaDB-optimized strategies if available
api = SearchAPI("config_examples/universal_retrieval_config.json")
results = api.search("security issue", top_k=5)
```

## 🧩 Adding New Components

### Adding a Universal Strategy

1. **Create the strategy file:**

```python
# retrieval/strategies/universal/my_new_strategy.py
"""My new strategy - description of what it does."""

from typing import List, Dict, Any
from ...base import RetrievalStrategy, RetrievalResult
from core.base import Document


class MyNewStrategy(RetrievalStrategy):
    def __init__(self, name: str = "MyNewStrategy", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.my_param = config.get("my_param", "default")
    
    def retrieve(self, query_embedding, vector_store, top_k=5, **kwargs):
        # Your retrieval logic here
        documents = vector_store.search(query_embedding=query_embedding, top_k=top_k)
        scores = [doc.metadata.get("similarity_score", 0.0) for doc in documents]
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={"strategy": self.name, "my_param": self.my_param}
        )
    
    def supports_vector_store(self, vector_store_type: str) -> bool:
        return True  # Universal strategy
```

2. **Add to the universal module:**

```python
# retrieval/strategies/universal/__init__.py
from .my_new_strategy import MyNewStrategy

# Add to __all__ list
__all__ = [
    "BasicSimilarityStrategy",
    "MetadataFilteredStrategy", 
    "MultiQueryStrategy",
    "RerankedStrategy",
    "HybridUniversalStrategy",
    "MyNewStrategy"  # Add here
]

# Add to UNIVERSAL_STRATEGIES metadata
UNIVERSAL_STRATEGIES = {
    # ... existing strategies ...
    "MyNewStrategy": {
        "class": MyNewStrategy,
        "version": "1.0.0",
        "description": "My new retrieval strategy",
        "aliases": ["my_new"],
        "use_cases": ["specific_use_case"],
        "performance": "medium",
        "complexity": "low"
    }
}
```

3. **Auto-registration (optional):**
The registry will automatically discover your new strategy through the `__init__.py` file.

3. **Add capability requirements:**

```python
# stores/capabilities.py - in STRATEGY_REQUIREMENTS
"MyNewStrategy": {
    "required_capabilities": ["basic_similarity"],
    "preferred_capabilities": ["metadata_filtering"],
    "description": "My new retrieval strategy"
}
```

### Adding Database-Specific Optimization

1. **Create optimized implementation:**

```python
# retrieval/strategies/database_specific.py
class MyDatabaseSpecificStrategy(RetrievalStrategy):
    def retrieve(self, query_embedding, vector_store, top_k=5, **kwargs):
        # Use database-specific optimizations
        if hasattr(vector_store, 'native_my_feature'):
            return vector_store.native_my_feature(query_embedding, top_k)
        else:
            # Fallback to universal approach
            return super().retrieve(query_embedding, vector_store, top_k, **kwargs)
```

2. **Register database-specific strategy:**

```python
# retrieval/registry.py - in _load_database_specific_strategies()
self.register_database_specific_strategy(
    "MyDatabase", 
    "MyNewStrategy", 
    MyDatabaseSpecificStrategy
)
```

### Adding a New Database

1. **Define capabilities:**

```python
# stores/capabilities.py - in DATABASE_CAPABILITIES
"MyVectorStore": {
    "supported": [
        "basic_similarity",
        "metadata_filtering",
        "my_special_feature"
    ],
    "distance_metrics": ["cosine", "euclidean"],
    "max_batch_size": 500,
    "native_filtering": True,
    "notes": "My custom vector database"
}
```

2. **Add any special capability:**

```python
# stores/capabilities.py - in CAPABILITIES
"my_special_feature": "Description of my special feature"
```

3. **Test compatibility:**

```python
from retrieval.registry import get_registry

registry = get_registry()
compatible = registry.get_compatible_strategies("MyVectorStore")
optimal = registry.get_optimal_strategy_name("MyVectorStore", "general")
```

## 🧪 Testing Your Contributions

### Test New Strategy
```python
from retrieval.registry import get_registry

registry = get_registry()

# Test strategy creation
strategy = registry.create_strategy("MyNewStrategy", {"my_param": "test"})
print(f"Created: {strategy.name}")

# Test compatibility
compatible = registry.get_compatible_strategies("ChromaStore")
assert "MyNewStrategy" in compatible
```

### Test Database Support
```python
# Test capability detection
db_info = registry.get_database_info("MyVectorStore")
print(f"Supported strategies: {db_info['supported_strategies']}")

# Test validation
validation = registry.validate_strategy_config(
    "MyVectorStore", 
    "MyNewStrategy", 
    {"my_param": "value"}
)
assert validation["valid"]
```

### Integration Test
```bash
# Test with your new components
uv run python test_retrieval_system.py
uv run python example_retrieval_usage.py
```

## 🔍 Debugging and Monitoring

### Strategy Selection Debug
```python
from retrieval.registry import get_registry

registry = get_registry()

# See what strategy would be selected
optimal = registry.get_optimal_strategy_name("ChromaStore", "general")
print(f"Would select: {optimal}")

# See compatibility scores
strategies = registry.get_compatible_strategies("ChromaStore")
for name, info in strategies.items():
    print(f"{name}: score={info['compatibility_score']:.2f}, preferred_met={info['preferred_met']}")
```

### Configuration Validation
```python
validation = registry.validate_strategy_config(
    "ChromaStore",
    "HybridUniversalStrategy", 
    {"distance_metric": "invalid_metric"}
)

print(f"Valid: {validation['valid']}")
print(f"Warnings: {validation['warnings']}")
print(f"Errors: {validation['errors']}")
```

## 📈 Performance Guidelines

### Strategy Selection by Use Case

| Use Case | Recommended Strategy | Why |
|----------|---------------------|-----|
| **Getting Started** | `BasicSimilarityStrategy` | Simple, fast, reliable |
| **Production General** | `HybridUniversalStrategy` | Balanced performance |
| **High Precision** | `RerankedStrategy` | Multi-factor ranking |
| **Filtered Content** | `MetadataFilteredStrategy` | Efficient filtering |
| **Complex Queries** | `MultiQueryStrategy` | Better recall |
| **High Performance** | `BasicSimilarityStrategy` | Minimal overhead |

### Optimization Tips

1. **Start Simple** - Begin with `BasicSimilarityStrategy`
2. **Measure Performance** - Use built-in compatibility scores
3. **Database-Specific** - Let the system choose optimized implementations
4. **Validate Config** - Use validation functions to catch issues early
5. **Monitor Results** - Check strategy metadata in results

## 🚀 Migration Guide

### From Database-Specific to Universal

**Old (Database-Specific):**
```json
{
  "retrieval_strategy": {
    "type": "ChromaHybridStrategy",
    "config": {...}
  }
}
```

**New (Universal):**
```json
{
  "retrieval_strategy": {
    "type": "HybridUniversalStrategy", 
    "config": {...}
  }
}
```

The system maintains backward compatibility while providing universal strategies that work across databases.

### Benefits of Migration

- **Database Independence** - Switch databases without changing strategy configs
- **Automatic Optimization** - System chooses best implementation for your database
- **Future Proof** - New databases automatically supported
- **Easier Contribution** - Single universal strategy works everywhere

---

This universal retrieval system makes it easy to contribute new strategies and database support while maintaining optimal performance across different vector databases. The plugin architecture ensures that improvements benefit the entire ecosystem! 🎯