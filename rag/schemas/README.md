# RAG Strategy Schema System

## Overview

The RAG system uses a unified strategy schema format that defines how document processing pipelines are configured. All strategies follow a consistent structure defined in `/rag/schema.yaml`.

## Schema Format

### Strategy Definition

Strategies are defined in YAML files with the following structure:

```yaml
strategies:
  - name: "strategy_name"
    description: "Strategy description"
    tags: ["tag1", "tag2"]
    use_cases: 
      - "Use case 1"
      - "Use case 2"
    
    components:
      parser:
        type: "ParserType"
        config: {...}
      
      extractors:
        - type: "ExtractorType"
          priority: 100
          enabled: true
          config: {...}
      
      embedder:
        type: "EmbedderType"
        config: {...}
      
      vector_store:
        type: "StoreType"
        config: {...}
      
      retrieval_strategy:
        type: "RetrievalType"
        config: {...}
    
    # Optional sections
    optimization:
      performance_priority: "balanced"
      resource_constraints: {...}
      batch_settings: {...}
      caching: {...}
    
    validation:
      max_document_length: 100000
      required_metadata: [...]
      content_filters: [...]
    
    monitoring:
      metrics_enabled: true
      log_level: "INFO"
```

## Key Features

### 1. Unified Format
All strategy configurations use the same schema, ensuring consistency across:
- Demo strategies (`/demos/demo_strategies.yaml`)
- Default strategies (`/default_strategies.yaml`)
- Config samples (`/config_samples/*.yaml`)

### 2. Component Priority
Extractors can have priority values (0-100) to control execution order:
```yaml
extractors:
  - type: "EntityExtractor"
    priority: 100  # Runs first
  - type: "SummaryExtractor"
    priority: 50   # Runs second
```

### 3. Tags and Use Cases
Strategies can be searched and filtered by tags and use cases:
```python
loader = StrategyLoader()
strategies = loader.get_strategies_by_tag("demo")
strategies = loader.get_strategies_by_use_case("Customer support")
```

### 4. Optimization Settings
Fine-tune performance with optimization settings:
```yaml
optimization:
  performance_priority: "accuracy"  # speed, accuracy, or balanced
  batch_settings:
    batch_size: 32
    max_batch_wait_ms: 100
  caching:
    enabled: true
    ttl_seconds: 3600
```

### 5. Validation Rules
Define document validation rules:
```yaml
validation:
  required_metadata: ["document_id", "source"]
  min_document_length: 100
  max_document_length: 100000
  content_filters: ["no_pii", "english_only"]
```

## Component Schemas

Each component type has its own schema definition in `/components/{type}/{name}/schema.yaml`:

```yaml
name: component_name
type: component_type
class_name: ComponentClass
description: "Component description"
config_schema:
  type: object
  properties:
    setting1:
      type: string
      description: "Setting description"
dependencies:
  required: ["package1"]
  optional: ["package2"]
use_cases: ["use case 1", "use case 2"]
```

## CI/CD Integration

### Schema Validation

The `schema_validator.py` script validates:
1. Main schema structure
2. Component schema definitions
3. Strategy configuration files
4. Schema compatibility

Run validation:
```bash
python rag/scripts/schema_validator.py
```

### Automated Validation

GitHub Actions automatically validates schemas on:
- Push to schema-related files
- Pull requests affecting schemas

### Consolidated Schema

The validator generates consolidated schema files:
- `schemas/consolidated.yaml` - YAML format
- `schemas/consolidated.json` - JSON format

These contain all component definitions for easy reference.

## Migration Guide

### From Legacy Format

Legacy format (top-level strategies):
```yaml
strategy_name:
  name: "Strategy Name"
  components: {...}
```

New unified format:
```yaml
strategies:
  - name: "strategy_name"
    description: "Strategy Name"
    components: {...}
```

### Backward Compatibility

The `StrategyLoader` supports both formats:
- Detects format automatically
- Converts legacy format on-the-fly
- No code changes required

## Best Practices

### 1. Use Descriptive Names
```yaml
name: "customer_support_optimization"  # Good
name: "strategy1"  # Bad
```

### 2. Add Comprehensive Tags
```yaml
tags: ["support", "tickets", "helpdesk", "production"]
```

### 3. Define Clear Use Cases
```yaml
use_cases:
  - "Customer support ticket routing"
  - "Knowledge base search"
  - "Support automation"
```

### 4. Set Appropriate Priorities
```yaml
extractors:
  - type: "PIIDetector"
    priority: 100  # Run first for compliance
  - type: "SummaryExtractor"
    priority: 50   # Run after critical extractors
```

### 5. Document Component Configs
```yaml
parser:
  type: "PDFParser"
  config:
    extract_metadata: true  # Include document properties
    ocr_enabled: false      # Disable for performance
    combine_pages: true     # Treat as single document
```

## Examples

### Minimal Strategy
```yaml
strategies:
  - name: "minimal"
    description: "Minimal configuration"
    tags: ["simple"]
    use_cases: ["Quick start"]
    
    components:
      parser:
        type: "PlainTextParser"
        config:
          chunk_size: 1000
      embedder:
        type: "OllamaEmbedder"
        config:
          model: "nomic-embed-text"
      vector_store:
        type: "ChromaStore"
        config:
          collection_name: "documents"
      retrieval_strategy:
        type: "BasicSimilarityStrategy"
        config:
          top_k: 10
```

### Production Strategy
```yaml
strategies:
  - name: "production"
    description: "Production-ready configuration"
    tags: ["production", "optimized"]
    use_cases: ["Enterprise search"]
    
    components:
      parser:
        type: "PDFParser"
        config:
          extract_metadata: true
          ocr_enabled: true
      
      extractors:
        - type: "EntityExtractor"
          priority: 100
          config:
            confidence_threshold: 0.8
        - type: "SummaryExtractor"
          priority: 90
      
      embedder:
        type: "OpenAIEmbedder"
        config:
          model: "text-embedding-3-small"
          batch_size: 100
      
      vector_store:
        type: "PineconeStore"
        config:
          index_name: "production"
      
      retrieval_strategy:
        type: "HybridUniversalStrategy"
        config:
          final_k: 10
    
    optimization:
      performance_priority: "balanced"
      caching:
        enabled: true
    
    monitoring:
      metrics_enabled: true
      log_level: "INFO"
```

## Troubleshooting

### Schema Validation Errors

If validation fails:
1. Check YAML syntax
2. Verify required fields are present
3. Ensure component types exist
4. Validate config parameters

### Loading Issues

If strategies don't load:
1. Check file path
2. Verify schema format
3. Check for missing dependencies
4. Review error logs

### Performance Issues

If processing is slow:
1. Adjust batch sizes
2. Enable caching
3. Use appropriate chunk sizes
4. Consider simpler extractors

### Metadata Handling

The system automatically handles nested metadata serialization:
- **Storage**: Vector stores may serialize nested objects as JSON strings
- **Retrieval**: Automatic parsing of JSON strings back to objects
- **ChromaDB**: Built-in support for nested metadata parsing
- **Extractors**: Complex metadata structures are preserved

## Contributing

When adding new strategies or components:
1. Follow the unified schema format
2. Add comprehensive documentation
3. Include example configurations
4. Update component schemas
5. Run schema validation
6. Add tests for new components