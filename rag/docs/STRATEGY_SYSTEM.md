# Strategy System Documentation

The RAG system uses a **strategy-first** approach where entire document processing and retrieval pipelines are configured through YAML files. This provides maximum flexibility while maintaining consistency and reusability.

## Table of Contents

1. [Overview](#overview)
2. [Strategy Configuration](#strategy-configuration)
3. [Components](#components)
4. [Retrieval Strategies](#retrieval-strategies)
5. [Creating Custom Strategies](#creating-custom-strategies)
6. [Best Practices](#best-practices)
7. [Examples](#examples)

## Overview

A strategy defines a complete RAG pipeline including:
- **Parsers**: How documents are read and processed
- **Extractors**: What metadata is extracted from documents
- **Embedders**: How text is converted to vectors
- **Vector Stores**: Where and how embeddings are stored
- **Retrieval**: How documents are searched and ranked

### Benefits

- **No Code Changes**: Modify behavior through configuration
- **Reusability**: Share strategies across projects
- **Consistency**: Ensure uniform processing
- **Version Control**: Track configuration changes
- **A/B Testing**: Easy comparison of approaches

## Strategy Configuration

### Basic Structure

According to the schema, strategies must be defined in an array under a `strategies` key:

```yaml
strategies:
  - name: "strategy_name"
    description: "Human-readable description"
    tags: ["tag1", "tag2"]  # Optional tags for categorization
    use_cases: 
      - "Use case 1"
      - "Use case 2"
    
    components:
      # Document parsing
      parser:
        type: "ParserType"
        config:
          setting1: value1
          setting2: value2
      
      # Metadata extraction
      extractors:
        - type: "ExtractorType1"
          config: {}
        - type: "ExtractorType2"
          config:
            setting: value
      
      # Embedding generation
      embedder:
        type: "EmbedderType"
        config:
          model: "model-name"
          dimension: 768
      
      # Vector storage
      vector_store:
        type: "StoreType"
        config:
          collection_name: "collection"
          persist_directory: "./vectordb"
      
      # Retrieval configuration
      retrieval_strategy:
        type: "StrategyType"
        config:
          top_k: 5
```

**Note:** Multiple strategies can be defined in the same file by adding more items to the `strategies` array.

### Complete Example

```yaml
strategies:
  - name: "research_papers_demo"
    description: "Optimized for academic papers with citations and entities"
    tags: ["academic", "research", "papers"]
    use_cases:
      - "Literature review"
      - "Research synthesis"
      - "Citation tracking"
    
    components:
      parser:
        type: "PDFParser"
        config:
          extract_images: false
          extract_metadata: true
          chunk_size: 1024
          chunk_overlap: 200
      
      extractors:
        - type: "HeadingExtractor"
          config:
            levels: [1, 2, 3]
            include_content: true
        
        - type: "CitationExtractor"
          config:
            formats: ["APA", "MLA", "Chicago"]
            validate: true
        
        - type: "EntityExtractor"
          config:
            entities: ["PERSON", "ORG", "DATE"]
            confidence_threshold: 0.8
        
        - type: "KeywordExtractor"
          config:
            max_keywords: 15
            algorithm: "yake"
      
      embedder:
        type: "OllamaEmbedder"
        config:
          model: "nomic-embed-text"
          dimension: 768
          batch_size: 32
          timeout: 60
      
      vector_store:
        type: "ChromaStore"
        config:
          collection_name: "research_papers"
          persist_directory: "./vectordb/research"
          distance_metric: "cosine"
      
      retrieval_strategy:
        type: "RerankedStrategy"
        config:
          initial_k: 20
          final_k: 5
          rerank_factors:
            recency_weight: 0.2
            citation_weight: 0.3
            relevance_weight: 0.5
```

## Components

### Parsers

Available parsers and their configurations:

#### TextParser
```yaml
parser:
  type: "TextParser"
  config:
    encoding: "utf-8"
    chunk_size: 512
    chunk_overlap: 50
    preserve_formatting: true
```

#### PDFParser
```yaml
parser:
  type: "PDFParser"
  config:
    extract_images: false
    extract_tables: true
    extract_metadata: true
    ocr_enabled: false
```

#### HTMLParser
```yaml
parser:
  type: "HTMLParser"
  config:
    extract_links: true
    preserve_structure: false
    remove_scripts: true
    remove_styles: true
```

#### CSVParser
```yaml
parser:
  type: "CSVParser"
  config:
    delimiter: ","
    has_header: true
    field_mapping:
      content_field: "description"
      id_field: "id"
```

#### MarkdownParser
```yaml
parser:
  type: "MarkdownParser"
  config:
    preserve_formatting: true
    extract_code_blocks: true
    extract_links: true
```

### Extractors

Available extractors for metadata enrichment:

#### EntityExtractor
```yaml
- type: "EntityExtractor"
  config:
    entities: ["PERSON", "ORG", "GPE", "DATE", "MONEY"]
    model: "en_core_web_sm"
    confidence_threshold: 0.7
```

#### KeywordExtractor
```yaml
- type: "KeywordExtractor"
  config:
    max_keywords: 10
    algorithm: "yake"  # or "rake", "tfidf"
    language: "en"
```

#### SummaryExtractor
```yaml
- type: "SummaryExtractor"
  config:
    max_sentences: 3
    include_keywords: true
    include_statistics: true
```

#### HeadingExtractor
```yaml
- type: "HeadingExtractor"
  config:
    levels: [1, 2, 3, 4]
    include_content: true
    max_content_length: 200
```

#### PatternExtractor
```yaml
- type: "PatternExtractor"
  config:
    patterns:
      email: "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"
      phone: "\\d{3}-\\d{3}-\\d{4}"
      custom: "your-regex-here"
    include_context: true
    context_window: 50
```

#### LinkExtractor
```yaml
- type: "LinkExtractor"
  config:
    validate_urls: false
    extract_anchor_text: true
    categorize: true  # internal vs external
```

#### ContentStatisticsExtractor
```yaml
- type: "ContentStatisticsExtractor"
  config:
    calculate_readability: true
    calculate_sentiment: false
    word_frequency: true
```

### Embedders

#### OllamaEmbedder
```yaml
embedder:
  type: "OllamaEmbedder"
  config:
    model: "nomic-embed-text"
    host: "http://localhost:11434"
    dimension: 768
    batch_size: 32
    timeout: 60
```

#### OpenAIEmbedder (future)
```yaml
embedder:
  type: "OpenAIEmbedder"
  config:
    model: "text-embedding-3-small"
    api_key: "${OPENAI_API_KEY}"
    dimension: 1536
    batch_size: 100
```

### Vector Stores

#### ChromaStore
```yaml
vector_store:
  type: "ChromaStore"
  config:
    collection_name: "my_collection"
    persist_directory: "./chromadb"
    distance_metric: "cosine"  # or "l2", "ip"
```

#### QdrantStore (future)
```yaml
vector_store:
  type: "QdrantStore"
  config:
    collection_name: "my_collection"
    host: "localhost"
    port: 6333
    vector_size: 768
    distance: "Cosine"
```

## Retrieval Strategies

### BasicSimilarityStrategy
Simple vector similarity search.

```yaml
retrieval_strategy:
  type: "BasicSimilarityStrategy"
  config:
    top_k: 5
    distance_metric: "cosine"
    score_threshold: 0.7
```

### MetadataFilteredStrategy
Filter results based on metadata before/after retrieval.

```yaml
retrieval_strategy:
  type: "MetadataFilteredStrategy"
  config:
    top_k: 10
    filters:
      priority: ["high", "critical"]
      date_after: "2024-01-01"
      category: "technical"
    filter_mode: "pre"  # or "post"
```

### RerankedStrategy
Re-rank results using multiple factors.

```yaml
retrieval_strategy:
  type: "RerankedStrategy"
  config:
    initial_k: 20
    final_k: 5
    rerank_factors:
      recency_weight: 0.2
      length_weight: 0.1
      metadata_boost:
        priority:
          high: 1.5
          medium: 1.0
          low: 0.8
```

### MultiQueryStrategy
Generate multiple query variations for better recall.

```yaml
retrieval_strategy:
  type: "MultiQueryStrategy"
  config:
    num_queries: 3
    aggregation: "rrf"  # reciprocal rank fusion
    top_k_per_query: 5
    final_k: 5
```

### HybridUniversalStrategy
Combine multiple strategies with weighted scoring.

```yaml
retrieval_strategy:
  type: "HybridUniversalStrategy"
  config:
    strategies:
      - type: "BasicSimilarityStrategy"
        weight: 0.6
        config:
          top_k: 10
      - type: "MetadataFilteredStrategy"
        weight: 0.4
        config:
          top_k: 10
    fusion_method: "weighted"  # or "rrf"
    final_k: 5
```

## Creating Custom Strategies

### Step 1: Define Your Requirements

Consider:
- Document types and formats
- Important metadata to extract
- Search patterns and use cases
- Performance requirements
- Storage constraints

### Step 2: Create Strategy YAML

Create a new file `my_strategies.yaml`:

```yaml
strategies:
  - name: "my_custom_strategy"
    description: "Custom strategy for my use case"
    tags: ["custom", "specific"]
    use_cases:
      - "Specific use case 1"
      - "Specific use case 2"
    
    components:
      # Your configuration here
```

### Step 3: Use with CLI

```bash
# Use your custom strategy file
python cli.py --strategy-file my_strategies.yaml \
    ingest documents/ --strategy my_custom_strategy

# Search using your strategy
python cli.py --strategy-file my_strategies.yaml \
    search "query" --strategy my_custom_strategy
```

### Step 4: Test and Iterate

```bash
# Test configuration
python cli.py --strategy-file my_strategies.yaml \
    test --strategy my_custom_strategy

# View strategy details
python cli.py --strategy-file my_strategies.yaml \
    strategies show my_custom_strategy
```

## Best Practices

### 1. Start Simple
Begin with a basic configuration and add complexity as needed:

```yaml
strategies:
  - name: "simple_start"
    description: "Simple configuration to get started"
    components:
      parser:
        type: "TextParser"
      embedder:
        type: "OllamaEmbedder"
      vector_store:
        type: "ChromaStore"
      retrieval_strategy:
        type: "BasicSimilarityStrategy"
```

### 2. Use Appropriate Chunk Sizes

| Document Type | Recommended Chunk Size | Overlap |
|--------------|------------------------|---------|
| Technical Docs | 512-1024 | 100-200 |
| Legal Text | 256-512 | 50-100 |
| Research Papers | 1024-2048 | 200-400 |
| Chat/Support | 256-512 | 50 |
| Code | Function/Class level | 0 |

### 3. Choose Extractors Wisely

Match extractors to your content:
- **Academic**: CitationExtractor, EntityExtractor
- **Business**: TableExtractor, ContentStatisticsExtractor
- **Technical**: HeadingExtractor, CodeExtractor
- **Support**: PatternExtractor, SentimentExtractor

### 4. Optimize Retrieval

Balance precision and recall:
- **High Precision**: Use filters and reranking
- **High Recall**: Use multi-query and larger top_k
- **Balanced**: Use hybrid strategies

### 5. Version Your Strategies

```yaml
strategies:
  - name: "my_strategy_v2"
    description: "Version 2.0 with improvements"
    tags: ["v2", "production"]
    metadata:
      version: "2.0"
      changelog: "Added entity extraction, increased chunk overlap"
    # ... rest of configuration
```

## Examples

### Example 1: Legal Document Strategy

```yaml
strategies:
  - name: "legal_documents"
    description: "Optimized for legal contracts and agreements"
    tags: ["legal", "contracts"]
    
    components:
      parser:
        type: "PDFParser"
        config:
          extract_metadata: true
          preserve_formatting: true
          chunk_size: 256
          chunk_overlap: 50
      
      extractors:
        - type: "ClauseExtractor"
          config:
            clause_types: ["indemnity", "liability", "termination"]
        - type: "EntityExtractor"
          config:
            entities: ["ORG", "PERSON", "DATE", "MONEY"]
        - type: "PatternExtractor"
          config:
            patterns:
              case_number: "[0-9]{2}-[A-Z]{2}-[0-9]{4}"
              statute: "\\d{1,3}\\s+U\\.S\\.C\\.\\s+§\\s+\\d+"
      
      retrieval_strategy:
        type: "MetadataFilteredStrategy"
        config:
          top_k: 10
          filters:
            document_type: "contract"
            jurisdiction: "US"
```

### Example 2: Multi-lingual Strategy

```yaml
multilingual_docs:
  description: "Handle documents in multiple languages"
  
  components:
    parser:
      type: "UniversalParser"
      config:
        detect_language: true
        supported_languages: ["en", "es", "fr", "de"]
    
    extractors:
      - type: "LanguageDetector"
      - type: "MultilingualEntityExtractor"
        config:
          model: "xx_ent_wiki_sm"
    
    embedder:
      type: "MultilingualEmbedder"
      config:
        model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        dimension: 768
    
    retrieval_strategy:
      type: "CrossLingualStrategy"
      config:
        enable_translation: true
        primary_language: "en"
```

### Example 3: Real-time Chat Strategy

```yaml
chat_support:
  description: "Real-time customer chat processing"
  
  components:
    parser:
      type: "ConversationParser"
      config:
        preserve_speaker: true
        chunk_by_turn: true
    
    extractors:
      - type: "IntentExtractor"
      - type: "SentimentExtractor"
        config:
          real_time: true
      - type: "UrgencyDetector"
    
    embedder:
      type: "FastEmbedder"
      config:
        model: "all-MiniLM-L6-v2"
        dimension: 384
        cache_enabled: true
    
    retrieval_strategy:
      type: "CachedStrategy"
      config:
        cache_size: 1000
        ttl: 3600
        fallback_strategy: "BasicSimilarityStrategy"
```

## Troubleshooting

### Strategy Not Loading
- Check YAML syntax
- Verify all component types exist
- Ensure configuration keys are correct

### Poor Search Results
- Adjust chunk size and overlap
- Add relevant extractors
- Try different retrieval strategies
- Increase top_k values

### Slow Performance
- Reduce chunk size
- Use smaller embedding models
- Enable caching
- Optimize batch sizes

### High Memory Usage
- Use disk-based vector stores
- Reduce embedding dimensions
- Enable compression/quantization
- Process in smaller batches

## Next Steps

- [Component Guide](COMPONENTS.md) - Detailed component documentation
- [CLI Guide](../cli/README.md) - Using strategies with CLI
- [API Reference](API_REFERENCE.md) - Programmatic strategy usage
- [Examples](../demos/) - See strategies in action