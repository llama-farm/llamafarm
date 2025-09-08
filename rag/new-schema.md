# New Schema Architecture Plan

## Overview
This document outlines the plan to restructure the LlamaFarm RAG schema to achieve better separation of concerns by extracting datastores and retrieval/embedding strategies from the main strategy schema into a cleaner, more modular structure.

## Current Architecture Problems
1. **Tight Coupling**: Strategies currently contain all components mixed together
2. **Limited Reusability**: Datastores cannot be easily shared across different pipelines
3. **No Hybrid Support**: Difficult to use multiple embedding/retrieval strategies for hybrid databases
4. **Configuration Overhead**: Each strategy must redefine its entire stack

## Proposed Architecture

### Single Schema Structure
The RAG system will use ONE unified schema (`schema.yaml`) with two top-level sections:

```yaml
rag:
  # Independent database definitions with their own strategies
  databases:
    - name: "main_vectordb"
      type: "chroma"
      config:
        persist_directory: "./chroma_db"
        collection_name: "documents"
      embedding_strategies:
        - name: "openai_ada"
          type: "OpenAIEmbedder"
          config:
            model: "text-embedding-ada-002"
        - name: "local_bert"
          type: "SentenceTransformerEmbedder"
          config:
            model: "all-MiniLM-L6-v2"
      retrieval_strategies:
        - name: "semantic_search"
          type: "VectorRetriever"
          config:
            top_k: 10
        - name: "hybrid_search"
          type: "HybridRetriever"
          config:
            vector_weight: 0.7
            keyword_weight: 0.3

  # Data processing pipelines (ingestion)
  data_processing_strategies:
    - name: "pdf_processing"
      parsers:
        - type: "PDFParser_LlamaIndex"
          config:
            chunk_strategy: "semantic"
        - type: "PDFParser_PyPDF2"
          config:
            fallback: true
      extractors:
        - type: "MetadataExtractor"
          config:
            extract_headers: true
        - type: "KeywordExtractor"
          config:
            method: "yake"
```

## Complete Example Configuration

```yaml
# schema.yaml - Single unified schema
rag:
  # Database configurations with embedded strategies
  databases:
    - name: "research_papers_db"
      type: "chroma"
      config:
        persist_directory: "./research_db"
        collection_name: "papers"
      # Multiple embeddings for hybrid search
      embedding_strategies:
        - name: "scientific_embedder"
          type: "OpenAIEmbedder"
          config:
            model: "text-embedding-3-large"
            dimensions: 1536
        - name: "code_embedder"
          type: "CodeBertEmbedder"
          config:
            model: "microsoft/codebert-base"
            for_code_blocks: true
      # Multiple retrieval strategies
      retrieval_strategies:
        - name: "semantic_retrieval"
          type: "VectorRetriever"
          config:
            top_k: 20
            similarity_threshold: 0.7
        - name: "citation_graph"
          type: "GraphRetriever"
          config:
            include_citations: true
            max_hops: 2
        - name: "reranked"
          type: "RerankedRetriever"
          config:
            reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    - name: "metadata_store"
      type: "elasticsearch"
      config:
        host: "localhost:9200"
        index: "paper_metadata"
      # No embedding strategies needed for keyword search
      retrieval_strategies:
        - name: "faceted_search"
          type: "ElasticRetriever"
          config:
            fields: ["title", "authors", "year", "keywords"]
        - name: "full_text"
          type: "BM25Retriever"
          config:
            k1: 1.2
            b: 0.75

  # Data processing strategies for ingestion
  data_processing_strategies:
    - name: "scientific_pdf_processing"
      description: "Process scientific PDFs with citations and equations"
      parsers:
        - type: "PDFParser_LlamaIndex"
          config:
            chunk_strategy: "semantic"
            chunk_size: 1000
            preserve_equations: true
            extract_images: true
          fallback_parser: "PDFParser_PyPDF2"
        - type: "PDFParser_PyPDF2"
          config:
            as_fallback: true
            extract_text_method: "both"
      extractors:
        - type: "ScientificMetadataExtractor"
          config:
            extract_title: true
            extract_authors: true
            extract_abstract: true
            extract_citations: true
            extract_doi: true
        - type: "EquationExtractor"
          config:
            formats: ["latex", "mathml"]
        - type: "CodeBlockExtractor"
          config:
            languages: ["python", "r", "matlab"]
            add_language_tags: true
    
    - name: "customer_support_processing"
      description: "Process customer support tickets and conversations"
      parsers:
        - type: "CSVParser_Pandas"
          config:
            chunk_size: 100
            key_column: "ticket_id"
        - type: "TextParser_Python"
          config:
            chunk_strategy: "sentences"
            chunk_size: 512
      extractors:
        - type: "CustomerInfoExtractor"
          config:
            extract_email: true
            extract_account_id: true
            mask_pii: true
        - type: "SentimentExtractor"
          config:
            model: "vader"
        - type: "IssueClassifier"
          config:
            categories: ["billing", "technical", "feature_request"]
            multi_label: true
```

## Schema Structure Details

### Database Schema
```yaml
database:
  type: object
  required: [name, type]
  properties:
    name:
      type: string
      description: "Unique identifier for the database"
    type:
      type: string
      enum: ["chroma", "qdrant", "weaviate", "milvus", "pinecone", "elasticsearch", "faiss"]
    config:
      type: object
      description: "Database-specific configuration"
    embedding_strategies:
      type: array
      items:
        type: object
        required: [name, type]
        properties:
          name:
            type: string
          type:
            type: string
            enum: ["OpenAIEmbedder", "CohereEmbedder", "SentenceTransformerEmbedder", "OllamaEmbedder", "CodeBertEmbedder"]
          config:
            type: object
          condition:
            type: string
            description: "Optional condition for when to use this embedding"
    retrieval_strategies:
      type: array
      items:
        type: object
        required: [name, type]
        properties:
          name:
            type: string
          type:
            type: string
            enum: ["VectorRetriever", "HybridRetriever", "BM25Retriever", "RerankedRetriever", "GraphRetriever", "ElasticRetriever"]
          config:
            type: object
```

### Data Processing Strategy Schema
```yaml
data_processing_strategy:
  type: object
  required: [name, parsers]
  properties:
    name:
      type: string
      description: "Unique identifier for the processing strategy"
    description:
      type: string
    parsers:
      type: array
      items:
        type: object
        properties:
          type:
            type: string
            enum: ["auto", "PDFParser_LlamaIndex", "PDFParser_PyPDF2", "CSVParser_Pandas", "CSVParser_Python", "CSVParser_LlamaIndex", "TextParser_Python", "TextParser_LlamaIndex", "MarkdownParser_Python", "MarkdownParser_LlamaIndex", "DocxParser_PythonDocx", "DocxParser_LlamaIndex", "ExcelParser_Pandas", "ExcelParser_OpenPyXL", "ExcelParser_LlamaIndex"]
            description: "Parser type or 'auto' for automatic file type detection"
          config:
            type: object
          file_extensions:
            type: array
            items:
              type: string
            description: "File extensions this parser should handle (when not using auto)"
          mime_types:
            type: array
            items:
              type: string
            description: "MIME types this parser should handle (when not using auto)"
          fallback_parser:
            type: string
            description: "Parser to use if this one fails"
    extractors:
      type: array
      items:
        type: object
        required: [type]
        properties:
          type:
            type: string
          config:
            type: object
          required_for:
            type: array
            items:
              type: string
            description: "Database names that require this extractor"
```

### **File Type Mapping System**
The system maintains comprehensive mappings in `components/parsers/parser_registry.py`:

**40+ File Extensions**: `.pdf`, `.csv`, `.xlsx`, `.docx`, `.md`, `.py`, `.js`, `.java`, `.cpp`, `.json`, `.yaml`, etc.

**30+ MIME Types**: `application/pdf`, `text/csv`, `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`, `text/markdown`, `text/plain`, etc.

Each mapping includes multiple parser options with priority and tool information for automatic parser selection.

## Migration Plan

### Phase 1: Update Core Schema (Week 1)
1. **Update `schema.yaml`**:
   - Add new top-level `rag` section
   - Define `databases` array with embedding/retrieval strategies
   - Define `data_processing_strategies` array
   - Add JSON Schema validation for new structure
   - Maintain backward compatibility with deprecation warnings

2. **Schema Migration Path**:
   ```yaml
   # Old format (deprecated)
   strategies:
     - name: "old_strategy"
       components:
         parser: {...}
         embedder: {...}
         store: {...}
         retriever: {...}
   
   # New format
   rag:
     databases:
       - name: "migrated_db"
         type: "chroma"
         embedding_strategies: [...]
         retrieval_strategies: [...]
     data_processing_strategies:
       - name: "migrated_processing"
         parsers: [...]
         extractors: [...]
   ```

### Phase 2: Component Updates (Week 2)
1. **Create Database Manager** (`components/databases/database_manager.py`):
   ```python
   class DatabaseManager:
       def __init__(self, config: Dict):
           self.databases = {}
           self.load_databases(config)
       
       def get_database(self, name: str) -> Database:
           return self.databases[name]
       
       def get_embedder(self, db_name: str, strategy_name: str = None):
           # Return specific or default embedding strategy
       
       def get_retriever(self, db_name: str, strategy_name: str = None):
           # Return specific or default retrieval strategy
   ```

2. **Create Processing Pipeline Manager** (`components/pipelines/processing_manager.py`):
   ```python
   class ProcessingManager:
       def __init__(self, config: Dict):
           self.strategies = {}
           self.load_strategies(config)
       
       def process(self, strategy_name: str, documents: List[Document]):
           strategy = self.strategies[strategy_name]
           # Run parsers and extractors in sequence
   ```

### Phase 3: Factory Updates (Week 3)
1. **Update Existing Factories**:
   - `parser_factory.py`: Load from `data_processing_strategies`
   - `embedder_factory.py`: Load from `databases[].embedding_strategies`
   - `store_factory.py`: Load from `databases[]` config
   - `retriever_factory.py`: Load from `databases[].retrieval_strategies`

2. **Create Strategy Router**:
   ```python
   class StrategyRouter:
       def route_to_database(self, document: Document) -> str:
           # Determine which database based on document metadata
       
       def select_embedding_strategy(self, document: Document, db_name: str) -> str:
           # Choose embedding strategy based on content type
       
       def select_retrieval_strategy(self, query: Query, db_name: str) -> str:
           # Choose retrieval strategy based on query type
   ```

### Phase 4: CLI Updates (Week 4)
1. **New CLI Commands**:
   ```bash
   # Database management
   llamafarm database list
   llamafarm database create --config db_config.yaml
   llamafarm database info <db_name>
   
   # Processing strategies
   llamafarm process --strategy <name> --input <path> --target-db <db_name>
   
   # Query with specific strategies
   llamafarm query --db <db_name> --retrieval <strategy_name> "query text"
   ```

2. **Update Existing Commands**:
   - `ingest`: Use `data_processing_strategies` and route to databases
   - `search`: Use database-specific retrieval strategies
   - `info`: Show database and strategy information

### Phase 5: Testing and Migration (Week 5)
1. **Migration Script** (`scripts/migrate_schema.py`):
   ```python
   def migrate_old_strategy(old_strategy: Dict) -> Dict:
       """Convert old strategy format to new schema format"""
       return {
           "databases": extract_databases(old_strategy),
           "data_processing_strategies": extract_processing(old_strategy)
       }
   ```

2. **Validation and Testing**:
   - Unit tests for new managers
   - Integration tests for end-to-end flow
   - Performance comparison tests
   - Migration validation tests

## Benefits of New Architecture

### 1. **Clear Separation**
- Databases are independent entities with their own strategies
- Processing pipelines are separate from storage concerns
- Clean boundary between ingestion and retrieval

### 2. **Hybrid Database Support**
- Multiple embedding strategies per database for different content types
- Multiple retrieval strategies for different query patterns
- Easy A/B testing of strategies

### 3. **Reusability**
- Share databases across different applications
- Reuse processing strategies for different data sources
- Mix and match components freely

### 4. **Simplified Configuration**
- Clearer mental model: databases + processing
- Less duplication in configuration
- Easier to understand and maintain

## Example Use Cases

### Use Case 1: Multi-Modal Research Database
```yaml
rag:
  databases:
    - name: "research_multimodal"
      type: "weaviate"
      embedding_strategies:
        - name: "text_embedder"
          type: "OpenAIEmbedder"
          condition: "doc.type == 'text'"
        - name: "image_embedder"  
          type: "CLIPEmbedder"
          condition: "doc.type == 'image'"
        - name: "code_embedder"
          type: "CodeBertEmbedder"
          condition: "doc.type == 'code'"
```

### Use Case 2: Hybrid Search System
```yaml
rag:
  databases:
    - name: "hybrid_search_db"
      type: "qdrant"
      embedding_strategies:
        - name: "dense"
          type: "SentenceTransformerEmbedder"
        - name: "sparse"
          type: "SPLADEEmbedder"
      retrieval_strategies:
        - name: "hybrid"
          type: "HybridRetriever"
          config:
            dense_weight: 0.7
            sparse_weight: 0.3
```

### Use Case 3: Multi-Database Query
```yaml
rag:
  databases:
    - name: "recent_docs"
      type: "chroma"
      config:
        in_memory: true
    - name: "archive_docs"
      type: "milvus"
      config:
        distributed: true
```

## Files to Change

### Core Files
1. **`schema.yaml`** (PRIMARY):
   - Complete restructure with new `rag` top-level
   - Add `databases` and `data_processing_strategies` sections
   - Update all component schemas

### New Files to Create
1. `components/databases/database_manager.py`
2. `components/pipelines/processing_manager.py`
3. `components/routing/strategy_router.py`
4. `scripts/migrate_schema.py`
5. `tests/test_new_schema.py`

### Files to Update
1. `components/parsers/parser_factory.py`
2. `components/embedders/embedder_factory.py`
3. `components/stores/store_factory.py`
4. `components/retrievers/retriever_factory.py`
5. `cli.py` - Add new commands
6. `core/strategy_loader.py` - Support new schema
7. All test files for compatibility

## Implementation Order

1. **Week 1**: Define and validate new schemas
2. **Week 2**: Implement core managers and orchestrators
3. **Week 3**: Update factories and registries
4. **Week 4**: CLI and API updates
5. **Week 5**: Migration, testing, and documentation

## Backward Compatibility

### Compatibility Layer
```python
class LegacyStrategyAdapter:
    """Adapter to support old strategy format during transition"""
    
    def load_legacy_strategy(self, old_config: Dict) -> Dict:
        # Convert to new format
        new_config = {
            "rag": {
                "databases": [],
                "data_processing_strategies": []
            }
        }
        # Migration logic...
        return new_config
```

### Deprecation Timeline
- **Version 2.0**: Introduce new schema with compatibility layer
- **Version 2.1**: Deprecation warnings for old format
- **Version 3.0**: Remove old format support

## Success Criteria

1. **All existing strategies work after migration**
2. **Support for multiple embeddings per database**
3. **Support for multiple retrieval strategies per database**
4. **Clean separation between processing and storage**
5. **No performance degradation**
6. **Improved configuration clarity**

## Next Steps

1. Review and approve this plan
2. Update `schema.yaml` with new structure
3. Implement database and processing managers
4. Create migration scripts
5. Update CLI and documentation
6. Test with existing configurations
7. Roll out with backward compatibility

## Conclusion

This restructuring provides a cleaner, more intuitive architecture that separates data processing from storage and retrieval concerns. The single schema approach with two top-level sections (`databases` and `data_processing_strategies`) makes the system easier to understand and configure while enabling powerful features like hybrid search and multi-modal embeddings.