# Claude.md - A Config-Based, Extensible Document Processing Pipeline

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Architecture](#core-architecture)
4. [Component Library](#component-library)
5. [Vector Database Integration](#vector-database-integration)
6. [Embedding Models Guide](#embedding-models-guide)
7. [Configuration System](#configuration-system)
8. [Domain-Specific Pipelines](#domain-specific-pipelines)
9. [Extension Guide](#extension-guide)
10. [Examples](#examples)

---

## Overview

This framework provides a modular, config-based approach to building RAG (Retrieval-Augmented Generation) pipelines. It's designed to be:

- **Framework-agnostic**: Works with or without LlamaIndex, Langchain, etc.
- **Extensible**: Easy to add custom components
- **Config-driven**: Define pipelines in YAML/JSON
- **Production-ready**: Includes monitoring, error handling, and optimization

### Key Features

- 🔧 **Modular component system**
- 🔄 **Hot-swappable pipeline components**
- 📊 **Multiple vector database support**
- 🎯 **Smart configuration recommendations**
- 🚀 **Domain-specific optimizations**
- 📈 **Performance monitoring**

### Philosophy

The framework follows these principles:
1. **Configuration over code**: Define behavior through config files
2. **Composition over inheritance**: Build complex pipelines from simple components
3. **Fail gracefully**: Continue processing even when individual components fail
4. **Optimize locally**: Each component can be optimized independently
5. **Monitor everything**: Built-in metrics and logging

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from pipeline import Pipeline, ComponentRegistry
from config import load_config

# Load configuration
config = load_config("configs/basic_rag.yaml")

# Build pipeline from config
pipeline = Pipeline.from_config(config)

# Process documents
results = pipeline.run("path/to/documents")
```

### Simple Configuration

```yaml
# configs/basic_rag.yaml
name: Basic RAG Pipeline
version: 1.0

components:
  - type: PDFParser
    config:
      extract_images: false
  
  - type: SentenceSplitter
    config:
      chunk_size: 512
      chunk_overlap: 50
  
  - type: EmbeddingGenerator
    config:
      model: all-MiniLM-L6-v2
      batch_size: 32
  
  - type: VectorIndexer
    config:
      database: chromadb
      collection: documents

monitoring:
  log_level: INFO
  metrics: true
```

---

## Core Architecture

### Component Model

Every component in the pipeline inherits from a base `Component` class:

```python
class Component(ABC):
    """Base class for all pipeline components"""
    
    def __init__(self, name: str = None, config: Dict = None):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = logging.getLogger(self.name)
    
    @abstractmethod
    def process(self, documents: List[Document]) -> ProcessingResult:
        """Process documents and return results"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate component configuration"""
        pass
```

### Component Types

1. **Parsers**: Extract content from various file formats
2. **Extractors**: Extract metadata and features from documents
3. **Transformers**: Modify or split documents
4. **Embedders**: Generate vector embeddings
5. **Indexers**: Store documents in vector databases
6. **Retrievers**: Search and retrieve relevant documents
7. **Rerankers**: Re-score retrieved results

### Document Model

```python
@dataclass
class Document:
    """Universal document representation"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    source: Optional[str] = None
    embeddings: Optional[List[float]] = None
    
    # Framework adapters
    def to_llamaindex(self): ...
    def to_langchain(self): ...
    def to_dict(self): ...
```

### Pipeline Architecture

```
Input → Parser → Transformer → Extractor → Embedder → Indexer → Output
                      ↓                         ↓
                  [Metadata]              [Vector Database]
```

### Error Handling

The pipeline uses a resilient error handling strategy:

```python
class ProcessingResult:
    documents: List[Document]
    errors: List[Error]
    metrics: Dict[str, Any]
```

Failed documents continue through the pipeline with error annotations.

---

## Component Library

### Parsers

#### PDF Parsers
- **PyMuPDFParser**: Fast, handles complex layouts
- **PDFMinerParser**: Better text extraction for scanned docs
- **CamelotParser**: Specialized for tables
- **OCRParser**: For scanned documents (uses Tesseract)

#### Document Parsers
- **DocxParser**: Microsoft Word documents
- **GoogleDocsParser**: Via Google Drive API
- **MarkdownParser**: Preserves structure
- **HTMLParser**: Web pages with metadata

#### Data Parsers
- **CSVParser**: Tabular data with schema detection
- **JSONParser**: Structured data with path extraction
- **XMLParser**: With XPath support

### Extractors

#### Text Extractors
- **KeywordExtractor**: YAKE, RAKE, or TF-IDF based
- **EntityExtractor**: Named entity recognition (spaCy)
- **SentimentExtractor**: TextBlob or VADER
- **SummaryExtractor**: LLM-based or extractive

#### Metadata Extractors
- **DateExtractor**: Temporal information
- **AuthorExtractor**: Document authorship
- **CitationExtractor**: Academic references
- **URLExtractor**: Links and references

#### Domain-Specific Extractors
- **MedicalEntityExtractor**: ICD codes, medications, conditions
- **PartNumberExtractor**: Aviation/manufacturing parts
- **PIIExtractor**: Personal information detection
- **CodeExtractor**: Functions, classes, imports

### Transformers

#### Text Transformers
- **SentenceSplitter**: Sentence-aware chunking
- **SemanticSplitter**: Topic-based splitting
- **WindowSplitter**: Overlapping windows
- **HierarchicalSplitter**: Parent-child chunks

#### Content Transformers
- **TextCleaner**: Remove noise, normalize
- **HTMLStripper**: Remove markup
- **CodeFormatter**: Language-specific formatting
- **TableTransformer**: Convert tables to text

#### Privacy Transformers
- **PIIRedactor**: Remove personal information
- **AnonymizerTransformer**: Replace with placeholders

### Embedders

#### Local Models
- **SentenceTransformerEmbedder**: Any HuggingFace model
- **CustomModelEmbedder**: Load your own models
- **MultiModalEmbedder**: Text + image embeddings

#### API-Based
- **OpenAIEmbedder**: text-embedding-3-*
- **CohereEmbedder**: Cohere embeddings
- **VoyageEmbedder**: Voyage AI embeddings

### Indexers

Support for all major vector databases with unified interface.

---

## Vector Database Integration

### Database Comparison Matrix

| Database | Type | Scale | Hybrid Search | Filtering | GPU Support | Ease of Use |
|----------|------|-------|---------------|-----------|-------------|-------------|
| ChromaDB | Local | <1M | ❌ | ✅ | ❌ | ⭐⭐⭐⭐⭐ |
| Qdrant | Local/Cloud | 1-10M | ✅ | ✅✅ | ✅ | ⭐⭐⭐⭐ |
| Weaviate | Local/Cloud | 1-10M | ✅ | ✅ | ❌ | ⭐⭐⭐ |
| Milvus | Distributed | >10M | ✅ | ✅ | ✅ | ⭐⭐ |
| FAISS | Library | Any | ❌ | ❌ | ✅ | ⭐⭐⭐ |
| LanceDB | Embedded | <10M | ✅ | ✅ | ❌ | ⭐⭐⭐⭐ |
| Pinecone | Cloud | Any | ❌ | ✅ | N/A | ⭐⭐⭐⭐⭐ |

### Configuration Examples

#### ChromaDB Configuration
```python
# config/vector_db/chromadb.py
CHROMADB_CONFIG = {
    "type": "chromadb",
    "connection": {
        "mode": "persistent",  # "memory", "persistent", "http"
        "path": "./chroma_db",
        # For HTTP mode
        "host": "localhost",
        "port": 8000,
        "ssl": False,
    },
    "collection": {
        "name": "documents",
        "metadata": {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 100,
            "hnsw:M": 16,
        }
    },
    "embedding": {
        "function": "sentence-transformers",
        "model": "all-MiniLM-L6-v2",
        "dimension": 384
    }
}
```

#### Qdrant Configuration
```python
# config/vector_db/qdrant.py
QDRANT_CONFIG = {
    "type": "qdrant",
    "connection": {
        "host": "localhost",
        "port": 6333,
        "grpc_port": 6334,
        "prefer_grpc": True,
        "api_key": "${QDRANT_API_KEY}",  # Environment variable
    },
    "collection": {
        "name": "documents",
        "vector_config": {
            "size": 768,
            "distance": "Cosine",
            "hnsw_config": {
                "m": 16,
                "ef_construct": 100,
                "full_scan_threshold": 10000,
            },
            "quantization_config": {
                "scalar": {
                    "type": "int8",
                    "quantile": 0.99,
                    "always_ram": True
                }
            }
        },
        "optimizers_config": {
            "deleted_threshold": 0.2,
            "vacuum_min_vector_number": 1000,
            "default_segment_number": 5,
        }
    },
    "search": {
        "always_ram": True,
        "indexing_threshold": 20000,
        "payload_indexing": {
            "fields": ["source", "date", "author"],
            "types": ["keyword", "datetime", "keyword"]
        }
    }
}
```

#### Milvus Configuration
```python
# config/vector_db/milvus.py
MILVUS_CONFIG = {
    "type": "milvus",
    "connection": {
        "host": "localhost",
        "port": 19530,
        "user": "root",
        "password": "${MILVUS_PASSWORD}",
        "pool_size": 10,
    },
    "collection": {
        "name": "documents",
        "schema": {
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "embedding", "type": "float_vector", "dim": 768},
                {"name": "content", "type": "varchar", "max_length": 65535},
                {"name": "metadata", "type": "json"}
            ],
            "enable_dynamic_field": True
        },
        "index": {
            "type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": 16,
                "efConstruction": 256
            }
        },
        "consistency_level": "Strong",
        "shards_num": 2,
        "replica_number": 1
    }
}
```

### Performance Optimization Settings

```python
# config/vector_db/optimization.py
OPTIMIZATION_PROFILES = {
    "speed": {
        "vector_db": "faiss",
        "index_type": "IVF",
        "quantization": "PQ8",
        "cache": "redis",
        "batch_size": 1000
    },
    "accuracy": {
        "vector_db": "qdrant",
        "index_type": "HNSW",
        "quantization": None,
        "reranking": True,
        "two_stage_retrieval": True
    },
    "scale": {
        "vector_db": "milvus",
        "sharding": True,
        "partitioning": "by_date",
        "distributed": True,
        "compression": "zstd"
    },
    "cost": {
        "vector_db": "chromadb",
        "index_type": "flat",
        "on_disk": True,
        "compression": "gzip"
    }
}
```

---

## Embedding Models Guide

### Model Selection Matrix

| Model | Dimensions | Size | Speed | Quality | Multi-lingual | Use Case |
|-------|------------|------|-------|---------|---------------|----------|
| all-MiniLM-L6-v2 | 384 | 80MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | ❌ | Prototypes, real-time |
| all-mpnet-base-v2 | 768 | 420MB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ❌ | General purpose |
| bge-small-en-v1.5 | 384 | 130MB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ❌ | Efficient production |
| bge-base-en-v1.5 | 768 | 440MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | ❌ | Balanced production |
| bge-large-en-v1.5 | 1024 | 1.3GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ❌ | High accuracy |
| e5-small-v2 | 384 | 130MB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ | Multi-lingual, fast |
| e5-base-v2 | 768 | 440MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ | Multi-lingual, balanced |
| e5-large-v2 | 1024 | 1.3GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ | Multi-lingual, best |
| instructor-xl | 768 | 5GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ | Task-specific |
| OpenAI-3-small | 1536 | API | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ | Easy integration |
| OpenAI-3-large | 3072 | API | ⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ | Highest quality |

### Embedding Configuration

```python
# config/embeddings.py
EMBEDDING_CONFIGS = {
    "development": {
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "normalize": True,
        "batch_size": 32,
        "device": "cpu",
        "cache": True
    },
    "production": {
        "model": "bge-base-en-v1.5",
        "dimension": 768,
        "normalize": True,
        "batch_size": 64,
        "device": "cuda",
        "precision": "fp16",
        "cache": True,
        "cache_backend": "redis"
    },
    "high_accuracy": {
        "model": "e5-large-v2",
        "dimension": 1024,
        "normalize": True,
        "batch_size": 32,
        "device": "cuda",
        "instruction": "query: ",  # For e5 models
        "max_length": 512
    },
    "multilingual": {
        "model": "e5-large-multilingual",
        "dimension": 1024,
        "normalize": True,
        "language_detection": True,
        "instruction_templates": {
            "en": "query: ",
            "zh": "查询：",
            "es": "consulta: "
        }
    },
    "hybrid": {
        "dense": {
            "model": "bge-base-en-v1.5",
            "dimension": 768
        },
        "sparse": {
            "model": "splade-v2",
            "vocab_size": 30522,
            "top_k": 100
        },
        "fusion": "reciprocal_rank",
        "alpha": 0.5  # 0=sparse only, 1=dense only
    }
}
```

### Optimization Strategies

```python
# config/embedding_optimization.py
OPTIMIZATION_STRATEGIES = {
    "quantization": {
        "enabled": True,
        "type": "int8",  # int8, int4, binary
        "calibration_samples": 1000
    },
    "caching": {
        "enabled": True,
        "backend": "redis",
        "ttl": 3600,
        "max_size": "10GB"
    },
    "batching": {
        "dynamic": True,
        "min_batch": 1,
        "max_batch": 128,
        "timeout_ms": 50
    },
    "gpu": {
        "multi_gpu": True,
        "mixed_precision": True,
        "compile": True,  # torch.compile
        "flash_attention": True
    },
    "pruning": {
        "enabled": False,
        "sparsity": 0.9,
        "structured": True
    }
}
```

---

## Configuration System

### Configuration Hierarchy

```
configs/
├── base.yaml              # Base configuration
├── environments/          # Environment-specific
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── pipelines/            # Pipeline definitions
│   ├── basic_rag.yaml
│   ├── medical_rag.yaml
│   └── code_analysis.yaml
├── components/           # Component configs
│   ├── parsers/
│   ├── extractors/
│   └── embedders/
└── databases/           # Database configs
    ├── chromadb.yaml
    ├── qdrant.yaml
    └── milvus.yaml
```

### Base Configuration

```yaml
# configs/base.yaml
version: "1.0"
project: "Document Processing Pipeline"

# Global settings
settings:
  log_level: INFO
  error_handling: continue  # continue, fail, retry
  max_retries: 3
  timeout_seconds: 300

# Default component settings
defaults:
  parser:
    type: AutoParser
    timeout: 60
  
  splitter:
    type: SentenceSplitter
    chunk_size: 512
    chunk_overlap: 50
  
  embedder:
    type: LocalEmbedder
    model: all-MiniLM-L6-v2
    batch_size: 32
  
  indexer:
    type: ChromaIndexer
    collection: documents

# Monitoring
monitoring:
  enabled: true
  backend: prometheus
  port: 9090
  
# Error handling
error_handling:
  max_errors_per_component: 10
  quarantine_failed_docs: true
  alert_threshold: 0.1  # 10% error rate
```

### Environment Configuration

```yaml
# configs/environments/production.yaml
extends: base.yaml

settings:
  log_level: WARNING
  error_handling: retry
  max_retries: 5

# Production overrides
embedder:
  model: bge-large-en-v1.5
  device: cuda
  batch_size: 128

indexer:
  type: QdrantIndexer
  host: qdrant.production.internal
  api_key: ${QDRANT_API_KEY}
  
monitoring:
  backend: datadog
  api_key: ${DATADOG_API_KEY}
  
scaling:
  auto_scale: true
  min_workers: 2
  max_workers: 10
  
security:
  encryption_at_rest: true
  tls_enabled: true
  audit_logging: true
```

### Pipeline Configuration

```yaml
# configs/pipelines/medical_rag.yaml
name: Medical Document Processing
version: "2.0"
extends: base.yaml

# Pipeline-specific settings
settings:
  compliance: HIPAA
  pii_handling: strict

# Component chain
components:
  - name: medical_parser
    type: MedicalPDFParser
    config:
      extract_images: true
      ocr_enabled: true
      
  - name: pii_detector
    type: MedicalPIIDetector
    config:
      entities: [SSN, MRN, DOB, NAME, ADDRESS]
      confidence_threshold: 0.8
      
  - name: pii_redactor
    type: PIIRedactor
    config:
      method: mask  # mask, hash, synthetic
      
  - name: medical_extractor
    type: MedicalEntityExtractor
    config:
      model: en_core_sci_md
      extract_codes: [ICD10, CPT, LOINC]
      
  - name: clinical_chunker
    type: ClinicalTextSplitter
    config:
      preserve_sections: true
      section_overlap: 100
      
  - name: medical_embedder
    type: BioBertEmbedder
    config:
      model: dmis-lab/biobert-v1.1
      pool_strategy: mean
      
  - name: secure_indexer
    type: SecureQdrantIndexer
    config:
      encryption: AES256
      access_control: true

# Validation rules
validation:
  required_metadata: [patient_id, document_type, date]
  phi_check: true
  
# Routing rules
routing:
  - condition: "doc.metadata.type == 'lab_result'"
    pipeline: lab_result_pipeline
  - condition: "doc.metadata.urgent == true"
    priority: high
```

### Dynamic Configuration

```python
# config/dynamic.py
class DynamicConfig:
    """Runtime configuration based on document characteristics"""
    
    @staticmethod
    def get_chunk_size(doc_length: int, doc_type: str) -> int:
        """Dynamically determine chunk size"""
        if doc_type == "legal":
            return min(256, doc_length // 10)
        elif doc_type == "technical":
            return min(1024, doc_length // 5)
        elif doc_length < 1000:
            return doc_length  # Don't chunk small docs
        else:
            return 512
    
    @staticmethod
    def select_embedding_model(
        accuracy_needed: float,
        latency_budget_ms: int,
        languages: List[str]
    ) -> str:
        """Select optimal embedding model"""
        
        if len(languages) > 1 or any(lang != "en" for lang in languages):
            # Need multilingual
            if latency_budget_ms < 50:
                return "e5-small-multilingual"
            elif accuracy_needed > 0.9:
                return "e5-large-multilingual"
            else:
                return "e5-base-multilingual"
        else:
            # English only
            if latency_budget_ms < 20:
                return "all-MiniLM-L6-v2"
            elif accuracy_needed > 0.95:
                return "bge-large-en-v1.5"
            else:
                return "bge-base-en-v1.5"
    
    @staticmethod
    def optimize_vector_db(
        num_documents: int,
        update_frequency: str,
        query_patterns: Dict
    ) -> Dict:
        """Optimize vector database configuration"""
        
        config = {}
        
        if num_documents < 100_000:
            config["database"] = "chromadb"
            config["index"] = "flat"
        elif num_documents < 10_000_000:
            config["database"] = "qdrant"
            config["index"] = "hnsw"
            config["quantization"] = "scalar" if num_documents > 1_000_000 else None
        else:
            config["database"] = "milvus"
            config["sharding"] = True
            config["partitions"] = max(2, num_documents // 10_000_000)
        
        # Optimize for update patterns
        if update_frequency == "realtime":
            config["wal_enabled"] = True
            config["async_index"] = True
        elif update_frequency == "batch":
            config["bulk_import"] = True
            config["index_after_import"] = True
            
        return config
```

---

## Domain-Specific Pipelines

### Medical Pipeline

```yaml
# configs/domains/medical.yaml
name: Medical Document Pipeline
domain: healthcare
compliance: [HIPAA, GDPR]

components:
  # Specialized medical components
  - type: DicomParser
    config:
      extract_metadata: true
      anonymize: true
      
  - type: HL7Parser
    config:
      version: "2.5"
      
  - type: ClinicalNoteParser
    config:
      formats: [CDA, FHIR]
      
  - type: MedicalAbbreviationExpander
    config:
      dictionary: medical_abbreviations.json
      
  - type: AnatomyExtractor
    config:
      ontology: SNOMED-CT
      
  - type: DrugInteractionChecker
    config:
      database: DrugBank
      
  - type: ClinicalTrialMatcher
    config:
      registry: ClinicalTrials.gov

quality_checks:
  - type: MedicalTermValidator
  - type: DosageConsistencyChecker
  - type: DateConsistencyChecker

output:
  format: FHIR
  deidentified: true
```

### Legal Pipeline

```yaml
# configs/domains/legal.yaml
name: Legal Document Pipeline
domain: legal
compliance: [GDPR, CCPA]

components:
  - type: LegalDocumentClassifier
    config:
      categories: [contract, brief, opinion, statute]
      
  - type: ClauseExtractor
    config:
      clause_types: [indemnification, limitation, termination]
      
  - type: CitationExtractor
    config:
      formats: [bluebook, APA, MLA]
      validate: true
      
  - type: JurisdictionDetector
    config:
      regions: [US, EU, UK]
      
  - type: LegalEntityExtractor
    config:
      types: [plaintiff, defendant, judge, court]
      
  - type: ContractAnalyzer
    config:
      extract_parties: true
      extract_dates: true
      extract_obligations: true

indexing:
  strategy: hierarchical
  preserve_structure: true
  enable_paragraph_search: true
```

### Financial Pipeline

```yaml
# configs/domains/financial.yaml
name: Financial Document Pipeline
domain: finance
compliance: [SOX, MiFID]

components:
  - type: FinancialStatementParser
    config:
      formats: [10-K, 10-Q, 8-K]
      extract_tables: true
      
  - type: XBRLParser
    config:
      taxonomy: US-GAAP
      
  - type: FinancialMetricExtractor
    config:
      metrics: [revenue, EBITDA, EPS, P/E]
      
  - type: CurrencyNormalizer
    config:
      target_currency: USD
      
  - type: RiskFactorExtractor
    config:
      categories: [market, credit, operational]
      
  - type: SentimentAnalyzer
    config:
      model: finbert
      aspects: [earnings, guidance, risk]

validation:
  - type: FinancialNumberValidator
  - type: DateConsistencyChecker
  - type: CalculationVerifier
```

### Code Analysis Pipeline

```yaml
# configs/domains/code_analysis.yaml
name: Code Analysis Pipeline
domain: software_engineering

components:
  - type: CodeParser
    config:
      languages: [python, javascript, java, go]
      parser: tree-sitter
      
  - type: ASTExtractor
    config:
      extract_functions: true
      extract_classes: true
      extract_imports: true
      
  - type: DocstringExtractor
    config:
      formats: [google, numpy, sphinx]
      
  - type: CodeComplexityAnalyzer
    config:
      metrics: [cyclomatic, cognitive, halstead]
      
  - type: DependencyExtractor
    config:
      include_versions: true
      
  - type: SecurityVulnerabilityScanner
    config:
      scanners: [bandit, semgrep]
      
  - type: CodeEmbedder
    config:
      model: codeBERT
      context_window: 512

chunking:
  strategy: ast_based
  preserve_context: true
  include_comments: true
```

### Customer Support Pipeline

```yaml
# configs/domains/customer_support.yaml
name: Customer Support Pipeline
domain: customer_service

components:
  - type: ConversationParser
    config:
      formats: [zendesk, intercom, slack]
      
  - type: ThreadReconstructor
    config:
      group_by: [ticket_id, conversation_id]
      
  - type: CustomerInfoExtractor
    config:
      fields: [name, email, account_id]
      mask_pii: true
      
  - type: IssueClassifier
    config:
      categories: [billing, technical, feature_request]
      multi_label: true
      
  - type: SentimentAnalyzer
    config:
      granularity: message  # message, conversation, customer
      
  - type: ResolutionExtractor
    config:
      extract_solution: true
      extract_time_to_resolve: true
      
  - type: AgentPerformanceAnalyzer
    config:
      metrics: [response_time, resolution_rate, csat]

routing:
  urgent_keywords: [urgent, critical, down, broken]
  escalation_threshold: 3  # messages
```

---

## Extension Guide

### Creating Custom Components

#### Basic Component Template

```python
# components/custom/my_extractor.py
from typing import Dict, List, Any
from core import Component, Document, ProcessingResult

class MyCustomExtractor(Component):
    """Extract custom information from documents"""
    
    def __init__(self, name: str = None, config: Dict = None):
        super().__init__(name, config)
        # Initialize your component
        self.important_terms = config.get("important_terms", [])
        
    def validate_config(self) -> bool:
        """Validate configuration"""
        if not isinstance(self.important_terms, list):
            raise ValueError("important_terms must be a list")
        return True
    
    def process(self, documents: List[Document]) -> ProcessingResult:
        """Process documents"""
        processed = []
        errors = []
        
        for doc in documents:
            try:
                # Your extraction logic
                extracted = self._extract(doc)
                doc.metadata.update(extracted)
                processed.append(doc)
            except Exception as e:
                errors.append({
                    "document_id": doc.id,
                    "error": str(e),
                    "component": self.name
                })
                processed.append(doc)  # Continue with original
        
        return ProcessingResult(
            documents=processed,
            errors=errors,
            metrics={"processed": len(processed), "errors": len(errors)}
        )
    
    def _extract(self, doc: Document) -> Dict[str, Any]:
        """Your extraction logic here"""
        found_terms = []
        for term in self.important_terms:
            if term.lower() in doc.content.lower():
                found_terms.append(term)
        
        return {
            "found_important_terms": found_terms,
            "term_count": len(found_terms)
        }
```

#### Async Component Template

```python
# components/custom/async_enricher.py
import asyncio
from typing import List
from core import AsyncComponent, Document

class AsyncEnricher(AsyncComponent):
    """Enrich documents with external data asynchronously"""
    
    async def process(self, documents: List[Document]) -> ProcessingResult:
        """Process documents asynchronously"""
        tasks = []
        for doc in documents:
            task = asyncio.create_task(self._enrich_document(doc))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed = []
        errors = []
        
        for doc, result in zip(documents, results):
            if isinstance(result, Exception):
                errors.append({
                    "document_id": doc.id,
                    "error": str(result)
                })
                processed.append(doc)
            else:
                doc.metadata.update(result)
                processed.append(doc)
        
        return ProcessingResult(documents=processed, errors=errors)
    
    async def _enrich_document(self, doc: Document) -> Dict:
        """Async enrichment logic"""
        # Example: Call external API
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.example.com/enrich/{doc.id}") as resp:
                return await resp.json()
```

### Plugin System

```python
# plugins/plugin_manager.py
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type

class PluginManager:
    """Manage and load plugins dynamically"""
    
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, Type[Component]] = {}
        
    def discover_plugins(self):
        """Discover all plugins in plugin directory"""
        for plugin_file in self.plugin_dir.glob("**/*.py"):
            if plugin_file.name.startswith("_"):
                continue
                
            module_path = str(plugin_file.relative_to(self.plugin_dir.parent))
            module_path = module_path.replace("/", ".").replace(".py", "")
            
            try:
                module = importlib.import_module(module_path)
                
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Component) and 
                        obj != Component):
                        self.register_plugin(name, obj)
                        
            except Exception as e:
                print(f"Failed to load plugin {module_path}: {e}")
    
    def register_plugin(self, name: str, plugin_class: Type[Component]):
        """Register a plugin"""
        self.plugins[name] = plugin_class
        print(f"Registered plugin: {name}")
    
    def get_plugin(self, name: str) -> Type[Component]:
        """Get a plugin by name"""
        if name not in self.plugins:
            raise ValueError(f"Plugin {name} not found")
        return self.plugins[name]
    
    def list_plugins(self) -> List[str]:
        """List all available plugins"""
        return list(self.plugins.keys())
```

### Custom Vector Database Adapter

```python
# adapters/custom_vector_db.py
from typing import List, Dict, Any
from core import VectorDBAdapter, Document

class CustomVectorDBAdapter(VectorDBAdapter):
    """Adapter for custom vector database"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Initialize your database connection
        
    def create_collection(self, name: str, dimension: int, **kwargs):
        """Create a new collection"""
        # Your implementation
        pass
    
    def insert(self, collection: str, documents: List[Document]):
        """Insert documents with embeddings"""
        # Your implementation
        pass
    
    def search(self, collection: str, query_vector: List[float], 
               top_k: int = 10, filters: Dict = None) -> List[Document]:
        """Search for similar documents"""
        # Your implementation
        pass
    
    def update(self, collection: str, document_id: str, 
               document: Document):
        """Update a document"""
        # Your implementation
        pass
    
    def delete(self, collection: str, document_ids: List[str]):
        """Delete documents"""
        # Your implementation
        pass
    
    def get_stats(self, collection: str) -> Dict[str, Any]:
        """Get collection statistics"""
        # Your implementation
        pass
```

### Monitoring Extension

```python
# extensions/monitoring.py
from typing import Dict, Any
import time
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PipelineMetrics:
    """Pipeline execution metrics"""
    start_time: datetime
    end_time: datetime
    total_documents: int
    successful_documents: int
    failed_documents: int
    component_metrics: Dict[str, Dict[str, Any]]
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return self.successful_documents / self.total_documents
    
    @property
    def throughput_per_second(self) -> float:
        if self.duration_seconds == 0:
            return 0.0
        return self.total_documents / self.duration_seconds

class MetricsCollector:
    """Collect and export pipeline metrics"""
    
    def __init__(self, export_backend: str = "prometheus"):
        self.backend = export_backend
        self.metrics: List[PipelineMetrics] = []
        
    def start_pipeline(self, pipeline_name: str) -> str:
        """Start tracking a pipeline execution"""
        execution_id = f"{pipeline_name}_{int(time.time())}"
        # Initialize tracking
        return execution_id
    
    def record_component_metric(self, execution_id: str, 
                              component_name: str, 
                              metric_name: str, 
                              value: Any):
        """Record a component-level metric"""
        # Your implementation
        pass
    
    def end_pipeline(self, execution_id: str, metrics: PipelineMetrics):
        """Finish tracking a pipeline execution"""
        self.metrics.append(metrics)
        self._export_metrics(metrics)
    
    def _export_metrics(self, metrics: PipelineMetrics):
        """Export metrics to configured backend"""
        if self.backend == "prometheus":
            self._export_to_prometheus(metrics)
        elif self.backend == "datadog":
            self._export_to_datadog(metrics)
        elif self.backend == "cloudwatch":
            self._export_to_cloudwatch(metrics)
            
    def _export_to_prometheus(self, metrics: PipelineMetrics):
        """Export to Prometheus"""
        # Implementation for Prometheus
        pass
```

---

## Examples

### Example 1: Simple Document Processing

```python
# examples/simple_processing.py
from pipeline import Pipeline
from components import PDFParser, SentenceSplitter, KeywordExtractor
from indexers import ChromaIndexer

# Build pipeline programmatically
pipeline = Pipeline("Simple Document Processing")

pipeline.add(PDFParser())
pipeline.add(SentenceSplitter(config={"chunk_size": 512}))
pipeline.add(KeywordExtractor(config={"method": "yake", "max_keywords": 10}))
pipeline.add(ChromaIndexer(config={"collection": "documents"}))

# Process documents
results = pipeline.run("./documents")

print(f"Processed {len(results.documents)} documents")
print(f"Errors: {len(results.errors)}")
```

### Example 2: Multi-Modal Pipeline

```python
# examples/multimodal_pipeline.py
from pipeline import Pipeline, load_config

# Load configuration
config = """
name: Multi-Modal Pipeline
components:
  - type: MultiModalParser
    config:
      extract_images: true
      extract_tables: true
      extract_text: true
      
  - type: ImageCaptionGenerator
    config:
      model: BLIP-2
      
  - type: TableStructureExtractor
    config:
      format: markdown
      
  - type: MultiModalEmbedder
    config:
      text_model: bge-base-en-v1.5
      image_model: CLIP-ViT-B-32
      
  - type: WeaviateIndexer
    config:
      enable_multi_modal: true
"""

pipeline = Pipeline.from_yaml(config)
results = pipeline.run("./mixed_documents")
```

### Example 3: Production Pipeline with Monitoring

```python
# examples/production_pipeline.py
import asyncio
from pipeline import AsyncPipeline
from monitoring import MetricsCollector, AlertManager

async def main():
    # Initialize monitoring
    metrics = MetricsCollector(export_backend="prometheus")
    alerts = AlertManager(webhook_url="https://alerts.example.com")
    
    # Load production configuration
    pipeline = await AsyncPipeline.from_config("configs/production.yaml")
    
    # Add monitoring
    pipeline.add_metrics_collector(metrics)
    pipeline.add_alert_manager(alerts)
    
    # Process with error handling
    try:
        results = await pipeline.run_async(
            source="s3://bucket/documents",
            parallel_workers=10,
            batch_size=100
        )
        
        # Check results
        if results.error_rate > 0.05:  # 5% error threshold
            await alerts.send_alert(
                level="warning",
                message=f"High error rate: {results.error_rate:.2%}"
            )
            
    except Exception as e:
        await alerts.send_alert(
            level="critical",
            message=f"Pipeline failed: {str(e)}"
        )
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 4: A/B Testing Configurations

```python
# examples/ab_testing.py
from pipeline import Pipeline, ABTestRunner
import numpy as np

# Configuration A: Speed optimized
config_a = {
    "name": "Speed Optimized",
    "embedder": {"model": "all-MiniLM-L6-v2", "dimension": 384},
    "chunker": {"size": 256, "overlap": 25},
    "indexer": {"type": "faiss", "index": "IVF"}
}

# Configuration B: Quality optimized
config_b = {
    "name": "Quality Optimized",
    "embedder": {"model": "bge-large-en-v1.5", "dimension": 1024},
    "chunker": {"size": 512, "overlap": 100},
    "indexer": {"type": "qdrant", "index": "HNSW"}
}

# Run A/B test
runner = ABTestRunner()
results = runner.compare(
    config_a=config_a,
    config_b=config_b,
    test_documents="./test_set",
    metrics=["latency", "precision@10", "recall@10", "cost"]
)

# Analyze results
print("A/B Test Results:")
print(f"Config A - Latency: {results.a.latency_p95}ms")
print(f"Config B - Latency: {results.b.latency_p95}ms")
print(f"Config A - Precision@10: {results.a.precision:.3f}")
print(f"Config B - Precision@10: {results.b.precision:.3f}")

# Statistical significance
if results.is_significant(metric="precision", confidence=0.95):
    winner = "A" if results.a.precision > results.b.precision else "B"
    print(f"Config {winner} is significantly better for precision")
```

### Example 5: Dynamic Pipeline Selection

```python
# examples/dynamic_pipeline.py
from pipeline import PipelineRouter, DocumentClassifier

# Initialize router with multiple pipelines
router = PipelineRouter()

# Register specialized pipelines
router.register("medical", Pipeline.from_config("configs/medical.yaml"))
router.register("legal", Pipeline.from_config("configs/legal.yaml"))
router.register("technical", Pipeline.from_config("configs/technical.yaml"))
router.register("general", Pipeline.from_config("configs/general.yaml"))

# Document classifier
classifier = DocumentClassifier(model="document-type-classifier")

# Process documents with automatic routing
async def process_documents(document_paths: List[str]):
    for doc_path in document_paths:
        # Classify document
        doc_type = classifier.classify(doc_path)
        
        # Route to appropriate pipeline
        pipeline = router.get_pipeline(doc_type)
        
        # Process with specialized pipeline
        result = await pipeline.run_async(doc_path)
        
        print(f"Processed {doc_path} with {doc_type} pipeline")

# Run processing
asyncio.run(process_documents(["doc1.pdf", "doc2.pdf", "doc3.pdf"]))
```

---

## Best Practices

### 1. Configuration Management
- Use environment variables for secrets
- Layer configurations (base → environment → pipeline)
- Version your configurations
- Validate configurations before deployment

### 2. Performance Optimization
- Profile your pipeline to identify bottlenecks
- Use appropriate batch sizes
- Enable caching where possible
- Consider GPU acceleration for embeddings

### 3. Error Handling
- Implement graceful degradation
- Log errors with context
- Set up alerts for critical failures
- Maintain error budgets

### 4. Monitoring
- Track key metrics (latency, throughput, accuracy)
- Set up dashboards
- Implement SLOs (Service Level Objectives)
- Regular performance reviews

### 5. Security
- Encrypt sensitive data at rest and in transit
- Implement access controls
- Audit logging
- Regular security scans

### 6. Testing
- Unit tests for each component
- Integration tests for pipelines
- Performance benchmarks
- A/B testing for optimizations

## Conclusion

This framework provides a flexible, extensible foundation for building document processing pipelines. By following the configuration-driven approach and leveraging the modular architecture, you can quickly build, test, and deploy sophisticated RAG systems tailored to your specific needs.

For questions, contributions, or support, please refer to our GitHub repository and documentation.


# Complete Guide to Embeddings and Vector Databases for RAG

## 1. Understanding Embeddings

### **Dense Embeddings**
Dense embeddings represent text as continuous vectors where most/all dimensions have non-zero values.

**Popular Dense Models:**

| Model | Dimensions | Size (MB) | Speed | Quality | Best For |
|-------|------------|-----------|--------|---------|----------|
| **all-MiniLM-L6-v2** | 384 | 80 | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Quick prototypes, real-time apps |
| **all-mpnet-base-v2** | 768 | 420 | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | General purpose, balanced |
| **bge-small-en-v1.5** | 384 | 130 | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Chinese/English, efficient |
| **bge-large-en-v1.5** | 1024 | 1340 | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy needs |
| **e5-large-v2** | 1024 | 1340 | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Multi-lingual, high quality |
| **instructor-xl** | 768 | 5000 | ⚡⚡ | ⭐⭐⭐⭐⭐ | Task-specific embeddings |
| **OpenAI text-embedding-3-small** | 1536 | API | ⚡⚡⚡ | ⭐⭐⭐⭐ | Cloud-based, easy |
| **OpenAI text-embedding-3-large** | 3072 | API | ⚡⚡ | ⭐⭐⭐⭐⭐ | Highest quality, cloud |

### **Sparse Embeddings**
Sparse embeddings represent text as high-dimensional vectors where most values are zero.

**Popular Sparse Models:**

| Model | Type | Dimensions | Best For |
|-------|------|------------|----------|
| **BM25** | Statistical | Variable | Keyword matching, exact terms |
| **TF-IDF** | Statistical | Vocab size | Document collections |
| **SPLADE** | Learned | ~30k | Neural + keyword benefits |
| **ColBERT** | Learned | 128 per token | Fine-grained matching |

### **Hybrid Approaches**
Combine dense and sparse for best of both worlds.

## 2. Dimensionality Trade-offs

| Dimensions | Storage | Speed | Quality | Use Case |
|------------|---------|--------|---------|----------|
| **128-256** | 0.5-1 KB/doc | ⚡⚡⚡⚡⚡ | ⭐⭐ | Mobile apps, edge devices |
| **384-512** | 1.5-2 KB/doc | ⚡⚡⚡⚡ | ⭐⭐⭐ | General web apps |
| **768-1024** | 3-4 KB/doc | ⚡⚡⚡ | ⭐⭐⭐⭐ | Enterprise search |
| **1536-3072** | 6-12 KB/doc | ⚡⚡ | ⭐⭐⭐⭐⭐ | Research, high accuracy |

### **Dimension Reduction Techniques**
- **PCA**: Linear reduction, fast but may lose information
- **UMAP**: Non-linear, better preservation of relationships
- **Matryoshka embeddings**: Models trained to be truncatable

## 3. Vector Database Comparison

### **Local/Open-Source Databases**

| Database | Type | Strengths | Weaknesses | Best For |
|----------|------|-----------|------------|----------|
| **ChromaDB** | Local/Cloud | Simple API, easy setup | Limited scaling | Prototypes, <1M vectors |
| **Qdrant** | Local/Cloud | Rust performance, filtering | Complex setup | Production, 1-10M vectors |
| **Weaviate** | Local/Cloud | Multi-modal, GraphQL | Resource heavy | Multi-modal data |
| **Milvus** | Distributed | Highly scalable | Complex ops | Large scale, >10M vectors |
| **FAISS** | Library | Fastest, flexible | No persistence | Research, benchmarking |
| **LanceDB** | Embedded | Serverless, simple | New, limited features | Edge deployments |
| **Pinecone** | Cloud | Fully managed | Vendor lock-in, cost | No-ops teams |

### **Database Feature Matrix**

| Feature | ChromaDB | Qdrant | Weaviate | Milvus | FAISS | LanceDB |
|---------|----------|---------|-----------|---------|--------|----------|
| **Hybrid Search** | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Filtering** | ✅ | ✅✅ | ✅ | ✅ | ❌ | ✅ |
| **Multi-tenancy** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **GPU Support** | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Memory Mapped** | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |

## 4. Chunking Strategies

### **Chunking Methods by Data Type**

| Data Type | Chunk Size | Overlap | Method | Rationale |
|-----------|------------|---------|---------|-----------|
| **Technical Docs** | 512-1024 | 100-200 | Semantic | Preserve context |
| **Legal Text** | 256-512 | 50-100 | Sentence | Precise references |
| **Conversations** | By turn | None | Speaker-aware | Natural boundaries |
| **Code** | Function/Class | None | AST-based | Logical units |
| **Research Papers** | 1024-2048 | 200 | Section-aware | Complete thoughts |
| **FAQ/Q&A** | Question+Answer | None | Pair-based | Atomic units |

### **Advanced Chunking Techniques**
1. **Semantic Chunking**: Break at topic changes
2. **Sliding Window**: Multiple overlapping chunks
3. **Hierarchical**: Parent-child relationships
4. **Dynamic**: Adjust size based on content

## 5. Decision Framework

### **Quick Decision Tree**

```
Start Here
    ↓
Is your data size > 10M documents?
    YES → Consider distributed (Milvus, Elasticsearch)
    NO ↓
    
Do you need keyword search too?
    YES → Hybrid approach (Qdrant, Weaviate)
    NO ↓
    
Is latency critical (<50ms)?
    YES → Small embeddings (384d) + FAISS/LanceDB
    NO ↓
    
Is accuracy most important?
    YES → Large embeddings (1024d+) + Qdrant
    NO → Balanced approach (768d + ChromaDB)
```

### **Configuration Recommendations by Use Case**

| Use Case | Embedding Model | Dimensions | Chunk Size | Vector DB | Rationale |
|----------|----------------|------------|------------|-----------|-----------|
| **Chatbot (General)** | all-mpnet-base-v2 | 768 | 512 | ChromaDB | Balanced performance |
| **Legal Research** | bge-large + BM25 | 1024 | 256 | Qdrant | Precision + keywords |
| **Code Search** | CodeBERT | 768 | Function-level | Weaviate | Code understanding |
| **Medical Records** | BioBERT + SPLADE | 768 | 256 | Qdrant | Domain-specific |
| **Customer Support** | e5-large-v2 | 1024 | Conversation | Milvus | Scale + quality |
| **Academic Papers** | SPECTER + BM25 | 768 | 1024 | Elasticsearch | Citations + search |
| **Product Search** | instructor-xl | 768 | Product desc | Pinecone | Task-specific |
| **Multi-lingual** | e5-large-v2 | 1024 | 512 | Qdrant | Language support |

### **Constraints-Based Selection**

| Constraint | Recommendation |
|------------|----------------|
| **Limited RAM (<4GB)** | 384d embeddings, LanceDB, chunk size 256 |
| **No GPU** | CPU-optimized models (MiniLM), FAISS-CPU |
| **High Throughput** | Batch processing, smaller models, caching |
| **Air-gapped** | Local models only, Qdrant/ChromaDB |
| **Real-time (<100ms)** | 384d, in-memory index, pre-compute |
| **Highest Accuracy** | 1536d+, re-ranking, hybrid search |

## 6. Optimization Strategies

### **Storage Optimization**
```python
# Quantization example
from qdrant_client import models

# Reduce memory 4x with scalar quantization
collection_config = models.Collection(
    vector_size=768,
    quantization_config=models.ScalarQuantization(
        type=models.ScalarType.INT8,
        quantile=0.99,
        always_ram=False
    )
)
```

### **Performance Optimization**
```python
# Batch processing
embeddings = model.encode(texts, batch_size=32, show_progress=True)

# Caching frequent queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text):
    return model.encode(text)
```

### **Quality Optimization**
```python
# Hybrid search example
def hybrid_search(query, k=10):
    # Dense search
    dense_results = vector_db.search(
        embedding=embed_model.encode(query),
        limit=k*2
    )
    
    # Sparse search (BM25)
    sparse_results = bm25_index.search(query, k=k*2)
    
    # Reciprocal Rank Fusion
    return fuse_results(dense_results, sparse_results, k=k)
```

## 7. Cost Analysis

### **Embedding Costs**

| Model | Cost per 1M tokens | Speed | Self-hosted |
|-------|-------------------|--------|-------------|
| OpenAI ada-002 | $0.10 | Fast | ❌ |
| OpenAI 3-small | $0.02 | Fast | ❌ |
| OpenAI 3-large | $0.13 | Medium | ❌ |
| Cohere embed-v3 | $0.10 | Fast | ❌ |
| Open source | $0 | Variable | ✅ |

### **Vector Database Costs (Monthly)**

| Database | 1M vectors | 10M vectors | 100M vectors |
|----------|------------|-------------|--------------|
| Pinecone | $70 | $700 | Custom |
| Qdrant Cloud | $25 | $250 | $2500 |
| Weaviate Cloud | $25 | $250 | Custom |
| Self-hosted | Infrastructure only | Infrastructure only | Infrastructure only |

## 8. Implementation Examples

### **Example 1: High-Performance Local RAG**
```python
# For speed-critical applications
config = {
    "embedding_model": "all-MiniLM-L6-v2",  # 384d, fast
    "chunk_size": 256,
    "chunk_overlap": 50,
    "vector_db": "lancedb",  # Embedded, fast
    "index_type": "IVF_FLAT",
    "quantization": "int8"
}
```

### **Example 2: High-Accuracy Enterprise RAG**
```python
# For accuracy-critical applications
config = {
    "embedding_model": "bge-large-en-v1.5",  # 1024d
    "chunk_size": 512,
    "chunk_overlap": 128,
    "vector_db": "qdrant",
    "hybrid_search": True,
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

### **Example 3: Scalable Multi-tenant RAG**
```python
# For SaaS applications
config = {
    "embedding_model": "e5-large-v2",  # Multi-lingual
    "chunk_size": 512,
    "chunk_overlap": 100,
    "vector_db": "milvus",
    "partitioning": "tenant_id",
    "caching": "redis"
}
```

## 9. Monitoring and Evaluation

### **Key Metrics to Track**

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Retrieval Precision** | >0.8 | Relevant docs in top-k |
| **Latency (P95)** | <200ms | End-to-end timing |
| **Embedding Time** | <50ms | Model inference only |
| **Index Time** | <10ms | Vector search only |
| **Memory Usage** | <75% | System monitoring |

### **A/B Testing Framework**
```python
def compare_configurations(config_a, config_b, test_queries):
    results = {
        "config_a": evaluate_rag(config_a, test_queries),
        "config_b": evaluate_rag(config_b, test_queries)
    }
    
    # Compare metrics
    for metric in ["precision", "recall", "latency", "cost"]:
        winner = "A" if results["config_a"][metric] > results["config_b"][metric] else "B"
        print(f"{metric}: Config {winner} wins")
```

## 10. Future Trends

### **Emerging Technologies**
1. **Matryoshka Embeddings**: Variable-dimension embeddings
2. **Late Interaction Models**: ColBERT-style architectures
3. **Learned Indices**: Neural network-based indexing
4. **Multi-modal Embeddings**: Text + image + code
5. **Streaming Embeddings**: Real-time updates

### **Best Practices Summary**
1. **Start simple**: MiniLM + ChromaDB for prototypes
2. **Measure everything**: A/B test configurations
3. **Hybrid when needed**: Dense + sparse for best results

---

## 11. Current Implementation Status (2024)

This section documents the **actual implemented features** of the RAG system as of the latest development cycle.

### **✅ Implemented Core Components**

#### **1. Universal Retrieval Strategies System**
- **Status**: ✅ **COMPLETE** - Fully implemented and tested
- **Location**: `retrieval/strategies/universal/`
- **Architecture**: Individual files for each strategy with auto-discovery

**Available Strategies:**
- `BasicSimilarityStrategy`: Simple vector similarity (fast, reliable baseline)
- `MetadataFilteredStrategy`: Intelligent metadata filtering with native/fallback support
- `MultiQueryStrategy`: Multiple query variations with configurable aggregation
- `RerankedStrategy`: Multi-factor re-ranking with recency, length, and metadata boosts
- `HybridUniversalStrategy`: Combines multiple strategies with weighted or rank fusion

**Key Features:**
- **Database Agnostic**: All strategies work with any vector database
- **Auto-Optimization**: Automatically uses database-specific optimizations when available
- **Plugin Architecture**: Easy addition of new strategies via individual files
- **Comprehensive Testing**: 40+ unit tests covering all functionality
- **Schema Validation**: JSON schemas for all strategy configurations

#### **2. Parser System**
- **Status**: ✅ **COMPLETE**
- **Parsers**: CSV (CustomerSupportCSVParser), PDF (basic text extraction)
- **Features**: Configurable field mapping, metadata extraction, flexible path resolution

#### **3. Embedding System** 
- **Status**: ✅ **COMPLETE**
- **Embedders**: OllamaEmbedder (local embeddings via Ollama)
- **Model**: nomic-embed-text (768 dimensions)
- **Features**: Batch processing, configurable timeout

#### **4. Vector Store System**
- **Status**: ✅ **COMPLETE** 
- **Stores**: ChromaStore with persistence
- **Features**: Collection management, metadata support, search functionality

#### **5. CLI System**
- **Status**: ✅ **COMPLETE**
- **Commands**: init, test, ingest, search, info
- **Features**: Progress bars with llama puns, flexible config paths, enhanced error handling

#### **6. API System**
- **Status**: ✅ **COMPLETE**
- **Features**: Internal SearchAPI, programmatic access, configurable parameters

#### **7. Configuration System**
- **Status**: ✅ **COMPLETE**
- **Format**: JSON-based with flexible path resolution
- **Examples**: 15+ configuration examples covering all use cases

### **🔄 Database Capability System**

**Implementation**: `stores/capabilities.py`
- **Purpose**: Maps database capabilities to optimize strategy selection
- **Coverage**: ChromaDB (complete), placeholders for Pinecone, Weaviate, etc.
- **Auto-Selection**: Registry automatically chooses optimal strategies

```python
DATABASE_CAPABILITIES = {
    "ChromaStore": {
        "supported": ["basic_similarity", "metadata_filtering", "batch_operations"],
        "distance_metrics": ["cosine", "euclidean", "manhattan"],
        "native_filtering": True
    }
}
```

### **📊 Test Coverage**
- **Unit Tests**: 40+ comprehensive tests for retrieval strategies
- **Integration Tests**: End-to-end workflow validation
- **Coverage**: Core retrieval system has >90% test coverage

### **📚 Documentation Status**
- **README.md**: ✅ Updated with universal strategies guide
- **retrieval/README.md**: ✅ Comprehensive strategy documentation  
- **Config Examples**: ✅ 15+ examples covering all strategies
- **Code Documentation**: ✅ Docstrings and type hints throughout

---

## 12. Current Limitations & Known Issues

### **⚠️ Areas Needing Development**

1. **Vector Database Support**
   - **Current**: ChromaDB only
   - **Needed**: Pinecone, Weaviate, Qdrant implementations
   - **Priority**: Medium (architecture supports it)

2. **Advanced Embedding Models**
   - **Current**: Ollama integration only
   - **Needed**: OpenAI, Cohere, local transformers
   - **Priority**: Medium

3. **Query Expansion** 
   - **Current**: Basic multi-query without actual expansion
   - **Needed**: LLM-based query variation generation
   - **Priority**: Low (strategy framework ready)

4. **Production Monitoring**
   - **Current**: Basic logging
   - **Needed**: Metrics, tracing, performance monitoring
   - **Priority**: High for production use

5. **Incremental Updates**
   - **Current**: Full re-ingestion required
   - **Needed**: Document update/delete capabilities
   - **Priority**: Medium

---

## 13. Future Development Roadmap

### **Phase 1: Production Readiness (Q1 2025)**
- [ ] Performance monitoring and metrics
- [ ] Incremental document updates
- [ ] Enhanced error handling and recovery
- [ ] Docker containerization
- [ ] CI/CD pipeline improvements

### **Phase 2: Advanced Retrieval (Q2 2025)**
- [ ] LLM-based query expansion for MultiQueryStrategy
- [ ] Learned sparse retrievers (SPLADE-style)
- [ ] Cross-encoder reranking models
- [ ] Question-answer pair mining from documents

### **Phase 3: Multi-Modal & Scale (Q3 2025)**
- [ ] Image and document visual understanding
- [ ] Multi-language support with language-specific strategies
- [ ] Distributed processing for large corpora
- [ ] Advanced chunking strategies (semantic, sliding window)

### **Phase 4: Enterprise Features (Q4 2025)**
- [ ] Multi-tenant support with isolation
- [ ] Role-based access control
- [ ] Audit logging and compliance features
- [ ] Enterprise database integrations

### **Phase 5: AI-Powered Optimization (2026)**
- [ ] Auto-tuning of retrieval parameters
- [ ] LLM-in-the-loop strategy selection
- [ ] Learned query understanding and routing
- [ ] Automated evaluation and A/B testing

---

## 14. Contributing to the Current Implementation

### **Easy Contribution Areas**

1. **Add New Universal Strategy**
   - Create file in `retrieval/strategies/universal/`
   - Follow existing pattern with comprehensive docstrings
   - Add to `__init__.py` metadata
   - Include unit tests

2. **Add Vector Database Support**
   - Implement VectorStore interface
   - Add capabilities to `stores/capabilities.py`
   - Create database-specific optimized strategies if needed

3. **Enhance Existing Strategies**
   - Improve reranking factors in RerankedStrategy
   - Add new aggregation methods to MultiQueryStrategy
   - Extend metadata filtering operators

4. **Configuration Examples**
   - Add domain-specific configuration examples
   - Document best practices for different use cases

### **Current Architecture Strengths**
- **Modular**: Clean separation of concerns
- **Testable**: Comprehensive test coverage
- **Extensible**: Plugin-style architecture
- **Universal**: Database-agnostic strategies
- **Production-Ready**: Error handling, logging, configuration validation

The current implementation provides a solid foundation for building production RAG systems while maintaining the flexibility to extend and customize for specific use cases.