# RAG Pipeline Refactoring Plan
## Radical Transformation to Fit LlamaFarm CLI Architecture

### Executive Summary
This plan outlines the complete refactoring of the RAG pipeline to align with the LlamaFarm CLI's file-by-file ingestion pattern. We will eliminate directory processing in favor of blob-based iterative parser selection, where files are sent individually from the server.

### Implementation Status
**Last Updated:** September 12, 2025
**Status:** Phase 1-3 COMPLETED ✅ | Phase 4-5 PENDING ⏳

### Key Changes Implemented

#### 🗑️ Removed Components
- **Standalone RAG CLI** (`/rag/cli.py`) - DELETED
- **directory_config** - Removed from schema.yaml (lines 1891-1954)
- **DirectoryParser references** - Eliminated from handler.py

#### ✨ New Components Created
- **`/rag/core/blob_processor.py`** - Centralized blob processing with pattern matching
- **`/rag/core/ingest_handler.py`** - LlamaFarm CLI integration handler
- **Pattern-based routing** - Using `file_include_patterns` and `file_exclude_patterns`

#### 🔄 Updated Components
- **`/rag/core/strategies/handler.py`**
  - Removed `directory_config` from `create_processing_config()`
  - Added `get_parsers_config()` to return all parsers
  - Kept deprecated `get_parser_config()` for compatibility
  
- **`/rag/components/parsers/base/base_parser.py`**
  - Added `parse_blob()` method for blob processing
  
- **Configuration Files**
  - `/config/templates/default.yaml` - Updated to single `universal_processor` strategy
  - `/llamafarm.yaml` - Updated with pattern-based parser/extractor routing
  - Removed ALL `directory_config` sections

#### 🎯 Architecture Changes
- **FROM:** Directory-based batch processing → **TO:** Blob-based individual file processing
- **FROM:** MIME type matching → **TO:** Pattern-based matching (glob-style with fnmatch)
- **FROM:** Single parser per strategy → **TO:** Multiple parsers with priority-based fallback
- **FROM:** Parser-level pattern matching → **TO:** Centralized pattern matching in BlobProcessor

---

## PHASE 1: Understanding Current Architecture

### 1.1 CLI Ingest Pattern Analysis
**Current Flow:**
```
CLI -> Server API -> Individual File Upload -> Processing
```

**Key Insights from `cli/cmd/datasets.go`:**
- Files are uploaded one-by-one via multipart POST to `/v1/projects/{namespace}/{project}/datasets/{dataset}/data`
- Each file is sent as a blob with metadata (filename, extension)
- Server handles processing individually, not in batches
- Strategy selection happens at dataset creation, not file upload

### 1.2 Current RAG Strategy Handling
**Problems with Current Approach:**
- `directory_config` assumes batch processing of directories
- Parser selection based on MIME types and extensions at strategy level
- Multiple parsers defined but no iterative selection mechanism
- DirectoryParser is always active (line 241 in handler.py)

---

## PHASE 2: New Architecture Design

### 2.1 File Extension-Based Parser Matching
**New Flow:**
```python
def process_file_blob(file_data: bytes, metadata: dict, strategy: dict) -> Document:
    """
    Process a single file blob by MATCHING the right parser based on extension.
    
    CRITICAL: The system finds the parser(s) that match the file extension.
    If multiple parsers support the same extension, try them by priority.
    
    Args:
        file_data: Raw file bytes
        metadata: {
            'filename': 'document.pdf',
            'extension': '.pdf',
            'size': 12345,
            'upload_time': '2024-01-01T00:00:00Z'
        }
        strategy: Data processing strategy configuration
    """
    file_extension = metadata.get('extension', '').lower()
    
    # STEP 1: Find parsers that match this file extension
    matching_parsers = []
    for parser_config in strategy.get('parsers', []):
        supported_extensions = [ext.lower() for ext in parser_config.get('file_extensions', [])]
        if file_extension in supported_extensions:
            matching_parsers.append(parser_config)
    
    if not matching_parsers:
        logger.warning(f"No parser found for extension {file_extension}, using text fallback")
        return TextParser().parse(file_data, metadata)
    
    # STEP 2: Sort matching parsers by priority (highest first)
    matching_parsers = sorted(
        matching_parsers,
        key=lambda x: x.get('priority', 0),
        reverse=True
    )
    
    # STEP 3: Try each matching parser until one succeeds
    for parser_config in matching_parsers:
        try:
            logger.info(f"Processing {file_extension} with {parser_config['type']}")
            parser = create_parser(parser_config)
            documents = parser.parse(file_data, metadata)
            
            if documents:  # Successfully parsed
                logger.info(f"✅ Successfully parsed with {parser_config['type']}")
                return documents
        except Exception as e:
            if len(matching_parsers) > 1:
                logger.warning(f"Parser {parser_config['type']} failed: {e}, trying next...")
                continue
            else:
                logger.error(f"Parser {parser_config['type']} failed: {e}")
                raise
    
    # Fallback to text parser if all fail
    logger.warning("All matching parsers failed, using text fallback")
    return TextParser().parse(file_data, metadata)
```

**Example Multi-Format Strategy:**
```yaml
data_processing_strategies:
  - name: "universal_processor"
    description: "Handles PDFs, CSVs, Word docs, and more"
    parsers:
      # PDF Parsers
      - type: "PDFParser_LlamaIndex"
        file_extensions: [".pdf", ".PDF"]
        priority: 100  # Primary PDF parser
        config:
          chunk_strategy: "semantic"
      
      - type: "PDFParser_PyPDF2"
        file_extensions: [".pdf", ".PDF"]
        priority: 50  # Fallback PDF parser
        config:
          chunk_size: 1000
      
      # CSV Parsers
      - type: "CSVParser_Pandas"
        file_extensions: [".csv", ".CSV", ".tsv", ".TSV"]
        priority: 100  # Primary CSV parser
        config:
          chunk_size: 500
      
      - type: "CSVParser_Python"
        file_extensions: [".csv", ".CSV"]
        priority: 50  # Fallback CSV parser
        config:
          encoding: "utf-8"
      
      # Word Document Parsers
      - type: "DocxParser_LlamaIndex"
        file_extensions: [".docx", ".DOCX"]
        priority: 100
        config:
          extract_tables: true
      
      - type: "DocxParser_PythonDocx"
        file_extensions: [".docx", ".DOCX", ".doc", ".DOC"]
        priority: 50
        config:
          extract_metadata: true
      
      # Markdown Parser
      - type: "MarkdownParser_Python"
        file_extensions: [".md", ".markdown", ".mdown"]
        priority: 100
        config:
          extract_code_blocks: true
      
      # Excel Parser
      - type: "ExcelParser_Pandas"
        file_extensions: [".xlsx", ".XLSX", ".xls", ".XLS"]
        priority: 100
        config:
          sheets: null  # Process all sheets
      
      # Text Parser (catches .txt and unknown)
      - type: "TextParser_Python"
        file_extensions: [".txt", ".text", ".log", ".json", ".xml", ".yaml", ".yml"]
        priority: 100
        config:
          encoding: "utf-8"
```

**File Processing Examples:**
```
Scenario 1: Processing "report.pdf"
1. Extension detected: .pdf
2. Find matching parsers:
   - PDFParser_LlamaIndex (priority: 100)
   - PDFParser_PyPDF2 (priority: 50)
3. Try PDFParser_LlamaIndex → Success ✅

Scenario 2: Processing "data.csv"
1. Extension detected: .csv
2. Find matching parsers:
   - CSVParser_Pandas (priority: 100)
   - CSVParser_Python (priority: 50)
3. Try CSVParser_Pandas → Success ✅

Scenario 3: Processing "document.docx"
1. Extension detected: .docx
2. Find matching parsers:
   - DocxParser_LlamaIndex (priority: 100)
   - DocxParser_PythonDocx (priority: 50)
3. Try DocxParser_LlamaIndex → Fails
4. Try DocxParser_PythonDocx → Success ✅
```

### 2.2 Parser Interface Changes
**New Parser Base Class:**
```python
class BaseParser:
    def __init__(self, config: dict):
        self.config = config
        self.supported_extensions = config.get('file_extensions', [])
    
    def can_handle(self, extension: str) -> bool:
        """Check if parser can handle this file extension."""
        return extension.lower() in [ext.lower() for ext in self.supported_extensions]
    
    def parse(self, blob: bytes, metadata: dict) -> List[Document]:
        """Parse file blob into documents."""
        raise NotImplementedError
```

---

## PHASE 3: Implementation Steps

### 3.1 Remove directory_config (COMPLETE REMOVAL)

**Files to Modify:**
1. `/rag/schema.yaml` - Remove lines 1891-1954 (directory_config definition)
2. `/rag/core/strategies/handler.py` - Remove directory processing logic
3. `/config/templates/default.yaml` - Remove all directory_config sections

**Specific Changes:**

#### 3.1.1 Schema Changes (`/rag/schema.yaml`)
```yaml
# REMOVE THIS ENTIRE SECTION (lines 1891-1954):
directory_config:
  type: object
  title: Directory Processing Configuration
  # ... ALL OF THIS GETS DELETED
  
# KEEP parsers but modify structure:
parsers:
  type: array
  items:
    type: object
    properties:
      type:
        type: string
        enum: [...]
      config:
        type: object
      file_extensions:  # This becomes the primary selector
        type: array
        items:
          type: string
          pattern: '^\\.[a-zA-Z0-9]+$'
      priority:  # Order of parser attempts
        type: integer
        default: 0
```

#### 3.1.2 Handler Changes (`/rag/core/strategies/handler.py`)
```python
# REMOVE get_parser_config method (lines 234-249)
# REPLACE WITH:
def get_parsers_config(self, proc_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get parsers configuration from processing strategy."""
    return proc_config.get('parsers', [])
```

### 3.2 Implement Blob Processing with Centralized Pattern Matching

**CRITICAL: Pattern matching is CENTRALIZED in BlobProcessor, NOT in individual parsers**

**New File: `/rag/core/blob_processor.py`**
```python
"""Single file blob processor for LlamaFarm ingestion."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import io
import fnmatch

from ..components.parsers.parser_factory import ToolAwareParserFactory
from ..core.document_manager import Document

logger = logging.getLogger(__name__)

class BlobProcessor:
    """Process individual file blobs with CENTRALIZED pattern matching.
    
    Pattern matching is handled HERE, not in individual parsers.
    Each parser declares supported patterns, BlobProcessor does the matching.
    """
    
    def __init__(self, strategy_config: Dict[str, Any]):
        """Initialize with data processing strategy."""
        self.strategy = strategy_config
        self.parsers = self._initialize_parsers()
    
    def _initialize_parsers(self) -> Dict[str, List[Any]]:
        """Initialize parsers organized by file extension."""
        parser_configs = self.strategy.get('parsers', [])
        
        # Organize parsers by extension
        parsers_by_extension = {}
        for config in parser_configs:
            parser_type = config['type']
            extensions = config.get('file_extensions', [])
            priority = config.get('priority', 0)
            
            # Create parser instance
            parser = ToolAwareParserFactory.create_parser(
                parser_name=parser_type,
                config=config.get('config', {})
            )
            
            if parser:
                # Store parser for each extension it supports
                for ext in extensions:
                    ext_lower = ext.lower()
                    if ext_lower not in parsers_by_extension:
                        parsers_by_extension[ext_lower] = []
                    parsers_by_extension[ext_lower].append({
                        'parser': parser,
                        'priority': priority,
                        'type': parser_type
                    })
        
        # Sort parsers by priority for each extension
        for ext in parsers_by_extension:
            parsers_by_extension[ext] = sorted(
                parsers_by_extension[ext],
                key=lambda x: x['priority'],
                reverse=True
            )
        
        return parsers_by_extension
    
    def _matches_patterns(self, filename: str, patterns: List[str]) -> bool:
        """Check if filename matches any of the glob patterns.
        
        Centralized pattern matching using fnmatch.
        """
        for pattern in patterns:
            if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                return True
        return False
    
    def _excluded_by_patterns(self, filename: str, patterns: List[str]) -> bool:
        """Check if filename is excluded by any pattern."""
        for pattern in patterns:
            if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                return True
        return False
    
    def process(self, file_data: bytes, metadata: Dict[str, Any]) -> List[Document]:
        """
        Process a single file blob by matching the right parser.
        
        CENTRALIZED PATTERN MATCHING:
        1. Check file_include_patterns (e.g., "*.pdf", "report_*.pdf")
        2. Check file_exclude_patterns (e.g., "*_draft.pdf", "*.tmp")
        3. Match parsers and try in priority order
        
        Args:
            file_data: Raw file bytes
            metadata: File metadata including filename and extension
            
        Returns:
            List of Document objects
        """
        filename = metadata.get('filename', 'unknown')
        
        logger.info(f"Processing file: {filename}")
        
        # Find parsers that match this file
        matching_parsers = []
        for parser_config in self.strategy.get('parsers', []):
            # Check include patterns
            include_patterns = parser_config.get('file_include_patterns', [])
            exclude_patterns = parser_config.get('file_exclude_patterns', [])
            
            # Must match at least one include pattern
            if include_patterns and not self._matches_patterns(filename, include_patterns):
                continue
            
            # Must not match any exclude pattern
            if exclude_patterns and self._excluded_by_patterns(filename, exclude_patterns):
                continue
            
            # Parser matches!
            matching_parsers.append(parser_config)
        
        if not matching_parsers:
            logger.warning(f"No parser found for {filename}, using text fallback")
            return self._fallback_parse(file_data, metadata)
        
        # Try each matching parser in priority order
        for parser_info in matching_parsers:
            parser = parser_info['parser']
            parser_type = parser_info['type']
            
            try:
                logger.debug(f"Trying parser: {parser_type} (priority: {parser_info['priority']})")
                
                # Create file-like object from bytes
                file_obj = io.BytesIO(file_data)
                file_obj.name = filename
                
                # Parse the document
                documents = parser.parse(file_obj)
                
                if documents:
                    # Add metadata to documents
                    for doc in documents:
                        doc.metadata.update(metadata)
                        doc.metadata['parser'] = parser_type
                    
                    logger.info(f"✅ Successfully parsed with {parser_type}")
                    return documents
                    
            except Exception as e:
                if len(matching_parsers) > 1:
                    logger.warning(f"Parser {parser_type} failed: {e}, trying next...")
                    continue
                else:
                    logger.error(f"Parser {parser_type} failed: {e}")
                    # Still fallback to text if only parser fails
        
        # Fallback: try text parser
        logger.warning(f"All parsers failed for {filename}, using text fallback")
        return self._fallback_parse(file_data, metadata)
    
    def apply_extractors(self, documents: List[Document], filename: str) -> List[Document]:
        """Apply matching extractors to parsed documents.
        
        CENTRALIZED EXTRACTOR MATCHING:
        - Each extractor has file_include_patterns and file_exclude_patterns
        - BlobProcessor matches extractors to files, not individual extractors
        
        Args:
            documents: Parsed documents
            filename: Original filename for pattern matching
            
        Returns:
            Documents with extracted metadata
        """
        # Find extractors that match this file
        matching_extractors = []
        for extractor_config in self.strategy.get('extractors', []):
            # Check include patterns
            include_patterns = extractor_config.get('file_include_patterns', ['*'])
            exclude_patterns = extractor_config.get('file_exclude_patterns', [])
            
            # Must match at least one include pattern
            if not self._matches_patterns(filename, include_patterns):
                continue
            
            # Must not match any exclude pattern
            if exclude_patterns and self._excluded_by_patterns(filename, exclude_patterns):
                continue
            
            # Extractor matches!
            matching_extractors.append(extractor_config)
        
        # Sort by priority
        matching_extractors = sorted(
            matching_extractors,
            key=lambda x: x.get('priority', 0),
            reverse=True
        )
        
        # Apply each matching extractor
        for extractor_config in matching_extractors:
            try:
                extractor = create_extractor(extractor_config)
                for doc in documents:
                    extracted_metadata = extractor.extract(doc.content)
                    doc.metadata.update(extracted_metadata)
                logger.debug(f"Applied extractor: {extractor_config['type']}")
            except Exception as e:
                logger.warning(f"Extractor {extractor_config['type']} failed: {e}")
        
        return documents
    
    def _fallback_parse(self, file_data: bytes, metadata: Dict[str, Any]) -> List[Document]:
        """Fallback text parsing."""
        try:
            text = file_data.decode('utf-8', errors='ignore')
            doc = Document(
                content=text,
                metadata={
                    **metadata,
                    'parser': 'fallback_text'
                }
            )
            return [doc]
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
            return []
```

### 3.3 Centralized Pattern Matching Architecture

**CRITICAL: All pattern matching is CENTRALIZED in BlobProcessor**

The pattern matching system works as follows:

1. **Parsers and Extractors declare patterns** - They don't implement matching
2. **BlobProcessor performs all matching** - Single source of truth
3. **Patterns use glob-style syntax** - Standard fnmatch patterns

**Pattern Types:**
```yaml
# Parser/Extractor configuration
file_include_patterns: ["*.pdf", "report_*.pdf", "invoice*.pdf"]
file_exclude_patterns: ["*_draft.pdf", "*.tmp.pdf", "~$*.pdf"]
```

**Matching Logic (in BlobProcessor):**
```python
import fnmatch

def matches_patterns(filename: str, patterns: List[str]) -> bool:
    """Centralized pattern matching using fnmatch."""
    for pattern in patterns:
        if fnmatch.fnmatch(filename.lower(), pattern.lower()):
            return True
    return False
```

**Benefits:**
- **Single implementation** - Pattern matching logic in one place
- **Consistent behavior** - All components use same matching
- **Easy to maintain** - Update logic in BlobProcessor only
- **Flexible patterns** - Support wildcards, prefixes, suffixes
- **No parser changes** - Parsers just declare patterns

### 3.4 Update Config Templates

#### 3.3.1 Update `/config/templates/default.yaml`
```yaml
rag:
  databases:
    # ... existing database config ...
  
  data_processing_strategies:
    - name: "pdf_processing"
      description: "PDF document processing"
      # NO directory_config HERE!
      parsers:
        - type: "PDFParser_PyPDF2"
          file_extensions: [".pdf", ".PDF"]
          priority: 10
          config:
            chunk_size: 1000
            chunk_overlap: 150
        - type: "PDFParser_LlamaIndex"
          file_extensions: [".pdf", ".PDF"]
          priority: 5  # Fallback
          config:
            chunk_strategy: "semantic"
      extractors:
        # ... existing extractors ...
```

### 3.4 LlamaFarm CLI Integration (NO STANDALONE RAG CLI)

**CRITICAL: The RAG system should ONLY work through LlamaFarm CLI commands**
- Remove `/rag/cli.py` - NO standalone RAG CLI
- All RAG operations go through `lf datasets` commands
- Configuration overrides passed through LF CLI flags

**Remove Standalone RAG CLI:**
```bash
# DELETE THIS FILE:
rm /rag/cli.py

# All RAG operations now through LlamaFarm CLI:
lf datasets add ...
lf datasets ingest ...
lf rag query ...  # Future: query functionality
```

**New File: `/rag/core/ingest_handler.py`** (NOT in cli/ directory!)
```python
"""LlamaFarm ingestion handler for RAG system."""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .blob_processor import BlobProcessor
from .strategies.handler import SchemaHandler
from .factories import create_embedder, create_vector_store
from .document_manager import DocumentManager

logger = logging.getLogger(__name__)

class IngestHandler:
    """Handle file ingestion from LlamaFarm CLI.
    
    This is NOT a CLI itself - it's called BY the LlamaFarm CLI.
    """
    
    def __init__(self, config_path: str = "llamafarm.yaml"):
        """Initialize with global LlamaFarm config file."""
        self.schema_handler = SchemaHandler(config_path)
        self.processors = {}  # Cache processors by strategy
    
    def ingest_file(
        self,
        file_data: bytes,
        metadata: Dict[str, Any],
        data_processing_strategy: str,
        database: str
    ) -> Dict[str, Any]:
        """
        Ingest a single file.
        
        Args:
            file_data: Raw file bytes
            metadata: File metadata
            data_processing_strategy: Name of processing strategy
            database: Name of database
            
        Returns:
            Ingestion result with document IDs
        """
        # Get processor for strategy
        processor = self._get_processor(data_processing_strategy)
        
        # Process file into documents
        documents = processor.process(file_data, metadata)
        
        if not documents:
            return {
                'status': 'failed',
                'error': 'No documents extracted'
            }
        
        # Get database configuration
        db_config = self.schema_handler.get_database_config(database)
        if not db_config:
            raise ValueError(f"Database '{database}' not found")
        
        # Create embedder and store
        embedder = self._create_embedder(db_config)
        vector_store = self._create_vector_store(db_config)
        
        # Store documents
        doc_manager = DocumentManager()
        doc_ids = []
        
        for doc in documents:
            # Generate embeddings
            embedding = embedder.embed([doc.content])[0]
            doc.embedding = embedding
            
            # Store in vector database
            doc_id = vector_store.add_document(doc)
            doc_ids.append(doc_id)
        
        return {
            'status': 'success',
            'documents_processed': len(documents),
            'document_ids': doc_ids
        }
    
    def _get_processor(self, strategy_name: str) -> BlobProcessor:
        """Get or create processor for strategy."""
        if strategy_name not in self.processors:
            strategy_config = self.schema_handler.get_processing_strategy_config(strategy_name)
            if not strategy_config:
                raise ValueError(f"Strategy '{strategy_name}' not found")
            self.processors[strategy_name] = BlobProcessor(strategy_config)
        return self.processors[strategy_name]
    
    def _create_embedder(self, db_config: Dict[str, Any]):
        """Create embedder from database config."""
        embedder_config = self.schema_handler.get_embedder_config(db_config)
        return create_embedder(
            embedder_config['type'],
            embedder_config.get('config', {})
        )
    
    def _create_vector_store(self, db_config: Dict[str, Any]):
        """Create vector store from database config."""
        store_config = self.schema_handler.get_vector_store_config(db_config)
        return create_vector_store(
            store_config['type'],
            store_config.get('config', {})
        )
```

### 3.5 Default Handling and Override Mechanism

**Default Selection in Strategies:**
```python
# In handler.py - Enhanced default handling
class SchemaHandler:
    def get_default_embedder(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get default embedder with proper fallback chain."""
        # 1. Check explicit default_embedding_strategy field
        default_name = db_config.get("default_embedding_strategy")
        if default_name:
            for strategy in db_config.get("embedding_strategies", []):
                if strategy.get("name") == default_name:
                    return strategy
        
        # 2. Check for strategy with default=true flag
        for strategy in db_config.get("embedding_strategies", []):
            if strategy.get("default", False):
                return strategy
        
        # 3. Use first available strategy
        strategies = db_config.get("embedding_strategies", [])
        if strategies:
            return strategies[0]
        
        # 4. Ultimate fallback
        return {
            "type": "OllamaEmbedder",
            "config": {
                "model": "nomic-embed-text",
                "base_url": "http://localhost:11434"
            }
        }
```

**CLI Override Mechanism:**
```go
// In cli/cmd/datasets.go - Add override flags
var (
    // Override flags for ingestion
    overrideChunkSize    int
    overrideChunkOverlap int
    overrideParser       string
    overrideEmbedder     string
)

// In datasetsIngestCmd:
datasetsIngestCmd.Flags().IntVar(&overrideChunkSize, "chunk-size", 0, "Override chunk size")
datasetsIngestCmd.Flags().IntVar(&overrideChunkOverlap, "chunk-overlap", 0, "Override chunk overlap")
datasetsIngestCmd.Flags().StringVar(&overrideParser, "parser", "", "Force specific parser type")
datasetsIngestCmd.Flags().StringVar(&overrideEmbedder, "embedder", "", "Override embedder type")
```

**Pass Overrides to RAG System:**
```python
# In ingest_handler.py
def ingest_file(
    self,
    file_data: bytes,
    metadata: Dict[str, Any],
    data_processing_strategy: str,
    database: str,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Ingest with optional overrides from CLI.
    
    Args:
        overrides: {
            'parser': 'PDFParser_PyPDF2',  # Force specific parser
            'chunk_size': 2000,            # Override chunk size
            'chunk_overlap': 300,           # Override overlap
            'embedder': 'OpenAIEmbedder'   # Override embedder
        }
    """
    # Apply overrides to strategy config
    if overrides:
        strategy_config = self._apply_overrides(strategy_config, overrides)
```

---

## PHASE 4: Testing Plan

### 4.1 Test Commands

```bash
# 1. Initialize project (already done)
lf init

# 2. Create dataset with strategy
lf datasets add --data-processing-strategy pdf_processing --database main_database test-pdfs

# 3. Ingest PDF files one by one
lf datasets ingest test-pdfs rag/demos/static_samples/747/ryanair-737-700-800-fcom-rev-30.pdf

# 4. Test multiple formats
lf datasets add --data-processing-strategy multi_format_llamaindex --database main_database test-mixed
lf datasets ingest test-mixed rag/demos/static_samples/business_reports/quarterly_financial_report.xlsx
lf datasets ingest test-mixed rag/demos/static_samples/customer_support/support_tickets.csv
lf datasets ingest test-mixed rag/demos/static_samples/code_documentation/api_reference.md

# 5. Verify ingestion
lf datasets list
```

### 4.2 Validation Points

1. **Parser Selection**: Verify correct parser is chosen based on extension
2. **Fallback**: Test unknown file types fall back to text parser
3. **Error Handling**: Ensure graceful failure when parser fails
4. **Metadata**: Confirm metadata is preserved through pipeline
5. **Database Storage**: Verify documents are stored correctly

---

## PHASE 5: Implementation Checklist

### Completed Actions ✅:

1. [x] **DELETE Standalone RAG CLI**
   - DELETED `/rag/cli.py` completely ✅
   - All operations through LlamaFarm CLI only

2. [x] **DELETE directory_config from schema.yaml**
   - Lines 1891-1954 REMOVED ✅
   - Removed ALL references to directory processing
   - Schema updated to support pattern-based matching

3. [x] **DELETE directory_config from all templates**
   - `/config/templates/default.yaml` - UPDATED to new structure ✅
   - `/llamafarm.yaml` - UPDATED to new structure ✅
   - Removed ALL directory_config sections

4. [x] **CREATE blob_processor.py in /rag/core/**
   - CREATED BlobProcessor class ✅
   - Implemented CENTRALIZED pattern matching with fnmatch
   - Added ITERATIVE parser selection (try each parser until success)
   - Sorts parsers by priority
   - Pattern-based matching (file_include_patterns/file_exclude_patterns)

5. [x] **UPDATE handler.py**
   - UPDATED create_processing_config to remove directory_config ✅
   - Added get_parsers_config (plural) - returns list of parsers ✅
   - Kept deprecated get_parser_config for backward compatibility
   - Removed DirectoryParser references

6. [x] **CREATE ingest_handler.py in /rag/core/**
   - CREATED IngestHandler class ✅
   - Implements single file ingestion
   - Connects to blob processor
   - Handles embeddings and vector storage

7. [x] **UPDATE all parser configs in strategy definitions**
   - UPDATED to use file_include_patterns and file_exclude_patterns ✅
   - Added priority field to all parsers
   - Created single universal_processor strategy
   - Pattern matching centralized in BlobProcessor

8. [x] **UPDATE base parser for blob support**
   - Added parse_blob() method to BaseParser ✅
   - Default implementation converts blob to temp file

### Pending Actions ⏳:

9. [ ] **TEST with LlamaFarm CLI**
   - Use `lf datasets` commands ONLY
   - Test iterative parser selection
   - Verify defaults work properly
   - Test fallback mechanisms

10. [ ] **Update CLI Integration**
    - Update Go code to use new ingest_handler.py
    - Pass file blobs to RAG system
    - Handle responses from IngestHandler

11. [ ] **Add Error Recovery**
    - Implement retry logic
    - Better error messages
    - Logging improvements

12. [ ] **Documentation Updates**
    - Update README with new architecture
    - Document pattern matching system
    - Create migration guide

### Critical Requirements:

- **NO BACKWARD COMPATIBILITY** - Delete all deprecated code
- **NO DIRECTORY PROCESSING** - Everything is single-file blob
- **ITERATIVE PARSER SELECTION** - Try parsers in priority order
- **EXTENSION-BASED MATCHING** - Use file extensions, not MIME types
- **SERVER SENDS ONE FILE AT A TIME** - Design for this pattern

### Success Criteria:

1. `lf datasets ingest` works with individual files
2. Correct parser selected based on extension
3. No references to directory_config remain
4. Schema validates without deprecated fields
5. All demo files process successfully

---

## APPENDIX: Quick Reference

### Parser Matching and Selection Example

**Scenario: Processing mixed file types with one strategy**

```yaml
# Strategy definition with parsers for different file types
data_processing_strategies:
  - name: "document_processor"
    parsers:
      # PDF parsers
      - type: "PDFParser_LlamaIndex"
        file_extensions: [".pdf"]
        priority: 100
      - type: "PDFParser_PyPDF2"
        file_extensions: [".pdf"]
        priority: 50
      
      # CSV parsers
      - type: "CSVParser_Pandas"
        file_extensions: [".csv", ".tsv"]
        priority: 100
      
      # Word doc parsers
      - type: "DocxParser_LlamaIndex"
        file_extensions: [".docx", ".doc"]
        priority: 100
```

**Processing Flow Examples:**

```
Example 1: Processing "invoice.pdf"
1. Extension detected: .pdf
2. MATCH parsers for .pdf:
   - PDFParser_LlamaIndex (priority: 100) ✓
   - PDFParser_PyPDF2 (priority: 50) ✓
   - CSVParser_Pandas ✗ (doesn't support .pdf)
   - DocxParser_LlamaIndex ✗ (doesn't support .pdf)
3. Try PDFParser_LlamaIndex → Success ✅

Example 2: Processing "data.csv"
1. Extension detected: .csv
2. MATCH parsers for .csv:
   - CSVParser_Pandas (priority: 100) ✓
   - PDFParser_LlamaIndex ✗ (doesn't support .csv)
   - PDFParser_PyPDF2 ✗ (doesn't support .csv)
   - DocxParser_LlamaIndex ✗ (doesn't support .csv)
3. Try CSVParser_Pandas → Success ✅

Example 3: Processing "report.docx"
1. Extension detected: .docx
2. MATCH parsers for .docx:
   - DocxParser_LlamaIndex (priority: 100) ✓
   - Others ✗ (don't support .docx)
3. Try DocxParser_LlamaIndex → Success ✅

Example 4: Processing corrupted "damaged.pdf"
1. Extension detected: .pdf
2. MATCH parsers for .pdf:
   - PDFParser_LlamaIndex (priority: 100) ✓
   - PDFParser_PyPDF2 (priority: 50) ✓
3. Try PDFParser_LlamaIndex → Fails ❌
4. Try PDFParser_PyPDF2 → Success ✅
```

**Key Points:**
- MATCHING: System finds parsers by file extension
- NO FILTERING: Each file processed individually
- FALLBACK: If multiple parsers match, try by priority
- ONE STRATEGY: Handles all file types in one configuration

## APPENDIX: Quick Reference

### File Extension to Parser Mapping
```python
EXTENSION_PARSER_MAP = {
    '.pdf': ['PDFParser_PyPDF2', 'PDFParser_LlamaIndex'],
    '.csv': ['CSVParser_Pandas', 'CSVParser_LlamaIndex', 'CSVParser_Python'],
    '.xlsx': ['ExcelParser_OpenPyXL', 'ExcelParser_Pandas', 'ExcelParser_LlamaIndex'],
    '.docx': ['DocxParser_PythonDocx', 'DocxParser_LlamaIndex'],
    '.md': ['MarkdownParser_Python', 'MarkdownParser_LlamaIndex'],
    '.txt': ['TextParser_Python', 'TextParser_LlamaIndex'],
}
```

### Parser Priority Guidelines
- 10+: Primary/preferred parser
- 5-9: Secondary/fallback parser
- 0-4: Last resort parser

### Metadata Structure
```python
metadata = {
    'filename': 'document.pdf',
    'extension': '.pdf',
    'size': 12345,
    'upload_time': '2024-01-01T00:00:00Z',
    'dataset': 'test-pdfs',
    'namespace': 'default',
    'project': 'llamafarm-1'
}
```