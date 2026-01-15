# LlamaIndex Patterns for RAG

Best practices for document parsing, chunking, and ingestion using LlamaIndex.

## Architecture Overview

```
File Input -> Parser (LlamaIndex Reader) -> Chunking (Node Parser) -> Document Objects
```

## Key Classes

### LlamaIndexParser Base

Location: `rag/components/parsers/base/llama_parser.py`

```python
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
)

class LlamaIndexParser(BaseParser):
    def __init__(self, config: dict[str, Any] = None):
        super().__init__(config)
        self.reader = None  # Set by subclass
        self.text_splitter = self._create_text_splitter()

    def _create_text_splitter(self):
        chunk_size = self.config.get("chunk_size", None)
        if chunk_size is None:
            return None  # No chunking

        chunk_overlap = self.config.get("chunk_overlap", 0)
        chunk_strategy = self.config.get("chunk_strategy", "characters")

        if chunk_strategy == "sentences":
            return SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif chunk_strategy == "paragraphs":
            return MarkdownNodeParser()
        else:  # characters
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
```

### Document Conversion

```python
def _llama_to_rag_documents(self, llama_docs: list) -> list[Document]:
    rag_docs = []
    for llama_doc in llama_docs:
        content = llama_doc.text if hasattr(llama_doc, "text") else str(llama_doc)
        metadata = llama_doc.metadata if hasattr(llama_doc, "metadata") else {}

        # Add parser metadata
        metadata["parser_type"] = self.__class__.__name__

        doc = self.create_document(
            content=content,
            metadata=metadata,
            doc_id=llama_doc.id_ if hasattr(llama_doc, "id_") else None,
            source=metadata.get("file_path"),
        )
        rag_docs.append(doc)
    return rag_docs
```

## Chunking Strategies

| Strategy | Use Case | Node Parser |
|----------|----------|-------------|
| `characters` | General text, code | `TokenTextSplitter` |
| `sentences` | Natural language | `SentenceSplitter` |
| `paragraphs` | Markdown, structured | `MarkdownNodeParser` |
| `semantic` | High-quality retrieval | `SemanticSplitterNodeParser` |

**Note**: Semantic chunking requires an embedder and falls back to sentence chunking if embedding fails.

## Code Review Checklist

### 1. Chunking Configuration

**Description**: Verify chunking is properly configured with appropriate size and overlap.

**Search Pattern**:
```bash
grep -rn "chunk_size\|chunk_overlap\|SentenceSplitter\|TokenTextSplitter" rag/
```

**Pass Criteria**:
- chunk_size is specified (typically 512-2048)
- chunk_overlap is 10-20% of chunk_size
- Appropriate splitter for content type

**Fail Criteria**:
- chunk_size=0 or None without intentional full-document mode
- chunk_overlap >= chunk_size
- Using TokenTextSplitter for natural language

**Severity**: Medium

**Recommendation**: Use SentenceSplitter for natural language with chunk_overlap at least 50 tokens.

---

### 2. Metadata Preservation

**Description**: Ensure metadata flows from LlamaIndex documents to RAG documents.

**Search Pattern**:
```bash
grep -rn "_llama_to_rag_documents\|llama_doc.metadata" rag/components/parsers/
```

**Pass Criteria**:
- Metadata is copied from LlamaIndex document
- Parser type is added to metadata
- Source path is preserved

**Fail Criteria**:
- Metadata is discarded or overwritten
- No parser_type tracking
- Source lost during conversion

**Severity**: Medium

**Recommendation**: Always copy and extend metadata, never replace.

---

### 3. Chunk Metadata

**Description**: Each chunk should have metadata linking it to the parent document.

**Search Pattern**:
```bash
grep -rn "chunk_num\|chunk_index\|total_chunks" rag/
```

**Pass Criteria**:
- chunk_index (0-based) is set
- total_chunks is set
- Original document ID is preserved

**Fail Criteria**:
- No chunk indexing
- Missing total_chunks
- Lost parent reference

**Severity**: Medium

**Recommendation**: Include chunk_index, total_chunks, and parent_doc_id in chunk metadata.

---

### 4. LlamaIndex Import Guards

**Description**: LlamaIndex imports should be guarded for optional dependency.

**Search Pattern**:
```bash
grep -rn "LLAMA_INDEX_AVAILABLE\|ImportError.*llama" rag/
```

**Pass Criteria**:
- LlamaIndex imports wrapped in try/except
- LLAMA_INDEX_AVAILABLE flag checked before use
- Clear error message if not available

**Fail Criteria**:
- Bare imports without guards
- No fallback behavior
- Unclear import errors

**Severity**: Low

**Recommendation**: Use lazy imports with availability flags.

---

### 5. Parser Error Handling

**Description**: Parser errors should be captured and reported, not silently ignored.

**Search Pattern**:
```bash
grep -rn "ProcessingResult.*errors\|errors.append" rag/components/parsers/
```

**Pass Criteria**:
- Errors captured in ProcessingResult.errors
- Error includes source file path
- Error includes parser name

**Fail Criteria**:
- Exceptions swallowed silently
- Errors not included in result
- Missing context in error

**Severity**: High

**Recommendation**: Always return errors in ProcessingResult with full context.

---

### 6. File Type Detection

**Description**: Parsers should properly detect supported file types.

**Search Pattern**:
```bash
grep -rn "can_parse\|supported_extensions\|mime_types" rag/components/parsers/
```

**Pass Criteria**:
- can_parse() checks extension and/or mime type
- Supported extensions defined in metadata
- Magic library used for content-based detection

**Fail Criteria**:
- Only extension-based detection
- Hardcoded type checking
- No mime type support

**Severity**: Low

**Recommendation**: Use python-magic for content-based detection with extension fallback.

## Anti-Patterns

### Ignoring Chunking Errors

```python
# BAD: Silent failure
try:
    chunks = self.text_splitter.split([llama_doc])
except:
    pass  # Chunks lost!

# GOOD: Fallback with logging
try:
    chunks = self.text_splitter.split([llama_doc])
except Exception as e:
    logger.warning(f"Chunking failed: {e}, using original document")
    chunked_docs.append(doc)
```

### Losing Source Information

```python
# BAD: No source tracking
doc = Document(content=text, metadata={})

# GOOD: Preserve source
doc = Document(
    content=text,
    metadata={"file_path": source_path, "parser": self.__class__.__name__},
    source=source_path,
)
```

### Hardcoded Chunk Sizes

```python
# BAD: Hardcoded values
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# GOOD: Configurable
splitter = SentenceSplitter(
    chunk_size=self.config.get("chunk_size", 512),
    chunk_overlap=self.config.get("chunk_overlap", 50),
)
```
