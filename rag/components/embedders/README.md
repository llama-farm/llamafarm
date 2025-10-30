# RAG Embedders

This directory contains embedder components for the LlamaFarm RAG system. Embedders convert text into vector representations for semantic search and retrieval.

## Available Embedders

### 1. Universal Embedder (Recommended)

**Location**: `universal_embedder/`

Uses the Universal Runtime to generate embeddings with any HuggingFace model.

**Features**:
- ✅ Any HuggingFace embedding model
- ✅ Hardware acceleration (MPS/CUDA/CPU)
- ✅ ONNX optimization (3x faster)
- ✅ Fully local, no external API calls
- ✅ OpenAI-compatible API
- ✅ Efficient batch processing

**Best for**: Production RAG systems, local deployment, maximum flexibility

**Example**:
```yaml
embedder:
  type: UniversalEmbedder
  config:
    model: sentence-transformers/all-MiniLM-L6-v2
    base_url: http://127.0.0.1:11540/v1
    batch_size: 32
```

[Full Documentation](./universal_embedder/universal_embedder.md)

---

### 2. Ollama Embedder

**Location**: `ollama_embedder/`

Uses Ollama for local embedding generation.

**Features**:
- ✅ Easy setup with Ollama
- ✅ Local inference
- ✅ Limited model selection

**Best for**: Quick local development with Ollama already installed

**Example**:
```yaml
embedder:
  type: OllamaEmbedder
  config:
    model: nomic-embed-text
    base_url: http://localhost:11434
```

[Documentation](./ollama_embedder/ollama_embedder.md)

---

### 3. OpenAI Embedder

**Location**: `openai_embedder/`

Uses OpenAI's embedding API.

**Features**:
- ✅ High quality embeddings
- ✅ No local setup required
- ❌ Requires API key and costs money
- ❌ Cloud-based (privacy concerns)

**Best for**: Cloud deployments where API costs are acceptable

**Example**:
```yaml
embedder:
  type: OpenAIEmbedder
  config:
    model: text-embedding-3-small
    api_key: sk-...
```

[Documentation](./openai_embedder/openai_embedder.md)

---

### 4. HuggingFace Embedder

**Location**: `huggingface_embedder/`

Direct HuggingFace Transformers integration.

**Features**:
- ✅ Direct model access
- ✅ No separate server needed
- ❌ In-process (memory overhead)
- ❌ Requires restart for updates

**Best for**: Simple scripts, single-process applications

**Example**:
```yaml
embedder:
  type: HuggingFaceEmbedder
  config:
    model: sentence-transformers/all-MiniLM-L6-v2
```

[Documentation](./huggingface_embedder/huggingface_embedder.md)

---

### 5. Sentence Transformer Embedder

**Location**: `sentence_transformer_embedder/`

Specialized for sentence-transformers library.

**Features**:
- ✅ Optimized for sentence embeddings
- ✅ Good performance
- ❌ In-process
- ❌ Limited to sentence-transformers models

**Best for**: Applications specifically using sentence-transformers

**Example**:
```yaml
embedder:
  type: SentenceTransformerEmbedder
  config:
    model: all-MiniLM-L6-v2
```

[Documentation](./sentence_transformer_embedder/sentence_transformer_embedder.md)

---

## Comparison

| Feature | Universal | Ollama | OpenAI | HuggingFace | SentenceTransformer |
|---------|-----------|--------|--------|-------------|---------------------|
| **Model Selection** | Any HF model | Limited | Limited | Any HF model | ST models only |
| **Hardware Accel** | MPS/CUDA/CPU | CPU | N/A | MPS/CUDA/CPU | CUDA/CPU |
| **ONNX Support** | ✅ (3x faster) | ❌ | N/A | ❌ | ❌ |
| **Local** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Privacy** | Fully private | Fully private | Cloud | Fully private | Fully private |
| **Cost** | Free | Free | $$ | Free | Free |
| **Setup** | Start server | Install Ollama | API key | pip install | pip install |
| **Isolation** | Separate process | Separate process | External | In-process | In-process |
| **Scalability** | Horizontal | Vertical | Unlimited | Vertical | Vertical |
| **Production Ready** | ✅ | ⚠️ | ✅ | ⚠️ | ⚠️ |

## Recommendations

### For Development
**Use Universal Embedder** with a fast model:
```yaml
embedder:
  type: UniversalEmbedder
  config:
    model: sentence-transformers/all-MiniLM-L6-v2  # 384 dims, very fast
    batch_size: 32
```

### For Production
**Use Universal Embedder** with a high-quality model:
```yaml
embedder:
  type: UniversalEmbedder
  config:
    model: BAAI/bge-base-en-v1.5  # 768 dims, excellent quality
    batch_size: 16
    timeout: 120
```

Enable ONNX for 3x speedup:
```bash
export RUNTIME_BACKEND=onnx
export ONNX_PROVIDER=CUDAExecutionProvider
```

### For Cloud Deployments
If local inference is not feasible, **use OpenAI Embedder**:
```yaml
embedder:
  type: OpenAIEmbedder
  config:
    model: text-embedding-3-small
    api_key: ${OPENAI_API_KEY}
```

### If You Already Use Ollama
Stick with **Ollama Embedder** for simplicity:
```yaml
embedder:
  type: OllamaEmbedder
  config:
    model: nomic-embed-text
```

## Performance Benchmarks

Tested on M2 Mac with 16GB RAM:

### Embedding Speed (docs/second)

| Embedder | Model | Speed | Dims | Quality |
|----------|-------|-------|------|---------|
| Universal (PyTorch) | all-MiniLM-L6-v2 | ~50 | 384 | Good |
| Universal (ONNX) | all-MiniLM-L6-v2 | ~150 | 384 | Good |
| Universal (PyTorch) | bge-base-en | ~30 | 768 | High |
| Universal (ONNX) | bge-base-en | ~90 | 768 | High |
| Ollama | nomic-embed-text | ~25 | 768 | Good |
| HuggingFace | all-MiniLM-L6-v2 | ~40 | 384 | Good |
| SentenceTransformer | all-MiniLM-L6-v2 | ~45 | 384 | Good |

**Note**: ONNX optimization provides 3x speedup for Universal Embedder.

## Getting Started

### 1. Choose an Embedder

For most use cases, we recommend **Universal Embedder**:

```bash
# Start Universal Runtime
cd runtimes/universal
uv run uvicorn server:app --port 11540 --reload
```

### 2. Configure Your Project

```yaml
version: v1
name: my-project
namespace: default

rag:
  databases:
    - id: main_db
      type: vector
      provider: chromadb
      embedder:
        type: UniversalEmbedder
        config:
          model: sentence-transformers/all-MiniLM-L6-v2
```

### 3. Start Using RAG

```bash
lf datasets create -s pdf_ingest -b main_db docs
lf datasets upload docs ./files/*.pdf
lf datasets process docs
lf rag query --database main_db "What are the key findings?"
```

## Creating Custom Embedders

To create a custom embedder:

1. Create a new directory: `my_embedder/`
2. Implement the `Embedder` base class from `core.base`
3. Required methods:
   - `embed(texts: List[str]) -> List[List[float]]`
   - `get_embedding_dimension() -> int`
   - `validate_config() -> bool`
4. Register in `core/factories.py`

Example structure:
```
my_embedder/
  __init__.py          # Export your embedder class
  my_embedder.py       # Implementation
  schema.yaml          # Configuration schema
  defaults.yaml        # Default configurations
  my_embedder.md       # Documentation
```

See `universal_embedder/` for a complete reference implementation.

## Resources

- [RAG Documentation](../../../docs/website/docs/rag/index.md)
- [Universal Runtime Guide](../../../runtimes/universal/README.md)
- [Configuration Guide](../../../docs/website/docs/configuration/index.md)

## Support

For issues or questions:
1. Check embedder-specific documentation
2. Verify server/runtime is running
3. Check logs for detailed errors
4. Review configuration schema

