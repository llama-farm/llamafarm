# Universal Embedder

Generate text embeddings using the Universal Runtime, which provides access to any HuggingFace embedding model with hardware acceleration.

## Overview

The Universal Embedder connects to the Universal Runtime server to generate embeddings for RAG systems. It supports any HuggingFace embedding model and provides:

- **Universal Model Support**: Use any HuggingFace sentence-transformers or embedding model
- **Hardware Acceleration**: Automatic MPS/CUDA/CPU detection
- **OpenAI Compatibility**: Standard OpenAI embeddings API
- **Batch Processing**: Efficient batch embedding generation
- **Local Inference**: No external API calls, full privacy
- **ONNX Optimization**: Optional 3x speedup with ONNX backend

## Prerequisites

The Universal Runtime server must be running:

```bash
cd runtimes/universal
uv run uvicorn server:app --port 11540 --reload
```

Verify the server is available:

```bash
curl http://127.0.0.1:11540/health
```

## Configuration

### Basic Configuration

```yaml
embedder:
  type: UniversalEmbedder
  config:
    model: sentence-transformers/all-MiniLM-L6-v2
    base_url: http://127.0.0.1:11540/v1
    batch_size: 32
```

### Advanced Configuration

```yaml
embedder:
  type: UniversalEmbedder
  config:
    model: BAAI/bge-base-en-v1.5
    base_url: http://127.0.0.1:11540/v1
    api_key: universal
    batch_size: 16
    timeout: 120
    normalize: true
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID |
| `base_url` | string | `http://127.0.0.1:11540/v1` | Universal Runtime endpoint |
| `api_key` | string | `universal` | API key (optional) |
| `batch_size` | integer | `32` | Batch processing size (1-128) |
| `timeout` | integer | `120` | Request timeout in seconds |
| `normalize` | boolean | `true` | Normalize embeddings to unit length |

## Supported Models

The Universal Embedder supports any HuggingFace model that can generate embeddings:

### Fast Models (Recommended for Development)
- `sentence-transformers/all-MiniLM-L6-v2` - 384 dimensions, very fast
- `sentence-transformers/all-MiniLM-L12-v2` - 384 dimensions, balanced

### High Quality Models (Recommended for Production)
- `BAAI/bge-base-en-v1.5` - 768 dimensions, excellent performance
- `BAAI/bge-large-en-v1.5` - 1024 dimensions, best quality
- `sentence-transformers/all-mpnet-base-v2` - 768 dimensions, high quality

### Long Context Models
- `nomic-ai/nomic-embed-text-v1.5` - 768 dimensions, optimized for long text

### Multilingual Models
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` - 768 dimensions

## Usage Examples

### Example 1: General Purpose RAG

```yaml
version: v1
name: rag-project
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
          batch_size: 32

  data_processing_strategies:
    - id: pdf_ingest
      parsers:
        - type: PDFParser
          config:
            chunk_size: 500
            chunk_overlap: 50
```

### Example 2: High Quality Production

```yaml
rag:
  databases:
    - id: production_db
      type: vector
      provider: chromadb
      embedder:
        type: UniversalEmbedder
        config:
          model: BAAI/bge-base-en-v1.5
          base_url: http://127.0.0.1:11540/v1
          batch_size: 16
          timeout: 120
```

### Example 3: Long Documents

```yaml
rag:
  databases:
    - id: docs_db
      type: vector
      provider: chromadb
      embedder:
        type: UniversalEmbedder
        config:
          model: nomic-ai/nomic-embed-text-v1.5
          batch_size: 16
```

## Performance

### Speed Comparison

| Model | Dimensions | Speed (docs/sec) | Quality |
|-------|-----------|------------------|---------|
| all-MiniLM-L6-v2 | 384 | ~50 | Good |
| all-mpnet-base-v2 | 768 | ~25 | High |
| bge-base-en-v1.5 | 768 | ~30 | High |
| bge-large-en-v1.5 | 1024 | ~15 | Excellent |

*Benchmarks on M2 Mac with MPS acceleration*

### ONNX Optimization

For 3x faster embeddings, enable ONNX backend in Universal Runtime:

```bash
export RUNTIME_BACKEND=onnx
export ONNX_PROVIDER=CUDAExecutionProvider
```

Expected speedup:
- PyTorch: ~15ms per embedding
- ONNX: ~5ms per embedding
- **3x faster!**

## Integration with LlamaFarm

The Universal Embedder integrates seamlessly with LlamaFarm workflows:

### Step 1: Start Universal Runtime

```bash
cd runtimes/universal
uv run uvicorn server:app --port 11540 --reload
```

### Step 2: Configure Your Project

Update `llamafarm.yaml`:

```yaml
rag:
  databases:
    - id: main_db
      embedder:
        type: UniversalEmbedder
        config:
          model: sentence-transformers/all-MiniLM-L6-v2
```

### Step 3: Ingest Documents

```bash
lf datasets create -s pdf_ingest -b main_db docs
lf datasets upload docs ./files/*.pdf
lf datasets process docs
```

### Step 4: Query

```bash
lf rag query --database main_db "What are the key findings?"
```

## Advantages

### vs Ollama Embedder

| Feature | Universal | Ollama |
|---------|-----------|--------|
| **Model Selection** | Any HuggingFace model | Limited to Ollama models |
| **Hardware Acceleration** | MPS/CUDA/CPU | CPU only by default |
| **ONNX Support** | Yes (3x faster) | No |
| **Batch Processing** | Efficient | One-by-one |
| **OpenAI Compatible** | Yes | Partial |
| **Model Loading** | On-demand | Must pre-pull |

### vs OpenAI Embedder

| Feature | Universal | OpenAI |
|---------|-----------|--------|
| **Privacy** | Fully local | Cloud-based |
| **Cost** | Free | Pay per token |
| **Latency** | Low (local) | Network dependent |
| **Model Choice** | Any HuggingFace | Limited options |
| **Offline** | Yes | No |

### vs HuggingFace Embedder

| Feature | Universal | HuggingFace |
|---------|-----------|-------------|
| **Isolation** | Separate process | In-process |
| **Memory** | Shared across workflows | Dedicated per process |
| **Updates** | Independent | Requires restart |
| **Scalability** | Horizontal | Vertical |

## Best Practices

### 1. Model Selection

Choose based on your use case:

- **Development**: `all-MiniLM-L6-v2` (fast, good quality)
- **Production**: `BAAI/bge-base-en-v1.5` (high quality)
- **Long documents**: `nomic-ai/nomic-embed-text-v1.5` (optimized for length)
- **Multilingual**: `paraphrase-multilingual-mpnet-base-v2` (100+ languages)

### 2. Batch Size Tuning

```yaml
# CPU: smaller batches
batch_size: 8

# GPU: larger batches
batch_size: 32

# Production: balance throughput and latency
batch_size: 16
```

### 3. Timeout Configuration

```yaml
# Development (local models cached)
timeout: 60

# Production (first-time model download)
timeout: 120

# Slow connection
timeout: 300
```

## Troubleshooting

### Universal Runtime Not Available

**Error**: `Universal Runtime not available at http://127.0.0.1:11540`

**Solution**:
```bash
# Start the Universal Runtime
cd runtimes/universal
uv run uvicorn server:app --port 11540 --reload

# Verify it's running
curl http://127.0.0.1:11540/health
```

### Model Not Loading

**Error**: `Timeout connecting to Universal Runtime`

**Solution**:
1. First model load takes time (downloading from HuggingFace)
2. Increase timeout: `timeout: 300`
3. Pre-download model:
   ```bash
   cd runtimes/universal
   uv run python -c "from transformers import AutoModel; AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
   ```

### Slow Performance

**Issue**: Embeddings generation is slow

**Solutions**:
1. **Use smaller model**: Switch to `all-MiniLM-L6-v2`
2. **Enable ONNX**: Set `RUNTIME_BACKEND=onnx` (3x faster)
3. **Increase batch size**: Set `batch_size: 64` (if you have GPU)
4. **Check hardware**: Verify GPU/MPS is being used:
   ```bash
   curl http://127.0.0.1:11540/health | jq '.device'
   ```

### Wrong Embedding Dimensions

**Error**: Vector store dimension mismatch

**Solution**: Check model dimensions and update store config:

```yaml
embedder:
  config:
    model: sentence-transformers/all-MiniLM-L6-v2  # 384 dims

store:
  config:
    dimension: 384  # Must match model
```

## Advanced Configuration

### Custom Universal Runtime Location

```yaml
embedder:
  type: UniversalEmbedder
  config:
    model: sentence-transformers/all-MiniLM-L6-v2
    base_url: http://my-runtime-server:8080/v1
    api_key: my-secret-key
```

### Multiple Databases with Different Models

```yaml
rag:
  databases:
    # Fast database for development
    - id: dev_db
      embedder:
        type: UniversalEmbedder
        config:
          model: sentence-transformers/all-MiniLM-L6-v2
          batch_size: 32

    # High-quality database for production
    - id: prod_db
      embedder:
        type: UniversalEmbedder
        config:
          model: BAAI/bge-large-en-v1.5
          batch_size: 16
```

## Production Deployment

For production, consider:

1. **Deploy Universal Runtime separately**:
   ```bash
   # Docker deployment
   docker run -p 11540:11540 llamafarm/universal-runtime
   ```

2. **Enable ONNX optimization**:
   ```bash
   export RUNTIME_BACKEND=onnx
   export ONNX_PROVIDER=CUDAExecutionProvider
   ```

3. **Use high-quality models**:
   ```yaml
   model: BAAI/bge-base-en-v1.5  # Best balance
   # or
   model: BAAI/bge-large-en-v1.5  # Best quality
   ```

4. **Tune batch size for your hardware**:
   ```yaml
   batch_size: 32  # GPU
   batch_size: 8   # CPU
   ```

## Resources

- [Universal Runtime Documentation](../../universal/README.md)
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)
- [BGE Embeddings](https://huggingface.co/BAAI)
- [Nomic Embeddings](https://huggingface.co/nomic-ai)

## Support

For issues:
1. Check Universal Runtime is running: `curl http://127.0.0.1:11540/health`
2. Verify model is supported on HuggingFace
3. Check logs for detailed error messages
4. Consult Universal Runtime troubleshooting guide

