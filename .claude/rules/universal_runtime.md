# Universal Runtime

## Overview
- Local ML inference server with OpenAI-compatible API
- Supports multiple model types: LLMs, embeddings, OCR, anomaly detection, classification
- Automatic hardware detection (MPS/CUDA/CPU)
- Model caching with configurable unload timeouts
- Supports HuggingFace transformers and GGUF quantized models

## Architecture

### Entry Points
- `runtimes/universal/server.py` - FastAPI server and endpoint definitions

### Directory Structure
- **models/** - Model type implementations
  - `language_model.py` - Text generation (transformers)
  - `gguf_language_model.py` - GGUF quantized LLMs (llama-cpp-python)
  - `encoder_model.py` - Embeddings (sentence-transformers, BERT)
  - `gguf_encoder_model.py` - GGUF embedding models
  - `classifier_model.py` - Text classification
  - `anomaly_model.py` - Anomaly detection
  - `document_model.py` - Document processing
  - `ocr_model.py` - Optical character recognition
- **routers/** - Additional API routes
  - `chat_completions/` - OpenAI-compatible chat completions endpoint
- **utils/** - Shared utilities
  - `device.py` - Hardware detection and device selection
  - `model_cache.py` - LRU model caching with TTL-based unloading
  - `model_format.py` - Model format detection (safetensors, GGUF, etc.)
  - `context_calculator.py` - Token/context length calculations
  - `feature_encoder.py` - Feature extraction utilities
  - `thinking.py` - Chain-of-thought processing
- **core/** - Infrastructure
  - `logging.py` - Structured logging

### Key APIs
- `POST /v1/chat/completions` - OpenAI-compatible chat (streaming supported)
- `POST /v1/embeddings` - Generate embeddings
- `POST /v1/classify` - Text classification
- `POST /v1/anomaly` - Anomaly detection
- `GET /v1/models` - List loaded/available models

### Environment Variables
- `MODEL_UNLOAD_TIMEOUT` - Seconds before unloading inactive models (default: 300)
- `CLEANUP_CHECK_INTERVAL` - Seconds between cleanup checks (default: 30)

## Development

### Running
- `nx start universal-runtime` or `uv run uvicorn runtimes.universal.server:app --reload --port 8001`
- Default port: 8001

### Testing
- `cd runtimes/universal && uv run pytest`
- Tests in `runtimes/universal/tests/`
