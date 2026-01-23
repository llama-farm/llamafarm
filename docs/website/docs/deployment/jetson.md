---
title: NVIDIA Jetson
sidebar_position: 2
---

# Deploying on NVIDIA Jetson

LlamaFarm runs natively on NVIDIA Jetson devices (Orin, Xavier, TX2, Nano) with CUDA acceleration. This guide covers setup, optimization, and troubleshooting for Jetson deployments.

## Supported Devices

| Device | Compute Capability | Memory | Performance |
|--------|-------------------|--------|-------------|
| Jetson Orin Nano | 8.7 | 8GB shared | 35+ tok/s |
| Jetson Orin NX | 8.7 | 8-16GB shared | 35+ tok/s |
| Jetson AGX Orin | 8.7 | 32-64GB shared | 40+ tok/s |
| Jetson Xavier NX | 7.2 | 8-16GB shared | 25+ tok/s |
| Jetson AGX Xavier | 7.2 | 16-32GB shared | 30+ tok/s |

## Quick Start (Automated)

LlamaFarm includes setup scripts that automate the Jetson configuration:

```bash
# 1. Start the Jetson container
docker run -it --rm --runtime=nvidia --network host \
  -v $HOME/.cache:/root/.cache \
  -v /path/to/llamafarm:/data/llamafarm \
  dustynv/l4t-pytorch:r36.2.0 \
  bash

# 2. Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 3. Run the automated setup
cd /data/llamafarm
./deployment/jetson/setup.sh

# 4. Start the server
./deployment/jetson/start.sh
```

The setup script automatically:
- Detects Jetson hardware
- Finds system Python with CUDA PyTorch
- Creates venv with `--system-site-packages` to inherit CUDA torch
- Installs dependencies and removes conflicting PyPI packages
- Creates `.env.jetson` with optimized environment variables

For custom model or context size:

```bash
./deployment/jetson/start.sh 'unsloth/Qwen3-0.6B-GGUF:Q4_K_M' 4096
```

## Prerequisites

1. **JetPack SDK** installed (includes CUDA, cuDNN, TensorRT)
2. **Python 3.10+** with UV package manager
3. **Custom llama.cpp build** (see below)

## Building llama.cpp for Jetson

Pre-built llama.cpp binaries don't include ARM64 CUDA. You must build from source:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# CRITICAL: Disable CUDA graphs for Tegra stability
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

### Why GGML_CUDA_GRAPHS=OFF?

CUDA graphs provide performance benefits on discrete GPUs but cause stability issues on Jetson's unified memory architecture:
- Graph compilation overhead exceeds benefits for small batch sizes
- Memory pressure from graph captures
- Unpredictable behavior on unified memory systems

### Architecture Flags

| Device | Flag |
|--------|------|
| Orin (Nano/NX/AGX) | `-DCMAKE_CUDA_ARCHITECTURES="87"` |
| Xavier (NX/AGX) | `-DCMAKE_CUDA_ARCHITECTURES="72"` |
| TX2 | `-DCMAKE_CUDA_ARCHITECTURES="62"` |
| Nano | `-DCMAKE_CUDA_ARCHITECTURES="53"` |

## Docker Container Setup (Recommended)

The easiest way to run LlamaFarm on Jetson is using NVIDIA's PyTorch container which includes CUDA-enabled PyTorch:

```bash
docker run -it --rm --runtime=nvidia --network host \
  -v /home/$USER/.cache:/root/.cache \
  -v /path/to/llamafarm:/data/llamafarm \
  dustynv/l4t-pytorch:r36.2.0 \
  bash
```

### Install UV Package Manager

Inside the container:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Setup Universal Runtime with CUDA PyTorch

The container has CUDA-enabled PyTorch (Python 3.10), but the venv must be configured to use it:

```bash
cd /data/llamafarm/runtimes/universal

# Create venv with system site packages (inherits CUDA torch)
uv venv .venv --python /usr/local/bin/python3 --system-site-packages

# Install dependencies (this pulls CPU torch from PyPI)
uv pip install -r pyproject.toml --python .venv/bin/python

# Remove CPU torch, fall back to system CUDA torch
uv pip uninstall torch torchvision --python .venv/bin/python

# Downgrade numpy (system torch needs numpy 1.x)
uv pip install "numpy<2" --python .venv/bin/python
```

### Verify CUDA PyTorch

```bash
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Should print: CUDA: True
```

This enables GPU acceleration for:
- **llama.cpp inference** (GGUF models) - 35+ tok/s
- **Transformers models** (classifiers, embeddings) - GPU accelerated
- **Anomaly detection** (sklearn) - CPU only

## Installing Built Libraries

Copy the built libraries to LlamaFarm's cache:

```bash
CACHE_DIR="$HOME/.cache/llamafarm-llama/jetson"
mkdir -p "$CACHE_DIR"

cp build/src/libllama.so "$CACHE_DIR/"
cp build/ggml/src/libggml*.so* "$CACHE_DIR/"

# Point LlamaFarm to the custom build
export LLAMAFARM_LLAMA_LIB_DIR="$CACHE_DIR"
```

## Running the Server

```bash
cd /path/to/llamafarm/runtimes/universal

# Start with conservative settings for 8GB memory
uv run python server.py \
    --model 'unsloth/Qwen3-1.7B-GGUF:Q4_K_M' \
    --ctx-len 2048
```

### Test the API

```bash
curl -s -X POST http://localhost:11540/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "n_ctx": 2048
  }'
```

**Important**: Always pass `n_ctx` in requests to avoid defaulting to the model's full context (40960 for Qwen3), which causes OOM.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMAFARM_SYNC_INFERENCE` | Force sync (`1`) or async (`0`) inference | Auto-detect |
| `LLAMAFARM_LLAMA_LIB_DIR` | Path to custom llama.cpp libraries | Auto |
| `CUDA_CACHE_DISABLE` | Disable CUDA cache (reduces fragmentation) | 0 |

## Memory Optimization

Jetson uses unified memory shared between CPU and GPU. Optimize with:

### 1. Use Quantized Models

Q4_K_M provides the best balance of quality and memory:

```
Model Size (Qwen3-1.7B):
- Q4_K_M: ~1.0 GB
- Q8_0: ~1.8 GB
- FP16: ~3.4 GB
```

### 2. Reduce Context Size

Start with `n_ctx=2048` and increase only if needed:

```
KV Cache Memory (Qwen3-1.7B):
- n_ctx=2048: ~224 MB
- n_ctx=4096: ~448 MB
- n_ctx=8192: ~896 MB
```

### 3. Enable KV Cache Quantization

For even lower memory, use quantized KV cache:

```python
# In your model configuration
cache_type_k="q4_0"  # 4-bit keys
cache_type_v="q4_0"  # 4-bit values
```

This reduces KV cache memory by ~4x.

## Troubleshooting

### "double free or corruption" crash

**Cause**: CUDA backend initialization from worker thread.

**Solution**: LlamaFarm automatically initializes the backend from the main thread. If you see this error, ensure you're using the latest version.

### Slow performance (< 25 tok/s)

**Check**:
1. Rebuild llama.cpp with `GGML_CUDA_GRAPHS=OFF`
2. Verify all layers on GPU: look for "offloaded 29/29 layers" in logs
3. Check `n_ctx` isn't too large (causes memory pressure)

### Out of memory

**Solutions**:
1. Reduce `n_ctx` (e.g., 2048 instead of 4096)
2. Use smaller quantization (Q4_K_M instead of Q8_0)
3. Enable KV cache quantization
4. Monitor with `tegrastats`

### Model loads on CPU

**Check**:
1. Verify CUDA: `nvcc --version`
2. Ensure libggml-cuda.so exists in library directory
3. Check logs for "CUDA0" device detection

## Performance Benchmarks

Tested on Jetson Orin Nano 8GB with Qwen3-1.7B Q4_K_M:

| Metric | Value |
|--------|-------|
| Tokens/second | 35-36 tok/s |
| Time to first token | ~70-100ms |
| Memory usage | ~2.5 GB |
| Power consumption | ~15W |

## systemd Service

Create `/etc/systemd/system/llamafarm.service`:

```ini
[Unit]
Description=LlamaFarm Universal Runtime
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/data/llamafarm/runtimes/universal
Environment=LLAMAFARM_LLAMA_LIB_DIR=/root/.cache/llamafarm-llama/jetson
Environment=LLAMAFARM_SYNC_INFERENCE=1
ExecStart=/root/.local/bin/uv run python server.py --model unsloth/Qwen3-1.7B-GGUF:Q4_K_M --ctx-len 2048
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable llamafarm
sudo systemctl start llamafarm
```

## See Also

- [Deployment Overview](./index.md)
- [Configuration Guide](../configuration/index.md)
- [Troubleshooting](../troubleshooting/index.md)
