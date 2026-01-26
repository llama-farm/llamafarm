# LlamaFarm on NVIDIA Jetson

Automated setup for running LlamaFarm Universal Runtime on NVIDIA Jetson devices with CUDA acceleration.

## Quick Start

### 1. Start the Jetson Container

```bash
docker run -it --rm --runtime=nvidia --network host \
  -v $HOME/.cache:/root/.cache \
  -v /path/to/llamafarm:/data/llamafarm \
  dustynv/l4t-pytorch:r36.2.0 \
  bash
```

### 2. Install UV Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 3. Run Setup Script

```bash
cd /data/llamafarm
./deployment/jetson/setup.sh
```

### 4. Start the Server

```bash
./deployment/jetson/start.sh
```

Or with custom settings:

```bash
./deployment/jetson/start.sh 'unsloth/Qwen3-0.6B-GGUF:Q4_K_M' 4096
```

## Building llama.cpp for Jetson

Pre-built binaries don't support ARM64 CUDA. Build from source:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# CRITICAL: CUDA_GRAPHS=OFF for Tegra stability
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)

# Copy to cache
mkdir -p ~/.cache/llamafarm-llama/jetson
cp build/src/libllama.so ~/.cache/llamafarm-llama/jetson/
cp build/ggml/src/libggml*.so* ~/.cache/llamafarm-llama/jetson/
```

### CUDA Architecture Flags

| Device | Flag |
|--------|------|
| Orin (Nano/NX/AGX) | `-DCMAKE_CUDA_ARCHITECTURES="87"` |
| Xavier (NX/AGX) | `-DCMAKE_CUDA_ARCHITECTURES="72"` |
| TX2 | `-DCMAKE_CUDA_ARCHITECTURES="62"` |
| Nano | `-DCMAKE_CUDA_ARCHITECTURES="53"` |

## What the Setup Script Does

1. **Detects Jetson hardware** via `/proc/device-tree/model`
2. **Finds system Python with CUDA PyTorch** (from dustynv container)
3. **Creates venv with `--system-site-packages`** to inherit CUDA torch
4. **Installs LlamaFarm dependencies**
5. **Removes PyPI torch** (CPU-only) to use system CUDA torch
6. **Downgrades numpy** to <2 for compatibility
7. **Creates `.env.jetson`** with optimized environment variables

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMAFARM_LLAMA_LIB_DIR` | Path to llama.cpp libraries | `~/.cache/llamafarm-llama/jetson` |
| `LLAMAFARM_SYNC_INFERENCE` | Force sync inference | `1` (recommended) |

## Troubleshooting

### CUDA not available after setup

Check that you're using the container's system Python:

```bash
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

If False, re-run `./deployment/jetson/setup.sh`

### Segfaults during inference

Ensure llama.cpp was built with `-DGGML_CUDA_GRAPHS=OFF`

### Out of memory

1. Reduce context: `--ctx-len 1024`
2. Use smaller quantization: `Q4_K_M` instead of `Q8_0`
3. Try a smaller model

## See Also

- [Full Jetson Documentation](../../docs/website/docs/deployment/jetson.md)
- [Universal Runtime](../../runtimes/universal/)
