# llamafarm-llama

LlamaFarm's custom llama.cpp Python bindings. Uses pre-built binaries from llama.cpp releases for easy installation and consistent behavior across platforms.

## Features

- **No compilation required**: Downloads pre-built binaries from llama.cpp releases
- **API compatible**: Drop-in replacement for llama-cpp-python
- **Automatic hardware detection**: Picks the best binary (CUDA, Metal, CPU) for your system
- **Hybrid distribution**: Minimal wheel on PyPI, platform wheels with bundled binaries available

## Installation

```bash
# Basic installation (downloads binary on first use)
pip install llamafarm-llama

# With specific backend via extra index
pip install llamafarm-llama --extra-index-url https://wheels.llamafarm.dev/cu121
```

## Quick Start

```python
from llamafarm_llama import Llama

# Load a GGUF model
llm = Llama(
    model_path="path/to/model.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,  # Use all GPU layers
)

# Chat completion (llama-cpp-python compatible API)
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    max_tokens=100,
    temperature=0.7,
)

print(response["choices"][0]["message"]["content"])
```

## Streaming

```python
for chunk in llm.create_chat_completion(
    messages=[{"role": "user", "content": "Tell me a story"}],
    max_tokens=500,
    stream=True,
):
    delta = chunk["choices"][0].get("delta", {})
    if "content" in delta:
        print(delta["content"], end="", flush=True)
```

## Embeddings

```python
# Initialize with embedding mode
llm = Llama(model_path="embedding-model.gguf", embedding=True)

response = llm.create_embedding("Hello world")
embedding = response["data"][0]["embedding"]
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLAMAFARM_BACKEND` | Force specific backend: `cpu`, `cuda12`, `cuda11`, `metal`, `vulkan` |
| `LLAMAFARM_CACHE_DIR` | Custom cache directory for downloaded binaries |
| `LLAMAFARM_LLAMA_VERSION` | Override llama.cpp version (default: b7693) |

### Binary Info

```python
from llamafarm_llama import get_binary_info

info = get_binary_info()
print(f"llama.cpp version: {info['version']}")
print(f"Platform: {info['platform_key']}")
print(f"Binary location: {info['lib_path']}")
```

## Supported Platforms

| Platform | Architecture | Backends |
|----------|--------------|----------|
| Linux | x86_64 | CPU, CUDA 11, CUDA 12, Vulkan |
| Linux | arm64 | CPU (source build), CUDA (source build), Vulkan (source build) |
| macOS | arm64 | Metal |
| macOS | x86_64 | CPU |
| Windows | x86_64 | CPU, CUDA 12, Vulkan |

## NVIDIA Jetson Support

Jetson devices (Orin, Xavier, TX2, Nano) require building llama.cpp from source because pre-built ARM64 CUDA binaries are not included in llama.cpp releases.

### Building for Jetson

See the llama.cpp [Jetson Build Guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build-jetson.md) for complete instructions. Quick start:

```bash
# Clone and build llama.cpp for Jetson Orin
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

### Jetson-Specific Environment Variables

| Variable | Description |
|----------|-------------|
| `LLAMAFARM_SYNC_INFERENCE` | Force sync (`1`) or async (`0`) inference. Default: auto-detect |
| `LLAMAFARM_LLAMA_LIB_DIR` | Path to custom-built llama.cpp libraries |

### Recommended Settings for Jetson

```python
from llamafarm_llama import Llama

llm = Llama(
    model_path="model.gguf",
    n_ctx=2048,              # Conservative for 8GB memory
    n_gpu_layers=-1,         # Offload all layers to GPU
    flash_attn=True,         # Enable flash attention
    use_mmap=False,          # Disable mmap for unified memory
    cache_type_k="q4_0",     # 4-bit KV cache for memory savings
    cache_type_v="q4_0",
)
```

### Performance on Jetson Orin Nano (8GB)

With Qwen3-1.7B Q4_K_M:
- **35+ tokens/second** generation speed
- **~140ms** time to first token
- **~2.5GB** memory usage

## License

MIT License - see LICENSE file for details.

llama.cpp is licensed under the MIT License.
