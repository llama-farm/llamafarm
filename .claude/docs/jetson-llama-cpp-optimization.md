# llama.cpp Optimization for Jetson Orin Nano and Edge Devices

Research compiled: 2026-01-10

## Executive Summary

This document provides practical optimization techniques for running llama.cpp on Jetson Orin Nano (8GB) and similar edge hardware. Key findings include:

- **Best model size**: 3B models at Q4_K_M quantization offer the optimal balance (28-55 tokens/sec)
- **Memory savings**: KV cache quantization can reduce memory by 4x (requires flash attention)
- **Context tradeoff**: Use 2048-4096 context for 7B models; smaller models can handle longer contexts
- **Power mode**: Use MAXN_SUPER (25W) for maximum performance with adequate cooling

---

## 1. Jetson Orin Nano Hardware Specifications

### Memory Architecture
- **Total RAM**: 8GB LPDDR5 (shared CPU/GPU)
- **Memory Bandwidth**: 68.3 GB/s theoretical (3x faster than original Jetson Nano)
- **CUDA Compute Capability**: 8.7 (Ampere architecture)

### Power Modes

| Mode | Power | CPU Cores | CPU Freq | GPU Freq | EMC Freq |
|------|-------|-----------|----------|----------|----------|
| 7W | 7W | 4 | 806.4 MHz | 306 MHz | Lower |
| 15W (default) | 15W | 6 | 1497.6 MHz | 612 MHz | 2133 MHz |
| 25W | 25W | 6 | 1728 MHz | 1020 MHz | 3199 MHz |
| MAXN_SUPER | Max | 6 | 1728 MHz | 1020 MHz | 3199 MHz |

### Thermal Thresholds
- Software throttling: 99.0C
- Hardware throttling: 103.0C
- Software shutdown: 104.5C
- Hardware shutdown: 105.0C

---

## 2. llama.cpp Parameter Reference

### KV Cache Quantization

Reduces memory usage for the key-value cache. **Requires flash attention enabled.**

```bash
# Server parameters
--cache-type-k q8_0    # KV cache key type (default: f16)
--cache-type-v q8_0    # KV cache value type (default: f16)
--flash-attn on        # Required for KV quantization
```

**Supported types**: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1

**Memory savings**:
- f16 -> q8_0: ~50% reduction
- f16 -> q4_0: ~75% reduction

**Performance impact**:
- q8_0: Minimal quality loss, ~10-15% slower generation
- q4_0: Slight quality degradation, ~25-30% slower generation

**Note**: KV cache quantization is NOT compatible with context shifting.

### GPU Layer Offloading

```bash
--gpu-layers 99        # Offload all layers to GPU (recommended for unified memory)
--gpu-layers -1        # Auto-detect optimal layer count
```

**Important for Jetson**: With shared memory architecture, offloading ALL layers to GPU is faster than partial offloading.

### Batch Size Tuning

```bash
--batch-size 512       # Logical batch size (default: 2048)
--ubatch-size 256      # Physical batch size (default: 512)
```

**Recommendations for 8GB RAM**:
- 7B models: `--batch-size 512 --ubatch-size 256`
- 3B models: `--batch-size 1024 --ubatch-size 512`
- 1B models: Default values work well

### Context Size

```bash
--ctx-size 4096        # Prompt context size
```

**Memory usage by context size** (approximate, varies by model):
- 2048 tokens: ~0.7 GB
- 4096 tokens: ~1.4 GB
- 8192 tokens: ~2.8 GB

### Memory Management

```bash
--mmap on              # Memory-map model file (default: on)
--mlock off            # Lock model in RAM (default: off)
--no-host              # Bypass host buffer for extra VRAM
--fit on               # Auto-adjust to fit device memory (default: on)
```

**For Jetson with limited RAM**: Keep mmap enabled to allow on-demand loading.

---

## 3. Model Selection Guidelines

### Recommended Quantizations

| Quantization | Bits | Quality Loss | Use Case |
|--------------|------|--------------|----------|
| Q4_K_M | 4-bit | +0.0535 ppl | **Recommended default** |
| Q4_0 | 4-bit | +0.2499 ppl | Legacy, simpler decode |
| Q5_K_M | 5-bit | +0.0203 ppl | Higher quality needs |
| Q3_K_M | 3-bit | +0.1535 ppl | Extreme memory constraints |

**Note**: Q4_K_M is almost always preferred over Q4_0 for the same file size with better quality.

### Model Size Recommendations for Jetson Orin Nano 8GB

| Model Size | Context | Memory | Performance | Recommended |
|------------|---------|--------|-------------|-------------|
| 1B Q4_K_M | 4096 | ~1.5 GB | 40-55 tok/s | Yes |
| 3B Q4_K_M | 4096 | ~2.5 GB | 28-40 tok/s | **Best balance** |
| 7B Q4_K_M | 2048 | ~5.0 GB | 8-15 tok/s | Limited use |
| 7B Q4_K_M | 4096 | ~6.5 GB | 6-12 tok/s | Tight fit |

---

## 4. Optimal Configuration for Jetson Orin Nano

### Recommended Server Configuration

```bash
llama-server \
  --model /path/to/model-3b-q4_k_m.gguf \
  --gpu-layers 99 \
  --ctx-size 4096 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --mmap on \
  --host 0.0.0.0 \
  --port 8080
```

### For Maximum Memory Efficiency (7B models)

```bash
llama-server \
  --model /path/to/model-7b-q4_k_m.gguf \
  --gpu-layers 99 \
  --ctx-size 2048 \
  --batch-size 256 \
  --ubatch-size 128 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --mmap on \
  --host 0.0.0.0 \
  --port 8080
```

### Universal Runtime Configuration (LlamaFarm)

When configuring models in LlamaFarm's Universal Runtime, use these parameters:

```yaml
runtime_options:
  n_gpu_layers: 99
  n_ctx: 4096
  n_batch: 512
  flash_attn: true
  cache_type_k: "q8_0"
  cache_type_v: "q8_0"
  use_mmap: true
```

---

## 5. System Optimization

### Setting Maximum Performance Mode

```bash
# Set to maximum performance (25W)
sudo nvpmodel -m 0

# Verify current mode
sudo nvpmodel -q

# Set maximum clock frequencies
sudo jetson_clocks

# With maximum fan speed (recommended for sustained inference)
sudo jetson_clocks --fan
```

### Building llama.cpp for Jetson Orin

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build with CUDA support
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

**Note**: CUDA compute capability 8.7 corresponds to the Orin's Ampere GPU.

---

## 6. Monitoring and Profiling

### Tegrastats (Real-time Monitoring)

```bash
# Basic monitoring (1 second interval)
tegrastats --interval 1000

# Log to file
tegrastats --interval 1000 --logfile inference_stats.log
```

**Output includes**:
- RAM/Swap usage
- CPU core utilization and frequencies
- GPU utilization and frequency
- Temperature readings
- Power consumption

### JTOP (Interactive Dashboard)

```bash
# Install
sudo pip3 install jetson-stats

# Run interactive dashboard
jtop
```

Features:
- Real-time GPU/CPU/memory visualization
- Temperature and power monitoring
- Fan control
- Clock frequency display

### Memory Usage Estimation

Approximate VRAM formula:
```
Total Memory = Model Size + KV Cache + Compute Buffers

Model Size (Q4_K_M) ≈ Parameters * 0.5 bytes
KV Cache ≈ 2 * n_layers * n_ctx * n_embd * 2 bytes (f16)
         ≈ 2 * n_layers * n_ctx * n_embd * 0.5 bytes (q4)
Compute Buffers ≈ batch_size * n_embd * 4 bytes
```

---

## 7. Advanced Optimizations

### Speculative Decoding

Use a smaller draft model to accelerate inference:

```bash
llama-server \
  --model /path/to/model-7b-q4_k_m.gguf \
  --model-draft /path/to/model-0.5b-q4_k_m.gguf \
  --draft-max 8 \
  --gpu-layers 99 \
  --gpu-layers-draft 99
```

**Expected speedup**: 1.5-2x for suitable model pairs.

### Continuous Batching

Enabled by default in llama-server. Allows efficient handling of multiple concurrent requests by dynamically batching tokens.

### Prompt Caching

For repeated system prompts, llama.cpp can share KV cache across sequences:

```bash
--cache-prompt on      # Cache common prompt prefixes
```

---

## 8. Troubleshooting

### GPU Not Detected

If llama.cpp reports "NO GPU DETECTED":

1. Verify CUDA installation: `nvcc --version`
2. Check GPU visibility: `nvidia-smi` or `tegrastats`
3. Rebuild with explicit CUDA architecture: `-DCMAKE_CUDA_ARCHITECTURES=87`

### Out of Memory Errors

1. Reduce context size: `--ctx-size 2048`
2. Enable KV cache quantization: `--cache-type-k q4_0 --cache-type-v q4_0`
3. Reduce batch size: `--batch-size 256 --ubatch-size 128`
4. Use smaller model or more aggressive quantization

### Thermal Throttling

1. Improve cooling (heatsink, fan)
2. Run `sudo jetson_clocks --fan` for maximum fan speed
3. Consider lower power mode for sustained workloads
4. Monitor with `tegrastats` or `jtop`

### Slow Performance

1. Ensure all layers on GPU: `--gpu-layers 99`
2. Enable flash attention: `--flash-attn on`
3. Verify power mode: `sudo nvpmodel -q`
4. Run `sudo jetson_clocks` for maximum clocks

---

## Sources

### llama.cpp Documentation
- [llama.cpp Server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [KV Cache Quantization Discussion](https://github.com/ggml-org/llama.cpp/discussions/5932)
- [GPU Layer Offloading Discussion](https://github.com/ggml-org/llama.cpp/discussions/7678)
- [Speculative Decoding](https://deepwiki.com/ggml-org/llama.cpp/7.2-speculative-decoding)

### Jetson Resources
- [NVIDIA Jetson Power and Performance Guide](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- [llama.cpp on Jetson Discussion](https://github.com/ggml-org/llama.cpp/discussions/5059)
- [Jetson Stats (jtop)](https://github.com/rbonghi/jetson_stats)
- [NVIDIA Jetson Forums - llama.cpp](https://forums.developer.nvidia.com/t/compile-llama-cpp-to-use-the-jetson-orin-nano-super-gpu/343758)

### Quantization Guides
- [GGUF Quantization Guide](https://enclaveai.app/blog/2025/11/12/practical-quantization-guide-iphone-mac-gguf/)
- [K-Quants vs I-Quants](https://kaitchup.substack.com/p/choosing-a-gguf-model-k-quants-i)
- [KV Cache Quantization Medium Article](https://medium.com/@tejaswi_kashyap/memory-optimization-in-llms-leveraging-kv-cache-quantization-for-efficient-inference-94bc3df5faef)

### Edge Deployment
- [Edge LLM Deployment Guide](https://www.shakudo.io/blog/edge-llm-deployment-guide)
- [On-Device LLM Inference on Jetson](https://www.genaiprotos.com/project/on-device-llm-inference-jetson-orin-nano/)
