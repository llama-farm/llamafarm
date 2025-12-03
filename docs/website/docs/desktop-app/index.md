---
title: Desktop App
sidebar_position: 2
---

# Desktop App

The LlamaFarm Desktop App provides a complete local AI environment with visual project management, dataset uploads, chat interface, and built-in model management — no command line required.

:::info Found a bug or have a feature request?
[Submit an issue on GitHub →](https://github.com/llama-farm/llamafarm/issues)
:::

## Downloads

<div style={{display: 'flex', gap: '16px', flexWrap: 'wrap', marginBottom: '24px', marginTop: '16px'}}>
  <a href="https://github.com/llama-farm/llamafarm/releases/download/v0.0.19/LlamaFarm-0.0.19-arm64-mac.zip" style={{display: 'inline-flex', alignItems: 'center', padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', borderRadius: '8px', textDecoration: 'none', fontWeight: '600', fontSize: '16px'}}>
    ⬇️ Mac (M1+)
  </a>
  <a href="https://github.com/llama-farm/llamafarm/releases/download/v0.0.19/LlamaFarm.Setup.0.0.19.exe" style={{display: 'inline-flex', alignItems: 'center', padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', borderRadius: '8px', textDecoration: 'none', fontWeight: '600', fontSize: '16px'}}>
    ⬇️ Windows
  </a>
  <a href="https://github.com/llama-farm/llamafarm/releases/download/v0.0.19/LlamaFarm-0.0.19.AppImage" style={{display: 'inline-flex', alignItems: 'center', padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', borderRadius: '8px', textDecoration: 'none', fontWeight: '600', fontSize: '16px'}}>
    ⬇️ Linux
  </a>
</div>

---

## Hardware Requirements

### Minimum Requirements

To run the desktop app with small models (1-3B parameters like Qwen 1.7B):

| Component | Mac (M1+) | Windows | Linux |
|-----------|-----------|---------|-------|
| **CPU** | Apple M1 or newer | Intel i5 / AMD Ryzen 5 (8th gen+) | Intel i5 / AMD Ryzen 5 (8th gen+) |
| **RAM** | 8 GB | 8 GB | 8 GB |
| **Storage** | 10 GB free | 10 GB free | 10 GB free |
| **OS** | macOS 12+ (Monterey) | Windows 10/11 (64-bit) | Ubuntu 22.04+ (tested) |
| **GPU** | Integrated (Metal) | Optional | Optional |

### Recommended Requirements

For larger models (7-8B parameters) and better performance:

| Component | Mac (M1+) | Windows | Linux |
|-----------|-----------|---------|-------|
| **CPU** | Apple M1 Pro/Max or M2+ | Intel i7 / AMD Ryzen 7 | Intel i7 / AMD Ryzen 7 |
| **RAM** | 16 GB+ | 16 GB+ | 16 GB+ |
| **Storage** | 50 GB+ SSD | 50 GB+ SSD | 50 GB+ SSD |
| **OS** | macOS 13+ (Ventura) | Windows 11 | Ubuntu 22.04+ |
| **GPU** | Unified Memory (Metal) | NVIDIA RTX 3060+ (8GB+ VRAM) | NVIDIA RTX 3060+ (8GB+ VRAM) |

---

## Model Memory Requirements

The default model is **Qwen 1.7B GGUF (Q4_K_M quantization)**, which works well on modest hardware.

| Model | Parameters | RAM Required | VRAM (GPU) | Notes |
|-------|------------|--------------|------------|-------|
| **Qwen 1.7B** (default) | 1.7B | 4 GB | 2 GB | Great for testing, fast responses |
| **Qwen 3B** | 3B | 6 GB | 4 GB | Better quality, still fast |
| **Llama 3.1 8B** | 8B | 10 GB | 6 GB | High quality, needs more resources |
| **Qwen 8B** | 8B | 10 GB | 6 GB | High quality reasoning |

:::tip Quantization Matters
GGUF models use quantization (Q4_K_M, Q5_K_M, Q8_0) to reduce memory usage. Q4_K_M offers the best balance of quality and speed for most users.
:::

---

## Platform-Specific Notes

### Mac (Apple Silicon)

- **Tested on**: M1, M1 Pro, M1 Max, M2, M3
- **Acceleration**: Uses Metal for GPU acceleration automatically
- **Memory**: Unified memory is shared between CPU and GPU — 16GB+ recommended for 8B models
- **Installation**: Unzip and drag to Applications folder

### Windows

- **Tested on**: Windows 10 (21H2+), Windows 11
- **Acceleration**: NVIDIA CUDA (if available), otherwise CPU
- **GPU Support**: NVIDIA GPUs with CUDA 11.8+ drivers
- **Installation**: Run the `.exe` installer

:::note Windows Defender
Windows Defender may scan the app on first launch. This is normal and should complete within a minute.
:::

### Linux

- **Tested on**: Ubuntu 22.04 LTS, Ubuntu 24.04 LTS
- **Format**: AppImage (portable, no installation needed)
- **Acceleration**: NVIDIA CUDA or Vulkan (if available)
- **Dependencies**: FUSE required for AppImage

```bash
# Make executable and run
chmod +x LlamaFarm-0.0.19.AppImage
./LlamaFarm-0.0.19.AppImage

# If FUSE is not installed:
sudo apt install fuse libfuse2
```

:::note Other Distributions
While Ubuntu is our primary test platform, the AppImage should work on most modern Linux distributions with glibc 2.31+. Community reports for Fedora, Arch, and Debian are welcome!
:::

---

## Features

The desktop app includes:

- **Visual Project Management** — Create, configure, and switch between projects
- **Dataset Uploads** — Drag-and-drop file uploads with real-time processing status
- **Chat Interface** — Test your AI with full RAG context
- **Model Management** — Download, switch, and configure models
- **Built-in Services** — No need to run Docker or manage background processes

---

## Troubleshooting

### App won't start

1. **Windows**: Allow through Windows Defender/Firewall
2. **Linux**: Ensure FUSE is installed, check AppImage is executable

### Out of memory

- Close other applications
- Use a smaller model (Qwen 1.7B instead of 8B)
- Use higher quantization (Q4_K_M instead of Q8_0)

### Model download fails

- Check internet connection
- Ensure sufficient disk space
- Try downloading again — downloads resume automatically

### Need help?

- [Submit an issue on GitHub](https://github.com/llama-farm/llamafarm/issues)
- [Join our Discord](https://discord.gg/RrAUXTCVNF)

---

## Next Steps

- [Quickstart Guide](../quickstart/index.md) — Get started with your first project
- [Configuration Guide](../configuration/index.md) — Customize your setup
- [CLI Reference](../cli/index.md) — For power users who want command-line access
