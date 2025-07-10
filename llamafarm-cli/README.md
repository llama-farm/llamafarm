# 🌾 LLaMA Farm CLI

Deploy AI models, agents, and databases into single deployable binaries - no cloud required.

> **🚧 Early Preview**: We're building the architecture for single-binary AI deployment. Some features are working 🐣, others are still incubating 🥚. Join us!

## Installation

```bash
npm install -g @llamafarm/llamafarm
```

## Quick Start

```bash
# Deploy a model
llamafarm plant llama3-8b

# Deploy with optimization
llamafarm plant llama3-8b --optimize

# Deploy to specific target
llamafarm plant mistral-7b --target raspberry-pi

# Development/Testing (no model download)
llamafarm plant llama3-8b --mock
```

## Complete Workflow Example

```bash
# 1. Plant - Configure your AI deployment
llamafarm plant llama3-8b \
  --device mac-arm \
  --agent chat-assistant \
  --rag \
  --database vector

# 2. Bale - Compile to single binary
llamafarm bale ./.llamafarm/llama3-8b \
  --device mac-arm \
  --optimize

# 3. Harvest - Deploy anywhere
llamafarm harvest llama3-8b-mac-arm-v1.0.0.bin --run

# Or just copy and run directly (no dependencies needed!)
./llama3-8b-mac-arm-v1.0.0.bin
```

## Features

### 🐣 Working Now
- ✅ CLI structure and all commands (with friendly placeholders)
- ✅ Plugin architecture for extensibility
- ✅ Mac platform detection
- ✅ Demo web UI
- ✅ Mock mode for testing

### 🥚 Coming Soon
- ⏳ Actual binary compilation
- ⏳ Model quantization and embedding
- ⏳ Vector database integration
- ⏳ GPU acceleration
- ⏳ Cross-platform binaries

## Commands

> **Note:** All commands are implemented but some features show placeholder messages while we build the core functionality.

### `plant` 🐣
Deploy a model to create a standalone binary.

```bash
llamafarm plant <model> [options]

Options:
  --target <platform>    Target platform (mac, linux, windows, raspberry-pi)
  --optimize            Enable size optimization
  --agent <name>        Include an agent
  --rag                 Enable RAG pipeline
  --database <type>     Include database (vector, sqlite)
```

### Examples

```bash
# Basic deployment
llamafarm plant llama3-8b

# Deploy with RAG and vector database
llamafarm plant mixtral-8x7b --rag --database vector

# Deploy optimized for Raspberry Pi
llamafarm plant llama3-8b --target raspberry-pi --optimize

# Deploy with custom agent
llamafarm plant llama3-8b --agent customer-service
```

### `bale` 🥚
🎯 **The Baler** - Compile your deployment into a single executable binary. *(Currently shows placeholder - actual compilation coming soon!)*

```bash
llamafarm bale <project-dir> [options]

Options:
  --device <platform>   Target platform (mac, linux, windows, raspberry-pi)
  --output <path>       Output binary path
  --optimize <level>    Optimization level (none, standard, max)
  --sign               Sign the binary for distribution
  --compress           Extra compression (slower but smaller)
```

The Baler packages everything into a single binary:
- 🧠 Quantized model (GGUF format)
- 🤖 Agent configuration & code
- 🗄️ Embedded vector database
- 🌐 Web UI
- 🚀 Node.js runtime
- 🔧 Platform-specific optimizations

**Supported Platforms:**
- `mac` / `mac-arm` / `mac-intel` - macOS with Metal acceleration
- `linux` / `linux-arm` - Linux with CUDA support
- `windows` - Windows with DirectML/CUDA
- `raspberry-pi` - Optimized for ARM devices
- `jetson` - NVIDIA Jetson edge devices

**Typical Binary Sizes:**
- 7B models: 4-8GB (depending on quantization)
- 13B models: 8-13GB
- Mixtral: 25-45GB

### Bale Examples

```bash
# Standard compilation
llamafarm bale ./.llamafarm/llama3-8b --device mac-arm

# Optimized for size
llamafarm bale ./.llamafarm/llama3-8b --device raspberry-pi --optimize max --compress

# Enterprise deployment with signing
llamafarm bale ./.llamafarm/mixtral --device linux --sign --output production.bin
```

### `harvest` 🥚
Deploy and run a compiled binary. *(Placeholder - will work once baler is complete!)*

```bash
llamafarm harvest <binary-or-url> [options]

Options:
  --run                 Run immediately after deployment
  --daemon             Run as background service
  --port <number>      Override default port
  --verify             Verify binary integrity
```

## Configuration

Create a `llamafarm.yaml` file for advanced configurations:

```yaml
name: my-assistant
base_model: llama3-8b
plugins:
  - vector_search
  - voice_recognition
data:
  - path: ./company-docs
    type: knowledge
optimization:
  quantization: int8
  target_size: 2GB
```

Then build:
```bash
llamafarm build
```

## Requirements

- Node.js 18+ 
- 8GB RAM (minimum)
- 10GB free disk space

## Documentation

For full documentation, visit [https://docs.llamafarm.ai](https://docs.llamafarm.ai)

## Support

- 📖 [Documentation](https://docs.llamafarm.ai)
- 💬 [Discord Community](https://discord.gg/llamafarm-ai)
- 🐛 [Issue Tracker](https://github.com/llamafarm-ai/llamafarm/issues)

## Baler FAQ

**Q: Can I run the binary on a different OS than where I compiled it?**
A: No, you need to compile for each target platform. Use `--device` to specify the target.

**Q: How much disk space do I need?**
A: During compilation, you need ~3x the final binary size. The final binary is typically 4-8GB for 7B models.

**Q: Can I update the model without recompiling?**
A: No, the model is embedded in the binary. This ensures zero dependencies but means updates require recompilation.

**Q: Does the binary need internet access?**
A: No! Everything runs completely offline once deployed.

## License

MIT © LLaMA Farm Team 