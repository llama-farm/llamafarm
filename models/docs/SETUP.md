# 🚀 LlamaFarm Models - Complete Setup Guide

## Overview

LlamaFarm Models features a **fully automatic setup system** that installs everything you need when you run any command. This ensures that any user can pull down the repository and immediately start using it without manual configuration.

## 🎯 Key Principle: Everything is Automatic

**You don't need to manually install anything!** The system automatically:
- Detects missing components
- Downloads and builds required tools
- Installs Python dependencies
- Sets up conversion utilities
- Configures local model servers

This happens transparently when you run **any command** in the `/models` directory.

## 📦 Auto-Setup Components

### 1. Python Dependencies
Automatically installed via `uv` when you run any command:
- PyTorch and transformers for model training
- PEFT for efficient fine-tuning
- Datasets and accelerate for training pipelines
- mistral-common for model conversion
- numpy, sentencepiece, protobuf for GGUF conversion

### 2. GGUF Converter (llama.cpp)
Automatically installed when you use conversion commands:
- Downloads llama.cpp repository
- Builds with CMake (replacing deprecated Makefile)
- Installs Python dependencies for conversion
- Persists between commands - installs once, uses forever

### 3. Ollama Integration
Automatically configured when using local models:
- Detects if Ollama is installed
- Provides installation instructions if missing
- Downloads required models on first use
- Manages model lifecycle

## 🚀 Quick Start

### Option 1: Run the Setup Script
```bash
cd models
uv run python setup.py
```

This will:
1. Check Python version (3.10+ required)
2. Verify uv is installed
3. Install all Python dependencies
4. Set up environment file from template
5. Install all components (Ollama, GGUF converter)
6. Test with mock model
7. Provide quick start commands

### Option 2: Just Start Using It!
```bash
cd models

# Any command auto-installs what it needs:
uv run python cli.py chat                    # Auto-sets up chat components
uv run python cli.py train --help            # Auto-sets up training components
uv run python cli.py convert --help          # Auto-sets up converters
uv run python demos/demo1_cloud_fallback.py  # Auto-sets up everything for demo
```

## 🔧 How Auto-Setup Works

### CLI Commands
Every CLI command checks for required components:

```python
# In cli.py convert_command:
if args.format in ["gguf", "ollama"]:
    from components.converters.llama_cpp_installer import get_llama_cpp_installer
    installer = get_llama_cpp_installer()
    if not installer.is_installed():
        print_info("Setting up conversion tools...")
        if not installer.install():
            print_error("Failed to set up conversion tools")
            sys.exit(1)
```

### Demo Scripts
All demos auto-setup at the start:

```python
# In every demo:
print("📦 Checking and installing requirements...")
success, stdout, _ = run_cli_command(
    'uv run python cli.py setup demos/strategies.yaml --auto --verbose',
    "Setting up components for this demo"
)
```

### Component Detection
The system intelligently detects what's installed:

```python
# llama_cpp_installer.py checks for:
- convert_hf_to_gguf.py (conversion script)
- build/bin/llama-quantize (quantization binary)
- CMake build system
- Python dependencies (mistral-common, etc.)
```

## 📋 Component Installation Details

### GGUF Converter Setup
Location: `components/converters/llama_cpp_installer.py`

1. **First Run**: Downloads and builds llama.cpp
   - Clones repository to `~/.llamafarm/tools/llama.cpp/`
   - Builds with CMake (supports macOS Metal, Linux CUDA, Windows)
   - Installs Python dependencies

2. **Subsequent Runs**: Detects existing installation
   - Checks for conversion scripts
   - Verifies quantization binaries
   - No reinstall needed

### Training Components
Location: `components/trainers/lora_trainer.py`

Automatically installs when running training commands:
- PyTorch with appropriate backend (CPU/CUDA/MPS)
- Transformers and PEFT libraries
- Datasets for data loading
- Accelerate for distributed training

### Ollama Setup
Location: `components/model_app.py`

1. Checks if Ollama is running
2. Provides installation instructions if missing
3. Downloads models on first use
4. Manages model lifecycle

## 🛠️ Manual Component Management

While everything is automatic, you can manually manage components:

### Check Component Status
```bash
# See what's installed
uv run python cli.py setup demos/strategies.yaml --verify-only --verbose

# Check specific strategy
uv run python cli.py info --strategy demo3_training
```

### Force Reinstall
```bash
# Remove llama.cpp to force reinstall
rm -rf ~/.llamafarm/tools/llama.cpp

# Next conversion will reinstall
uv run python cli.py convert model.safetensors model.gguf --format gguf
```

### Install Everything Upfront
```bash
# Install all components for all strategies
uv run python cli.py setup demos/strategies.yaml --auto --verbose
```

## 🐛 Troubleshooting

### Issue: "Makefile build is deprecated"
**Solution**: Already fixed! The system now uses CMake automatically.

### Issue: "ModuleNotFoundError: No module named 'mistral_common'"
**Solution**: Already fixed! Dependencies are auto-installed.

### Issue: "llama.cpp not found. Installing..."
**Solution**: This is normal on first run. It persists after installation.

### Issue: Conversion fails with "convert.py not found"
**Solution**: Fixed! Now looks for correct script name: `convert_hf_to_gguf.py`

### Issue: Components reinstall every time
**Solution**: Fixed! Proper detection of installed components.

## 🔍 Environment Variables

Optional configuration via `.env` file:
```bash
# API Keys (optional - for cloud models)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...

# Paths (optional - defaults work fine)
LLAMAFARM_TOOLS_DIR=~/.llamafarm/tools
LLAMAFARM_MODELS_DIR=~/.llamafarm/models
```

## 📊 Storage Locations

Default locations for installed components:
- **llama.cpp**: `~/.llamafarm/tools/llama.cpp/`
- **Downloaded models**: `~/.llamafarm/models/`
- **Fine-tuned models**: `./fine_tuned_models/`
- **Converted models**: Output path you specify

## 🚦 Testing Auto-Setup

### Test Complete Setup
```bash
# Run complete setup test
uv run python setup.py

# Test with mock model (no dependencies)
uv run python demos/demo_mock_model.py

# Test conversion setup
uv run python cli.py convert --help

# Test training setup  
uv run python cli.py train --help
```

### Verify Installation
```bash
# Check llama.cpp
ls ~/.llamafarm/tools/llama.cpp/build/bin/

# Check Python packages
uv pip list | grep -E "torch|transformers|peft"

# Check Ollama
ollama list
```

## 💡 Design Philosophy

1. **Zero Friction**: Users should never have to manually install dependencies
2. **Smart Detection**: Only install what's needed, when it's needed
3. **Persistence**: Components install once and persist between sessions
4. **Transparency**: Clear messages about what's being installed and why
5. **Resilience**: Graceful fallbacks if components can't be installed

## 🎯 Summary

The auto-setup system ensures that:
- ✅ Any user can clone and immediately use the repo
- ✅ Dependencies are installed automatically on first use
- ✅ Components persist between commands (no reinstalls)
- ✅ Everything works with a single command
- ✅ No manual configuration required

Just run any command and the system handles the rest!

## 📚 Related Documentation

- [CLI Reference](CLI.md) - All available commands
- [Strategies Guide](STRATEGIES.md) - Configuration system
- [Training Guide](TRAINING.md) - Fine-tuning models
- [Conversion Guide](CONVERSION.md) - GGUF and Ollama formats