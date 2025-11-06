# LlamaFarm Model Catalog

This directory contains the Model Farm - a structured catalog of pre-configured model families organized by capability.

## Directory Structure

```
models/
├── schema.yaml                    # JSON schema for validation
├── text-generation/              # Language models
│   ├── llama.yaml                # Meta Llama family
│   ├── qwen.yaml                 # Qwen 2.5 family
│   ├── qwen3.yaml                # Qwen 3 family (latest)
│   ├── mistral.yaml              # Mistral AI family
│   ├── deepseek.yaml             # DeepSeek-R1 reasoning models
│   ├── granite.yaml              # IBM Granite 4.0 family
│   ├── phi.yaml                  # Microsoft Phi family
│   ├── codellama.yaml            # Meta Code Llama family
│   └── tinyllama.yaml            # TinyLlama compact model
├── image-generation/             # Future: SD, FLUX, etc.
├── image-recognition/            # Future: CLIP, ViT, etc.
├── audio/                        # Future: Whisper, etc.
└── embedding/                    # Future: BGE, nomic-embed, etc.
```

## What is the Model Catalog?

The Model Catalog is a data-driven registry of model families and their variants. Each family YAML file contains:

- **Family metadata**: Organization, license, website, strengths
- **Model variants**: Different sizes (1B, 7B, 70B, etc.) with specs
- **Provider configurations**: How to access each model via different runtimes
- **Hardware requirements**: Min RAM/VRAM, recommended GPUs
- **Benchmarks & use cases**: Performance scores and recommended applications

## Runtime Compatibility

Models support different runtimes based on their format. See [RUNTIME_COMPATIBILITY.md](./RUNTIME_COMPATIBILITY.md) for complete details.

**Quick Reference:**
- **GGUF models**: Run on Ollama, Lemonade (llamacpp backend)
- **Transformers models**: Run on Universal, Lemonade (transformers backend)
- **ONNX models**: Run on Lemonade (onnx backend)

## Provider Types

Each model variant can be accessed through multiple providers:

### 🏆 Universal (Primary)
- **Runtime**: LlamaFarm Universal Runtime
- **Format**: transformers (HuggingFace models)
- **URL**: `http://127.0.0.1:11540`
- **Download**: Auto-downloads on first use
- **Features**:
  - Works with any HuggingFace model
  - Integrated download API for Designer UI
  - Supports text, image, audio, multimodal models
  - Auto-detects hardware (MPS, CUDA, CPU)

### 🦙 Ollama
- **Runtime**: Ollama
- **Format**: gguf (quantized models)
- **URL**: `http://localhost:11434`
- **Download**: `ollama pull model:tag`
- **Features**:
  - Optimized quantized models (smaller, faster)
  - Simple command-line interface
  - CPU-friendly with GPU acceleration
  - Pre-built model library

### 🍋 Lemonade
- **Runtime**: Lemonade SDK
- **Formats**: gguf, transformers, onnx
- **URL**: `http://127.0.0.1:11534` (configurable)
- **Download**: `uv run lemonade-server-dev pull user.ModelName --checkpoint HF/Repo --recipe BACKEND`
- **Features**:
  - Hardware-aware optimization (NPU/GPU/CPU)
  - Multiple backends: llamacpp, transformers, onnx
  - Excellent performance on Apple Silicon
  - Flexible format support

## Recommended Models

Families can designate recommended models for different use cases:

```yaml
# At family level
recommended:
  - category: "Small & Fast"
    description: "Efficient models for quick responses"
    models:
      - variant_id: "qwen3:0.6b"
        priority: 1
      - variant_id: "qwen3:1.7b"
        priority: 2

  - category: "Balanced"
    description: "Best balance of performance and resource usage"
    models:
      - variant_id: "qwen3:4b"
        priority: 1
```

These recommendations appear at the top of the model selection UI.

## Model Variant Schema

Each variant includes runtime information and download commands:

```yaml
- id: qwen3:4b                           # Unique identifier
  display_name: "Qwen3 4B"               # UI display name
  description: "..."                     # Brief description
  parameters: "4b"                       # Model size
  download_size: "8 GB"                  # Approximate size
  context_window: 32768                  # Max tokens
  hardware_requirements:                 # Minimum specs
    min_ram: "12 GB"
    min_vram: "6 GB"
    recommended_gpu: ["Apple M1 Pro", "NVIDIA RTX 3060"]
  providers:                             # How to access
    universal:
      provider: universal
      runtime: universal                 # Runtime: universal, ollama, lemonade, openai
      format: transformers               # Format: transformers, gguf, onnx, api
      model_id: "Qwen/Qwen3-4B"
      base_url: "http://127.0.0.1:11540"
      download_command: "Auto-downloads from HuggingFace on first use"
      notes: "Best balance of performance and resource usage"

    ollama:
      provider: ollama
      runtime: ollama
      format: gguf                       # Ollama uses GGUF format
      model_id: "qwen3:4b"
      base_url: "http://localhost:11434"
      download_command: "ollama pull qwen3:4b"
      notes: "Optimized GGUF version"

    lemonade:
      provider: lemonade
      runtime: lemonade
      format: gguf
      backend: llamacpp                  # llamacpp, transformers, or onnx
      model_id: "user.Qwen3-4B"
      checkpoint: "unsloth/Qwen3-4B-GGUF:Q4_K_M"
      recipe: "llamacpp"
      download_command: "uv run lemonade-server-dev pull user.Qwen3-4B --checkpoint unsloth/Qwen3-4B-GGUF:Q4_K_M --recipe llamacpp"
      notes: "Recommended for most use cases. Excellent NPU performance."

  use_cases: [chat, coding, reasoning]   # Recommended uses
  benchmark_scores:                      # Optional benchmarks
    mmlu: 70.5
    humaneval: 65.2
```

## How the Designer Uses This

The LlamaFarm Designer UI:
1. Loads model catalogs at build time
2. Displays model families/variants in the "Add or change models" tab
3. Shows hardware requirements and specs for each model
4. Uses the **universal provider** for download functionality
5. Allows users to pick provider when adding to project

## Adding a New Model Family

1. **Create a new YAML file** in the appropriate capability directory:
   ```bash
   touch models/text-generation/gemma.yaml
   ```

2. **Follow the schema**:
   ```yaml
   family_id: gemma
   family_name: Gemma
   organization: Google
   description: "Google's lightweight open models"
   website: https://ai.google.dev/gemma
   license: Gemma License
   tags: [chat, coding, lightweight]
   strengths:
     - "Efficient architecture"
     - "Strong performance for size"

   variants:
     - id: gemma:2b
       display_name: "Gemma 2B"
       parameters: "2b"
       download_size: "4 GB"
       context_window: 8192
       hardware_requirements:
         min_ram: "8 GB"
         min_vram: "4 GB"
         recommended_gpu: ["Apple M1", "NVIDIA GTX 1660"]
       providers:
         universal:
           provider: universal
           model_id: "google/gemma-2b-it"
           base_url: "http://127.0.0.1:11540"
           notes: "Auto-downloads from HuggingFace"
         ollama:
           provider: ollama
           model_id: "gemma:2b"
           pull_command: "ollama pull gemma:2b"
       use_cases: [chat, lightweight-inference]
   ```

3. **Validate against schema**:
   ```bash
   # TODO: Add validation script
   ```

4. **Submit PR** with your new model family

## Guidelines

### DO:
✅ Include multiple size variants (1B, 7B, 70B, etc.)
✅ List universal provider FIRST (primary for Designer)
✅ Include accurate hardware requirements
✅ Add benchmark scores when available
✅ Use official HuggingFace model IDs
✅ Include use case tags

### DON'T:
❌ Add models without proper license information
❌ Include proprietary/closed models
❌ Guess at hardware requirements
❌ Skip provider configurations

## Future Capabilities

As we add support for other model types, we'll create:

- **image-generation/**: Stable Diffusion, FLUX, DALL-E 3, etc.
- **image-recognition/**: CLIP, ViT, DINOv2, etc.
- **audio/**: Whisper, Wav2Vec2, Bark, etc.
- **embedding/**: BGE, nomic-embed, E5, etc.

## Questions?

- Check the [schema.yaml](./schema.yaml) for full specification
- See [LlamaFarm Models Documentation](../docs/website/docs/models/)
- Review existing family YAMLs for examples

---

**Happy cataloging! 🦙🚀**
