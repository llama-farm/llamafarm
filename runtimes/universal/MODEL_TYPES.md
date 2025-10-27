# Universal Runtime - Supported Model Types

This document describes all model types supported by the Universal Runtime and their use cases.

## Overview

The Universal Runtime supports mainstream HuggingFace models from both the `transformers` and `diffusers` libraries, organized into 6 major categories:

| Model Type | Library | Use Cases | Example Models |
|------------|---------|-----------|----------------|
| **CausalLM** | transformers | Text generation, chat | GPT-2, Llama, Mistral, Qwen, Phi |
| **Encoder** | transformers | Embeddings, classification | BERT, sentence-transformers, RoBERTa |
| **Diffusion** | diffusers | Image generation | Stable Diffusion, SDXL, FLUX |
| **Vision** | transformers | Image classification, CLIP | ViT, CLIP, DINOv2, ResNet |
| **Audio** | transformers | Speech-to-text | Whisper, Wav2Vec2 |
| **Multimodal** | transformers | Vision-language tasks | BLIP, LLaVA, Florence |

---

## 1. Causal Language Models (Text Generation)

**Purpose:** Generate text continuations, chat responses, code completion

**Architecture:** Autoregressive transformers (GPT-style)

### Example Models
```python
# Small/efficient
"microsoft/phi-2"                    # 2.7B params, fast
"Qwen/Qwen2.5-0.5B-Instruct"       # 0.5B params, tiny

# Medium
"meta-llama/Llama-3.2-3B-Instruct" # 3B params
"mistralai/Mistral-7B-Instruct-v0.3" # 7B params

# Large (requires good GPU)
"meta-llama/Llama-3.1-8B-Instruct" # 8B params
```

### API Endpoint
```bash
POST /v1/chat/completions

curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/phi-2",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

---

## 2. Encoder Models (Embeddings & Classification)

**Purpose:** Convert text to vectors, classify text

**Architecture:** Bidirectional transformers (BERT-style)

### Example Models

**Embeddings:**
```python
# General purpose
"sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast
"BAAI/bge-base-en-v1.5"                   # 768-dim, better quality
"nomic-ai/nomic-embed-text-v1.5"          # 768-dim, long context

# Multilingual
"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

**Classification:**
```python
"distilbert-base-uncased-finetuned-sst-2-english"  # Sentiment
"facebook/bart-large-mnli"                          # Zero-shot
```

### API Endpoints

**Embeddings (OpenAI-compatible):**
```bash
POST /v1/embeddings

curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": ["Hello world", "How are you?"]
  }'
```

**Use Case:** Critical for RAG systems! These embeddings power semantic search.

---

## 3. Diffusion Models (Image Generation)

**Purpose:** Generate images from text prompts

**Architecture:** U-Net with diffusion process (NOT traditional transformers)

### Example Models
```python
# Standard
"stabilityai/stable-diffusion-xl-base-1.0"  # SDXL, 1024x1024
"stabilityai/stable-diffusion-2-1"          # SD 2.1, 768x768

# Fast/Turbo
"stabilityai/sdxl-turbo"                     # 1-4 steps, fast
"sd-turbo"                                   # Ultra-fast

# Specialized
"runwayml/stable-diffusion-inpainting"      # Inpainting
"black-forest-labs/FLUX.1-dev"              # Latest, high quality
```

### API Endpoints

**Generation:**
```bash
POST /v1/images/generations

curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a serene mountain landscape at sunset",
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "size": "1024x1024",
    "n": 1,
    "num_inference_steps": 50,
    "guidance_scale": 7.5
  }'
```

**Inpainting:**
```bash
POST /v1/images/edits
```

**Image-to-Image:**
```bash
POST /v1/images/variations
```

---

## 4. Vision Models (Image Understanding)

**Purpose:** Classify images, extract features, zero-shot classification

**Architecture:** Vision transformers (ViT) or CNN-based

### Example Models
```python
# Classification
"google/vit-base-patch16-224"              # ViT, 224x224
"microsoft/resnet-50"                       # ResNet-50

# Zero-shot (CLIP)
"openai/clip-vit-base-patch32"             # CLIP base
"openai/clip-vit-large-patch14"            # CLIP large

# Feature extraction
"facebook/dinov2-base"                      # DINOv2, self-supervised
```

### API Endpoints

**Classification:**
```bash
POST /v1/vision/classify

curl -X POST http://localhost:11540/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/vit-base-patch16-224",
    "images": ["<base64_image>"],
    "top_k": 5
  }'
```

**Zero-shot (CLIP):**
```bash
POST /v1/vision/clip

curl -X POST http://localhost:11540/v1/vision/clip \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/clip-vit-base-patch32",
    "images": ["<base64_image>"],
    "candidate_labels": ["dog", "cat", "bird", "car"]
  }'
```

---

## 5. Audio Models (Speech-to-Text)

**Purpose:** Transcribe speech, translate audio

**Architecture:** Encoder-decoder transformers (Whisper)

### Example Models
```python
# Whisper family (recommended)
"openai/whisper-tiny"                       # 39M params, fast
"openai/whisper-base"                       # 74M params
"openai/whisper-small"                      # 244M params
"openai/whisper-medium"                     # 769M params
"openai/whisper-large-v3"                   # 1.5B params, best

# Distilled (faster)
"distil-whisper/distil-large-v3"           # 50% faster, similar quality
```

### API Endpoints

**Transcription (OpenAI-compatible):**
```bash
POST /v1/audio/transcriptions

curl -X POST http://localhost:11540/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d '{
    "file": "<base64_audio>",
    "model": "openai/whisper-large-v3",
    "language": "en",
    "response_format": "json"
  }'
```

**Translation (to English):**
```bash
POST /v1/audio/translations

curl -X POST http://localhost:11540/v1/audio/translations \
  -H "Content-Type: application/json" \
  -d '{
    "file": "<base64_audio>",
    "model": "openai/whisper-large-v3"
  }'
```

---

## 6. Multimodal Models (Vision-Language)

**Purpose:** Image captioning, visual question answering, visual chat

**Architecture:** Combined vision + language transformers

### Example Models
```python
# Image captioning
"Salesforce/blip-image-captioning-base"    # BLIP base
"Salesforce/blip-image-captioning-large"   # BLIP large

# Visual QA
"Salesforce/blip-vqa-base"                  # BLIP VQA
"dandelin/vilt-b32-finetuned-vqa"          # ViLT VQA

# Visual chat (advanced)
"llava-hf/llava-1.5-7b-hf"                 # LLaVA 7B
"microsoft/Florence-2-base"                 # Florence-2 (unified)
```

### API Endpoints

**Image Captioning:**
```bash
POST /v1/multimodal/caption

curl -X POST http://localhost:11540/v1/multimodal/caption \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Salesforce/blip-image-captioning-base",
    "image": "<base64_image>",
    "max_length": 50
  }'
```

**Visual Question Answering:**
```bash
POST /v1/multimodal/vqa

curl -X POST http://localhost:11540/v1/multimodal/vqa \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Salesforce/blip-vqa-base",
    "image": "<base64_image>",
    "question": "What color is the car?"
  }'
```

---

## Quickstart Guide

### 1. Start the server
```bash
cd runtimes/universal
bash start.sh
```

### 2. Try each model type

**Text Generation:**
```bash
curl http://localhost:11540/v1/chat/completions \
  -d '{"model":"microsoft/phi-2","messages":[{"role":"user","content":"Hi!"}]}'
```

**Embeddings:**
```bash
curl http://localhost:11540/v1/embeddings \
  -d '{"model":"sentence-transformers/all-MiniLM-L6-v2","input":"Hello"}'
```

**Image Generation:**
```bash
curl http://localhost:11540/v1/images/generations \
  -d '{"prompt":"sunset","model":"stabilityai/stable-diffusion-xl-base-1.0"}'
```

---

## Model Selection Guide

### For RAG Systems
- **Embeddings:** `BAAI/bge-base-en-v1.5` or `nomic-ai/nomic-embed-text-v1.5`
- **Text Generation:** `Qwen/Qwen2.5-3B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.3`

### For Chat Applications
- **Best Quality:** `meta-llama/Llama-3.1-8B-Instruct`
- **Best Speed:** `Qwen/Qwen2.5-0.5B-Instruct` or `microsoft/phi-2`

### For Image Tasks
- **Generation:** `stabilityai/stable-diffusion-xl-base-1.0` (quality) or `stabilityai/sdxl-turbo` (speed)
- **Understanding:** `openai/clip-vit-base-patch32` (zero-shot) or `google/vit-base-patch16-224` (classification)

### For Audio
- **Best:** `openai/whisper-large-v3`
- **Fastest:** `openai/whisper-tiny` or `distil-whisper/distil-large-v3`

### For Multimodal
- **Captioning:** `Salesforce/blip-image-captioning-large`
- **VQA:** `Salesforce/blip-vqa-base`
- **Advanced:** `llava-hf/llava-1.5-7b-hf`

---

## Hardware Requirements

| Model Type | Min RAM | Recommended | GPU VRAM | Notes |
|------------|---------|-------------|----------|-------|
| CausalLM (small) | 4GB | 8GB | 4GB+ | Phi-2, Qwen-0.5B |
| CausalLM (medium) | 8GB | 16GB | 8GB+ | Llama-3B, Mistral-7B |
| Encoder | 2GB | 4GB | 2GB+ | Fast inference |
| Diffusion | 8GB | 16GB | 8GB+ | SDXL needs 8GB+ VRAM |
| Vision | 2GB | 4GB | 2GB+ | Fast |
| Audio | 4GB | 8GB | 4GB+ | Whisper-large |
| Multimodal | 8GB | 16GB | 8GB+ | LLaVA needs 16GB+ |

---

## Next Steps

1. **Try the examples** in each section above
2. **Browse HuggingFace** for more models: https://huggingface.co/models
3. **Check the server logs** to see model loading progress
4. **Use `/health` endpoint** to see loaded models and device info

```bash
curl http://localhost:11540/health
```
