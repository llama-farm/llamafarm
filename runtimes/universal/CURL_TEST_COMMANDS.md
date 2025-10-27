# Universal Runtime - cURL Test Commands

This document provides comprehensive cURL commands to test all endpoints of the Universal Runtime server.

## Prerequisites

1. **Start the server:**
   ```bash
   cd runtimes/universal
   uv run uvicorn server:app --host 0.0.0.0 --port 11540 --reload
   ```

2. **Verify server is running:**
   ```bash
   curl http://localhost:11540/health
   ```

Expected response:
```json
{
  "status": "healthy",
  "device": "mps",
  "available_memory": "..."
}
```

---

## 1. Text Generation (CausalLM)

### OpenAI-compatible Chat Completions

**Basic text generation:**
```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Streaming response:**
```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Count from 1 to 10."}
    ],
    "stream": true,
    "max_tokens": 50
  }'
```

**With RAG context:**
```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are an expert on French history."},
      {"role": "user", "content": "Tell me about the Eiffel Tower."}
    ],
    "max_tokens": 200,
    "temperature": 0.5
  }'
```

---

## 2. Text Embeddings (EncoderModel)

### OpenAI-compatible Embeddings

**Single text embedding:**
```bash
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Hello, world!"
  }'
```

**Batch embeddings:**
```bash
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": [
      "The quick brown fox jumps over the lazy dog.",
      "Machine learning is transforming technology.",
      "Python is a popular programming language."
    ]
  }'
```

**With normalization:**
```bash
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Normalize this embedding",
    "encoding_format": "float"
  }'
```

---

## 3. Image Generation (DiffusionModel)

### OpenAI-compatible Image Generation

> **New!** Image endpoints now support content negotiation via `Accept` header:
> - `Accept: application/json` → JSON with base64
> - `Accept: image/jpeg` → Raw JPEG bytes (**default**, smallest size)
> - `Accept: image/png` → Raw PNG bytes (lossless)
> - `Accept: image/webp` → Raw WebP bytes (modern format)
> - `Accept: image/*` or `*/*` → Raw JPEG bytes (default)
>
> See `CONTENT_NEGOTIATION_GUIDE.md` for details.

**Basic image generation (JSON response):**
```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1",
    "prompt": "A serene mountain landscape at sunset",
    "n": 1,
    "size": "512x512"
  }' \
  --output image_generation_response.json
```

**Multiple images with negative prompt:**
```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A cute robot playing guitar",
    "negative_prompt": "blurry, distorted, low quality",
    "n": 2,
    "size": "512x512",
    "guidance_scale": 7.5
  }' \
  --output robot_images.json
```

**With seed for reproducibility:**
```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A magical forest with glowing mushrooms",
    "seed": 42,
    "num_inference_steps": 30,
    "size": "512x512"
  }' \
  --output magical_forest.json
```

**Save base64 image from response:**
```bash
# Generate and save image (default response format)
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A sunset over the ocean"
  }' | jq -r '.data[0].b64_json' | base64 -d > sunset.png

# View the image
open sunset.png  # macOS
# xdg-open sunset.png  # Linux
```

**Using data URL response format:**
```bash
# Generate with data URL format (can be used directly in HTML/browsers)
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A sunset over the ocean",
    "response_format": "url"
  }' | jq -r '.data[0].url' | sed 's/^data:image\/png;base64,//' | base64 -d > sunset.png

# Or save the data URL directly for use in HTML
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A sunset over the ocean",
    "response_format": "url"
  }' | jq -r '.data[0].url' > sunset_data_url.txt

# Then use in HTML: <img src="$(cat sunset_data_url.txt)" />
```

**Get raw JPEG bytes (binary response - smallest size!):**
```bash
# Generate and save directly as JPEG (no base64 decoding, smallest size!)
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Accept: image/jpeg" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A sunset over the ocean"
  }' > sunset.jpg

# Or just use image/* which defaults to JPEG
curl ... -H "Accept: image/*" ... > sunset.jpg

# View immediately
open sunset.jpg
```

**Get PNG for lossless quality:**
```bash
# Use PNG when you need lossless or transparency
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Accept: image/png" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A diagram with text"
  }' > diagram.png
```

**Get WebP for modern browsers:**
```bash
# WebP offers great compression with transparency
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Accept: image/webp" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A modern web image"
  }' > modern.webp
```

**Process binary response with ImageMagick:**
```bash
# Generate JPEG and resize in one pipeline
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Accept: image/jpeg" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A sunset"
  }' | convert - -resize 256x256 thumbnail.jpg

# Or pipe to viewer
curl ... -H "Accept: image/jpeg" ... | open -f -a Preview  # macOS
```

---

## 4. Audio Transcription (AudioModel)

### OpenAI-compatible Audio Transcription

**Transcribe audio from file:**
```bash
# First, create a test audio file (or use an existing one)
# For testing, you can download a sample:
# curl -o test_audio.mp3 "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav"

curl -X POST http://localhost:11540/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_audio.mp3" \
  -F "model=openai/whisper-tiny" \
  -F "language=en"
```

**Transcribe with verbose output:**
```bash
# Note: timestamp_granularities parameter not yet implemented
curl -X POST http://localhost:11540/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_audio.mp3" \
  -F "model=openai/whisper-tiny" \
  -F "response_format=verbose_json"
```

**Get plain text response:**
```bash
curl -X POST http://localhost:11540/v1/audio/transcriptions \
  -F "file=@test_audio.mp3" \
  -F "model=openai/whisper-tiny" \
  -F "response_format=text"
```

**Translate audio to English:**
```bash
curl -X POST http://localhost:11540/v1/audio/translations \
  -H "Content-Type: multipart/form-data" \
  -F "file=@spanish_audio.mp3" \
  -F "model=openai/whisper-tiny"
```

---

## 5. Image Classification (VisionModel)

**Classify an image:**
```bash
# Using a base64-encoded image
IMAGE_BASE64=$(base64 -i cat.jpg)

curl -X POST http://localhost:11540/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"google/vit-base-patch16-224\",
    \"image\": \"$IMAGE_BASE64\"
  }"
```

**Batch classification:**
```bash
IMAGE1=$(base64 -i cat.jpg)
IMAGE2=$(base64 -i dog.jpg)

curl -X POST http://localhost:11540/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"google/vit-base-patch16-224\",
    \"images\": [\"$IMAGE1\", \"$IMAGE2\"]
  }"
```

**CLIP zero-shot classification:**
```bash
IMAGE_BASE64=$(base64 -i animal.jpg)

curl -X POST http://localhost:11540/v1/vision/clip \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"openai/clip-vit-base-patch32\",
    \"image\": \"$IMAGE_BASE64\",
    \"labels\": [\"cat\", \"dog\", \"bird\", \"fish\"]
  }"
```

---

## 6. Multimodal (Vision-Language)

### Image Captioning

**Generate caption:**
```bash
IMAGE_BASE64=$(base64 -i photo.jpg)

curl -X POST http://localhost:11540/v1/multimodal/caption \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$IMAGE_BASE64\"
  }"
```

### Visual Question Answering (VQA)

**Ask question about image:**
```bash
IMAGE_BASE64=$(base64 -i scene.jpg)

curl -X POST http://localhost:11540/v1/multimodal/vqa \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$IMAGE_BASE64\",
    \"question\": \"What color is the car?\"
  }"
```

**Multiple questions:**
```bash
IMAGE_BASE64=$(base64 -i street.jpg)

# Question 1
curl -X POST http://localhost:11540/v1/multimodal/vqa \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$IMAGE_BASE64\",
    \"question\": \"How many people are in the image?\"
  }"

# Question 2
curl -X POST http://localhost:11540/v1/multimodal/vqa \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$IMAGE_BASE64\",
    \"question\": \"What is the weather like?\"
  }"
```

---

## 7. Model Information & Health

**List available models:**
```bash
curl http://localhost:11540/models
```

**Get model details:**
```bash
curl http://localhost:11540/models/sentence-transformers/all-MiniLM-L6-v2
```

**Health check:**
```bash
curl http://localhost:11540/health
```

**Server info:**
```bash
curl http://localhost:11540/
```

---

## Test Workflow Examples

### Example 1: RAG Pipeline Test

```bash
# 1. Generate embeddings for documents
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": [
      "The Eiffel Tower is in Paris.",
      "Paris is the capital of France.",
      "The Louvre Museum is a famous art museum."
    ]
  }' > document_embeddings.json

# 2. Generate query embedding
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Where is the Eiffel Tower located?"
  }' > query_embedding.json

# 3. Use retrieved context in chat
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {
        "role": "system",
        "content": "Use the following context: The Eiffel Tower is in Paris. Paris is the capital of France."
      },
      {"role": "user", "content": "Where is the Eiffel Tower?"}
    ]
  }'
```

### Example 2: Multimodal Analysis

```bash
IMAGE_BASE64=$(base64 -i vacation_photo.jpg)

# 1. Caption the image
curl -X POST http://localhost:11540/v1/multimodal/caption \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$IMAGE_BASE64\"
  }" > caption.json

# 2. Classify the image
curl -X POST http://localhost:11540/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"google/vit-base-patch16-224\",
    \"image\": \"$IMAGE_BASE64\"
  }" > classification.json

# 3. Ask specific questions
curl -X POST http://localhost:11540/v1/multimodal/vqa \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$IMAGE_BASE64\",
    \"question\": \"Is this indoors or outdoors?\"
  }" > vqa_result.json
```

### Example 3: Full Content Creation Pipeline

```bash
# 1. Generate an image
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A futuristic city skyline at night",
    "size": "512x512"
  }' | jq -r '.data[0].b64_json' | base64 -d > city.png

# 2. Caption the generated image
CITY_IMAGE=$(base64 -i city.png)
curl -X POST http://localhost:11540/v1/multimodal/caption \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$CITY_IMAGE\"
  }" > generated_caption.json

# 3. Create a story based on the caption
CAPTION=$(jq -r '.caption' generated_caption.json)
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen/Qwen2.5-0.5B-Instruct\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Write a short story about: $CAPTION\"}
    ],
    \"max_tokens\": 300
  }"
```

---

## Helper Scripts

### Create test image from URL:
```bash
#!/bin/bash
# download_test_image.sh
curl -o test_image.jpg "https://picsum.photos/512/512"
echo "Test image downloaded: test_image.jpg"
```

### Test all endpoints:
```bash
#!/bin/bash
# test_all_endpoints.sh

echo "Testing Universal Runtime Endpoints..."

# 1. Health check
echo -e "\n1. Health Check:"
curl -s http://localhost:11540/health | jq

# 2. Text generation
echo -e "\n2. Text Generation:"
curl -s -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 20
  }' | jq

# 3. Embeddings
echo -e "\n3. Embeddings:"
curl -s -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Test embedding"
  }' | jq '.data[0].embedding[:5]'

echo -e "\nAll tests completed!"
```

### Performance benchmarking:
```bash
#!/bin/bash
# benchmark.sh

echo "Benchmarking embedding endpoint..."

# Run 10 requests and measure time
time for i in {1..10}; do
  curl -s -X POST http://localhost:11540/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "input": "Benchmark test"
    }' > /dev/null
done

echo "10 requests completed"
```

---

## Troubleshooting

### Server not responding:
```bash
# Check if server is running
ps aux | grep uvicorn

# Check port
lsof -i :11540

# Restart server
pkill -f uvicorn
cd runtimes/universal && uv run uvicorn server:app --host 0.0.0.0 --port 11540 --reload
```

### Model not loading:
```bash
# Check server logs
tail -f server.log

# Test specific model
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "test"}' \
  -v
```

### Base64 encoding issues:
```bash
# macOS/Linux base64 difference
# macOS: base64 -i file.jpg
# Linux: base64 -w 0 file.jpg

# Test base64 encoding
echo "test" | base64
# Should output: dGVzdAo=
```

---

## Next Steps

1. **Save these commands** to a file for easy reference
2. **Create test scripts** for automated testing
3. **Monitor performance** with the benchmark script
4. **Add authentication** if deploying publicly
5. **Set up logging** for production debugging

For production deployment, see `PRODUCTION_READY_CHECKLIST.md`.
