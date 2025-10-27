# Manual Testing Examples - Copy & Paste Commands

Ready-to-use curl commands for testing the Universal Runtime server.

## Setup

1. **Start the server in one terminal:**
   ```bash
   cd runtimes/universal
   uv run uvicorn server:app --host 0.0.0.0 --port 11540 --reload
   ```

2. **In another terminal, run these commands:**

---

## Quick Health Check

```bash
curl http://localhost:11540/health | jq
```

Expected output:
```json
{
  "status": "healthy",
  "device": "mps",
  "torch_version": "2.x.x",
  "available_memory": "..."
}
```

---

## 1. TEXT GENERATION (CausalLM)

### Simple Question
```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50
  }' | jq
```

### With System Prompt
```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a pirate. Always respond like a pirate."},
      {"role": "user", "content": "How are you today?"}
    ],
    "max_tokens": 100,
    "temperature": 0.8
  }' | jq
```

### Streaming Response
```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Write a haiku about coding."}
    ],
    "max_tokens": 100,
    "stream": true
  }'
```

---

## 2. TEXT EMBEDDINGS (EncoderModel)

### Single Embedding
```bash
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "The quick brown fox jumps over the lazy dog."
  }' | jq '.data[0].embedding[0:10]'
```

### Batch Embeddings (for RAG)
```bash
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": [
      "Document 1: Machine learning is a subset of AI.",
      "Document 2: Neural networks are inspired by the brain.",
      "Document 3: Python is widely used for data science."
    ]
  }' | jq '.data[].index'
```

### Query Embedding (for semantic search)
```bash
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "What is machine learning?"
  }' | jq '.usage'
```

---

## 3. IMAGE GENERATION (DiffusionModel)

### Basic Generation
```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A serene mountain landscape at sunset, photorealistic",
    "size": "512x512",
    "num_inference_steps": 25
  }' | jq -r '.data[0].b64_json' | base64 -d > mountain.png && echo "Image saved to mountain.png"
```

### With Negative Prompt
```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A cute robot playing guitar, digital art",
    "negative_prompt": "blurry, distorted, low quality, watermark",
    "size": "512x512",
    "guidance_scale": 7.5,
    "num_inference_steps": 30
  }' | jq -r '.data[0].b64_json' | base64 -d > robot.png && open robot.png
```

### Multiple Images
```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A futuristic city skyline at night",
    "n": 2,
    "size": "512x512"
  }' | jq '.data[].revised_prompt'
```

### With Seed (Reproducible)
```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A magical glowing forest with fireflies",
    "seed": 42,
    "size": "512x512",
    "num_inference_steps": 30
  }' | jq -r '.data[0].b64_json' | base64 -d > forest_seed42.png
```

---

## 4. AUDIO TRANSCRIPTION (AudioModel)

### First, download test audio:
```bash
curl -o test_audio.mp3 "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav"
```

### Transcribe Audio File
```bash
curl -X POST http://localhost:11540/v1/audio/transcriptions \
  -F "file=@test_audio.mp3" \
  -F "model=openai/whisper-tiny" \
  -F "language=en" | jq
```

### With Timestamps
```bash
curl -X POST http://localhost:11540/v1/audio/transcriptions \
  -F "file=@test_audio.mp3" \
  -F "model=openai/whisper-tiny" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=segment" | jq
```

### Translate to English
```bash
curl -X POST http://localhost:11540/v1/audio/translations \
  -F "file=@test_audio.mp3" \
  -F "model=openai/whisper-tiny" | jq
```

---

## 5. IMAGE CLASSIFICATION (VisionModel)

### First, download test image:
```bash
curl -o cat.jpg "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512"
```

### Classify Image
```bash
IMAGE_BASE64=$(base64 -i cat.jpg)

curl -X POST http://localhost:11540/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"google/vit-base-patch16-224\",
    \"image\": \"$IMAGE_BASE64\"
  }" | jq '.predictions[0:5]'
```

### CLIP Zero-Shot Classification
```bash
IMAGE_BASE64=$(base64 -i cat.jpg)

curl -X POST http://localhost:11540/v1/vision/clip \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"openai/clip-vit-base-patch32\",
    \"image\": \"$IMAGE_BASE64\",
    \"labels\": [\"cat\", \"dog\", \"bird\", \"car\", \"tree\"]
  }" | jq
```

---

## 6. MULTIMODAL (Vision-Language)

### Image Captioning
```bash
IMAGE_BASE64=$(base64 -i cat.jpg)

curl -X POST http://localhost:11540/v1/multimodal/caption \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$IMAGE_BASE64\"
  }" | jq
```

### Visual Question Answering (VQA)
```bash
IMAGE_BASE64=$(base64 -i cat.jpg)

curl -X POST http://localhost:11540/v1/multimodal/vqa \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$IMAGE_BASE64\",
    \"question\": \"What color is this animal?\"
  }" | jq
```

### Multiple Questions on Same Image
```bash
IMAGE_BASE64=$(base64 -i cat.jpg)

# Question 1
curl -s -X POST http://localhost:11540/v1/multimodal/vqa \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$IMAGE_BASE64\",
    \"question\": \"What animal is this?\"
  }" | jq -r '.answer'

# Question 2
curl -s -X POST http://localhost:11540/v1/multimodal/vqa \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$IMAGE_BASE64\",
    \"question\": \"Is this indoors or outdoors?\"
  }" | jq -r '.answer'
```

---

## 7. COMPLETE WORKFLOW EXAMPLES

### Example A: RAG Pipeline

**Step 1: Embed Documents**
```bash
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": [
      "The Eiffel Tower is located in Paris, France.",
      "Paris is the capital and largest city of France.",
      "The Louvre Museum is one of the world largest art museums."
    ]
  }' > documents.json

echo "Documents embedded and saved to documents.json"
```

**Step 2: Embed Query**
```bash
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Where is the Eiffel Tower?"
  }' > query.json

echo "Query embedded and saved to query.json"
```

**Step 3: Use Retrieved Context in Chat**
```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {
        "role": "system",
        "content": "Answer based on this context: The Eiffel Tower is located in Paris, France. Paris is the capital of France."
      },
      {"role": "user", "content": "Where is the Eiffel Tower and what is special about its location?"}
    ],
    "max_tokens": 150
  }' | jq -r '.choices[0].message.content'
```

### Example B: Content Creation Pipeline

**Step 1: Generate Image**
```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A cozy coffee shop interior with warm lighting",
    "size": "512x512",
    "num_inference_steps": 25
  }' | jq -r '.data[0].b64_json' | base64 -d > coffee_shop.png

echo "Generated image: coffee_shop.png"
```

**Step 2: Caption the Image**
```bash
IMAGE_BASE64=$(base64 -i coffee_shop.png)

CAPTION=$(curl -s -X POST http://localhost:11540/v1/multimodal/caption \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Salesforce/blip-image-captioning-base\",
    \"image\": \"$IMAGE_BASE64\"
  }" | jq -r '.caption')

echo "Caption: $CAPTION"
```

**Step 3: Generate Story from Caption**
```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen/Qwen2.5-0.5B-Instruct\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"Write a short story (2-3 sentences) inspired by this scene: $CAPTION\"
      }
    ],
    \"max_tokens\": 150
  }" | jq -r '.choices[0].message.content'
```

### Example C: Multimodal Analysis

**Analyze an image in multiple ways:**
```bash
IMAGE_BASE64=$(base64 -i cat.jpg)

echo "1. Caption:"
curl -s -X POST http://localhost:11540/v1/multimodal/caption \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"Salesforce/blip-image-captioning-base\", \"image\": \"$IMAGE_BASE64\"}" \
  | jq -r '.caption'

echo -e "\n2. Classification:"
curl -s -X POST http://localhost:11540/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"google/vit-base-patch16-224\", \"image\": \"$IMAGE_BASE64\"}" \
  | jq -r '.predictions[0] | "\(.label) (\(.score * 100 | round)%)"'

echo -e "\n3. Questions:"
curl -s -X POST http://localhost:11540/v1/multimodal/vqa \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"Salesforce/blip-image-captioning-base\", \"image\": \"$IMAGE_BASE64\", \"question\": \"What is the main subject?\"}" \
  | jq -r '"Q: What is the main subject? A: " + .answer'
```

---

## Performance Testing

### Latency Test (Embeddings)
```bash
time for i in {1..10}; do
  curl -s -X POST http://localhost:11540/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "input": "Test embedding"
    }' > /dev/null
done
```

### Concurrent Requests
```bash
for i in {1..5}; do
  curl -X POST http://localhost:11540/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "input": "Concurrent test '$i'"
    }' &
done
wait
echo "All concurrent requests completed"
```

---

## Debugging Commands

### Check Server Status
```bash
curl -s http://localhost:11540/ | jq
```

### List Loaded Models
```bash
curl -s http://localhost:11540/models | jq
```

### Get Model Info
```bash
curl -s http://localhost:11540/models/sentence-transformers/all-MiniLM-L6-v2 | jq
```

### Health Check with Details
```bash
curl -s http://localhost:11540/health | jq
```

### Test with Verbose Output
```bash
curl -v -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Debug test"
  }'
```

---

## Tips

1. **Save responses for debugging:**
   ```bash
   curl ... > response.json
   cat response.json | jq
   ```

2. **Extract specific fields:**
   ```bash
   curl ... | jq -r '.choices[0].message.content'
   ```

3. **Pretty print JSON:**
   ```bash
   curl ... | jq .
   ```

4. **Time requests:**
   ```bash
   time curl ...
   ```

5. **Silent mode (no progress):**
   ```bash
   curl -s ...
   ```

6. **Save generated images:**
   ```bash
   curl ... | jq -r '.data[0].b64_json' | base64 -d > output.png
   ```

---

## Common Issues

### Issue: "Connection refused"
**Fix:** Make sure server is running:
```bash
ps aux | grep uvicorn
```

### Issue: "Model not found"
**Fix:** Check model ID spelling:
```bash
curl http://localhost:11540/models | jq
```

### Issue: "Out of memory"
**Fix:** Use smaller models or CPU:
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU
```

### Issue: Base64 encoding
**macOS vs Linux difference:**
```bash
# macOS
base64 -i file.jpg

# Linux
base64 -w 0 file.jpg
```

---

## Next Steps

- Run automated tests: `./quick_test.sh`
- Full test suite: `./test_server.sh`
- View API docs: http://localhost:11540/docs
- See production guide: `PRODUCTION_READY_CHECKLIST.md`
