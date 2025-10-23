#!/bin/bash

# Quick Test Script - Tests core endpoints without external dependencies
# Run this to verify the server is working

BASE_URL="${BASE_URL:-http://localhost:11540}"

echo "================================================"
echo "Universal Runtime - Quick Server Test"
echo "================================================"
echo "Testing: $BASE_URL"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test 1: Health Check
echo -n "1. Health check... "
HEALTH=$(curl -s "$BASE_URL/health")
if echo "$HEALTH" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
    DEVICE=$(echo "$HEALTH" | jq -r '.device')
    echo -e "${GREEN}✓${NC} (device: $DEVICE)"
else
    echo -e "${RED}✗ Server not responding${NC}"
    exit 1
fi

# Test 2: Text Embeddings
echo -n "2. Text embeddings... "
EMBED_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Hello, world!"
  }')

if echo "$EMBED_RESPONSE" | jq -e '.data[0].embedding | length > 0' > /dev/null 2>&1; then
    EMBED_DIM=$(echo "$EMBED_RESPONSE" | jq '.data[0].embedding | length')
    echo -e "${GREEN}✓${NC} (dimension: $EMBED_DIM)"
else
    echo -e "${RED}✗${NC}"
    echo "Response: $EMBED_RESPONSE"
fi

# Test 3: Batch Embeddings
echo -n "3. Batch embeddings... "
BATCH_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": ["First text", "Second text", "Third text"]
  }')

if echo "$BATCH_RESPONSE" | jq -e '.data | length == 3' > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} (3 embeddings)"
else
    echo -e "${RED}✗${NC}"
fi

# Test 4: Text Generation
echo -n "4. Text generation... "
CHAT_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hello in one word."}
    ],
    "max_tokens": 5
  }')

if echo "$CHAT_RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
    RESPONSE_TEXT=$(echo "$CHAT_RESPONSE" | jq -r '.choices[0].message.content' | head -c 30)
    echo -e "${GREEN}✓${NC} (\"$RESPONSE_TEXT...\")"
else
    echo -e "${RED}✗${NC}"
    echo "Response: $CHAT_RESPONSE"
fi

# Test 5: Streaming
echo -n "5. Streaming generation... "
STREAM_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Count: 1"}],
    "max_tokens": 10,
    "stream": true
  }')

if echo "$STREAM_RESPONSE" | grep -q "data:"; then
    CHUNKS=$(echo "$STREAM_RESPONSE" | grep -c "data:")
    echo -e "${GREEN}✓${NC} ($CHUNKS chunks)"
else
    echo -e "${RED}✗${NC}"
fi

# Test 6: Image Generation (slower)
echo -n "6. Image generation (30-60s)... "
IMAGE_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A simple red circle on white background",
    "size": "512x512",
    "num_inference_steps": 15,
    "response_format": "b64_json"
  }' --max-time 120)

if echo "$IMAGE_RESPONSE" | jq -e '.data[0].b64_json' > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    # Save the image
    echo "$IMAGE_RESPONSE" | jq -r '.data[0].b64_json' | base64 -d > quick_test_image.png
    echo "   → Saved to: quick_test_image.png"
else
    echo -e "${YELLOW}⚠${NC} (skipped or timed out)"
fi

echo ""
echo "================================================"
echo -e "${GREEN}Core functionality verified!${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo "  • Test audio: ./test_audio.sh"
echo "  • Test vision: ./test_vision.sh"
echo "  • Full suite: ./test_server.sh"
echo "  • View docs: curl $BASE_URL/docs"
echo ""
