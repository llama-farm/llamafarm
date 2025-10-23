#!/bin/bash

# Universal Runtime Server Test Suite
# Tests all major endpoints with realistic examples

set -e  # Exit on error

BASE_URL="http://localhost:11540"
FAILED_TESTS=0
PASSED_TESTS=0

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
function test_endpoint() {
    local name="$1"
    local command="$2"

    echo -e "\n${YELLOW}Testing: $name${NC}"

    if eval "$command"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED_TESTS++))
    fi
}

function check_response() {
    local response="$1"
    local expected_field="$2"

    if echo "$response" | jq -e "$expected_field" > /dev/null 2>&1; then
        return 0
    else
        echo "Response: $response"
        return 1
    fi
}

# Start tests
echo "================================================"
echo "Universal Runtime Server Test Suite"
echo "================================================"
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Health Check
test_endpoint "Health Check" \
    "curl -s $BASE_URL/health | jq -e '.status == \"healthy\"'"

# Test 2: Server Info
test_endpoint "Server Info" \
    "curl -s $BASE_URL/ | jq -e '.name'"

# Test 3: Text Generation (Chat Completions)
test_endpoint "Text Generation" \
    "curl -s -X POST $BASE_URL/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{
        \"model\": \"Qwen/Qwen2.5-0.5B-Instruct\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}],
        \"max_tokens\": 10
      }' | jq -e '.choices[0].message.content'"

# Test 4: Streaming Text Generation
test_endpoint "Streaming Text Generation" \
    "curl -s -X POST $BASE_URL/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{
        \"model\": \"Qwen/Qwen2.5-0.5B-Instruct\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Count to 3\"}],
        \"max_tokens\": 20,
        \"stream\": true
      }' | grep -q 'data:'"

# Test 5: Embeddings (Single)
test_endpoint "Single Text Embedding" \
    "curl -s -X POST $BASE_URL/v1/embeddings \
      -H 'Content-Type: application/json' \
      -d '{
        \"model\": \"sentence-transformers/all-MiniLM-L6-v2\",
        \"input\": \"Hello world\"
      }' | jq -e '.data[0].embedding | length > 0'"

# Test 6: Embeddings (Batch)
test_endpoint "Batch Text Embeddings" \
    "curl -s -X POST $BASE_URL/v1/embeddings \
      -H 'Content-Type: application/json' \
      -d '{
        \"model\": \"sentence-transformers/all-MiniLM-L6-v2\",
        \"input\": [\"Hello\", \"World\"]
      }' | jq -e '.data | length == 2'"

# Test 7: Image Generation
echo -e "\n${YELLOW}Testing: Image Generation (this may take 30-60 seconds)${NC}"
RESPONSE=$(curl -s -X POST $BASE_URL/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A simple red circle",
    "n": 1,
    "size": "512x512",
    "num_inference_steps": 20,
    "response_format": "b64_json"
  }')

if echo "$RESPONSE" | jq -e '.data[0].b64_json' > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED_TESTS++))

    # Save the image
    echo "$RESPONSE" | jq -r '.data[0].b64_json' | base64 -d > test_generated_image.png
    echo "  Image saved to: test_generated_image.png"
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "Response: $RESPONSE"
    ((FAILED_TESTS++))
fi

# Test 8: Image Generation with Seed
test_endpoint "Image Generation with Seed" \
    "curl -s -X POST $BASE_URL/v1/images/generations \
      -H 'Content-Type: application/json' \
      -d '{
        \"model\": \"stabilityai/stable-diffusion-2-1-base\",
        \"prompt\": \"A blue square\",
        \"seed\": 42,
        \"size\": \"512x512\",
        \"num_inference_steps\": 15
      }' | jq -e '.data[0].b64_json or .data[0].url'"

# Test 9: Audio Transcription (requires audio file)
if [ -f "test_audio.mp3" ] || [ -f "test_audio.wav" ]; then
    AUDIO_FILE=$(ls test_audio.* | head -1)
    test_endpoint "Audio Transcription" \
        "curl -s -X POST $BASE_URL/v1/audio/transcriptions \
          -F 'file=@$AUDIO_FILE' \
          -F 'model=openai/whisper-tiny' | jq -e '.text'"
else
    echo -e "\n${YELLOW}Skipping: Audio Transcription (no test_audio.mp3/wav file)${NC}"
    echo "  To test audio: curl -o test_audio.mp3 'https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav'"
fi

# Test 10: Image Classification (requires test image)
if [ -f "test_image.jpg" ] || [ -f "test_image.png" ]; then
    TEST_IMAGE=$(ls test_image.* | head -1)
    IMAGE_BASE64=$(base64 -i "$TEST_IMAGE" 2>/dev/null || base64 -w 0 "$TEST_IMAGE")

    test_endpoint "Image Classification" \
        "curl -s -X POST $BASE_URL/v1/vision/classify \
          -H 'Content-Type: application/json' \
          -d '{
            \"model\": \"google/vit-base-patch16-224\",
            \"image\": \"$IMAGE_BASE64\"
          }' | jq -e '.predictions[0].label'"
else
    echo -e "\n${YELLOW}Skipping: Image Classification (no test_image.jpg/png file)${NC}"
    echo "  To test vision: curl -o test_image.jpg 'https://picsum.photos/512/512'"
fi

# Test 11: Multimodal Caption (requires test image)
if [ -f "test_image.jpg" ] || [ -f "test_image.png" ]; then
    TEST_IMAGE=$(ls test_image.* | head -1)
    IMAGE_BASE64=$(base64 -i "$TEST_IMAGE" 2>/dev/null || base64 -w 0 "$TEST_IMAGE")

    test_endpoint "Image Captioning" \
        "curl -s -X POST $BASE_URL/v1/multimodal/caption \
          -H 'Content-Type: application/json' \
          -d '{
            \"model\": \"Salesforce/blip-image-captioning-base\",
            \"image\": \"$IMAGE_BASE64\"
          }' | jq -e '.caption'"
fi

# Summary
echo ""
echo "================================================"
echo "Test Summary"
echo "================================================"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo "Total: $((PASSED_TESTS + FAILED_TESTS))"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
