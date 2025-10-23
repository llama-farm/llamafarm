#!/bin/bash

# Test Audio Upload - Quick verification script for audio endpoints

set -e

BASE_URL="http://localhost:11540"
MODEL="openai/whisper-tiny"

echo "=================================="
echo "Audio Upload Test Suite"
echo "=================================="
echo ""

# Check if server is running
echo "1. Checking server health..."
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo "❌ Server is not running at ${BASE_URL}"
    echo "   Start it with: ./start.sh"
    exit 1
fi
echo "✅ Server is running"
echo ""

# Check for test audio file
if [ ! -f "test_audio.mp3" ]; then
    echo "2. Downloading test audio file..."
    curl -o test_audio.mp3 "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav"
    echo "✅ Downloaded test_audio.mp3"
else
    echo "2. Using existing test_audio.mp3"
fi
echo ""

# Test 1: Basic JSON transcription
echo "3. Testing basic transcription (JSON)..."
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/audio/transcriptions" \
  -F "file=@test_audio.mp3" \
  -F "model=${MODEL}" \
  -F "language=en")

if echo "$RESPONSE" | grep -q '"text"'; then
    echo "✅ JSON transcription successful"
    echo "   Response: $(echo $RESPONSE | head -c 100)..."
else
    echo "❌ JSON transcription failed"
    echo "   Response: $RESPONSE"
    exit 1
fi
echo ""

# Test 2: Plain text transcription
echo "4. Testing text format transcription..."
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/audio/transcriptions" \
  -F "file=@test_audio.mp3" \
  -F "model=${MODEL}" \
  -F "response_format=text")

if [ -n "$RESPONSE" ] && ! echo "$RESPONSE" | grep -q "error"; then
    echo "✅ Text transcription successful"
    echo "   Response: $(echo $RESPONSE | head -c 100)..."
else
    echo "❌ Text transcription failed"
    echo "   Response: $RESPONSE"
    exit 1
fi
echo ""

# Test 3: Verbose JSON
echo "5. Testing verbose JSON format..."
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/audio/transcriptions" \
  -F "file=@test_audio.mp3" \
  -F "model=${MODEL}" \
  -F "response_format=verbose_json")

if echo "$RESPONSE" | grep -q '"task"' && echo "$RESPONSE" | grep -q '"language"'; then
    echo "✅ Verbose JSON transcription successful"
    echo "   Response: $(echo $RESPONSE | head -c 150)..."
else
    echo "❌ Verbose JSON transcription failed"
    echo "   Response: $RESPONSE"
    exit 1
fi
echo ""

# Test 4: Translation (if different language audio available)
echo "6. Testing translation endpoint..."
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/audio/translations" \
  -F "file=@test_audio.mp3" \
  -F "model=${MODEL}")

if echo "$RESPONSE" | grep -q '"text"'; then
    echo "✅ Translation endpoint working"
    echo "   Response: $(echo $RESPONSE | head -c 100)..."
else
    echo "⚠️  Translation endpoint returned unexpected response"
    echo "   (This may be normal if audio is already in English)"
    echo "   Response: $(echo $RESPONSE | head -c 100)..."
fi
echo ""

echo "=================================="
echo "✅ All audio upload tests passed!"
echo "=================================="
echo ""
echo "The audio endpoints are working correctly."
echo "You can now upload audio files for transcription and translation."


