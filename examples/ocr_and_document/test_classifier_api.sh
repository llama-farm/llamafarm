#!/bin/bash
# Test script for SetFit-based text classifier via LlamaFarm API
# Tests the /v1/ml/classifier/* endpoints with intent classification

set -e

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}/v1/ml"
MODEL_NAME="intent-classifier-test"

echo "========================================"
echo "Testing SetFit Text Classifier"
echo "via LlamaFarm API (/v1/ml proxy)"
echo "Base URL: $BASE_URL"
echo "Model: $MODEL_NAME"
echo "========================================"

# Check if LlamaFarm server is running
echo ""
echo "1. Checking LlamaFarm API health..."
if ! curl -sf "http://localhost:${PORT}/health" > /dev/null; then
    echo "   ERROR: LlamaFarm API not responding at port $PORT"
    echo "   Start with: nx start server"
    exit 1
fi
echo "   LlamaFarm API is healthy"

# Check Universal Runtime via proxy
echo ""
echo "2. Checking Universal Runtime health via proxy..."
if ! curl -sf "$BASE_URL/health" > /dev/null; then
    echo "   ERROR: Universal Runtime not responding"
    echo "   Start with: nx start universal"
    exit 1
fi
echo "   Universal Runtime is healthy"

# Training data: ~40 examples across 4 intents (10 per class)
# Intent classes: booking, cancellation, inquiry, complaint
echo ""
echo "3. Training classifier with 40 examples (10 per class)..."

TRAIN_RESPONSE=$(curl -sf -X POST "$BASE_URL/classifier/fit" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_NAME"'",
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "training_data": [
            {"text": "I need to book a flight to New York", "label": "booking"},
            {"text": "Can you reserve a hotel room for me?", "label": "booking"},
            {"text": "I want to schedule an appointment", "label": "booking"},
            {"text": "Please book me a table for dinner", "label": "booking"},
            {"text": "I would like to make a reservation", "label": "booking"},
            {"text": "Reserve two tickets for the show", "label": "booking"},
            {"text": "Book a rental car for next week", "label": "booking"},
            {"text": "I need to schedule a meeting room", "label": "booking"},
            {"text": "Can I get a booking for Saturday?", "label": "booking"},
            {"text": "Make a reservation at the restaurant", "label": "booking"},

            {"text": "I want to cancel my reservation", "label": "cancellation"},
            {"text": "Please cancel my flight booking", "label": "cancellation"},
            {"text": "I need to cancel my hotel room", "label": "cancellation"},
            {"text": "Cancel my appointment please", "label": "cancellation"},
            {"text": "I would like to cancel my order", "label": "cancellation"},
            {"text": "Please cancel the dinner reservation", "label": "cancellation"},
            {"text": "I need to cancel my subscription", "label": "cancellation"},
            {"text": "Cancel my meeting for tomorrow", "label": "cancellation"},
            {"text": "I want to cancel everything", "label": "cancellation"},
            {"text": "Please remove my booking", "label": "cancellation"},

            {"text": "What time does the flight depart?", "label": "inquiry"},
            {"text": "How much does a room cost?", "label": "inquiry"},
            {"text": "What are your business hours?", "label": "inquiry"},
            {"text": "Can you tell me about the menu?", "label": "inquiry"},
            {"text": "What services do you offer?", "label": "inquiry"},
            {"text": "How long is the wait time?", "label": "inquiry"},
            {"text": "Do you have availability tomorrow?", "label": "inquiry"},
            {"text": "What is the price for two people?", "label": "inquiry"},
            {"text": "Can you provide more information?", "label": "inquiry"},
            {"text": "Where is your location?", "label": "inquiry"},

            {"text": "I am very unhappy with the service", "label": "complaint"},
            {"text": "This is unacceptable quality", "label": "complaint"},
            {"text": "I want to file a complaint", "label": "complaint"},
            {"text": "The food was terrible", "label": "complaint"},
            {"text": "Your staff was very rude", "label": "complaint"},
            {"text": "I had a horrible experience", "label": "complaint"},
            {"text": "This is the worst service ever", "label": "complaint"},
            {"text": "I am extremely disappointed", "label": "complaint"},
            {"text": "I demand a refund immediately", "label": "complaint"},
            {"text": "This is completely unacceptable", "label": "complaint"}
        ],
        "num_iterations": 20,
        "batch_size": 16
    }')

echo "   Training response:"
echo "$TRAIN_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$TRAIN_RESPONSE"

# Check if training succeeded
if echo "$TRAIN_RESPONSE" | grep -q '"status"'; then
    echo "   Training successful!"
else
    echo "   ERROR: Training failed"
    exit 1
fi

# Extract versioned model name from response
VERSIONED_MODEL=$(echo "$TRAIN_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('versioned_name', ''))" 2>/dev/null)
if [ -z "$VERSIONED_MODEL" ]; then
    VERSIONED_MODEL="$MODEL_NAME"
fi
echo "   Versioned model: $VERSIONED_MODEL"

# Test data: 20 examples (5 per class)
echo ""
echo "4. Testing classifier with 20 new examples..."
echo "   Using: ${MODEL_NAME}-latest (resolves to $VERSIONED_MODEL)"

TEST_RESPONSE=$(curl -sf -X POST "$BASE_URL/classifier/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_NAME"'-latest",
        "texts": [
            "Book me a flight to Paris tomorrow",
            "I need a room for three nights",
            "Schedule a consultation for Monday",
            "Reserve a spot at the conference",
            "I want to book a spa treatment",

            "Cancel my upcoming trip",
            "I need to cancel the reservation I made",
            "Please cancel my account",
            "Remove my booking for the weekend",
            "I want to cancel my membership",

            "What are the check-in times?",
            "How do I get to your office?",
            "What payment methods do you accept?",
            "Is breakfast included?",
            "Do you offer student discounts?",

            "This is absolutely terrible service",
            "I want to speak to a manager",
            "The quality is very poor",
            "I am not satisfied at all",
            "Your company has let me down"
        ]
    }')

echo "   Prediction results:"
echo "$TEST_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$TEST_RESPONSE"

# Analyze results
echo ""
echo "5. Analyzing accuracy..."

# Expected labels for the 20 test examples
EXPECTED_LABELS=("booking" "booking" "booking" "booking" "booking" "cancellation" "cancellation" "cancellation" "cancellation" "cancellation" "inquiry" "inquiry" "inquiry" "inquiry" "inquiry" "complaint" "complaint" "complaint" "complaint" "complaint")

# Extract predicted labels
PREDICTED_LABELS=$(echo "$TEST_RESPONSE" | python3 -c "
import json
import sys
data = json.load(sys.stdin)
for item in data['data']:
    print(item['label'])
" 2>/dev/null)

if [ -z "$PREDICTED_LABELS" ]; then
    echo "   Could not parse predictions"
else
    # Count correct predictions
    CORRECT=0
    TOTAL=20
    i=0
    while IFS= read -r pred; do
        expected="${EXPECTED_LABELS[$i]}"
        if [ "$pred" = "$expected" ]; then
            CORRECT=$((CORRECT + 1))
        else
            echo "   Mismatch at index $i: expected '$expected', got '$pred'"
        fi
        i=$((i + 1))
    done <<< "$PREDICTED_LABELS"

    ACCURACY=$(echo "scale=1; $CORRECT * 100 / $TOTAL" | bc)
    echo ""
    echo "   Correct: $CORRECT / $TOTAL"
    echo "   Accuracy: ${ACCURACY}%"

    if [ "$CORRECT" -ge 16 ]; then
        echo "   PASSED (>= 80% accuracy)"
    else
        echo "   WARNING: Accuracy below 80% threshold"
    fi
fi

# Note: Models now autosave during fit, no explicit save endpoint needed

# List saved models
echo ""
echo "6. Listing saved classifier models..."
LIST_RESPONSE=$(curl -sf "$BASE_URL/classifier/models")
echo "$LIST_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$LIST_RESPONSE"

# Test loading the model using -latest (simulating server restart)
echo ""
echo "7. Testing model load using -latest suffix..."
LOAD_RESPONSE=$(curl -sf -X POST "$BASE_URL/classifier/load" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_NAME"'-latest"
    }')

echo "   Load response:"
echo "$LOAD_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$LOAD_RESPONSE"

# Test prediction after load
echo ""
echo "8. Testing prediction after reload..."
RELOAD_TEST=$(curl -sf -X POST "$BASE_URL/classifier/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_NAME"'-latest",
        "texts": ["I want to book a flight", "Cancel my order"]
    }')

echo "   Predictions after reload:"
echo "$RELOAD_TEST" | python3 -m json.tool 2>/dev/null || echo "$RELOAD_TEST"

# Cleanup - delete the test model (use versioned name)
echo ""
echo "9. Cleaning up test model ($VERSIONED_MODEL)..."
DELETE_RESPONSE=$(curl -sf -X DELETE "$BASE_URL/classifier/models/$VERSIONED_MODEL")
echo "   Delete response:"
echo "$DELETE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$DELETE_RESPONSE"

echo ""
echo "========================================"
echo "Classifier test complete!"
echo "========================================"
