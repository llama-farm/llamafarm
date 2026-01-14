#!/bin/bash
# Test Dataset Quality Audit (Cleanlab) endpoint
#
# This script demonstrates:
# 1. Auditing dataset for label errors
# 2. Getting label quality scores
# 3. Finding duplicate samples
#
# Usage: ./test_dataset_audit.sh [PORT]
#   PORT defaults to LF_RUNTIME_PORT from .env (fallback: 11540)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi


PORT=${1:-${LF_RUNTIME_PORT:-11540}}
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Dataset Quality Audit (Cleanlab) Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check health
echo -e "${YELLOW}Checking server health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# ============================================================================
# Test 1: Audit Dataset with Label Errors
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Detect Label Errors${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Generate dataset with intentional label errors
DATASET=$(python3 -c "
import json
import random
random.seed(42)

n_samples = 50
n_classes = 3

# True labels
true_labels = [i % n_classes for i in range(n_samples)]

# Create prediction probabilities (high confidence for true labels)
pred_probs = []
for i, label in enumerate(true_labels):
    probs = [0.05, 0.05, 0.05]
    probs[label] = 0.85 + random.uniform(0, 0.1)
    remaining = 1 - probs[label]
    for j in range(n_classes):
        if j != label:
            probs[j] = remaining / 2
    pred_probs.append([round(p, 3) for p in probs])

# Add label noise (flip 20% of labels)
labels = true_labels.copy()
for i in range(0, n_samples, 5):  # Every 5th sample
    wrong = [l for l in range(n_classes) if l != true_labels[i]]
    labels[i] = random.choice(wrong)

print(json.dumps({'labels': labels, 'pred_probs': pred_probs}))
")

LABELS=$(echo "$DATASET" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['labels']))")
PROBS=$(echo "$DATASET" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['pred_probs']))")

echo -e "${YELLOW}Dataset: 50 samples, 3 classes, ~20% label noise${NC}"
echo ""

echo -e "${YELLOW}Running audit...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/dataset/audit" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"labels\": ${LABELS},
        \"pred_probs\": ${PROBS},
        \"label_names\": [\"cat\", \"dog\", \"bird\"]
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'summary' in data:
        summary = data['summary']
        print(f'  Summary:')
        print(f'    Total samples: {summary.get(\"total_samples\", 0)}')
        print(f'    Label issues found: {summary.get(\"num_label_issues\", 0)}')
        print(f'    Overall quality: {summary.get(\"overall_quality_score\", 0):.2%}')
    if 'label_issues' in data:
        issues = data['label_issues'][:5]
        if issues:
            print(f'')
            print(f'  Top label issues:')
            for issue in issues:
                idx = issue.get('index', '?')
                given = issue.get('given_label', '?')
                suggested = issue.get('suggested_label', '?')
                quality = issue.get('quality_score', 0)
                print(f'    Sample {idx}: {given} → {suggested} (quality: {quality:.2%})')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 2: Get Label Quality Scores
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Label Quality Scores${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Getting per-sample quality scores...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/dataset/quality-scores" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"labels\": ${LABELS},
        \"pred_probs\": ${PROBS}
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'scores' in data:
        scores = data['scores']
        print(f'  ✓ Got {len(scores)} quality scores')

        # Find worst samples
        indexed = list(enumerate(scores))
        worst = sorted(indexed, key=lambda x: x[1])[:5]
        print(f'')
        print(f'  Lowest quality samples:')
        for idx, score in worst:
            print(f'    Sample {idx}: {score:.4f}')

        # Statistics
        avg = sum(scores) / len(scores)
        low_quality = sum(1 for s in scores if s < 0.5)
        print(f'')
        print(f'  Statistics:')
        print(f'    Average quality: {avg:.2%}')
        print(f'    Low quality (<50%): {low_quality} samples')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "API Endpoints:"
echo ""
echo "  POST /v1/dataset/audit"
echo "    {"
echo "      \"labels\": [0, 1, 2, ...],"
echo "      \"pred_probs\": [[0.8, 0.1, 0.1], ...],"
echo "      \"label_names\": [\"cat\", \"dog\", \"bird\"]  // Optional"
echo "    }"
echo ""
echo "  POST /v1/dataset/quality-scores"
echo "    {"
echo "      \"labels\": [0, 1, 2, ...],"
echo "      \"pred_probs\": [[0.8, 0.1, 0.1], ...]"
echo "    }"
echo ""
echo "Use cases:"
echo "  - Find mislabeled training data"
echo "  - Identify low-quality annotations"
echo "  - Clean datasets before training"
echo ""
