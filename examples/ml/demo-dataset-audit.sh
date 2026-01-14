#!/bin/bash
# Demo: Dataset Quality Audit with Cleanlab
# Phase 16 of Universal Runtime ML Enhancements
#
# Finds mislabeled examples and near-duplicates in classification datasets.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi
RUNTIME_URL="${RUNTIME_URL:-http://127.0.0.1:${LF_RUNTIME_PORT:-11540}}"

echo "=========================================="
echo "Dataset Quality Audit Demo (Cleanlab)"
echo "=========================================="
echo ""

# Check if server is running
echo "1. Checking server health..."
if ! curl -s "$RUNTIME_URL/health" > /dev/null 2>&1; then
    echo "   ERROR: Server not running at $RUNTIME_URL"
    echo "   Start with: cd runtimes/universal && uv run python server.py"
    exit 1
fi
echo "   Server is healthy!"
echo ""

# Generate training data with label noise
echo "2. Generating Training Data with Label Noise"
echo "   Creating a 3-class dataset (cat=0, dog=1, bird=2)..."
echo "   Intentionally mislabeling ~20% of samples..."
echo ""

# Generate data: labels, pred_probs, and introduce noise
DATASET=$(python3 -c "
import random
import json

random.seed(42)

# True labels from 'model predictions'
n_samples = 50
n_classes = 3
labels = []
pred_probs = []

for i in range(n_samples):
    # True class (what the model predicts)
    true_class = random.randint(0, n_classes - 1)

    # Create prediction probabilities (high confidence for true class)
    probs = [0.05] * n_classes
    probs[true_class] = 0.85 + random.uniform(0, 0.1)
    # Normalize
    total = sum(probs)
    probs = [p / total for p in probs]
    pred_probs.append([round(p, 4) for p in probs])

    # Usually use true label, but sometimes flip (label noise)
    if random.random() < 0.2:  # 20% noise
        # Pick a different label
        wrong_labels = [l for l in range(n_classes) if l != true_class]
        labels.append(random.choice(wrong_labels))
    else:
        labels.append(true_class)

# Also create feature vectors for duplicate detection
features = []
for i in range(n_samples):
    feat = [random.gauss(0, 1) for _ in range(5)]
    features.append([round(f, 4) for f in feat])

# Create some near-duplicates
for i in range(3):
    features[i * 2 + 1] = [f + random.gauss(0, 0.01) for f in features[i * 2]]

print(json.dumps({
    'labels': labels,
    'pred_probs': pred_probs,
    'features': features
}))
")

LABELS=$(echo "$DATASET" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['labels']))")
PRED_PROBS=$(echo "$DATASET" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['pred_probs']))")
FEATURES=$(echo "$DATASET" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['features']))")

echo "   Generated $(($(echo "$LABELS" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))"))) samples"
echo ""

# Run the audit
echo "3. Running Dataset Quality Audit"
echo "   Analyzing labels with Cleanlab..."
echo ""

RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/dataset/audit" \
    -H "Content-Type: application/json" \
    -d "{
        \"labels\": $LABELS,
        \"pred_probs\": $PRED_PROBS,
        \"features\": $FEATURES,
        \"label_names\": [\"cat\", \"dog\", \"bird\"],
        \"check_duplicates\": true,
        \"duplicate_threshold\": 0.95
    }")

echo "   Audit Results:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Extract and summarize label issues
echo "4. Summary: Potentially Mislabeled Samples"
echo ""

echo "$RESPONSE" | python3 -c "
import sys
import json

data = json.load(sys.stdin)
if 'label_issues' in data and data['label_issues']:
    print('   Found potentially mislabeled samples:')
    for issue in data['label_issues'][:5]:  # Top 5
        given = issue.get('given_label_name', issue['given_label'])
        suggested = issue.get('suggested_label_name', issue['suggested_label'])
        conf_given = issue['given_confidence']
        conf_suggested = issue['suggested_confidence']
        print(f'     Index {issue[\"index\"]}: Labeled as \"{given}\" (conf: {conf_given:.2f})')
        print(f'                  Should be \"{suggested}\"? (conf: {conf_suggested:.2f})')
        print()
else:
    print('   No obvious label issues found!')
"

# Show duplicates
echo "5. Summary: Near-Duplicate Samples"
echo ""

echo "$RESPONSE" | python3 -c "
import sys
import json

data = json.load(sys.stdin)
if 'duplicates' in data and data['duplicates']:
    print('   Found near-duplicate pairs:')
    for dup in data['duplicates'][:5]:  # Top 5
        print(f'     Samples {dup[\"index_a\"]} and {dup[\"index_b\"]}: similarity = {dup[\"similarity\"]:.4f}')
else:
    print('   No near-duplicates found!')
"
echo ""

# Get quality scores
echo "6. Label Quality Scores (per-sample)"
echo "   Getting confidence that each label is correct..."
echo ""

RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/dataset/quality-scores" \
    -H "Content-Type: application/json" \
    -d "{
        \"labels\": $LABELS,
        \"pred_probs\": $PRED_PROBS
    }")

echo "   Quality Statistics:"
echo "$RESPONSE" | python3 -c "
import sys
import json

data = json.load(sys.stdin)
print(f'     Mean quality:  {data[\"mean_quality\"]:.3f}')
print(f'     Min quality:   {data[\"min_quality\"]:.3f}')
print(f'     Max quality:   {data[\"max_quality\"]:.3f}')
print()
print('   Lowest quality samples (most likely mislabeled):')
scores = data['scores']
indexed = [(i, s) for i, s in enumerate(scores)]
indexed.sort(key=lambda x: x[1])
for idx, score in indexed[:5]:
    print(f'     Sample {idx}: quality = {score:.3f}')
"
echo ""

echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Key insights:"
echo "  - Cleanlab finds label errors using model predictions"
echo "  - Low quality scores indicate likely mislabeled samples"
echo "  - Near-duplicate detection helps find redundant data"
echo "  - Use this to clean training data before retraining"
echo ""
echo "Use cases:"
echo "  - Improving classifier accuracy by fixing labels"
echo "  - Finding and removing duplicate training examples"
echo "  - Auditing data quality in production pipelines"
