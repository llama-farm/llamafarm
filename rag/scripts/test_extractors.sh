#!/bin/bash

# Test script for extractors (non-interactive)
# Quick verification that all extractors work

set -e

echo "🧪 Testing all extractors..."

# Test each extractor
extractors=("yake" "rake" "entities" "datetime" "statistics" "tfidf")

for extractor in "${extractors[@]}"; do
    echo "Testing $extractor..."
    if uv run python cli.py extractors test --extractor "$extractor" --text "Machine learning and AI are transforming technology. Contact us at test@example.com on January 1, 2024." >/dev/null 2>&1; then
        echo "✅ $extractor: PASS"
    else
        echo "❌ $extractor: FAIL"
    fi
done

echo "🎉 Extractor testing complete!"