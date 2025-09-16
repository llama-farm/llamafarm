#!/bin/bash

echo "=== Testing RAG CLI Integration ==="
echo ""

# Test 1: Without any RAG flags (should use config defaults)
echo "1. Testing without --rag flag (config defaults):"
echo "   Command: lf run 'What is transformer architecture?'"
/Users/robthelen/llamafarm-1/lf run "What is transformer architecture?" --debug 2>&1 | grep -E "JSON DATA:|rag_enabled|RAG enabled by default" | head -3
echo ""

# Test 2: With explicit --rag flag (enables RAG explicitly)
echo "2. Testing with --rag flag:"
echo "   Command: lf run --rag 'What is transformer architecture?'"
/Users/robthelen/llamafarm-1/lf run --rag "What is transformer architecture?" --debug 2>&1 | grep -E "JSON DATA:|rag_enabled" | head -2
echo ""

# Test 3: With --rag and custom database
echo "3. Testing with --rag and specific database:"
echo "   Command: lf run --rag --rag-database main_database 'What is attention mechanism?'"
/Users/robthelen/llamafarm-1/lf run --rag --rag-database main_database "What is attention mechanism?" --debug 2>&1 | grep -E "JSON DATA:|rag_database" | head -2
echo ""

# Test 4: With --rag and custom top-k
echo "4. Testing with --rag and custom top-k:"
echo "   Command: lf run --rag --rag-top-k 10 'How do neural networks work?'"
/Users/robthelen/llamafarm-1/lf run --rag --rag-top-k 10 "How do neural networks work?" --debug 2>&1 | grep -E "JSON DATA:|rag_top_k" | head -2
echo ""

# Test 5: With all RAG parameters
echo "5. Testing with all RAG parameters:"
echo "   Command: lf run --rag --rag-database main_database --rag-top-k 3 --rag-score-threshold 0.5 'Explain BERT model'"
/Users/robthelen/llamafarm-1/lf run --rag --rag-database main_database --rag-top-k 3 --rag-score-threshold 0.5 "Explain BERT model" --debug 2>&1 | grep -E "JSON DATA:|rag_" | head -2
echo ""

echo "=== Config Check ==="
echo "Current config RAG settings:"
grep -A 5 "^rag:" /Users/robthelen/.llamafarm/projects/default/llamafarm-1/llamafarm.yaml | head -6
echo ""
echo "Current datasets:"
grep -A 3 "^datasets:" /Users/robthelen/.llamafarm/projects/default/llamafarm-1/llamafarm.yaml | head -4
echo ""

echo "=== Summary ==="
echo "- Without --rag flag: Should use config defaults (RAG enabled if datasets exist)"
echo "- With --rag flag: Explicitly enables RAG"
echo "- Database parameter: Overrides default database selection"
echo "- Top-k parameter: Overrides default top-k value"
echo "- Score threshold: Filters results by minimum score"