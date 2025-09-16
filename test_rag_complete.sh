#!/bin/bash

echo "=== Complete RAG Test Suite ==="
echo ""

# Clean session
rm -f ~/.llamafarm/session_context.yaml 2>/dev/null

echo "1. Test simple query without RAG flag (should use config defaults):"
echo "   Command: lf run 'What is 2+2?'"
/Users/robthelen/llamafarm-1/lf run "What is 2+2?" 2>&1 | grep -v "^$" | tail -5
echo ""

echo "2. Test with explicit RAG flag:"
echo "   Command: lf run --rag 'What is transformer architecture?'"  
/Users/robthelen/llamafarm-1/lf run --rag "What is transformer architecture?" 2>&1 | grep -v "^$" | tail -10
echo ""

echo "3. Test with RAG and specific parameters:"
echo "   Command: lf run --rag --rag-top-k 3 'Explain attention mechanism'"
/Users/robthelen/llamafarm-1/lf run --rag --rag-top-k 3 "Explain attention mechanism" 2>&1 | grep -v "^$" | tail -10
echo ""

echo "=== Config Status ==="
echo "System prompts configured:"
grep -A 10 "^prompts:" /Users/robthelen/.llamafarm/projects/default/llamafarm-1/llamafarm.yaml | head -11
echo ""
echo "Datasets configured:"
grep -A 3 "^datasets:" /Users/robthelen/.llamafarm/projects/default/llamafarm-1/llamafarm.yaml | head -4
echo ""
echo "RAG enabled in config: $(grep -c "^rag:" /Users/robthelen/.llamafarm/projects/default/llamafarm-1/llamafarm.yaml) (1 = yes, 0 = no)"