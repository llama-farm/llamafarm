#!/bin/bash
# Demo 05: Parse-Only API Endpoint
#
# This demo shows how to use the new /rag/parse endpoint to parse
# documents without storing them to the database.
#
# Prerequisites:
#   - Server running on port 8005 (or PORT env var)
#   - A project configured (namespace/project)
#
# Usage:
#   bash demo_05_parse_api.sh

set -e

# Configuration - read from llamafarm.yaml if available
SCRIPT_DIR_TEMP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR_TEMP}/llamafarm.yaml" ]; then
    # Read namespace and name from llamafarm.yaml
    NAMESPACE="${NAMESPACE:-$(grep '^namespace:' "${SCRIPT_DIR_TEMP}/llamafarm.yaml" | awk '{print $2}' | tr -d '"')}"
    PROJECT="${PROJECT:-$(grep '^name:' "${SCRIPT_DIR_TEMP}/llamafarm.yaml" | awk '{print $2}' | tr -d '"')}"
fi
SERVER_URL="${SERVER_URL:-http://localhost:8005}"
NAMESPACE="${NAMESPACE:-default}"
PROJECT="${PROJECT:-new-rag-demo}"
API_URL="${SERVER_URL}/v1/projects/${NAMESPACE}/${PROJECT}/rag/parse"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "LLAMAFARM RAG - Parse-Only API Demo"
echo "============================================================"
echo ""
echo "Server URL: ${SERVER_URL}"
echo "API Endpoint: ${API_URL}"
echo ""

# Find a sample file to parse
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(dirname "$SCRIPT_DIR")"

# Look for sample files
SAMPLE_PDF=""
SAMPLE_TXT=""

# Check common locations for sample PDFs
for dir in "${EXAMPLES_DIR}/fda_rag/files" "${EXAMPLES_DIR}/rag_legal_filings/files" "${EXAMPLES_DIR}/rag_pipeline/sample_files"; do
    if [ -d "$dir" ]; then
        # Find first PDF
        if [ -z "$SAMPLE_PDF" ]; then
            SAMPLE_PDF=$(find "$dir" -name "*.pdf" -type f 2>/dev/null | head -1)
        fi
    fi
done

# Create a sample text file if no PDF found
if [ -z "$SAMPLE_PDF" ]; then
    SAMPLE_TXT="/tmp/sample_for_parse_demo.txt"
    cat > "$SAMPLE_TXT" << 'EOF'
# LlamaFarm RAG System

## Introduction

LlamaFarm is a powerful platform for building AI applications with retrieval-augmented generation (RAG).

This document demonstrates the parse-only API endpoint, which allows you to:
- Parse documents without storing to database
- Preview content before ingestion
- Test parser configurations
- Extract metadata from documents

## Key Features

### Document Parsing
The system supports multiple file formats including PDF, DOCX, XLSX, and more.

### Smart Chunking
Content can be chunked using various strategies:
- Character-based chunking
- Sentence-based chunking
- Paragraph-based chunking
- Section-based chunking (using markdown headers)

### Metadata Extraction
Parsers extract useful metadata including page counts, headings, and document structure.

## Conclusion

The parse-only API is useful for testing and previewing before committing to database storage.
EOF
    echo "Created sample text file at: ${SAMPLE_TXT}"
fi

# Function to check if server is running
check_server() {
    echo -n "Checking server availability... "
    if curl -s -o /dev/null -w "%{http_code}" "${SERVER_URL}/health" | grep -q "200"; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${YELLOW}Server may not be running${NC}"
        echo "Note: This demo requires the LlamaFarm server to be running."
        echo "Start it with: nx start server"
        return 1
    fi
}

# Demo 1: Parse a text file with default settings
demo_parse_text_default() {
    echo ""
    echo "============================================================"
    echo "Demo 1: Parse Text File (Default Settings)"
    echo "============================================================"
    echo ""

    if [ -n "$SAMPLE_TXT" ]; then
        echo "curl -X POST '${API_URL}' \\"
        echo "  -F 'file=@${SAMPLE_TXT}'"
        echo ""

        response=$(curl -s -X POST "${API_URL}" \
            -F "file=@${SAMPLE_TXT}" \
            2>&1) || true

        echo "Response:"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    else
        echo "No sample text file available"
    fi
}

# Demo 2: Parse with chunking enabled
demo_parse_with_chunking() {
    echo ""
    echo "============================================================"
    echo "Demo 2: Parse with Sentence Chunking"
    echo "============================================================"
    echo ""

    local file="${SAMPLE_TXT:-$SAMPLE_PDF}"
    if [ -n "$file" ] && [ -f "$file" ]; then
        echo "curl -X POST '${API_URL}' \\"
        echo "  -F 'file=@${file}' \\"
        echo "  -F 'chunk_strategy=sentences' \\"
        echo "  -F 'chunk_size=200'"
        echo ""

        response=$(curl -s -X POST "${API_URL}" \
            -F "file=@${file}" \
            -F "chunk_strategy=sentences" \
            -F "chunk_size=200" \
            2>&1) || true

        echo "Response (truncated):"
        echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'Format: {data.get(\"format\", \"N/A\")}')
    print(f'Parser: {data.get(\"parser_used\", \"N/A\")}')
    print(f'Processing time: {data.get(\"processing_time_ms\", 0):.2f}ms')
    print(f'Total chunks: {len(data.get(\"chunks\", []))}')
    if data.get('chunks'):
        print(f'\\nFirst chunk preview:')
        print(data['chunks'][0]['content'][:200] + '...')
except:
    print(sys.stdin.read())
" 2>/dev/null || echo "$response"
    fi
}

# Demo 3: Parse PDF with Docling (if available)
demo_parse_pdf() {
    echo ""
    echo "============================================================"
    echo "Demo 3: Parse PDF with Docling"
    echo "============================================================"
    echo ""

    if [ -n "$SAMPLE_PDF" ] && [ -f "$SAMPLE_PDF" ]; then
        echo "File: ${SAMPLE_PDF}"
        echo ""
        echo "curl -X POST '${API_URL}' \\"
        echo "  -F 'file=@${SAMPLE_PDF}' \\"
        echo "  -F 'output_format=markdown' \\"
        echo "  -F 'chunk_strategy=hybrid' \\"
        echo "  -F 'chunk_size=512'"
        echo ""

        response=$(curl -s -X POST "${API_URL}" \
            -F "file=@${SAMPLE_PDF}" \
            -F "output_format=markdown" \
            -F "chunk_strategy=hybrid" \
            -F "chunk_size=512" \
            2>&1) || true

        echo "Response (summary):"
        echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'Format: {data.get(\"format\", \"N/A\")}')
    print(f'Parser: {data.get(\"parser_used\", \"N/A\")}')
    print(f'Processing time: {data.get(\"processing_time_ms\", 0):.2f}ms')
    print(f'Filename: {data.get(\"filename\", \"N/A\")}')
    meta = data.get('metadata', {})
    print(f'Page count: {meta.get(\"page_count\", \"N/A\")}')
    print(f'Total chunks: {len(data.get(\"chunks\", []))}')
    content = data.get('content', '')
    print(f'Content length: {len(content)} chars')
    print(f'\\nContent preview:')
    print(content[:500] + '...')
except Exception as e:
    print(f'Error: {e}')
    print(sys.stdin.read())
" 2>/dev/null || echo "$response"
    else
        echo "No PDF file available. Skipping PDF demo."
        echo "To test PDF parsing, add a PDF to examples/fda_rag/files/"
    fi
}

# Demo 4: Get plain text output
demo_plain_text_output() {
    echo ""
    echo "============================================================"
    echo "Demo 4: Plain Text Output Format"
    echo "============================================================"
    echo ""

    local file="${SAMPLE_TXT:-$SAMPLE_PDF}"
    if [ -n "$file" ] && [ -f "$file" ]; then
        echo "curl -X POST '${API_URL}' \\"
        echo "  -F 'file=@${file}' \\"
        echo "  -F 'output_format=text'"
        echo ""

        response=$(curl -s -X POST "${API_URL}" \
            -F "file=@${file}" \
            -F "output_format=text" \
            2>&1) || true

        echo "Response:"
        echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'Format: {data.get(\"format\", \"N/A\")}')
    content = data.get('content', '')
    print(f'\\nPlain text output (first 400 chars):')
    print('-' * 40)
    print(content[:400])
    print('-' * 40)
except:
    print(sys.stdin.read())
" 2>/dev/null || echo "$response"
    fi
}

# Demo 5: Error handling
demo_error_handling() {
    echo ""
    echo "============================================================"
    echo "Demo 5: Error Handling (Unsupported File Type)"
    echo "============================================================"
    echo ""

    # Create a file with unsupported extension
    echo "test content" > /tmp/test_unsupported.xyz

    echo "curl -X POST '${API_URL}' \\"
    echo "  -F 'file=@/tmp/test_unsupported.xyz'"
    echo ""

    response=$(curl -s -X POST "${API_URL}" \
        -F "file=@/tmp/test_unsupported.xyz" \
        2>&1) || true

    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

    rm -f /tmp/test_unsupported.xyz
}

# Main execution
main() {
    # Check if server is running (but continue anyway for demo purposes)
    check_server || echo "(Demo will show expected curl commands even if server is not running)"

    demo_parse_text_default
    demo_parse_with_chunking
    demo_parse_pdf
    demo_plain_text_output
    demo_error_handling

    echo ""
    echo "============================================================"
    echo "Demo Complete!"
    echo "============================================================"
    echo ""
    echo "Key API Features:"
    echo "  - POST /rag/parse - Parse without database storage"
    echo "  - output_format: markdown, text, json"
    echo "  - chunk_strategy: none, sentences, paragraphs, sections, hybrid"
    echo "  - chunk_size: 50-10000 (characters or tokens)"
    echo "  - chunk_overlap: 0-1000"
    echo ""
    echo "Supported File Types:"
    echo "  PDF, DOCX, PPTX, XLSX, TXT, MD, CSV, HTML"

    # Clean up
    if [ -f "/tmp/sample_for_parse_demo.txt" ]; then
        rm -f /tmp/sample_for_parse_demo.txt
    fi
}

main "$@"
