#!/bin/bash

# Example: Directory Upload with LlamaFarm CLI
# This script demonstrates various ways to upload files and directories

echo "================================================"
echo "LlamaFarm Directory Upload Examples"
echo "================================================"

# Configuration
DATASET="example-dataset"
# Determine LF command based on where script is run from
if [ -f "./lf" ]; then
    LF="./lf"  # Running from root
elif [ -f "../../lf" ]; then
    LF="../../lf"  # Running from examples/rag_pipeline
else
    LF="lf"  # Try system PATH
fi

# Create dataset if it doesn't exist
echo ""
echo "1. Creating dataset (if not exists)..."
$LF datasets add $DATASET -s universal_processor -b main_database 2>/dev/null || echo "Dataset already exists"

echo ""
echo "================================================"
echo "Example 1: Upload all files in a directory (non-recursive)"
echo "================================================"
echo "Command: $LF datasets ingest $DATASET examples/rag_pipeline/sample_files/research_papers/"
$LF datasets ingest $DATASET examples/rag_pipeline/sample_files/research_papers/

echo ""
echo "================================================"
echo "Example 2: Upload files recursively using /**/* pattern"
echo "================================================"
echo "Command: $LF datasets ingest $DATASET \"examples/rag_pipeline/sample_files/**/*\""
echo "(This uploads ALL files in all subdirectories)"
# Note: Using quotes to ensure the glob is passed to the CLI for expansion
$LF datasets ingest $DATASET "examples/rag_pipeline/sample_files/**/*"

echo ""
echo "================================================"
echo "Example 3: Upload specific file types using glob pattern"
echo "================================================"
echo "Command: $LF datasets ingest $DATASET examples/rag_pipeline/sample_files/code_documentation/*.md"
$LF datasets ingest $DATASET examples/rag_pipeline/sample_files/code_documentation/*.md

echo ""
echo "================================================"
echo "Example 4: Upload from multiple sources"
echo "================================================"
echo "Command: $LF datasets ingest $DATASET examples/rag_pipeline/sample_files/code/*.py examples/rag_pipeline/sample_files/fda/*.pdf"
$LF datasets ingest $DATASET examples/rag_pipeline/sample_files/code/*.py examples/rag_pipeline/sample_files/fda/*.pdf

echo ""
echo "================================================"
echo "Example 5: Mixed - directory and specific files"
echo "================================================"
echo "Command: $LF datasets ingest $DATASET examples/rag_pipeline/sample_files/news_articles/ examples/rag_pipeline/sample_files/code/example.py"
$LF datasets ingest $DATASET examples/rag_pipeline/sample_files/news_articles/ examples/rag_pipeline/sample_files/code/example.py

echo ""
echo "================================================"
echo "Summary"
echo "================================================"
echo "The enhanced ingest command supports:"
echo "  • Single files: ./file.pdf"
echo "  • Multiple files: file1.txt file2.md"  
echo "  • Glob patterns: *.pdf, docs/*.txt"
echo "  • Directories: ./docs/"
echo "  • Recursive: ./docs/**/* (includes all files in subdirectories)"
echo "  • Mixed: ./docs/ *.pdf specific.txt"
echo ""
echo "Features:"
echo "  • Batch upload with progress display"
echo "  • Automatic duplicate detection"
echo "  • Continues on errors"
echo "  • Clear success/failure summary"