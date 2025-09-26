#!/bin/bash

# Example: Directory Upload with LlamaFarm CLI
# This script demonstrates various ways to upload files and directories

echo "================================================"
echo "LlamaFarm Directory Upload Examples"
echo "================================================"

# Configuration
DATASET="example-dataset"
LF="../../lf"  # Adjust path to your lf binary

# Create dataset if it doesn't exist
echo ""
echo "1. Creating dataset (if not exists)..."
$LF datasets add $DATASET -s universal_processor -b main_database 2>/dev/null || echo "Dataset already exists"

echo ""
echo "================================================"
echo "Example 1: Upload all files in a directory (non-recursive)"
echo "================================================"
echo "Command: $LF datasets ingest $DATASET ./sample_files/research_papers/"
$LF datasets ingest $DATASET ./sample_files/research_papers/

echo ""
echo "================================================"
echo "Example 2: Upload files recursively from directory"
echo "================================================"
echo "Command: $LF datasets ingest $DATASET ./sample_files/ --recursive"
echo "(This would upload ALL files in all subdirectories - skipping for demo)"
# $LF datasets ingest $DATASET ./sample_files/ --recursive

echo ""
echo "================================================"
echo "Example 3: Upload specific file types using glob pattern"
echo "================================================"
echo "Command: $LF datasets ingest $DATASET ./sample_files/code_documentation/*.md"
$LF datasets ingest $DATASET ./sample_files/code_documentation/*.md

echo ""
echo "================================================"
echo "Example 4: Upload from multiple sources"
echo "================================================"
echo "Command: $LF datasets ingest $DATASET ./sample_files/code/*.py ./sample_files/fda/*.pdf"
$LF datasets ingest $DATASET ./sample_files/code/*.py ./sample_files/fda/*.pdf

echo ""
echo "================================================"
echo "Example 5: Mixed - directory and specific files"
echo "================================================"
echo "Command: $LF datasets ingest $DATASET ./sample_files/news_articles/ ./sample_files/code/example.py"
$LF datasets ingest $DATASET ./sample_files/news_articles/ ./sample_files/code/example.py

echo ""
echo "================================================"
echo "Summary"
echo "================================================"
echo "The enhanced ingest command supports:"
echo "  • Single files: ./file.pdf"
echo "  • Multiple files: file1.txt file2.md"  
echo "  • Glob patterns: *.pdf, docs/*.txt"
echo "  • Directories: ./docs/ (use --recursive for subdirectories)"
echo "  • Mixed: ./docs/ *.pdf specific.txt"
echo ""
echo "Features:"
echo "  • Batch upload with progress display"
echo "  • Automatic duplicate detection"
echo "  • Continues on errors"
echo "  • Clear success/failure summary"