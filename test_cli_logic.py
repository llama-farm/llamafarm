#!/usr/bin/env python3
"""
Test the logic we implemented for path detection
"""

import os
import glob
from pathlib import Path

def determine_method(paths):
    """Python version of the Go determineIngestMethod logic"""
    has_files = False
    has_dirs = False  
    has_patterns = False
    
    for path in paths:
        # Check for glob patterns
        if '*' in path or '?' in path or '[' in path:
            has_patterns = True
        else:
            # Check if exists and what type
            if os.path.exists(path):
                if os.path.isdir(path):
                    has_dirs = True
                else:
                    has_files = True
            else:
                # If doesn't exist, check if parent dir exists (could be pattern)
                parent = os.path.dirname(path)
                if os.path.exists(parent):
                    has_patterns = True
    
    # Determine method
    if has_files and not has_dirs and not has_patterns:
        return "files"
    elif (has_dirs or has_patterns) and not has_files:
        return "paths"
    else:
        return "mixed"

def expand_paths(paths, recursive=False, pattern=None):
    """Expand paths including glob patterns and directories"""
    all_files = []
    
    for path in paths:
        # Handle glob patterns
        if '*' in path or '?' in path or '[' in path:
            matches = glob.glob(path, recursive=recursive)
            all_files.extend(matches)
        elif os.path.isdir(path):
            # Handle directory
            if pattern:
                search_pattern = os.path.join(path, '**' if recursive else '', pattern)
            else:
                search_pattern = os.path.join(path, '**' if recursive else '*')
            
            matches = glob.glob(search_pattern, recursive=recursive)
            all_files.extend([f for f in matches if os.path.isfile(f)])
        elif os.path.isfile(path):
            # Handle regular file
            all_files.append(path)
    
    return all_files

# Test cases
test_cases = [
    {
        "paths": ["examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt"],
        "description": "Single file"
    },
    {
        "paths": ["examples/rag_pipeline/sample_files/research_papers/"],
        "description": "Directory"
    },
    {
        "paths": ["examples/rag_pipeline/sample_files/research_papers/*.txt"],
        "description": "Glob pattern"
    },
    {
        "paths": ["examples/rag_pipeline/sample_files/code_documentation/*.md"],
        "description": "Glob pattern for markdown files"
    },
    {
        "paths": ["examples/rag_pipeline/sample_files/"],
        "recursive": True,
        "description": "Directory with recursive"
    },
    {
        "paths": ["examples/rag_pipeline/sample_files/"],
        "recursive": True,
        "pattern": "*.pdf",
        "description": "Directory recursive with pattern filter"
    },
    {
        "paths": [
            "examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt",
            "examples/rag_pipeline/sample_files/code/",
            "examples/rag_pipeline/sample_files/fda/*.pdf"
        ],
        "description": "Mixed input"
    }
]

print("Testing Path Detection and Expansion Logic")
print("=" * 60)

for test in test_cases:
    print(f"\n🧪 Test: {test['description']}")
    print(f"   Input paths: {test['paths']}")
    
    method = determine_method(test['paths'])
    print(f"   Detection method: {method}")
    
    recursive = test.get('recursive', False)
    pattern = test.get('pattern', None)
    
    expanded = expand_paths(test['paths'], recursive, pattern)
    print(f"   Found {len(expanded)} files:")
    for f in expanded[:5]:  # Show first 5
        print(f"     - {os.path.basename(f)}")
    if len(expanded) > 5:
        print(f"     ... and {len(expanded) - 5} more")

print("\n✅ Logic test completed successfully!")