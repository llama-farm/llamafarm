#!/usr/bin/env python3
"""Quick test script for PathExtractor functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rag.core.base import Document
from rag.components.extractors.path_extractor import PathExtractor


def test_path_extractor():
    """Test various PathExtractor configurations."""
    
    print("🧪 Testing PathExtractor Component\n")
    
    # Test 1: Basic path extraction
    print("1️⃣ Basic Path Extraction")
    extractor = PathExtractor("basic", {
        "store_full_path": True,
        "store_filename": True,
        "store_directory": True,
        "store_extension": True
    })
    
    doc = Document(
        content="Test document content",
        metadata={},
        source="/path/to/documents/test_file.pdf"
    )
    
    [extracted_doc] = extractor.extract([doc])
    print(f"   Source: {doc.source}")
    print(f"   Extracted metadata: {extracted_doc.metadata}")
    print()
    
    # Test 2: Minimal extraction
    print("2️⃣ Minimal Path Extraction (filename only)")
    extractor = PathExtractor("minimal", {
        "store_full_path": False,
        "store_filename": True,
        "store_directory": False,
        "store_extension": False
    })
    
    doc = Document(
        content="Test document content",
        metadata={},
        source="/path/to/documents/test_file.pdf"
    )
    
    [extracted_doc] = extractor.extract([doc])
    print(f"   Source: {doc.source}")
    print(f"   Extracted metadata: {extracted_doc.metadata}")
    print()
    
    # Test 3: Normalized paths
    print("3️⃣ Normalized Path Extraction")
    extractor = PathExtractor("normalized", {
        "store_full_path": True,
        "store_filename": True,
        "store_directory": True,
        "store_extension": True,
        "normalize_paths": True
    })
    
    doc = Document(
        content="Test document content",
        metadata={},
        source="C:\\Users\\Documents\\test_file.pdf"  # Windows-style path
    )
    
    [extracted_doc] = extractor.extract([doc])
    print(f"   Source: {doc.source}")
    print(f"   Normalized path: {extracted_doc.metadata.get('file_path', 'Not found')}")
    print()
    
    # Test 4: Relative paths
    print("4️⃣ Relative Path Extraction")
    extractor = PathExtractor("relative", {
        "store_full_path": True,
        "store_filename": True,
        "store_directory": True,
        "store_extension": True,
        "normalize_paths": True,
        "relative_to": "."
    })
    
    # Use current file's path as example
    current_file = Path(__file__).resolve()
    doc = Document(
        content="Test document content",
        metadata={},
        source=str(current_file)
    )
    
    [extracted_doc] = extractor.extract([doc])
    print(f"   Source: {doc.source}")
    print(f"   Relative path: {extracted_doc.metadata.get('file_path', 'Not found')}")
    print()
    
    # Test 5: Document without source
    print("5️⃣ Document Without Source")
    doc = Document(
        content="Test document content",
        metadata={},
        source=None
    )
    
    [extracted_doc] = extractor.extract([doc])
    print(f"   Source: {doc.source}")
    print(f"   Extracted metadata: {extracted_doc.metadata}")
    print()
    
    print("✅ PathExtractor tests complete!")


if __name__ == "__main__":
    test_path_extractor()