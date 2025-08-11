#!/usr/bin/env python3
"""
Hash utilities for document and chunk deduplication.
Provides consistent hashing for content, documents, and source tracking.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def hash_content(content: str) -> str:
    """
    Generate a consistent hash for content (chunk-level).
    Uses SHA-256 for reliable deduplication.
    
    Args:
        content: Text content to hash
        
    Returns:
        Hexadecimal hash string (64 characters)
    """
    # Normalize content: strip whitespace and convert to consistent encoding
    normalized = content.strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def hash_document(file_path: str, content: Optional[str] = None) -> str:
    """
    Generate a document-level hash combining file path and content.
    This helps identify when the same document is re-ingested.
    
    Args:
        file_path: Path to the source document
        content: Document content (if available, for more accurate hashing)
        
    Returns:
        Hexadecimal hash string (64 characters)
    """
    # Create a composite identifier
    doc_data = {
        "file_path": str(Path(file_path).resolve()),  # Normalize path
        "file_name": Path(file_path).name,
    }
    
    # Add content hash if available for stronger identification
    if content:
        doc_data["content_hash"] = hash_content(content)
    
    # Create consistent JSON representation
    json_str = json.dumps(doc_data, sort_keys=True)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def hash_source_identity(file_path: str) -> str:
    """
    Generate a hash based solely on source identity (file path + name).
    Used for tracking document sources and preventing duplicate ingestion.
    
    Args:
        file_path: Path to the source document
        
    Returns:
        Hexadecimal hash string (64 characters)
    """
    path_obj = Path(file_path)
    source_data = {
        "resolved_path": str(path_obj.resolve()),
        "file_name": path_obj.name,
        "parent_dir": str(path_obj.parent)
    }
    
    json_str = json.dumps(source_data, sort_keys=True)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def generate_chunk_id(document_hash: str, chunk_index: int, content_hash: str) -> str:
    """
    Generate a deterministic chunk ID that ties chunks to their source document.
    
    Args:
        document_hash: Hash of the parent document
        chunk_index: Index of chunk within document
        content_hash: Hash of chunk content
        
    Returns:
        Structured chunk ID for consistent identification
    """
    return f"doc_{document_hash[:12]}_chunk_{chunk_index}_{content_hash[:8]}"


def generate_document_metadata(file_path: str, content: str) -> Dict[str, Any]:
    """
    Generate comprehensive document metadata including all hash types.
    
    Args:
        file_path: Path to source document
        content: Full document content
        
    Returns:
        Dictionary with all hash and tracking metadata
    """
    path_obj = Path(file_path)
    
    # Generate all hash types
    content_hash = hash_content(content)
    document_hash = hash_document(file_path, content)
    source_hash = hash_source_identity(file_path)
    
    return {
        # Hash identifiers
        "document_hash": document_hash,
        "source_hash": source_hash,
        "content_hash": content_hash,
        
        # Source tracking
        "file_path": str(path_obj.resolve()),
        "file_name": path_obj.name,
        "file_extension": path_obj.suffix,
        "file_size": len(content.encode('utf-8')),
        "parent_directory": str(path_obj.parent),
        
        # Timestamps
        "ingestion_timestamp": datetime.utcnow().isoformat(),
        "content_length": len(content),
        "word_count": len(content.split()),
        
        # Deduplication tracking
        "is_duplicate": False,
        "original_document_id": None,
    }


def generate_chunk_metadata(
    parent_doc_metadata: Dict[str, Any],
    chunk_content: str,
    chunk_index: int,
    total_chunks: int
) -> Dict[str, Any]:
    """
    Generate chunk metadata linked to parent document.
    
    Args:
        parent_doc_metadata: Metadata from parent document
        chunk_content: Content of this specific chunk
        chunk_index: Index of chunk within document (0-based)
        total_chunks: Total number of chunks in document
        
    Returns:
        Dictionary with chunk metadata
    """
    chunk_hash = hash_content(chunk_content)
    chunk_id = generate_chunk_id(
        parent_doc_metadata["document_hash"],
        chunk_index,
        chunk_hash
    )
    
    # Inherit key metadata from parent document
    chunk_metadata = {
        # Chunk-specific identifiers
        "chunk_hash": chunk_hash,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        
        # Parent document references
        "document_hash": parent_doc_metadata["document_hash"],
        "source_hash": parent_doc_metadata["source_hash"],
        "parent_document_content_hash": parent_doc_metadata["content_hash"],
        
        # Source information (inherited)
        "file_path": parent_doc_metadata["file_path"],
        "file_name": parent_doc_metadata["file_name"],
        "file_extension": parent_doc_metadata["file_extension"],
        "parent_directory": parent_doc_metadata["parent_directory"],
        
        # Chunk-specific properties
        "chunk_length": len(chunk_content),
        "chunk_word_count": len(chunk_content.split()),
        
        # Timestamps
        "ingestion_timestamp": parent_doc_metadata["ingestion_timestamp"],
        
        # Deduplication tracking
        "is_duplicate_chunk": False,
        "original_chunk_id": None,
    }
    
    return chunk_metadata


class DeduplicationTracker:
    """
    Track document and chunk hashes to prevent duplicates during ingestion.
    """
    
    def __init__(self):
        self.document_hashes: Dict[str, str] = {}  # document_hash -> document_id
        self.chunk_hashes: Dict[str, str] = {}      # chunk_hash -> chunk_id
        self.source_hashes: Dict[str, str] = {}     # source_hash -> document_hash
        
    def is_duplicate_document(self, document_hash: str) -> bool:
        """Check if document hash already exists."""
        return document_hash in self.document_hashes
    
    def is_duplicate_chunk(self, chunk_hash: str) -> bool:
        """Check if chunk hash already exists."""
        return chunk_hash in self.chunk_hashes
    
    def is_duplicate_source(self, source_hash: str) -> bool:
        """Check if source has already been processed."""
        return source_hash in self.source_hashes
    
    def register_document(self, document_hash: str, document_id: str, source_hash: str):
        """Register a new document in the tracker."""
        self.document_hashes[document_hash] = document_id
        self.source_hashes[source_hash] = document_hash
    
    def register_chunk(self, chunk_hash: str, chunk_id: str):
        """Register a new chunk in the tracker."""
        self.chunk_hashes[chunk_hash] = chunk_id
    
    def get_original_document_id(self, document_hash: str) -> Optional[str]:
        """Get the original document ID for a hash."""
        return self.document_hashes.get(document_hash)
    
    def get_original_chunk_id(self, chunk_hash: str) -> Optional[str]:
        """Get the original chunk ID for a hash."""
        return self.chunk_hashes.get(chunk_hash)
    
    def get_document_by_source(self, source_hash: str) -> Optional[str]:
        """Get document hash by source hash."""
        return self.source_hashes.get(source_hash)


def extract_document_hash_from_chunk_id(chunk_id: str) -> Optional[str]:
    """
    Extract document hash from chunk ID.
    
    Args:
        chunk_id: Chunk ID in format 'doc_{doc_hash}_chunk_{index}_{content_hash}'
        
    Returns:
        Document hash portion or None if invalid format
    """
    try:
        if not chunk_id.startswith('doc_'):
            return None
        
        parts = chunk_id.split('_')
        if len(parts) >= 2:
            # Extract the document hash (12 characters after 'doc_')
            return parts[1]
        
    except Exception:
        pass
    
    return None


def validate_hash_format(hash_value: str, expected_length: int = 64) -> bool:
    """
    Validate that a hash has the expected format.
    
    Args:
        hash_value: Hash string to validate
        expected_length: Expected length (default 64 for SHA-256)
        
    Returns:
        True if hash is valid format
    """
    if not isinstance(hash_value, str):
        return False
    
    if len(hash_value) != expected_length:
        return False
    
    # Check if it's hexadecimal
    try:
        int(hash_value, 16)
        return True
    except ValueError:
        return False