"""Document management system with hash-based tracking, versioning, and lifecycle management."""

import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from core.base import Document
from utils.hash_utils import extract_document_hash_from_chunk_id

from core.logging import RAGStructLogger
logger = RAGStructLogger("rag.core.document_manager")


class DeletionStrategy(Enum):
    """Deletion strategy types."""
    HARD_DELETE = "hard_delete"
    SOFT_DELETE = "soft_delete"
    ARCHIVE_DELETE = "archive_delete"
    CASCADE_DELETE = "cascade_delete"


class UpdateStrategy(Enum):
    """Document update strategy types."""
    REPLACE_ALL = "replace_all"
    INCREMENTAL = "incremental"
    VERSIONING = "versioning"
    COPY_ON_WRITE = "copy_on_write"


@dataclass
class DocumentMetadata:
    """Comprehensive document metadata for management."""
    # Identity
    doc_id: str
    chunk_id: str
    chunk_index: int = 0
    total_chunks: int = 1
    
    # Source Information
    filename: str = ""
    filepath: str = ""
    source_system: str = ""
    file_size: int = 0
    
    # Hashes
    file_hash: str = ""
    chunk_hash: str = ""
    metadata_hash: str = ""
    
    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    indexed_at: str = ""
    expires_at: str = ""
    
    # Versioning
    version: int = 1
    is_active: bool = True
    previous_version: str = ""
    change_summary: str = ""
    
    # Content
    page_number: int = 0
    section: str = ""
    
    # Relationships
    parent_doc: str = ""
    related_docs: List[str] = None
    supersedes: str = ""
    
    # Permissions & Classification
    access_level: str = "public"
    department: str = ""
    tags: List[str] = None
    retention_category: str = "standard"
    
    # Analysis
    summary: str = ""
    keywords: List[str] = None
    language: str = "en"
    reading_time_minutes: float = 0.0
    
    def __post_init__(self):
        """Initialize default values."""
        if self.related_docs is None:
            self.related_docs = []
        if self.tags is None:
            self.tags = []
        if self.keywords is None:
            self.keywords = []
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.indexed_at:
            self.indexed_at = self.created_at


class DocumentHashManager:
    """Manages document hashing for deduplication and change detection."""
    
    def __init__(self, hash_algorithm: str = "sha256"):
        self.hash_algorithm = hash_algorithm
        self._hash_func = getattr(hashlib, hash_algorithm)
    
    def generate_file_hash(self, file_path: Path) -> str:
        """Generate hash for entire file."""
        hash_obj = self._hash_func()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            return f"{self.hash_algorithm}:{hash_obj.hexdigest()}"
        except Exception as e:
            logger.error(f"Failed to generate file hash for {file_path}: {e}")
            return ""
    
    def generate_content_hash(self, content: str) -> str:
        """Generate hash for text content (normalized)."""
        # Normalize content for consistent hashing
        normalized = content.strip().lower()
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        
        hash_obj = self._hash_func()
        hash_obj.update(normalized.encode('utf-8'))
        return f"{self.hash_algorithm}:{hash_obj.hexdigest()}"
    
    def generate_metadata_hash(self, metadata: Dict[str, Any]) -> str:
        """Generate hash for metadata (excluding timestamps)."""
        # Create filtered metadata for hashing
        hash_metadata = {k: v for k, v in metadata.items() 
                        if k not in ['created_at', 'updated_at', 'indexed_at', 'metadata_hash']}
        
        # Sort for consistent hashing
        metadata_str = json.dumps(hash_metadata, sort_keys=True)
        hash_obj = self._hash_func()
        hash_obj.update(metadata_str.encode('utf-8'))
        return f"{self.hash_algorithm}:{hash_obj.hexdigest()}"
    
    def generate_composite_hash(self, *components: str) -> str:
        """Generate hash from multiple components."""
        combined = "|".join(str(c) for c in components)
        hash_obj = self._hash_func()
        hash_obj.update(combined.encode('utf-8'))
        return f"{self.hash_algorithm}:{hash_obj.hexdigest()}"


class DocumentLifecycleManager:
    """Manages document lifecycle including versioning, expiration, and cleanup."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hash_manager = DocumentHashManager(
            config.get("hash_algorithm", "sha256")
        )
        self.retention_policy = config.get("retention_policy", {})
        self.enable_versioning = config.get("enable_versioning", True)
        self.enable_soft_delete = config.get("enable_soft_delete", True)
        
    def enhance_document_metadata(
        self, 
        document: Document, 
        file_path: Optional[Path] = None,
        chunk_metadata_config: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Enhance document with comprehensive metadata."""
        
        # Generate basic metadata
        current_time = datetime.utcnow().isoformat() + "Z"
        
        # Create base metadata
        base_metadata = {
            "doc_id": document.metadata.get("doc_id", document.id),
            "chunk_id": document.id,
            "filename": file_path.name if file_path else document.metadata.get("filename", ""),
            "filepath": str(file_path.parent) if file_path else "",
            "created_at": current_time,
            "updated_at": current_time,
            "indexed_at": current_time,
            "version": 1,
            "is_active": True,
        }
        
        # Generate hashes
        if file_path and file_path.exists():
            base_metadata["file_hash"] = self.hash_manager.generate_file_hash(file_path)
            base_metadata["file_size"] = file_path.stat().st_size
        
        base_metadata["chunk_hash"] = self.hash_manager.generate_content_hash(document.content)
        
        # Add chunk-specific metadata if configured
        if chunk_metadata_config:
            chunk_metadata = self._generate_chunk_metadata(document, chunk_metadata_config)
            base_metadata.update(chunk_metadata)
        
        # Set expiration based on retention policy
        if self.retention_policy.get("default_ttl_days"):
            expiry_date = datetime.utcnow() + timedelta(
                days=self.retention_policy["default_ttl_days"]
            )
            base_metadata["expires_at"] = expiry_date.isoformat() + "Z"
        
        # Merge with existing metadata
        document.metadata.update(base_metadata)
        
        # Generate final metadata hash
        document.metadata["metadata_hash"] = self.hash_manager.generate_metadata_hash(
            document.metadata
        )
        
        return document
    
    def _generate_chunk_metadata(
        self, 
        document: Document, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate chunk-specific metadata based on configuration."""
        metadata = {}
        
        # Generate summary
        if config.get("generate_summary", False):
            metadata["summary"] = self._generate_summary(document.content)
        
        # Extract keywords  
        if config.get("extract_keywords", False):
            metadata["keywords"] = self._extract_keywords(document.content)
        
        # Include statistics
        if config.get("include_statistics", False):
            stats = self._calculate_content_statistics(document.content)
            metadata.update(stats)
        
        # Process custom fields
        custom_fields = config.get("custom_fields", {})
        for field_name, processing_type in custom_fields.items():
            metadata[field_name] = self._process_custom_field(
                document.content, processing_type
            )
        
        # Add content analysis
        content_analysis = config.get("content_analysis", {})
        if content_analysis:
            analysis = self._perform_content_analysis(document.content, content_analysis)
            metadata.update(analysis)
        
        return metadata
    
    def _generate_summary(self, content: str, max_length: int = 150) -> str:
        """Generate a summary of the content."""
        # Simple extractive summary - first meaningful sentences
        sentences = content.split('. ')
        summary = ""
        for sentence in sentences:
            if len(summary + sentence) > max_length:
                break
            summary += sentence + ". "
        return summary.strip()
    
    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from content."""
        # Simple keyword extraction based on word frequency
        import re
        from collections import Counter
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this',
            'that', 'these', 'those', 'are', 'was', 'were', 'been', 'have',
            'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might'
        }
        keywords = [word for word in words if word not in stop_words]
        
        # Get most common keywords
        word_counts = Counter(keywords)
        return [word for word, _ in word_counts.most_common(max_keywords)]
    
    def _calculate_content_statistics(self, content: str) -> Dict[str, Any]:
        """Calculate content statistics."""
        words = len(content.split())
        characters = len(content)
        sentences = len([s for s in content.split('.') if s.strip()])
        
        return {
            "word_count": words,
            "character_count": characters,
            "sentence_count": sentences,
            "reading_time_minutes": words / 200.0,  # Assume 200 words per minute
            "avg_sentence_length": words / max(sentences, 1)
        }
    
    def _process_custom_field(self, content: str, processing_type: str) -> Any:
        """Process custom fields based on type."""
        if processing_type == "auto_classify":
            return self._auto_classify_content(content)
        elif processing_type == "auto_analyze":
            return self._analyze_sentiment(content)
        elif processing_type == "calculate":
            return self._calculate_score(content)
        else:
            return None
    
    def _auto_classify_content(self, content: str) -> str:
        """Auto-classify content into categories."""
        content_lower = content.lower()
        if any(word in content_lower for word in ['error', 'bug', 'issue', 'problem']):
            return "technical_issue"
        elif any(word in content_lower for word in ['password', 'login', 'access']):
            return "access_issue"
        elif any(word in content_lower for word in ['payment', 'billing', 'invoice']):
            return "billing_issue"
        else:
            return "general"
    
    def _analyze_sentiment(self, content: str) -> str:
        """Simple sentiment analysis."""
        positive_words = {'good', 'great', 'excellent', 'happy', 'satisfied', 'thank'}
        negative_words = {'bad', 'terrible', 'awful', 'angry', 'frustrated', 'urgent'}
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "neutral"
    
    def _calculate_score(self, content: str) -> float:
        """Calculate a relevance/importance score."""
        urgent_indicators = ['urgent', 'emergency', 'critical', 'asap', 'immediately']
        content_lower = content.lower()
        score = 0.5  # Base score
        
        for indicator in urgent_indicators:
            if indicator in content_lower:
                score += 0.2
        
        # Longer content might be more complex/important
        if len(content) > 500:
            score += 0.1
        
        return min(score, 1.0)
    
    def _perform_content_analysis(
        self, 
        content: str, 
        analysis_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform various content analysis tasks."""
        results = {}
        
        if analysis_config.get("detect_language", False):
            results["language"] = self._detect_language(content)
        
        if analysis_config.get("estimate_reading_time", False):
            word_count = len(content.split())
            results["reading_time_minutes"] = word_count / 200.0
        
        if analysis_config.get("extract_key_phrases", False):
            results["key_phrases"] = self._extract_key_phrases(content)
        
        if analysis_config.get("identify_document_type", False):
            results["document_type"] = self._identify_document_type(content)
        
        return results
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection."""
        # This is a simplified implementation
        # In production, you'd use a proper language detection library
        common_english_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
        }
        words = content.lower().split()
        english_word_count = sum(1 for word in words[:50] if word in common_english_words)
        
        return "en" if english_word_count > 5 else "unknown"
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content."""
        # Simple n-gram extraction
        import re
        
        # Extract 2-3 word phrases that might be meaningful
        phrases = []
        words = re.findall(r'\b[A-Za-z]+\b', content)
        
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i + 1]) > 3:
                phrase = f"{words[i]} {words[i + 1]}"
                if phrase not in phrases:
                    phrases.append(phrase)
        
        return phrases[:10]  # Return top 10 phrases
    
    def _identify_document_type(self, content: str) -> str:
        """Identify document type based on content patterns."""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['contract', 'agreement', 'terms']):
            return "legal_document"
        elif any(term in content_lower for term in ['report', 'analysis', 'findings']):
            return "report"
        elif any(term in content_lower for term in ['manual', 'instructions', 'guide']):
            return "documentation"
        elif any(term in content_lower for term in ['email', 'message', 'correspondence']):
            return "communication"
        else:
            return "general_document"


class DocumentDeletionManager:
    """Manages document deletion with various strategies."""
    
    def __init__(self, vector_store, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.config = config
        self.enable_soft_delete = config.get("enable_soft_delete", True)
        self.archive_location = config.get("retention_policy", {}).get("archive_location")
        
    def delete_by_time(
        self, 
        older_than_days: int, 
        strategy: DeletionStrategy = DeletionStrategy.SOFT_DELETE
    ) -> Dict[str, Any]:
        """Delete documents older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        cutoff_iso = cutoff_date.isoformat() + "Z"
        
        # Find documents to delete
        filter_criteria = {
            "created_at": {"$lt": cutoff_iso}
        }
        
        if self.enable_soft_delete and strategy == DeletionStrategy.SOFT_DELETE:
            filter_criteria["is_active"] = True
        
        return self._execute_deletion(filter_criteria, strategy)
    
    def delete_by_document(
        self, 
        doc_ids: List[str], 
        strategy: DeletionStrategy = DeletionStrategy.SOFT_DELETE
    ) -> Dict[str, Any]:
        """Delete specific documents by ID."""
        filter_criteria = {
            "doc_id": {"$in": doc_ids}
        }
        
        if self.enable_soft_delete and strategy == DeletionStrategy.SOFT_DELETE:
            filter_criteria["is_active"] = True
        
        return self._execute_deletion(filter_criteria, strategy)
    
    def delete_by_filename(
        self, 
        filenames: List[str], 
        strategy: DeletionStrategy = DeletionStrategy.SOFT_DELETE
    ) -> Dict[str, Any]:
        """Delete documents by filename."""
        filter_criteria = {
            "filename": {"$in": filenames}
        }
        
        if self.enable_soft_delete and strategy == DeletionStrategy.SOFT_DELETE:
            filter_criteria["is_active"] = True
        
        return self._execute_deletion(filter_criteria, strategy)
    
    def delete_by_hash(
        self, 
        content_hashes: List[str], 
        strategy: DeletionStrategy = DeletionStrategy.SOFT_DELETE
    ) -> Dict[str, Any]:
        """Delete documents by content hash."""
        filter_criteria = {
            "chunk_hash": {"$in": content_hashes}
        }
        
        if self.enable_soft_delete and strategy == DeletionStrategy.SOFT_DELETE:
            filter_criteria["is_active"] = True
        
        return self._execute_deletion(filter_criteria, strategy)
        
    def delete_by_document_hash(
        self, 
        document_hashes: List[str], 
        strategy: DeletionStrategy = DeletionStrategy.HARD_DELETE
    ) -> Dict[str, Any]:
        """Delete all chunks belonging to specific documents by their hashes."""
        results = {
            "strategy": strategy.value,
            "document_hashes": document_hashes,
            "deleted_count": 0,
            "errors": []
        }
        
        try:
            total_deleted = 0
            for doc_hash in document_hashes:
                # Use vector store's hash-based deletion if available
                if hasattr(self.vector_store, 'delete_by_document_hash'):
                    success = self.vector_store.delete_by_document_hash(doc_hash)
                    if success:
                        # We don't get exact count, so estimate
                        total_deleted += 1  # At least one document group deleted
                else:
                    # Fallback: try to delete by metadata filter
                    filter_criteria = {"document_hash": doc_hash}
                    delete_result = self._execute_deletion(filter_criteria, strategy)
                    total_deleted += delete_result.get("deleted_count", 0)
                    
            results["deleted_count"] = total_deleted
            
        except Exception as e:
            error_msg = f"Failed to delete documents by hash: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
            
        return results
        
    def delete_by_source_path(
        self, 
        source_paths: List[str], 
        strategy: DeletionStrategy = DeletionStrategy.HARD_DELETE
    ) -> Dict[str, Any]:
        """Delete all documents from specific source files."""
        results = {
            "strategy": strategy.value,
            "source_paths": source_paths,
            "deleted_count": 0,
            "errors": []
        }
        
        try:
            total_deleted = 0
            for source_path in source_paths:
                # Use vector store's source-based deletion if available
                if hasattr(self.vector_store, 'delete_by_source'):
                    success = self.vector_store.delete_by_source(source_path)
                    if success:
                        total_deleted += 1  # At least one source group deleted
                else:
                    # Fallback: try multiple metadata fields
                    for field in ['source', 'file_path', 'filepath']:
                        filter_criteria = {field: source_path}
                        delete_result = self._execute_deletion(filter_criteria, strategy)
                        total_deleted += delete_result.get("deleted_count", 0)
                        
            results["deleted_count"] = total_deleted
            
        except Exception as e:
            error_msg = f"Failed to delete documents by source: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
            
        return results
    
    def delete_expired_documents(
        self, 
        strategy: DeletionStrategy = DeletionStrategy.SOFT_DELETE
    ) -> Dict[str, Any]:
        """Delete documents that have passed their expiration date."""
        current_time = datetime.utcnow().isoformat() + "Z"
        
        filter_criteria = {
            "expires_at": {"$lt": current_time}
        }
        
        if self.enable_soft_delete and strategy == DeletionStrategy.SOFT_DELETE:
            filter_criteria["is_active"] = True
        
        return self._execute_deletion(filter_criteria, strategy)
    
    def cleanup_old_versions(
        self, 
        keep_versions: int = 5
    ) -> Dict[str, Any]:
        """Clean up old versions, keeping only the specified number of latest versions."""
        # This would require querying for documents grouped by doc_id
        # and keeping only the latest N versions
        # Implementation depends on vector store capabilities
        
        results = {
            "action": "cleanup_versions",
            "deleted_count": 0,
            "archived_count": 0,
            "errors": []
        }
        
        logger.info(f"Version cleanup requested (keep {keep_versions} versions)")
        # TODO: Implement version cleanup logic
        
        return results
    
    def _execute_deletion(
        self, 
        filter_criteria: Dict[str, Any], 
        strategy: DeletionStrategy
    ) -> Dict[str, Any]:
        """Execute deletion based on strategy."""
        results = {
            "strategy": strategy.value,
            "filter_criteria": filter_criteria,
            "deleted_count": 0,
            "archived_count": 0,
            "errors": []
        }
        
        try:
            if strategy == DeletionStrategy.SOFT_DELETE:
                # Mark as inactive instead of deleting
                update_metadata = {
                    "is_active": False,
                    "deleted_at": datetime.utcnow().isoformat() + "Z"
                }
                results["deleted_count"] = self.vector_store.update_metadata(
                    filter_criteria, update_metadata
                )
            
            elif strategy == DeletionStrategy.HARD_DELETE:
                # Permanently delete
                results["deleted_count"] = self.vector_store.delete_documents(
                    filter_criteria
                )
            
            elif strategy == DeletionStrategy.ARCHIVE_DELETE:
                # Archive then delete
                if self.archive_location:
                    results["archived_count"] = self._archive_documents(filter_criteria)
                results["deleted_count"] = self.vector_store.delete_documents(
                    filter_criteria
                )
            
            else:
                results["errors"].append(f"Unsupported deletion strategy: {strategy}")
        
        except Exception as e:
            results["errors"].append(f"Deletion failed: {str(e)}")
            logger.error(f"Deletion failed: {e}")
        
        return results
    
    def _archive_documents(self, filter_criteria: Dict[str, Any]) -> int:
        """Archive documents before deletion."""
        # This would involve exporting documents to archive location
        # Implementation depends on requirements
        logger.info(f"Archiving documents with criteria: {filter_criteria}")
        return 0  # Placeholder


class DocumentManager:
    """Main document management interface."""
    
    def __init__(self, vector_store, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.config = config
        self.lifecycle_manager = DocumentLifecycleManager(config)
        self.deletion_manager = DocumentDeletionManager(vector_store, config)
        self.hash_manager = DocumentHashManager(
            config.get("hash_algorithm", "sha256")
        )
    
    def process_documents(
        self, 
        documents: List[Document], 
        file_path: Optional[Path] = None,
        chunk_metadata_config: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process documents with comprehensive metadata enhancement."""
        enhanced_documents = []
        
        for doc in documents:
            enhanced_doc = self.lifecycle_manager.enhance_document_metadata(
                doc, file_path, chunk_metadata_config
            )
            enhanced_documents.append(enhanced_doc)
        
        return enhanced_documents
    
    def delete_documents(
        self, 
        doc_ids: List[str] = None,
        filenames: List[str] = None,
        content_hashes: List[str] = None,
        document_hashes: List[str] = None,
        source_paths: List[str] = None,
        older_than: int = None,
        expired: bool = False,
        strategy: DeletionStrategy = DeletionStrategy.SOFT_DELETE
    ) -> Dict[str, Any]:
        """Delete documents based on various criteria with enhanced hash-based support."""
        
        if doc_ids:
            # Check if any doc_ids are actually chunk IDs that we can extract document hashes from
            document_hashes_from_ids = []
            regular_doc_ids = []
            
            for doc_id in doc_ids:
                doc_hash = extract_document_hash_from_chunk_id(doc_id)
                if doc_hash:
                    document_hashes_from_ids.append(doc_hash)
                else:
                    regular_doc_ids.append(doc_id)
            
            # Delete by document hash first (more efficient)
            results = {"deleted_count": 0, "errors": []}
            if document_hashes_from_ids:
                hash_result = self.deletion_manager.delete_by_document_hash(
                    document_hashes_from_ids, strategy
                )
                results["deleted_count"] += hash_result.get("deleted_count", 0)
                results["errors"].extend(hash_result.get("errors", []))
            
            # Delete remaining regular IDs
            if regular_doc_ids:
                regular_result = self.deletion_manager.delete_by_document(regular_doc_ids, strategy)
                results["deleted_count"] += regular_result.get("deleted_count", 0)
                results["errors"].extend(regular_result.get("errors", []))
            
            return results
            
        elif document_hashes:
            return self.deletion_manager.delete_by_document_hash(document_hashes, strategy)
        elif source_paths:
            return self.deletion_manager.delete_by_source_path(source_paths, strategy)
        elif filenames:
            return self.deletion_manager.delete_by_filename(filenames, strategy)
        elif content_hashes:
            return self.deletion_manager.delete_by_hash(content_hashes, strategy)
        elif older_than is not None:
            return self.deletion_manager.delete_by_time(older_than, strategy)
        elif expired:
            return self.deletion_manager.delete_expired_documents(strategy)
        else:
            return {
                "error": "No deletion criteria specified",
                "deleted_count": 0
            }
    
    def find_duplicates(self, documents: List[Document]) -> Dict[str, List[str]]:
        """Find duplicate documents based on content hash."""
        hash_to_docs = {}
        
        for doc in documents:
            content_hash = doc.metadata.get("chunk_hash", "")
            if content_hash:
                if content_hash not in hash_to_docs:
                    hash_to_docs[content_hash] = []
                hash_to_docs[content_hash].append(doc.id)
        
        # Return only hashes with multiple documents
        return {h: docs for h, docs in hash_to_docs.items() if len(docs) > 1}
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get comprehensive document statistics."""
        try:
            # Use enhanced vector store method if available
            if hasattr(self.vector_store, 'get_collection_stats'):
                return self.vector_store.get_collection_stats()
            else:
                # Fallback to basic collection info
                info = self.vector_store.get_collection_info()
                return {
                    "total_documents": info.get("count", 0),
                    "active_documents": info.get("count", 0),
                    "deleted_documents": 0,
                    "expired_documents": 0,
                    "total_versions": 0,
                    "storage_size_mb": 0,
                }
        except Exception as e:
            logger.error(f"Failed to get document stats: {e}")
            return {
                "total_documents": 0,
                "active_documents": 0,
                "deleted_documents": 0,
                "expired_documents": 0,
                "total_versions": 0,
                "storage_size_mb": 0,
            }