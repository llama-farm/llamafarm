"""Document management system with hash-based tracking and lifecycle management."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.base import Document
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.core.document_manager")


class DocumentHashManager:
    """Manages document hashing for deduplication and change detection."""

    def __init__(self, hash_algorithm: str = "sha256"):
        self.hash_algorithm = hash_algorithm
        self._hash_func = getattr(hashlib, hash_algorithm)

    def generate_file_hash(self, file_path: Path) -> str:
        """Generate hash for entire file."""
        hash_obj = self._hash_func()
        try:
            with open(file_path, "rb") as f:
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
        normalized = " ".join(normalized.split())  # Normalize whitespace

        hash_obj = self._hash_func()
        hash_obj.update(normalized.encode("utf-8"))
        return f"{self.hash_algorithm}:{hash_obj.hexdigest()}"

    def generate_metadata_hash(self, metadata: dict[str, Any]) -> str:
        """Generate hash for metadata (excluding timestamps)."""
        # Create filtered metadata for hashing
        hash_metadata = {
            k: v
            for k, v in metadata.items()
            if k not in ["created_at", "updated_at", "indexed_at", "metadata_hash"]
        }

        # Sort for consistent hashing
        metadata_str = json.dumps(hash_metadata, sort_keys=True)
        hash_obj = self._hash_func()
        hash_obj.update(metadata_str.encode("utf-8"))
        return f"{self.hash_algorithm}:{hash_obj.hexdigest()}"

    def generate_composite_hash(self, *components: str) -> str:
        """Generate hash from multiple components."""
        combined = "|".join(str(c) for c in components)
        hash_obj = self._hash_func()
        hash_obj.update(combined.encode("utf-8"))
        return f"{self.hash_algorithm}:{hash_obj.hexdigest()}"


class DocumentLifecycleManager:
    """Manages document lifecycle including versioning, expiration, and cleanup."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.hash_manager = DocumentHashManager(config.get("hash_algorithm", "sha256"))
        self.retention_policy = config.get("retention_policy", {})
        self.enable_versioning = config.get("enable_versioning", True)
        self.enable_soft_delete = config.get("enable_soft_delete", True)

    def enhance_document_metadata(
        self,
        document: Document,
        file_path: Path | None = None,
        chunk_metadata_config: dict[str, Any] | None = None,
    ) -> Document:
        """Enhance document with comprehensive metadata."""

        # Generate basic metadata
        current_time = datetime.utcnow().isoformat() + "Z"

        # Create base metadata
        base_metadata = {
            "doc_id": document.metadata.get("doc_id", document.id),
            "chunk_id": document.id,
            "filename": file_path.name
            if file_path
            else document.metadata.get("filename", ""),
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

        base_metadata["chunk_hash"] = self.hash_manager.generate_content_hash(
            document.content
        )

        # Add chunk-specific metadata if configured
        if chunk_metadata_config:
            chunk_metadata = self._generate_chunk_metadata(
                document, chunk_metadata_config
            )
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
        self, document: Document, config: dict[str, Any]
    ) -> dict[str, Any]:
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
            analysis = self._perform_content_analysis(
                document.content, content_analysis
            )
            metadata.update(analysis)

        return metadata

    def _generate_summary(self, content: str, max_length: int = 150) -> str:
        """Generate a summary of the content."""
        # Simple extractive summary - first meaningful sentences
        sentences = content.split(". ")
        summary = ""
        for sentence in sentences:
            if len(summary + sentence) > max_length:
                break
            summary += sentence + ". "
        return summary.strip()

    def _extract_keywords(self, content: str, max_keywords: int = 10) -> list[str]:
        """Extract keywords from content."""
        # Simple keyword extraction based on word frequency
        import re
        from collections import Counter

        # Clean and tokenize
        words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())
        # Remove common stop words
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "are",
            "was",
            "were",
            "been",
            "have",
            "has",
            "had",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
        }
        keywords = [word for word in words if word not in stop_words]

        # Get most common keywords
        word_counts = Counter(keywords)
        return [word for word, _ in word_counts.most_common(max_keywords)]

    def _calculate_content_statistics(self, content: str) -> dict[str, Any]:
        """Calculate content statistics."""
        words = len(content.split())
        characters = len(content)
        sentences = len([s for s in content.split(".") if s.strip()])

        return {
            "word_count": words,
            "character_count": characters,
            "sentence_count": sentences,
            "reading_time_minutes": words / 200.0,  # Assume 200 words per minute
            "avg_sentence_length": words / max(sentences, 1),
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
        if any(word in content_lower for word in ["error", "bug", "issue", "problem"]):
            return "technical_issue"
        elif any(word in content_lower for word in ["password", "login", "access"]):
            return "access_issue"
        elif any(word in content_lower for word in ["payment", "billing", "invoice"]):
            return "billing_issue"
        else:
            return "general"

    def _analyze_sentiment(self, content: str) -> str:
        """Simple sentiment analysis."""
        positive_words = {"good", "great", "excellent", "happy", "satisfied", "thank"}
        negative_words = {"bad", "terrible", "awful", "angry", "frustrated", "urgent"}

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
        urgent_indicators = ["urgent", "emergency", "critical", "asap", "immediately"]
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
        self, content: str, analysis_config: dict[str, Any]
    ) -> dict[str, Any]:
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
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
        }
        words = content.lower().split()
        english_word_count = sum(
            1 for word in words[:50] if word in common_english_words
        )

        return "en" if english_word_count > 5 else "unknown"

    def _extract_key_phrases(self, content: str) -> list[str]:
        """Extract key phrases from content."""
        # Simple n-gram extraction
        import re

        # Extract 2-3 word phrases that might be meaningful
        phrases = []
        words = re.findall(r"\b[A-Za-z]+\b", content)

        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i + 1]) > 3:
                phrase = f"{words[i]} {words[i + 1]}"
                if phrase not in phrases:
                    phrases.append(phrase)

        return phrases[:10]  # Return top 10 phrases

    def _identify_document_type(self, content: str) -> str:
        """Identify document type based on content patterns."""
        content_lower = content.lower()

        if any(term in content_lower for term in ["contract", "agreement", "terms"]):
            return "legal_document"
        elif any(term in content_lower for term in ["report", "analysis", "findings"]):
            return "report"
        elif any(term in content_lower for term in ["manual", "instructions", "guide"]):
            return "documentation"
        elif any(
            term in content_lower for term in ["email", "message", "correspondence"]
        ):
            return "communication"
        else:
            return "general_document"


class DocumentDeletionManager:
    """Manages document deletion by file_hash."""

    def __init__(self, vector_store, config: dict[str, Any] | None = None):
        self.vector_store = vector_store
        self.config = config or {}

    def delete_by_file_hash(self, file_hash: str) -> dict[str, Any]:
        """Delete all chunks belonging to a file by its content hash.

        This is the primary deletion interface. The file_hash is the SHA-256
        hash of the original file content, which is stored in chunk metadata.

        Args:
            file_hash: SHA-256 hash of the original file content.

        Returns:
            Dictionary with deletion results.
        """
        results = {
            "file_hash": file_hash,
            "deleted_count": 0,
            "error": None,
        }

        try:
            # Step 1: Find all documents with this file_hash
            documents = self.vector_store.get_documents_by_metadata(
                {"file_hash": file_hash}
            )

            if not documents:
                logger.info(f"No chunks found with file_hash {file_hash[:16]}...")
                return results

            # Step 2: Delete by document IDs
            doc_ids = [doc.id for doc in documents]
            deleted_count = self.vector_store.delete_documents(doc_ids)

            results["deleted_count"] = deleted_count
            logger.info(
                f"Deleted {deleted_count} chunks for file_hash {file_hash[:16]}..."
            )

        except Exception as e:
            error_msg = f"Failed to delete by file_hash: {e}"
            results["error"] = error_msg
            logger.error(error_msg)

        return results


class DocumentManager:
    """Main document management interface."""

    def __init__(self, vector_store, config: dict[str, Any] | None = None):
        self.vector_store = vector_store
        self.config = config or {}
        self.lifecycle_manager = DocumentLifecycleManager(self.config)
        self.deletion_manager = DocumentDeletionManager(vector_store, self.config)
        self.hash_manager = DocumentHashManager(
            self.config.get("hash_algorithm", "sha256")
        )

    def process_documents(
        self,
        documents: list[Document],
        file_path: Path | None = None,
        chunk_metadata_config: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Process documents with comprehensive metadata enhancement."""
        enhanced_documents = []

        for doc in documents:
            enhanced_doc = self.lifecycle_manager.enhance_document_metadata(
                doc, file_path, chunk_metadata_config
            )
            enhanced_documents.append(enhanced_doc)

        return enhanced_documents

    def delete_by_file_hash(self, file_hash: str) -> dict[str, Any]:
        """Delete all chunks belonging to a file by its content hash.

        This is the primary deletion interface. The file_hash is the SHA-256
        hash of the original file content.

        Args:
            file_hash: SHA-256 hash of the original file content.

        Returns:
            Dictionary with deletion results including deleted_count.
        """
        return self.deletion_manager.delete_by_file_hash(file_hash)

    def get_document_stats(self) -> dict[str, Any]:
        """Get document statistics from the vector store."""
        try:
            if hasattr(self.vector_store, "get_collection_stats"):
                return self.vector_store.get_collection_stats()
            else:
                info = self.vector_store.get_collection_info()
                return {
                    "total_documents": info.get("count", 0),
                }
        except Exception as e:
            logger.error(f"Failed to get document stats: {e}")
            return {"total_documents": 0}
