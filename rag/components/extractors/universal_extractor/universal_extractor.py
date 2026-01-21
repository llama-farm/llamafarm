"""Universal Extractor for comprehensive metadata extraction.

This extractor enhances documents with rich metadata including:
- Document-level: name, path, type, size, author, title, dates
- Chunk-level: position, word/character/sentence counts
- Extracted features: keywords, summary, language detection
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from components.extractors.base import BaseExtractor
from core.base import Document
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.extractors.universal_extractor")

# Try to import optional dependencies
try:
    import yake

    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    yake = None  # type: ignore

# GLiNER is optional for entity extraction
try:
    from gliner import GLiNER

    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    GLiNER = None  # type: ignore


class UniversalExtractor(BaseExtractor):
    """Universal metadata extractor for comprehensive document enrichment.

    Features:
    - Document-level metadata: name, path, type, size, author, title, dates
    - Chunk-level metadata: position labels, word/character/sentence counts
    - Keyword extraction via YAKE
    - Entity extraction via GLiNER (optional)
    - Summary generation (first N sentences)
    - Language detection
    - Table/code block detection
    - Graceful degradation when dependencies unavailable
    """

    def __init__(self, name: str | None = None, config: dict[str, Any] | None = None):
        """Initialize the UniversalExtractor.

        Args:
            name: Extractor name (default: UniversalExtractor)
            config: Configuration with optional keys:
                - keyword_count: Number of keywords to extract (default: 10)
                - use_gliner: Enable GLiNER entity extraction (default: False)
                - extract_entities: Extract named entities (default: True)
                - generate_summary: Generate chunk summaries (default: True)
                - summary_sentences: Number of sentences in summary (default: 3)
                - detect_language: Detect document language (default: True)
        """
        super().__init__(name=name or "UniversalExtractor", config=config)

        # Extract config with defaults
        self.keyword_count = self.config.get("keyword_count", 10)
        self.use_gliner = self.config.get("use_gliner", False)
        self.extract_entities = self.config.get("extract_entities", True)
        self.generate_summary = self.config.get("generate_summary", True)
        self.summary_sentences = self.config.get("summary_sentences", 3)
        self.detect_language = self.config.get("detect_language", True)

        # Initialize YAKE keyword extractor
        self._yake_extractor = None
        if YAKE_AVAILABLE:
            try:
                self._yake_extractor = yake.KeywordExtractor(
                    n=3,  # max n-gram size
                    dedupLim=0.7,  # deduplication threshold
                    top=self.keyword_count,
                    features=None,
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize YAKE: {e}")

        # Initialize GLiNER if requested
        self._gliner_model = None
        if self.use_gliner and GLINER_AVAILABLE:
            try:
                self._gliner_model = GLiNER.from_pretrained(
                    "urchade/gliner_small-v2.1"
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize GLiNER: {e}")

        self.logger.info(
            "UniversalExtractor initialized",
            keyword_count=self.keyword_count,
            yake_available=YAKE_AVAILABLE and self._yake_extractor is not None,
            gliner_available=GLINER_AVAILABLE and self._gliner_model is not None,
        )

    def get_dependencies(self) -> list[str]:
        """Get required dependencies for this extractor.

        Returns:
            List of required package names (none required, all optional)
        """
        # All dependencies are optional for graceful degradation
        return []

    def extract(self, documents: list[Document]) -> list[Document]:
        """Extract and enhance metadata for documents.

        Args:
            documents: List of documents to process

        Returns:
            List of enhanced documents with extracted metadata
        """
        if not documents:
            return documents

        enhanced_docs: list[Document] = []
        processing_time = datetime.now(UTC).isoformat()

        for doc in documents:
            try:
                enhanced_doc = self._enhance_document(doc, processing_time)
                enhanced_docs.append(enhanced_doc)
            except Exception as e:
                self.logger.error(f"Failed to enhance document: {e}", doc_id=doc.id)
                enhanced_docs.append(doc)

        self.logger.info(
            "Extraction complete",
            documents_processed=len(enhanced_docs),
        )

        return enhanced_docs

    def _enhance_document(
        self, doc: Document, processing_time: str
    ) -> Document:
        """Enhance a single document with extracted metadata.

        Args:
            doc: Document to enhance
            processing_time: ISO timestamp of processing

        Returns:
            Enhanced document
        """
        content = doc.content
        metadata = dict(doc.metadata)  # Copy existing metadata

        # Add processing timestamp
        metadata["processed_at"] = processing_time
        metadata["extractor"] = "UniversalExtractor"

        # Extract document-level metadata from source path (only if not already set)
        if doc.source:
            source_path = Path(doc.source)
            if source_path.exists():
                file_metadata = self._extract_file_metadata(source_path, metadata)
                # Only add keys that don't exist in metadata
                for key, value in file_metadata.items():
                    if key not in metadata:
                        metadata[key] = value

        # Extract chunk-level metadata
        metadata.update(self._extract_chunk_metadata(content, metadata))

        # Extract features
        metadata.update(self._extract_features(content))

        # Update document
        doc.metadata = metadata
        return doc

    def _extract_file_metadata(
        self, file_path: Path, existing_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract metadata from file system if not already present.

        Args:
            file_path: Path to the source file
            existing_metadata: The document's current metadata

        Returns:
            Dictionary with file metadata (only fields not in existing_metadata)
        """
        metadata: dict[str, Any] = {}

        try:
            stat = file_path.stat()

            # Basic file info - only set if not already present
            if "document_name" not in existing_metadata:
                metadata["document_name"] = file_path.name
            if "document_path" not in existing_metadata:
                metadata["document_path"] = str(file_path.absolute())
            if "document_type" not in existing_metadata:
                metadata["document_type"] = file_path.suffix.lower()
            if "document_size" not in existing_metadata:
                metadata["document_size"] = stat.st_size

            # Timestamps - only set if not already present
            if "created_date" not in existing_metadata:
                metadata["created_date"] = datetime.fromtimestamp(
                    stat.st_birthtime
                    if hasattr(stat, "st_birthtime")
                    else stat.st_ctime,
                    tz=UTC,
                ).isoformat()
            if "modified_date" not in existing_metadata:
                metadata["modified_date"] = datetime.fromtimestamp(
                    stat.st_mtime, tz=UTC
                ).isoformat()

        except Exception as e:
            self.logger.warning(f"Failed to extract file metadata: {e}")

        return metadata

    def _extract_chunk_metadata(
        self, content: str, existing_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract chunk-level metadata from content.

        Args:
            content: Chunk content
            existing_metadata: Existing metadata (may have chunk_index, etc.)

        Returns:
            Dictionary with chunk metadata
        """
        metadata: dict[str, Any] = {}

        # Character, word, sentence counts
        metadata["character_count"] = len(content)
        metadata["word_count"] = len(content.split())
        metadata["sentence_count"] = self._count_sentences(content)

        # Chunk position label if chunk info available
        chunk_index = existing_metadata.get("chunk_index")
        total_chunks = existing_metadata.get("total_chunks")

        if chunk_index is not None and total_chunks is not None:
            # Label: "1/5" format
            metadata["chunk_label"] = f"{chunk_index + 1}/{total_chunks}"

            # Position: start, middle, end, only
            if total_chunks == 1:
                metadata["chunk_position"] = "only"
            elif chunk_index == 0:
                metadata["chunk_position"] = "start"
            elif chunk_index == total_chunks - 1:
                metadata["chunk_position"] = "end"
            else:
                metadata["chunk_position"] = "middle"

        return metadata

    def _extract_features(self, content: str) -> dict[str, Any]:
        """Extract features from content (keywords, summary, etc.).

        Args:
            content: Chunk content

        Returns:
            Dictionary with extracted features
        """
        features: dict[str, Any] = {}

        # Extract keywords
        if self._yake_extractor and content.strip():
            features["keywords"] = self._extract_keywords(content)

        # Extract entities
        if self.extract_entities and self._gliner_model:
            features["entities"] = self._extract_entities(content)

        # Generate summary
        if self.generate_summary:
            features["summary"] = self._generate_summary(content)

        # Detect language (simple heuristic)
        if self.detect_language:
            features["language"] = self._detect_language(content)

        # Detect tables and code blocks
        features["has_tables"] = self._has_tables(content)
        features["has_code"] = self._has_code_blocks(content)

        return features

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords using YAKE.

        Args:
            text: Text to extract keywords from

        Returns:
            List of keywords
        """
        if not self._yake_extractor:
            return []

        try:
            keywords = self._yake_extractor.extract_keywords(text)
            # YAKE returns (keyword, score) tuples, lower score = more important
            return [kw for kw, _ in keywords[:self.keyword_count]]
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {e}")
            return []

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract named entities using GLiNER.

        Args:
            text: Text to extract entities from

        Returns:
            Dictionary mapping entity types to entity values
        """
        if not self._gliner_model:
            return {}

        try:
            labels = ["person", "organization", "location", "date", "product"]
            entities = self._gliner_model.predict_entities(
                text, labels, threshold=0.5
            )

            # Group by entity type
            result: dict[str, list[str]] = {}
            for entity in entities:
                entity_type = entity.get("label", "unknown")
                entity_text = entity.get("text", "")
                if entity_type not in result:
                    result[entity_type] = []
                if entity_text and entity_text not in result[entity_type]:
                    result[entity_type].append(entity_text)

            return result
        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {e}")
            return {}

    def _generate_summary(self, text: str) -> str:
        """Generate a summary from the first N sentences.

        Args:
            text: Text to summarize

        Returns:
            Summary string
        """
        sentences = self._split_sentences(text)
        summary_sentences = sentences[: self.summary_sentences]
        return " ".join(summary_sentences)

    def _detect_language(self, text: str) -> str:
        """Detect language using simple heuristics.

        Args:
            text: Text to analyze

        Returns:
            Language code (default: 'en')
        """
        # Simple heuristic: check for common non-English characters
        text_lower = text.lower()

        # Spanish indicators
        spanish_chars = "áéíóúñ¿¡"
        if any(c in text_lower for c in spanish_chars):
            return "es"

        # French indicators
        french_chars = "àâäéèêëïîôùûç"
        if any(c in text_lower for c in french_chars):
            return "fr"

        # German indicators
        german_chars = "äöüß"
        if any(c in text_lower for c in german_chars):
            return "de"

        # Chinese indicators
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        if chinese_pattern.search(text):
            return "zh"

        # Japanese indicators (Hiragana/Katakana)
        japanese_pattern = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
        if japanese_pattern.search(text):
            return "ja"

        # Default to English
        return "en"

    def _has_tables(self, text: str) -> bool:
        """Check if text contains table-like structures.

        Args:
            text: Text to check

        Returns:
            True if tables detected
        """
        # Markdown table pattern (header row + separator row with dashes/colons)
        table_pattern = re.compile(r"\|[^\n]+\|\s*\n\s*\|[-:\s|]+\|")
        if table_pattern.search(text):
            return True

        # Tab-separated rows pattern
        lines = text.split("\n")
        tab_rows = sum(1 for line in lines if line.count("\t") >= 2)
        return tab_rows >= 3

    def _has_code_blocks(self, text: str) -> bool:
        """Check if text contains code blocks.

        Args:
            text: Text to check

        Returns:
            True if code blocks detected
        """
        # Markdown code fence
        if "```" in text:
            return True

        # Indented code block (4+ spaces at start)
        indented_pattern = re.compile(r"^\s{4,}\S", re.MULTILINE)
        if indented_pattern.search(text):
            return True

        # Common code patterns
        code_patterns = [
            r"def\s+\w+\s*\(",  # Python function
            r"function\s+\w+\s*\(",  # JS function
            r"class\s+\w+\s*[:{]",  # Class definition
            r"import\s+\w+",  # Import statement
            r"from\s+\w+\s+import",  # Python import
        ]
        return any(re.search(pattern, text) for pattern in code_patterns)

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text.

        Args:
            text: Text to analyze

        Returns:
            Sentence count
        """
        # Return 0 for empty or whitespace-only content
        if not text or not text.strip():
            return 0
        # Count sentence-ending punctuation followed by space or end
        endings = len(re.findall(r"[.!?]+(?:\s|$)", text))
        return max(endings, 1)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Split on sentence boundaries
        sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        sentences = sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
