"""Lightweight Extractor using YAKE + GLiNER for RAG metadata enrichment.

This extractor provides:
- YAKE: Fast unsupervised keyword extraction (no ML models needed)
- GLiNER: Zero-shot named entity recognition (lightweight transformer)
- Summary generation: First N sentences extraction

All components are lightweight and work without heavy dependencies like
llama-index or complex NLP pipelines.
"""

from typing import Any

from components.extractors.base import BaseExtractor
from core.base import Document, ProcessingResult
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.extractors.lightweight")


class LightweightExtractor(BaseExtractor):
    """Lightweight metadata extractor using YAKE and GLiNER.

    Extracts:
    - Keywords (via YAKE - unsupervised, fast)
    - Entities (via GLiNER - zero-shot NER)
    - Summary (first N sentences)

    All without heavy dependencies or large model downloads.
    """

    def __init__(
        self, name: str = "LightweightExtractor", config: dict[str, Any] | None = None
    ):
        """Initialize LightweightExtractor.

        Args:
            name: Extractor name
            config: Configuration dictionary with options:
                - keyword_count: Number of keywords to extract (default: 5)
                - summary_sentences: Number of sentences for summary (default: 3)
                - entity_labels: Labels for GLiNER NER (default: common entities)
                - gliner_model: GLiNER model name (default: urchade/gliner_small-v2.1)
                - use_gliner: Enable GLiNER (default: True if available)
                - yake_language: YAKE language (default: en)
                - yake_max_ngram: Max n-gram size (default: 3)
        """
        super().__init__(name, config)

        # Configuration with defaults
        self.keyword_count = self.config.get("keyword_count", 5)
        self.summary_sentences = self.config.get("summary_sentences", 3)
        self.entity_labels = self.config.get(
            "entity_labels",
            [
                "person",
                "organization",
                "location",
                "software",
                "date",
                "product",
                "technology",
                "company",
            ],
        )
        self.gliner_model = self.config.get("gliner_model", "urchade/gliner_small-v2.1")
        self.use_gliner = self.config.get("use_gliner", True)
        self.yake_language = self.config.get("yake_language", "en")
        self.yake_max_ngram = self.config.get("yake_max_ngram", 3)

        # Initialize YAKE
        self._yake_available = False
        self._yake_extractor = None
        try:
            import yake

            self._yake_extractor = yake.KeywordExtractor(
                lan=self.yake_language,
                n=self.yake_max_ngram,
                top=self.keyword_count,
                features=None,
            )
            self._yake_available = True
            logger.info("YAKE keyword extractor initialized")
        except ImportError:
            logger.warning("YAKE not available. Install with: pip install yake")

        # Initialize GLiNER (lazy load to avoid slow startup)
        self._gliner_model = None
        self._gliner_available = False
        if self.use_gliner:
            try:
                from gliner import GLiNER

                self._gliner_class = GLiNER
                self._gliner_available = True
                logger.info("GLiNER available (will lazy-load model)")
            except ImportError:
                logger.warning("GLiNER not available. Install with: pip install gliner")
                self._gliner_class = None

    def _load_gliner(self):
        """Lazy-load GLiNER model on first use."""
        if self._gliner_model is None and self._gliner_available:
            logger.info(f"Loading GLiNER model: {self.gliner_model}")
            self._gliner_model = self._gliner_class.from_pretrained(self.gliner_model)
            logger.info("GLiNER model loaded")

    def extract(self, documents: list[Document]) -> list[Document]:
        """Extract metadata from documents and enhance them.

        Args:
            documents: List of documents to process

        Returns:
            List of enhanced documents with extracted metadata
        """
        enriched_documents = []

        for doc in documents:
            try:
                enriched_doc = self._enrich_document(doc)
                enriched_documents.append(enriched_doc)
            except Exception as e:
                logger.error(f"Failed to enrich document {doc.id}: {e}")
                # Keep original document on error
                enriched_documents.append(doc)

        return enriched_documents

    def process(self, documents: list[Document]) -> ProcessingResult:
        """Process documents and extract metadata (alternative interface).

        Args:
            documents: List of documents to process

        Returns:
            ProcessingResult with enriched documents
        """
        enriched_documents = self.extract(documents)

        return ProcessingResult(
            documents=enriched_documents,
            errors=[],
            metrics={
                "total_documents": len(documents),
                "enriched_documents": len(enriched_documents),
                "extractor": self.name,
            },
        )

    def _enrich_document(self, doc: Document) -> Document:
        """Enrich a single document with extracted metadata.

        Args:
            doc: Document to enrich

        Returns:
            Document with enriched metadata
        """
        content = doc.content
        metadata = doc.metadata.copy()

        # Extract keywords with YAKE
        if self._yake_available:
            keywords = self._extract_keywords(content)
            metadata["keywords"] = keywords
            logger.debug(f"Extracted {len(keywords)} keywords")

        # Extract entities with GLiNER
        if self._gliner_available and self.use_gliner:
            entities = self._extract_entities(content)
            metadata["entities"] = entities
            logger.debug(f"Extracted {len(entities)} entities")

        # Generate summary
        summary = self._generate_summary(content)
        metadata["summary"] = summary

        # Add extractor info
        metadata["extractor"] = self.name
        metadata["extraction_methods"] = []
        if self._yake_available:
            metadata["extraction_methods"].append("yake_keywords")
        if self._gliner_available and self.use_gliner:
            metadata["extraction_methods"].append("gliner_entities")
        metadata["extraction_methods"].append("sentence_summary")

        return Document(
            content=doc.content,
            metadata=metadata,
            id=doc.id,
            source=doc.source,
            embeddings=doc.embeddings,
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords using YAKE.

        YAKE is unsupervised and language-independent, making it
        fast and reliable without needing external APIs.

        Args:
            text: Text to extract keywords from

        Returns:
            List of keywords
        """
        if not self._yake_available or not self._yake_extractor:
            return []

        try:
            # YAKE returns list of (keyword, score) tuples
            # Lower score = more important keyword
            keywords = self._yake_extractor.extract_keywords(text)
            return [kw for kw, score in keywords]
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    def _extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities using GLiNER.

        GLiNER provides zero-shot NER - it can extract any entity type
        you specify without fine-tuning.

        Args:
            text: Text to extract entities from

        Returns:
            List of entity dictionaries with 'text', 'label', 'score'
        """
        if not self._gliner_available:
            return []

        try:
            # Lazy-load the model
            self._load_gliner()

            if self._gliner_model is None:
                return []

            # Truncate very long texts (GLiNER has context limits)
            max_length = 4096
            if len(text) > max_length:
                text = text[:max_length]

            # Extract entities
            entities = self._gliner_model.predict_entities(
                text, self.entity_labels, threshold=0.5
            )

            # Convert to serializable format
            return [
                {
                    "text": ent["text"],
                    "label": ent["label"],
                    "score": round(float(ent["score"]), 4),
                }
                for ent in entities
            ]

        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    def _generate_summary(self, text: str) -> str:
        """Generate summary by extracting first N sentences.

        Simple but effective - no ML models needed.

        Args:
            text: Text to summarize

        Returns:
            Summary string
        """
        import re

        # Split into sentences (simple regex approach)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        # Take first N sentences
        summary_sentences = sentences[: self.summary_sentences]

        # Join and return
        summary = " ".join(summary_sentences)

        # Truncate if too long
        max_length = 500
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(" ", 1)[0] + "..."

        return summary

    def get_dependencies(self) -> list[str]:
        """Return extractor dependencies.

        Returns:
            List of required package names
        """
        return ["yake"]

    def validate_config(self) -> bool:
        """Validate configuration."""
        if self.keyword_count < 1:
            raise ValueError("keyword_count must be at least 1")
        if self.summary_sentences < 1:
            raise ValueError("summary_sentences must be at least 1")
        if not self.entity_labels:
            raise ValueError("entity_labels cannot be empty")
        return True
