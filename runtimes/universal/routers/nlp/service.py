"""NLP service for text analysis operations.

Handles:
- Language detection
- Keyword extraction
- PII detection and redaction
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from services.model_manager import ModelManager, ModelType, model_manager
from utils.device import get_optimal_device

if TYPE_CHECKING:
    from models import LanguageDetectionModel, PIIModel
    from utils.keyword_extractor import KeywordExtractor

logger = logging.getLogger(__name__)

# Global device cache
_current_device: str | None = None

# Global keyword extractor (singleton)
_keyword_extractor: "KeywordExtractor | None" = None


def get_device() -> str:
    """Get the optimal device for the current platform."""
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device


class NLPService:
    """Service for NLP operations."""

    def __init__(self, manager: ModelManager | None = None):
        """Initialize the NLP service.

        Args:
            manager: Optional ModelManager instance. Uses global singleton if not provided.
        """
        self._manager = manager or model_manager

    # =========================================================================
    # Language Detection
    # =========================================================================

    async def load_language_model(
        self, model_name: str = "papluca/xlm-roberta-base-language-detection"
    ) -> "LanguageDetectionModel":
        """Load a language detection model.

        Args:
            model_name: HuggingFace model name

        Returns:
            Loaded LanguageDetectionModel instance
        """
        from models import LanguageDetectionModel

        cache_key = f"language:{model_name}"

        async def loader():
            logger.info(f"Loading language detection model: {model_name}")
            device = get_device()
            model = LanguageDetectionModel(
                model_id=cache_key,
                device=device,
                hf_model_name=model_name,
            )
            await model.load()
            return model

        return await self._manager.get_or_load(
            ModelType.LANG_DETECTION, cache_key, loader
        )

    async def detect_language(
        self,
        text: str,
        top_k: int = 5,
        model_name: str = "papluca/xlm-roberta-base-language-detection",
    ) -> dict:
        """Detect the language of a text.

        Args:
            text: Text to analyze
            top_k: Number of top predictions
            model_name: Model to use

        Returns:
            Detection result with language, confidence, and all scores
        """
        model = await self.load_language_model(model_name)
        result = await model.detect(text, top_k=top_k)
        return {
            "language": result["language"],
            "language_name": result["language_name"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
        }

    async def detect_language_batch(
        self,
        texts: list[str],
        top_k: int = 1,
        model_name: str = "papluca/xlm-roberta-base-language-detection",
    ) -> list[dict]:
        """Detect languages for multiple texts.

        Args:
            texts: Texts to analyze
            top_k: Number of top predictions per text
            model_name: Model to use

        Returns:
            List of detection results
        """
        model = await self.load_language_model(model_name)
        results = await model.detect_batch(texts, top_k=top_k)
        return [
            {
                "language": r["language"],
                "language_name": r["language_name"],
                "confidence": r["confidence"],
                "all_scores": r["all_scores"],
            }
            for r in results
        ]

    # =========================================================================
    # Keyword Extraction
    # =========================================================================

    async def get_keyword_extractor(self) -> "KeywordExtractor":
        """Get or create the keyword extractor singleton.

        Returns:
            KeywordExtractor instance
        """
        global _keyword_extractor
        if _keyword_extractor is None:
            from utils.keyword_extractor import KeywordExtractor

            _keyword_extractor = KeywordExtractor()
            logger.info("Initialized keyword extractor")
        return _keyword_extractor

    async def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        ngram_range: tuple[int, int] = (1, 3),
        diversity: float = 0.5,
    ) -> list[dict]:
        """Extract keywords from text.

        Args:
            text: Text to analyze
            top_k: Number of keywords to return
            ngram_range: Min and max n-gram sizes
            diversity: Diversity parameter (0-1)

        Returns:
            List of keyword dicts with 'keyword' and 'score' keys
        """
        extractor = await self.get_keyword_extractor()
        loop = asyncio.get_running_loop()
        keywords = await loop.run_in_executor(
            None,
            lambda: extractor.extract(
                text,
                top_k=top_k,
                ngram_range=ngram_range,
                diversity=diversity,
            ),
        )
        return keywords

    async def extract_keywords_batch(
        self,
        texts: list[str],
        top_k: int = 10,
        ngram_range: tuple[int, int] = (1, 3),
        diversity: float = 0.5,
    ) -> list[list[dict]]:
        """Extract keywords from multiple texts.

        Args:
            texts: Texts to analyze
            top_k: Number of keywords per text
            ngram_range: Min and max n-gram sizes
            diversity: Diversity parameter

        Returns:
            List of keyword lists
        """
        extractor = await self.get_keyword_extractor()
        loop = asyncio.get_running_loop()

        results = []
        for text in texts:
            keywords = await loop.run_in_executor(
                None,
                lambda t=text: extractor.extract(
                    t,
                    top_k=top_k,
                    ngram_range=ngram_range,
                    diversity=diversity,
                ),
            )
            results.append(keywords)
        return results

    # =========================================================================
    # PII Detection and Redaction
    # =========================================================================

    async def load_pii_model(
        self, model_name: str = "urchade/gliner_small-v2.1"
    ) -> "PIIModel":
        """Load a PII detection model.

        Args:
            model_name: HuggingFace GLiNER model name

        Returns:
            Loaded PIIModel instance
        """
        from models import PIIModel

        cache_key = f"pii:{model_name}"

        async def loader():
            logger.info(f"Loading PII detection model: {model_name}")
            device = get_device()
            model = PIIModel(
                model_id=cache_key,
                device=device,
                hf_model_name=model_name,
            )
            await model.load()
            return model

        return await self._manager.get_or_load(ModelType.PII, cache_key, loader)

    async def detect_pii(
        self,
        text: str,
        entity_types: list[str] | None = None,
        threshold: float = 0.5,
        use_regex: bool = True,
    ) -> list[dict]:
        """Detect PII in text.

        Args:
            text: Text to analyze
            entity_types: Custom entity types (default: standard PII)
            threshold: Detection confidence threshold
            use_regex: Whether to also use regex patterns

        Returns:
            List of detected entities
        """
        model = await self.load_pii_model()
        entities = await model.detect(
            text,
            entity_types=entity_types,
            threshold=threshold,
            use_regex=use_regex,
        )
        return entities

    async def redact_pii(
        self,
        text: str,
        entity_types: list[str] | None = None,
        replacement: str = "[REDACTED]",
        replacement_map: dict[str, str] | None = None,
        threshold: float = 0.5,
        use_regex: bool = True,
    ) -> dict:
        """Detect and redact PII from text.

        Args:
            text: Text to redact
            entity_types: Entity types to redact
            replacement: Default replacement string
            replacement_map: Per-type replacement strings
            threshold: Detection confidence threshold
            use_regex: Whether to also use regex patterns

        Returns:
            Dict with redacted_text and entities
        """
        model = await self.load_pii_model()
        result = await model.redact(
            text,
            entity_types=entity_types,
            replacement=replacement,
            replacement_map=replacement_map,
            threshold=threshold,
            use_regex=use_regex,
        )
        return result


# Global service instance
nlp_service = NLPService()
