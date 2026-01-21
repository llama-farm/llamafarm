"""NLP API router with 6 endpoints for text analysis.

Endpoints:
- Language detection: POST /v1/ml/nlp/language, POST /v1/ml/nlp/language/batch
- Keyword extraction: POST /v1/ml/nlp/keywords, POST /v1/ml/nlp/keywords/batch
- PII handling: POST /v1/ml/nlp/pii/detect, POST /v1/ml/nlp/redact
"""

import logging

from fastapi import APIRouter, HTTPException

from .service import nlp_service
from .types import (
    KeywordExtractBatchRequest,
    KeywordExtractRequest,
    LanguageDetectBatchRequest,
    LanguageDetectRequest,
    PIIDetectRequest,
    PIIRedactRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# Language Detection
# =============================================================================


@router.post("/v1/ml/nlp/language")
async def detect_language(request: LanguageDetectRequest):
    """
    Detect the language of a text.

    Uses XLM-RoBERTa fine-tuned for language detection. Supports 20 languages:
    Arabic, Bulgarian, German, Greek, English, Spanish, French, Hindi, Italian,
    Japanese, Dutch, Polish, Portuguese, Russian, Swahili, Thai, Turkish,
    Urdu, Vietnamese, Chinese.

    Example request:
    ```json
    {
        "text": "Hello, how are you today?",
        "top_k": 5
    }
    ```

    Response:
    ```json
    {
        "object": "language_detection",
        "language": "en",
        "language_name": "English",
        "confidence": 0.99,
        "all_scores": {"en": 0.99, "de": 0.005, ...}
    }
    ```
    """
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty",
            )

        result = await nlp_service.detect_language(request.text, top_k=request.top_k)

        return {
            "object": "language_detection",
            "language": result["language"],
            "language_name": result["language_name"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_language: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/ml/nlp/language/batch")
async def detect_language_batch(request: LanguageDetectBatchRequest):
    """
    Detect the language of multiple texts.

    More efficient than calling single endpoint multiple times.

    Example request:
    ```json
    {
        "texts": ["Hello world", "Bonjour le monde", "Hallo Welt"],
        "top_k": 1
    }
    ```

    Response:
    ```json
    {
        "object": "list",
        "data": [
            {"language": "en", "language_name": "English", "confidence": 0.99},
            {"language": "fr", "language_name": "French", "confidence": 0.98},
            {"language": "de", "language_name": "German", "confidence": 0.97}
        ],
        "total_count": 3
    }
    ```
    """
    try:
        if not request.texts:
            raise HTTPException(
                status_code=400,
                detail="At least one text is required",
            )

        results = await nlp_service.detect_language_batch(
            request.texts, top_k=request.top_k
        )

        return {
            "object": "list",
            "data": results,
            "total_count": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_language_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Keyword Extraction
# =============================================================================


@router.post("/v1/ml/nlp/keywords")
async def extract_keywords(request: KeywordExtractRequest):
    """
    Extract keywords and keyphrases from text.

    Uses sentence embeddings to find the most relevant n-grams in the text.
    Supports diversity parameter to avoid redundant keywords.

    Example request:
    ```json
    {
        "text": "Machine learning is a subset of artificial intelligence...",
        "top_k": 10,
        "diversity": 0.5,
        "ngram_range": [1, 3]
    }
    ```

    Response:
    ```json
    {
        "object": "keyword_extraction",
        "keywords": [
            {"keyword": "machine learning", "score": 0.87},
            {"keyword": "artificial intelligence", "score": 0.82}
        ],
        "count": 10
    }
    ```
    """
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty",
            )

        if len(request.ngram_range) != 2:
            raise HTTPException(
                status_code=400,
                detail="ngram_range must be a list of exactly 2 integers [min, max]",
            )

        keywords = await nlp_service.extract_keywords(
            request.text,
            top_k=request.top_k,
            ngram_range=tuple(request.ngram_range),
            diversity=request.diversity,
        )

        return {
            "object": "keyword_extraction",
            "keywords": keywords,
            "count": len(keywords),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_keywords: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/v1/ml/nlp/keywords/batch")
async def extract_keywords_batch(request: KeywordExtractBatchRequest):
    """
    Extract keywords from multiple texts.

    Example request:
    ```json
    {
        "texts": ["First document...", "Second document..."],
        "top_k": 5
    }
    ```

    Response:
    ```json
    {
        "object": "list",
        "data": [
            {"keywords": [...], "count": 5},
            {"keywords": [...], "count": 5}
        ],
        "total_count": 2
    }
    ```
    """
    try:
        if not request.texts:
            raise HTTPException(
                status_code=400,
                detail="At least one text is required",
            )

        if len(request.ngram_range) != 2:
            raise HTTPException(
                status_code=400,
                detail="ngram_range must be a list of exactly 2 integers [min, max]",
            )

        results = await nlp_service.extract_keywords_batch(
            request.texts,
            top_k=request.top_k,
            ngram_range=tuple(request.ngram_range),
            diversity=request.diversity,
        )

        data = [{"keywords": kw, "count": len(kw)} for kw in results]

        return {
            "object": "list",
            "data": data,
            "total_count": len(data),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_keywords_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


# =============================================================================
# PII Detection and Redaction
# =============================================================================


@router.post("/v1/ml/nlp/pii/detect")
async def detect_pii(request: PIIDetectRequest):
    """
    Detect PII (Personally Identifiable Information) in text.

    Uses GLiNER for zero-shot entity detection plus regex patterns for
    common PII formats. Supports custom entity types.

    Default entity types detected:
    - person, email, phone number, social security number
    - credit card number, address, date of birth
    - passport number, driver license, bank account, ip address

    Example request:
    ```json
    {
        "text": "Contact John at john@email.com or 555-123-4567",
        "threshold": 0.5
    }
    ```

    Response:
    ```json
    {
        "object": "pii_detection",
        "entities": [
            {"text": "John", "label": "person", "start": 8, "end": 12, "score": 0.95},
            {"text": "john@email.com", "label": "email", "start": 16, "end": 30, "score": 1.0}
        ],
        "entity_count": 2
    }
    ```
    """
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty",
            )

        entities = await nlp_service.detect_pii(
            request.text,
            entity_types=request.entity_types,
            threshold=request.threshold,
            use_regex=request.use_regex,
        )

        return {
            "object": "pii_detection",
            "entities": entities,
            "entity_count": len(entities),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_pii: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/ml/nlp/redact")
async def redact_pii(request: PIIRedactRequest):
    """
    Detect and redact PII from text.

    Returns the text with PII replaced by configurable replacement strings.
    Supports per-entity-type replacement patterns.

    Example request:
    ```json
    {
        "text": "Contact John at john@email.com",
        "replacement": "[REDACTED]",
        "replacement_map": {"email": "[EMAIL]", "person": "[NAME]"}
    }
    ```

    Response:
    ```json
    {
        "object": "pii_redaction",
        "redacted_text": "Contact [NAME] at [EMAIL]",
        "entities": [...],
        "entity_count": 2
    }
    ```
    """
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty",
            )

        result = await nlp_service.redact_pii(
            request.text,
            entity_types=request.entity_types,
            replacement=request.replacement,
            replacement_map=request.replacement_map,
            threshold=request.threshold,
            use_regex=request.use_regex,
        )

        return {
            "object": "pii_redaction",
            "redacted_text": result["redacted_text"],
            "entities": result["entities"],
            "entity_count": len(result["entities"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in redact_pii: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
