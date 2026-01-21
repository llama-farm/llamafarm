"""Pydantic models for NLP endpoints."""

from pydantic import BaseModel, Field

# =============================================================================
# Language Detection
# =============================================================================


class LanguageDetectRequest(BaseModel):
    """Language detection request."""

    text: str  # Single text to detect
    top_k: int = 5  # Number of top predictions


class LanguageDetectBatchRequest(BaseModel):
    """Language detection batch request."""

    texts: list[str]  # List of texts to detect
    top_k: int = 1  # Number of top predictions per text


# =============================================================================
# Keyword Extraction
# =============================================================================


class KeywordExtractRequest(BaseModel):
    """Keyword extraction request."""

    text: str  # Text to extract keywords from
    top_k: int = Field(
        default=10, ge=1, le=100, description="Number of keywords to return"
    )
    diversity: float = Field(
        default=0.5, ge=0, le=1, description="Diversity parameter (0-1)"
    )
    ngram_range: list[int] = Field(default_factory=lambda: [1, 3])  # Min and max n-gram size


class KeywordExtractBatchRequest(BaseModel):
    """Keyword extraction batch request."""

    texts: list[str]  # List of texts
    top_k: int = Field(default=10, ge=1, le=100, description="Keywords per text")
    diversity: float = Field(
        default=0.5, ge=0, le=1, description="Diversity parameter (0-1)"
    )
    ngram_range: list[int] = Field(default_factory=lambda: [1, 3])


# =============================================================================
# PII Detection and Redaction
# =============================================================================


class PIIDetectRequest(BaseModel):
    """PII detection request."""

    text: str  # Text to analyze
    entity_types: list[str] | None = None  # Custom entity types (default: standard PII)
    threshold: float = Field(
        default=0.5, ge=0, le=1, description="Detection confidence threshold"
    )
    use_regex: bool = True  # Also use regex patterns


class PIIRedactRequest(BaseModel):
    """PII redaction request."""

    text: str  # Text to redact
    entity_types: list[str] | None = None  # Entity types to redact
    replacement: str = "[REDACTED]"  # Default replacement
    replacement_map: dict[str, str] | None = None  # Per-type replacements
    threshold: float = Field(
        default=0.5, ge=0, le=1, description="Detection confidence threshold"
    )
    use_regex: bool = True
