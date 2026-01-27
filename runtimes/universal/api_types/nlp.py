"""NLP types for embeddings, reranking, classification, and NER endpoints."""

from typing import Literal

from pydantic import BaseModel

# =============================================================================
# Embeddings
# =============================================================================


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = "float"
    user: str | None = None
    extra_body: dict | None = None


class EmbeddingData(BaseModel):
    """Single embedding result."""

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str  # float list or base64 string


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict[str, int]


# =============================================================================
# Reranking
# =============================================================================


class RerankRequest(BaseModel):
    """Reranking request for cross-encoder models."""

    model: str
    query: str
    documents: list[str]
    top_k: int | None = None
    return_documents: bool = True


class RerankResult(BaseModel):
    """Single reranking result."""

    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(BaseModel):
    """Reranking response."""

    object: Literal["list"] = "list"
    data: list[RerankResult]
    model: str
    usage: dict[str, int]


# =============================================================================
# Classification
# =============================================================================


class ClassifyRequest(BaseModel):
    """Text classification request."""

    model: str  # HuggingFace model ID
    texts: list[str]  # Texts to classify
    max_length: int | None = None  # Optional max sequence length


class ClassifyResult(BaseModel):
    """Single classification result."""

    index: int
    label: str
    score: float
    all_scores: dict[str, float]


class ClassifyResponse(BaseModel):
    """Classification response."""

    object: Literal["list"] = "list"
    data: list[ClassifyResult]
    total_count: int
    model: str
    usage: dict[str, int] | None = None


# =============================================================================
# Named Entity Recognition (NER)
# =============================================================================


class NERRequest(BaseModel):
    """Named entity recognition request."""

    model: str  # HuggingFace model ID (e.g., "dslim/bert-base-NER")
    texts: list[str]  # Texts for entity extraction
    max_length: int | None = None  # Optional max sequence length


class EntityResult(BaseModel):
    """Single entity extraction result."""

    text: str
    label: str
    start: int
    end: int
    score: float


class NERResult(BaseModel):
    """NER result for a single text."""

    index: int
    entities: list[EntityResult]


class NERResponse(BaseModel):
    """NER response."""

    object: Literal["list"] = "list"
    data: list[NERResult]
    model: str
    usage: dict[str, int] | None = None
