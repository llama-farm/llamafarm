"""Pydantic models for NLP endpoints."""

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request for text embeddings."""

    input: str | list[str]
    model: str
    encoding_format: str = "float"
    dimensions: int | None = None


class RerankRequest(BaseModel):
    """Request to rerank documents by relevance."""

    query: str
    documents: list[str]
    model: str
    top_n: int | None = None
    return_documents: bool = True


class ClassifyRequest(BaseModel):
    """Request for text classification."""

    input: str | list[str]
    model: str
    labels: list[str] | None = None


class NERRequest(BaseModel):
    """Request for named entity recognition."""

    input: str | list[str]
    model: str
