"""
Request/response types for embedding-related endpoints.

Includes embeddings, reranking, classification, and NER.
"""

from typing import Literal

from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = "float"
    user: str | None = None
    extra_body: dict | None = None


class RerankRequest(BaseModel):
    """Reranking request for cross-encoder models."""

    model: str
    query: str
    documents: list[str]
    top_k: int | None = None
    return_documents: bool = True


class ClassifyRequest(BaseModel):
    """Text classification request."""

    model: str  # HuggingFace model ID (e.g., "distilbert-base-uncased-finetuned-sst-2-english")
    texts: list[str]  # Texts to classify
    max_length: int | None = None  # Optional max sequence length


class NERRequest(BaseModel):
    """Named entity recognition request."""

    model: str  # HuggingFace model ID (e.g., "dslim/bert-base-NER")
    texts: list[str]  # Texts for entity extraction
    max_length: int | None = None  # Optional max sequence length
