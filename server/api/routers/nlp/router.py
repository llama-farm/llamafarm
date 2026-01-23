"""
NLP Router - Endpoints for embeddings, reranking, classification, and NER.

Proxies to Universal Runtime's NLP endpoints:
- POST /v1/embeddings - Generate text embeddings
- POST /v1/rerank - Rerank documents by relevance
- POST /v1/classify - Zero-shot or trained text classification
- POST /v1/ner - Named entity recognition
"""

import logging
from typing import Any

from fastapi import APIRouter
from server.services.universal_runtime_service import UniversalRuntimeService

from .types import ClassifyRequest, EmbeddingRequest, NERRequest, RerankRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nlp", tags=["nlp"])


@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest) -> dict[str, Any]:
    """Generate embeddings for text input.

    Creates dense vector representations of text that capture semantic meaning.
    Useful for similarity search, clustering, and as input to other models.

    Example request:
    ```json
    {
        "input": ["Hello world", "How are you?"],
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    ```

    Returns OpenAI-compatible embedding response with vectors.
    """
    return await UniversalRuntimeService.embeddings(
        input=request.input,
        model=request.model,
        encoding_format=request.encoding_format,
        dimensions=request.dimensions,
    )


@router.post("/rerank")
async def rerank_documents(request: RerankRequest) -> dict[str, Any]:
    """Rerank documents by relevance to a query.

    Takes a query and a list of documents, returns them reordered by
    semantic relevance to the query. Useful for RAG retrieval refinement.

    Example request:
    ```json
    {
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of AI",
            "The weather is nice today",
            "Deep learning uses neural networks"
        ],
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_n": 2
    }
    ```

    Returns documents with relevance scores, sorted by score descending.
    """
    return await UniversalRuntimeService.rerank(
        query=request.query,
        documents=request.documents,
        model=request.model,
        top_n=request.top_n,
        return_documents=request.return_documents,
    )


@router.post("/classify")
async def classify_text(request: ClassifyRequest) -> dict[str, Any]:
    """Classify text using zero-shot or trained classifier.

    For zero-shot classification, provide candidate labels:
    ```json
    {
        "input": "I love this product!",
        "model": "facebook/bart-large-mnli",
        "labels": ["positive", "negative", "neutral"]
    }
    ```

    For trained classifiers (SetFit), omit labels:
    ```json
    {
        "input": ["I need help", "Cancel my order"],
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    ```

    Returns predicted labels with confidence scores.
    """
    return await UniversalRuntimeService.classify(
        input=request.input,
        model=request.model,
        labels=request.labels,
    )


@router.post("/ner")
async def extract_entities(request: NERRequest) -> dict[str, Any]:
    """Extract named entities from text.

    Identifies and classifies entities like people, organizations,
    locations, dates, etc.

    Example request:
    ```json
    {
        "input": "Apple Inc. was founded by Steve Jobs in California.",
        "model": "dslim/bert-base-NER"
    }
    ```

    Returns list of entities with their type, position, and confidence.
    """
    return await UniversalRuntimeService.ner(
        input=request.input,
        model=request.model,
    )
