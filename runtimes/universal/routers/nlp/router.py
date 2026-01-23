"""NLP router for embeddings, reranking, classification, and NER.

This router handles:
- /v1/embeddings - Generate text embeddings
- /v1/rerank - Cross-encoder document reranking
- /v1/classify - Text classification
- /v1/ner - Named entity recognition
"""

import base64
import logging
import struct

from fastapi import APIRouter, HTTPException

from api_types import (
    ClassifyRequest,
    ClassifyResponse,
    ClassifyResult,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EntityResult,
    NERRequest,
    NERResponse,
    NERResult,
    RerankRequest,
    RerankResponse,
    RerankResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["NLP"])


# Dependency injection: These will be set by the server during startup
_load_encoder_fn = None


def set_encoder_loader(load_encoder_fn):
    """Set the encoder loading function.

    This is called during server startup to inject the load_encoder function
    which handles model caching and loading.
    """
    global _load_encoder_fn
    _load_encoder_fn = load_encoder_fn


async def _get_encoder(model_id: str, task: str, **kwargs):
    """Get or load an encoder model."""
    if _load_encoder_fn is None:
        raise RuntimeError(
            "Encoder loader not initialized. Call set_encoder_loader first."
        )
    return await _load_encoder_fn(model_id, task=task, **kwargs)


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """OpenAI-compatible embeddings endpoint.

    Supports any HuggingFace encoder model for text embeddings.
    Model names can include quantization suffix (e.g., "model:Q4_K_M").
    """
    try:
        # Import parsing utility
        from utils.model_format import parse_model_with_quantization

        # Parse model name to extract quantization if present
        model_id, gguf_quantization = parse_model_with_quantization(request.model)

        model = await _get_encoder(
            model_id, task="embedding", preferred_quantization=gguf_quantization
        )

        # Normalize input to list
        texts = [request.input] if isinstance(request.input, str) else request.input

        # Generate embeddings
        embeddings = await model.embed(texts, normalize=True)

        # Format response
        data = []
        for idx, embedding in enumerate(embeddings):
            if request.encoding_format == "base64":
                embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
                embedding_data = base64.b64encode(embedding_bytes).decode("utf-8")
            else:
                embedding_data = embedding

            data.append(
                EmbeddingData(
                    object="embedding",
                    index=idx,
                    embedding=embedding_data,
                )
            )

        return EmbeddingResponse(
            object="list",
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": 0,  # TODO: Implement token counting
                "total_tokens": 0,
            },
        )

    except Exception as e:
        logger.error(f"Error in create_embeddings: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while generating embeddings"
        ) from e


@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest) -> RerankResponse:
    """Cross-encoder reranking endpoint.

    Reranks documents based on relevance to the query using proper
    cross-encoder architecture (query and document jointly encoded).

    This is significantly more accurate than bi-encoder similarity
    and 10-100x faster than LLM-based reranking.
    """
    try:
        model = await _get_encoder(request.model, task="reranking")

        # Rerank documents
        results = await model.rerank(
            query=request.query, documents=request.documents, top_k=request.top_k
        )

        # Format response
        data = []
        for result in results:
            data.append(
                RerankResult(
                    index=result["index"],
                    relevance_score=result["relevance_score"],
                    document=result["document"] if request.return_documents else None,
                )
            )

        return RerankResponse(
            object="list",
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        )

    except Exception as e:
        logger.error(f"Error in rerank_documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while reranking documents"
        ) from e


@router.post("/v1/classify", response_model=ClassifyResponse)
async def classify_texts(request: ClassifyRequest) -> ClassifyResponse:
    """Text classification endpoint.

    Classify texts using any HuggingFace sequence classification model.
    Supports sentiment analysis, spam detection, intent routing, etc.

    Popular models:
    - distilbert-base-uncased-finetuned-sst-2-english (sentiment)
    - facebook/bart-large-mnli (zero-shot classification)
    - cardiffnlp/twitter-roberta-base-sentiment-latest (social media sentiment)
    """
    try:
        # Import parsing utility
        from utils.model_format import parse_model_with_quantization

        # Parse model name
        model_id, _ = parse_model_with_quantization(request.model)

        model = await _get_encoder(
            model_id,
            task="classification",
            max_length=request.max_length,
        )

        # Run classification
        results = await model.classify(request.texts)

        # Format response
        data = []
        for idx, result in enumerate(results):
            data.append(
                ClassifyResult(
                    index=idx,
                    label=result["label"],
                    score=result["score"],
                    all_scores=result["all_scores"],
                )
            )

        return ClassifyResponse(
            object="list",
            data=data,
            model=request.model,
            total_count=len(data),
        )

    except Exception as e:
        logger.error(f"Error in classify_texts: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while classifying text"
        ) from e


@router.post("/v1/ner", response_model=NERResponse)
async def extract_entities(request: NERRequest) -> NERResponse:
    """Named Entity Recognition endpoint.

    Extract named entities (people, organizations, locations, etc.) from text
    using HuggingFace token classification models.

    Popular models:
    - dslim/bert-base-NER (English, PERSON/ORG/LOC/MISC)
    - Jean-Baptiste/roberta-large-ner-english (English, high accuracy)
    - xlm-roberta-large-finetuned-conll03-english (multilingual)
    """
    try:
        # Import parsing utility
        from utils.model_format import parse_model_with_quantization

        # Parse model name
        model_id, _ = parse_model_with_quantization(request.model)

        model = await _get_encoder(
            model_id,
            task="ner",
            max_length=request.max_length,
        )

        # Run NER
        results = await model.extract_entities(request.texts)

        # Format response
        data = []
        for idx, entities in enumerate(results):
            data.append(
                NERResult(
                    index=idx,
                    entities=[
                        EntityResult(
                            text=e.text,
                            label=e.label,
                            start=e.start,
                            end=e.end,
                            score=e.score,
                        )
                        for e in entities
                    ],
                )
            )

        return NERResponse(
            object="list",
            data=data,
            model=request.model,
        )

    except Exception as e:
        logger.error(f"Error in extract_entities: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while extracting entities"
        ) from e
