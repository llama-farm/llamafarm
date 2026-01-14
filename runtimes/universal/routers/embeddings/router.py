"""
Encoder-based endpoints: embeddings, reranking, classification, and NER.
"""

import base64

from fastapi import APIRouter, HTTPException

from core.logging import UniversalRuntimeLogger
from utils.model_format import parse_model_with_quantization

from .service import load_encoder
from .types import ClassifyRequest, EmbeddingRequest, NERRequest, RerankRequest

router = APIRouter()
logger = UniversalRuntimeLogger("universal-runtime.embeddings")


@router.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.

    Supports any HuggingFace encoder model for text embeddings.
    Model names can include quantization suffix (e.g., "model:Q4_K_M").
    """
    try:
        # Parse model name to extract quantization if present
        model_id, gguf_quantization = parse_model_with_quantization(request.model)

        model = await load_encoder(
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
                import struct

                embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
                embedding_data = base64.b64encode(embedding_bytes).decode("utf-8")
            else:
                embedding_data = embedding

            data.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding_data,
                }
            )

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": 0,  # TODO: Implement token counting
                "total_tokens": 0,
            },
        }

    except Exception as e:
        logger.error(f"Error in create_embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/rerank")
async def rerank_documents(request: RerankRequest):
    """
    Cross-encoder reranking endpoint.

    Reranks documents based on relevance to the query using proper
    cross-encoder architecture (query and document jointly encoded).

    This is significantly more accurate than bi-encoder similarity
    and 10-100x faster than LLM-based reranking.
    """
    try:
        model = await load_encoder(request.model, task="reranking")

        # Rerank documents
        results = await model.rerank(
            query=request.query, documents=request.documents, top_k=request.top_k
        )

        # Format response
        data = []
        for result in results:
            item = {
                "index": result["index"],
                "relevance_score": result["relevance_score"],
            }
            if request.return_documents:
                item["document"] = result["document"]
            data.append(item)

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": 0,  # TODO: Implement token counting
                "total_tokens": 0,
            },
        }

    except Exception as e:
        logger.error(f"Error in rerank_documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/classify")
async def classify_texts(request: ClassifyRequest):
    """
    Text classification endpoint.

    Classify texts using any HuggingFace sequence classification model.
    Supports sentiment analysis, spam detection, intent routing, etc.

    Popular models:
    - distilbert-base-uncased-finetuned-sst-2-english (sentiment)
    - facebook/bart-large-mnli (zero-shot classification)
    - cardiffnlp/twitter-roberta-base-sentiment-latest (social media sentiment)

    Example request:
    ```json
    {
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "texts": ["I love this product!", "This is terrible."]
    }
    ```
    """
    try:
        # Parse model name
        model_id, _ = parse_model_with_quantization(request.model)

        model = await load_encoder(
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
                {
                    "index": idx,
                    "label": result["label"],
                    "score": result["score"],
                    "all_scores": result["all_scores"],
                }
            )

        return {
            "object": "list",
            "data": data,
            "total_count": len(data),
            "model": request.model,
            "usage": {
                "texts_processed": len(request.texts),
            },
        }

    except Exception as e:
        logger.error(f"Error in classify_texts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/ner")
async def extract_entities(request: NERRequest):
    """
    Named Entity Recognition endpoint.

    Extract named entities (people, organizations, locations, etc.) from text
    using HuggingFace token classification models.

    Popular models:
    - dslim/bert-base-NER (English, PERSON/ORG/LOC/MISC)
    - Jean-Baptiste/roberta-large-ner-english (English, high accuracy)
    - xlm-roberta-large-finetuned-conll03-english (multilingual)

    Example request:
    ```json
    {
        "model": "dslim/bert-base-NER",
        "texts": ["John works at Google in San Francisco."]
    }
    ```

    Response entities include:
    - text: The extracted entity text
    - label: Entity type (PERSON, ORG, LOC, etc.)
    - start/end: Character offsets in the original text
    - score: Confidence score
    """
    try:
        # Parse model name
        model_id, _ = parse_model_with_quantization(request.model)

        model = await load_encoder(
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
                {
                    "index": idx,
                    "entities": [
                        {
                            "text": e.text,
                            "label": e.label,
                            "start": e.start,
                            "end": e.end,
                            "score": e.score,
                        }
                        for e in entities
                    ],
                }
            )

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "texts_processed": len(request.texts),
            },
        }

    except Exception as e:
        logger.error(f"Error in extract_entities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
