"""Lightweight RAG API endpoints for server."""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from services.rag.rag_service import LightweightRAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag-lite", tags=["rag-lite"])

# Global service instance
_rag_service: LightweightRAGService | None = None


def get_rag_service() -> LightweightRAGService:
    """Get or create RAG service instance."""
    global _rag_service
    if _rag_service is None:
        logger.info("Initializing lightweight RAG service...")
        _rag_service = LightweightRAGService()
    return _rag_service


class SearchRequest(BaseModel):
    """Search request model."""

    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    """Search response model."""

    success: bool
    query: str
    results: list[dict[str, Any]]
    count: int
    error: str | None = None


class IngestResponse(BaseModel):
    """Ingest response model."""

    success: bool
    file: str
    chunks: int | None = None
    characters: int | None = None
    document_ids: list[str] | None = None
    error: str | None = None


class StatsResponse(BaseModel):
    """Stats response model."""

    document_count: int
    collection: str
    embedding_dimension: int
    embedding_model: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    chunk_strategy: str = "semantic",
):
    """Ingest a file into the RAG system.

    Args:
        file: File to ingest

    Returns:
        Ingestion results
    """
    try:
        rag = get_rag_service()

        # Save uploaded file temporarily
        import tempfile

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename or "file").suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Build metadata (exclude None values - ChromaDB doesn't support them)
        ingest_metadata = {"filename": file.filename or "unknown"}
        if file.content_type:
            ingest_metadata["content_type"] = file.content_type

        # Ingest file with specified chunk strategy
        result = await rag.ingest_file(
            tmp_path,
            metadata=ingest_metadata,
            chunk_strategy=chunk_strategy,
        )

        # Clean up temp file
        import os

        os.unlink(tmp_path)

        return IngestResponse(**result)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for documents relevant to query.

    Args:
        request: Search request with query and top_k

    Returns:
        Search results
    """
    try:
        rag = get_rag_service()
        result = await rag.search(request.query, request.top_k)
        return SearchResponse(**result)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get RAG system statistics.

    Returns:
        System stats
    """
    try:
        rag = get_rag_service()
        stats = rag.get_stats()
        return StatsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
