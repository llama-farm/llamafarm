"""
Client utilities for interacting with RAG Celery tasks.

These helpers centralize the logic for building task signatures, dispatching
them, and polling for completion so that both FastAPI handlers and Celery
tasks can reuse the same behavior.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from celery import signature

from core.celery import app


def build_ingest_signature(
    project_dir: str,
    data_processing_strategy_name: str,
    database_name: str,
    source_path: str,
    filename: str | None = None,
    dataset_name: str | None = None,
):
    """
    Build a Celery signature for the rag.ingest_file task.

    This helper ensures all signatures are constructed consistently.
    """

    return signature(
        "rag.ingest_file",
        args=[
            project_dir,
            data_processing_strategy_name,
            database_name,
            source_path,
            filename,
            dataset_name,
        ],
        app=app,
    )


async def ingest_file_with_rag(
    project_dir: str,
    data_processing_strategy_name: str,
    database_name: str,
    source_path: str,
    filename: str | None = None,
    dataset_name: str | None = None,
    timeout: int = 300,
    poll_interval: float = 2.0,
) -> tuple[bool, dict[str, Any]]:
    """
    Dispatch rag.ingest_file and poll for completion from async contexts.
    """

    sig = build_ingest_signature(
        project_dir=project_dir,
        data_processing_strategy_name=data_processing_strategy_name,
        database_name=database_name,
        source_path=source_path,
        filename=filename,
        dataset_name=dataset_name,
    )
    result = sig.apply_async()

    waited = 0.0
    while waited < timeout:
        try:
            status = result.status
            if status not in ("PENDING", "STARTED"):
                break
        except Exception:
            await asyncio.sleep(poll_interval)
            waited += poll_interval
            continue

        await asyncio.sleep(poll_interval)
        waited += poll_interval

    try:
        final_status = result.status
    except Exception as exc:  # pragma: no cover - defensive
        return False, {"error": f"Failed to get task status: {exc}"}

    if final_status == "SUCCESS":
        try:
            return result.result
        except Exception as exc:  # pragma: no cover - defensive
            return False, {"error": f"Failed to get task result: {exc}"}

    if final_status == "FAILURE":
        # Propagate the remote traceback if available
        if hasattr(result, "traceback") and result.traceback:
            raise Exception(f"Task failed: {result.traceback}")  # noqa: BLE001
        raise Exception("Task failed")  # noqa: BLE001

    return (
        False,
        {"error": f"Task timed out or failed with status: {final_status}"},
    )


def search_with_rag_database(
    project_dir: str,
    database: str,
    query: str,
    top_k: int = 5,
    retrieval_strategy: str | None = None,
    score_threshold: float | None = None,
    metadata_filters: dict[str, Any] | None = None,
    distance_metric: str | None = None,
    hybrid_alpha: float | None = None,
    rerank_model: str | None = None,
    query_expansion: bool | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run a search query via rag.search_with_database.
    """

    task = signature(
        "rag.search_with_database",
        args=[
            project_dir,
            database,
            query,
            top_k,
            retrieval_strategy,
            score_threshold,
            metadata_filters,
            distance_metric,
            hybrid_alpha,
            rerank_model,
            query_expansion,
            max_tokens,
        ],
        app=app,
    )
    return _run_sync_task_with_polling(task, timeout=30, poll_interval=0.5) or []


def handle_rag_query(
    project_dir: str,
    database: str,
    query: str,
    context: dict[str, Any] | None = None,
    top_k: int = 5,
    retrieval_strategy: str | None = None,
) -> dict[str, Any]:
    """
    Execute rag.handle_rag_query and return the response.
    """

    task = signature(
        "rag.handle_rag_query",
        args=[project_dir, database, query, context, top_k, retrieval_strategy],
        app=app,
    )
    return _run_sync_task_with_polling(task, timeout=60, poll_interval=1) or {
        "query": query,
        "database": database,
        "results": [],
        "total_results": 0,
        "retrieval_strategy": retrieval_strategy,
        "context": context,
        "error": "Task timed out or failed",
    }


def get_rag_health(project_dir: str, database: str) -> dict[str, Any]:
    """
    Fetch RAG health diagnostics via rag.health_check_database.
    """

    task = signature(
        "rag.health_check_database",
        args=[project_dir, database],
        app=app,
    )
    return _run_sync_task_with_polling(task, timeout=30, poll_interval=0.5) or {
        "status": "degraded",
        "message": "Health check task timed out or failed",
        "database": database,
        "checks": {},
        "metrics": {},
        "errors": ["Unable to contact RAG worker"],
    }


def batch_search(
    project_dir: str,
    database: str,
    queries: list[str],
    top_k: int = 5,
    retrieval_strategy: str | None = None,
) -> list[dict[str, Any]]:
    """
    Execute rag.batch_search for a list of queries.
    """

    task = signature(
        "rag.batch_search",
        args=[project_dir, database, queries, top_k, retrieval_strategy],
        app=app,
    )
    timeout = max(30, len(queries) * 10)
    return _run_sync_task_with_polling(task, timeout=timeout, poll_interval=1) or []


def _run_sync_task_with_polling(task_signature, timeout: float, poll_interval: float):
    """
    Helper used by synchronous contexts to poll a Celery AsyncResult safely.
    """

    result = task_signature.apply_async()
    waited = 0.0

    while waited < timeout:
        try:
            status = result.status
            if status not in ("PENDING", "STARTED"):
                break
        except Exception:
            time.sleep(poll_interval)
            waited += poll_interval
            continue
        time.sleep(poll_interval)
        waited += poll_interval

    try:
        final_status = result.status
    except Exception as exc:  # pragma: no cover - defensive
        raise Exception(f"Failed to get task status: {exc}")  # noqa: BLE001

    if final_status == "SUCCESS":
        try:
            return result.result
        except Exception as exc:  # pragma: no cover - defensive
            raise Exception(f"Failed to get task result: {exc}")  # noqa: BLE001

    if final_status == "FAILURE":
        if hasattr(result, "traceback") and result.traceback:
            raise Exception(f"Task failed: {result.traceback}")  # noqa: BLE001
        raise Exception("Task failed")  # noqa: BLE001

    return None
