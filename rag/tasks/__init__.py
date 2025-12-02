"""
RAG Celery Tasks Package

This package contains all Celery task implementations for the RAG service.
Tasks are organized by functionality:
- search_tasks: RAG search and database queries
- ingest_tasks: File ingestion and processing
- delete_tasks: File deletion from vector stores
- query_tasks: Complex RAG query operations
- health_tasks: Health monitoring and diagnostics
"""

from .delete_tasks import delete_file_task
from .health_tasks import rag_health_check_task, rag_ping_task
from .ingest_tasks import ingest_file_with_rag_task
from .query_tasks import handle_rag_query_task
from .search_tasks import search_with_rag_database_task
from .stats_tasks import rag_get_database_stats_task, rag_list_database_documents_task

__all__ = [
    "search_with_rag_database_task",
    "ingest_file_with_rag_task",
    "delete_file_task",
    "handle_rag_query_task",
    "rag_health_check_task",
    "rag_ping_task",
    "rag_get_database_stats_task",
    "rag_list_database_documents_task",
]
