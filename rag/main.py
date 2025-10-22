#!/usr/bin/env python3
"""
RAG Service Celery Worker Entry Point

This module serves as the main entry point for the RAG container when running
as a Celery worker service. It connects to the Celery broker and handles
RAG-related tasks from the server.
"""

import sys

from celery_app import app, run_worker
from core.logging import RAGStructLogger, setup_logging
from core.settings import settings

logger = RAGStructLogger("rag.main")

setup_logging()


def main():
    """Main entry point for RAG Celery worker."""
    logger.info("Starting RAG Celery worker service")

    # Log environment info
    logger.info(
        "RAG worker configuration",
        extra={
            "data_dir": settings.LF_DATA_DIR,
            "python_path": sys.path[:3],  # First few paths for debugging
        },
    )

    # Log registered tasks for debugging
    try:
        registered_tasks = list(app.tasks.keys())
        logger.info(
            "Registered Celery tasks",
            extra={
                "task_count": len(registered_tasks),
                "tasks": registered_tasks,
                "rag_tasks": [
                    task for task in registered_tasks if task.startswith("rag.")
                ],
            },
        )
    except Exception as e:
        logger.warning(f"Could not list registered tasks: {e}")

    run_worker()


if __name__ == "__main__":
    main()
