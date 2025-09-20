"""
RAG Celery App Configuration

This module configures the Celery application for the RAG service worker.
It sets up the broker connection, task routing, and imports all RAG tasks.
"""

import os
import sys
from pathlib import Path

from celery import Celery
# Note: Celery signals import removed to avoid compatibility issues

# Create Celery app instance
app = Celery("RAG-Worker")

# Get data directory from environment - use same default as server
lf_data_dir = os.environ.get("LF_DATA_DIR", str(Path.home() / ".llamafarm"))

# Create necessary broker directories
_folders = [
    f"{lf_data_dir}/broker/in",
    f"{lf_data_dir}/broker/processed",
    f"{lf_data_dir}/broker/results",
]

for folder in _folders:
    os.makedirs(folder, exist_ok=True)

# Configure broker based on environment variables
celery_broker_url = os.environ.get("CELERY_BROKER_URL", "")
celery_result_backend = os.environ.get("CELERY_RESULT_BACKEND", "")

if celery_broker_url and celery_result_backend:
    # Use external broker (Redis, RabbitMQ, etc.)
    app.conf.update(
        {
            "broker_url": celery_broker_url,
            "result_backend": celery_result_backend,
            "result_persistent": True,
            "task_serializer": "json",
            "accept_content": ["json"],
            "result_serializer": "json",
            "timezone": "UTC",
            "enable_utc": True,
            # Task routing - only handle rag.* tasks
            "task_routes": {
                "rag.*": {"queue": "rag"},
            },
            # Import task modules - ensure they're imported at startup
            "imports": [
                "tasks.search_tasks",
                "tasks.ingest_tasks",
                "tasks.query_tasks",
            ],
            # Ensure tasks are discovered
            "include": [
                "tasks.search_tasks",
                "tasks.ingest_tasks",
                "tasks.query_tasks",
            ],
        }
    )
else:
    # Use default filesystem broker (same as server)
    app.conf.update(
        {
            "broker_url": "filesystem://",
            "broker_transport_options": {
                "data_folder_in": f"{lf_data_dir}/broker/in",
                "data_folder_out": f"{lf_data_dir}/broker/in",  # Must be same as data_folder_in
                "data_folder_processed": f"{lf_data_dir}/broker/processed",
            },
            "result_backend": f"file://{lf_data_dir}/broker/results",
            "result_persistent": True,
            "task_serializer": "json",
            "accept_content": ["json"],
            "result_serializer": "json",
            "timezone": "UTC",
            "enable_utc": True,
            # Task routing - only handle rag.* tasks
            "task_routes": {
                "rag.*": {"queue": "rag"},
            },
            # Import task modules - ensure they're imported at startup
            "imports": [
                "tasks.search_tasks",
                "tasks.ingest_tasks",
                "tasks.query_tasks",
            ],
            # Ensure tasks are discovered
            "include": [
                "tasks.search_tasks",
                "tasks.ingest_tasks",
                "tasks.query_tasks",
            ],
        }
    )


# Note: Celery logging setup removed to avoid signal compatibility issues
# Celery will use its default logging configuration

# Explicitly import task modules to ensure they are registered
# This is more reliable than relying on the 'imports' configuration
try:
    import tasks.ingest_tasks  # noqa: F401
    import tasks.query_tasks  # noqa: F401
    import tasks.search_tasks  # noqa: F401
except ImportError as e:
    print(f"Warning: Failed to import task modules: {e}", file=sys.stderr)
    # Don't fail completely - let Celery try to import them later
