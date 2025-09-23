"""
RAG Celery App Configuration

This module configures the Celery application for the RAG service worker.
It sets up the broker connection, task routing, and imports all RAG tasks.
"""

import logging
import os
from pathlib import Path

from celery import Celery, signals

logger = logging.getLogger(__name__)

# Create Celery app instance
app = Celery("LlamaFarm-RAG-Worker")

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
        broker_url=celery_broker_url,
        result_backend=celery_result_backend,
        result_persistent=True,
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # Task routing - only handle rag.* tasks
        task_routes={
            "rag.*": {"queue": "rag"},
        },
        # Import task modules
        imports=(
            "tasks.search_tasks",
            "tasks.ingest_tasks",
            "tasks.query_tasks",
            "tasks.health_tasks",
        ),
    )
else:
    # Use default filesystem broker (same as server)
    app.conf.update(
        broker_url="filesystem://",
        broker_transport_options={
            "data_folder_in": f"{lf_data_dir}/broker/in",
            "data_folder_out": f"{lf_data_dir}/broker/in",  # Must be same as data_folder_in
            "data_folder_processed": f"{lf_data_dir}/broker/processed",
        },
        result_backend=f"file://{lf_data_dir}/broker/results",
        result_persistent=True,
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # Task routing - only handle rag.* tasks
        task_routes={
            "rag.*": {"queue": "rag"},
        },
        # Import task modules
        imports=(
            "tasks.search_tasks",
            "tasks.ingest_tasks",
            "tasks.query_tasks",
            "tasks.health_tasks",
        ),
    )


# Intentionally empty function to prevent Celery from overriding root logger config
@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    pass


def run_worker():
    try:
        # Only consume from the 'rag' queue, not the 'celery' queue
        app.worker_main(argv=["worker", "-P", "solo", "-Q", "rag"])
    except KeyboardInterrupt:
        logger.info("RAG worker shutting down due to keyboard interrupt")
    except Exception as e:
        logger.error("RAG worker failed to start", extra={"error": str(e)})
        raise


# Ensure task modules are imported when celery_app is imported
# This fixes the issue where Celery's auto-import doesn't work reliably
try:
    # Import all task modules to register them with the Celery app
    import tasks.search_tasks  # noqa: F401
    import tasks.ingest_tasks  # noqa: F401
    import tasks.query_tasks  # noqa: F401
    import tasks.health_tasks  # noqa: F401

    logger.info("RAG task modules imported successfully")
except Exception as e:
    logger.warning(f"Failed to import RAG task modules: {e}")


# Only run worker if this module is the main entry point
if __name__ == "__main__":
    run_worker()
