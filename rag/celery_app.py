"""
RAG Celery App Configuration

This module configures the Celery application for the RAG service worker.
It sets up the broker connection, task routing, and imports all RAG tasks.
"""

import os
import sys
import multiprocessing

from celery import Celery, signals

from core.logging import RAGStructLogger
from core.settings import settings

# Fix macOS fork safety issues by using spawn instead of fork
if sys.platform == "darwin":  # macOS
    multiprocessing.set_start_method("spawn", force=True)

logger = RAGStructLogger("rag.celery")

# Create Celery app instance
app = Celery("LlamaFarm-RAG-Worker")

# Get data directory from environment - use same default as server
lf_data_dir = settings.LF_DATA_DIR

# Create necessary broker directories
_folders = [
    f"{lf_data_dir}/broker/in",
    f"{lf_data_dir}/broker/processed",
    f"{lf_data_dir}/broker/results",
]

for folder in _folders:
    os.makedirs(folder, exist_ok=True)

# Configure broker based on environment variables
celery_broker_url = settings.CELERY_BROKER_URL
celery_result_backend = settings.CELERY_RESULT_BACKEND

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
    # Convert Windows backslashes to forward slashes for file:// URL
    result_backend_path = f"{lf_data_dir}/broker/results".replace("\\", "/")
    # Ensure proper file:// URL format (file:/// for absolute paths on Windows)
    if (
        sys.platform == "win32"
        and len(result_backend_path) > 1
        and result_backend_path[1] == ":"
    ):
        # Windows absolute path (e.g., C:/Users/...) needs file:///C:/...
        result_backend_url = f"file:///{result_backend_path}"
    else:
        # Unix absolute path needs file:///path or relative path needs file://path
        result_backend_url = f"file://{result_backend_path}"

    app.conf.update(
        broker_url="filesystem://",
        broker_transport_options={
            "data_folder_in": f"{lf_data_dir}/broker/in",
            "data_folder_out": f"{lf_data_dir}/broker/in",  # Must be same as data_folder_in
            "data_folder_processed": f"{lf_data_dir}/broker/processed",
        },
        result_backend=result_backend_url,
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
        # Always use thread pool to avoid:
        # 1. Fork/spawn issues on macOS/Windows
        # 2. SQLite "database is locked" errors with ChromaDB's persistent client
        #    (prefork pool spawns multiple processes that can't share SQLite connections)
        worker_args = ["worker", "-Q", "rag", "--pool=threads"]

        # Allow concurrency override via environment variable
        # If not set, Celery defaults to number of CPUs (reasonable for I/O-bound tasks)
        concurrency = os.getenv("LF_CELERY_CONCURRENCY")
        if concurrency:
            worker_args.extend(["--concurrency", concurrency])
            logger.info(
                f"Starting RAG worker with thread pool (concurrency={concurrency})"
            )
        else:
            logger.info("Starting RAG worker with thread pool (concurrency=auto)")

        # Only consume from the 'rag' queue, not the 'celery' queue
        app.worker_main(argv=worker_args)
    except KeyboardInterrupt:
        logger.info("RAG worker shutting down due to keyboard interrupt")
    except Exception as e:
        logger.error("RAG worker failed to start", extra={"error": str(e)})
        raise


# Ensure task modules are imported when celery_app is imported
# This fixes the issue where Celery's auto-import doesn't work reliably
try:
    # Import all task modules to register them with the Celery app
    import tasks.health_tasks  # noqa: F401
    import tasks.ingest_tasks  # noqa: F401
    import tasks.query_tasks  # noqa: F401
    import tasks.search_tasks  # noqa: F401

    logger.info("RAG task modules imported successfully")
except Exception as e:
    logger.warning(f"Failed to import RAG task modules: {e}")


# Only run worker if this module is the main entry point
if __name__ == "__main__":
    run_worker()
