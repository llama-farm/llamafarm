import os
import sys
import threading

from celery import Celery, signals

from core.settings import settings

app = Celery("LlamaFarm")


_folders = [
    f"{settings.lf_data_dir}/broker/in",
    f"{settings.lf_data_dir}/broker/processed",
    f"{settings.lf_data_dir}/broker/results",
]

for folder in _folders:
    os.makedirs(folder, exist_ok=True)

# Configure broker based on settings
if settings.celery_broker_url and settings.celery_result_backend:
    # Use external broker (Redis, RabbitMQ, etc.)
    app.conf.update(
        {
            "broker_url": settings.celery_broker_url,
            "result_backend": settings.celery_result_backend,
            "result_persistent": True,
            "task_serializer": "json",
            "accept_content": ["json"],
            "result_serializer": "json",
            "timezone": "UTC",
            "enable_utc": True,
            # Task routing - only handle server tasks, ignore rag.* tasks
            "task_routes": {
                "rag.*": {"queue": "rag"},
                "core.celery.tasks.*": {"queue": "server"},
            },
        }
    )
else:
    # Use default filesystem broker
    # Convert Windows backslashes to forward slashes for file:// URL
    result_backend_path = f"{settings.lf_data_dir}/broker/results".replace("\\", "/")
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
        {
            "broker_url": "filesystem://",
            "broker_transport_options": {
                "data_folder_in": f"{settings.lf_data_dir}/broker/in",
                "data_folder_out": f"{settings.lf_data_dir}/broker/in",  # has to be the same as 'data_folder_in'  # noqa: E501
                "data_folder_processed": f"{settings.lf_data_dir}/broker/processed",
            },
            "result_backend": result_backend_url,
            "result_persistent": True,
            # Task routing - only handle server tasks, ignore rag.* tasks
            "task_routes": {
                "rag.*": {"queue": "rag"},
                "core.celery.tasks.*": {"queue": "server"},
            },
        }
    )


# Intentionally empty function to prevent Celery from overriding root logger config
@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    pass


# app.log.setup()

# Create a thread and run the worker in it. This is not a long-term solution.
# Eventually we should use a proper broker and backend for this like Redis and
# we can remove this code.


# Code to start the worker


def run_worker():
    # Only consume from the 'celery' queue, not the 'rag' queue
    app.worker_main(argv=["worker", "-P", "solo", "--uid", "0", "-Q", "server"])


t = threading.Thread(target=run_worker, daemon=True)

t.start()
