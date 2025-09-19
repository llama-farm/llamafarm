import os
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from server.core.celery import app as celery_app
from server.core.logging import setup_logging
from server.core.settings import settings

from server.core.celery.celery import get_celery_config

from server.core.celery.tasks.rag_tasks import rag_query_task, rag_ingest_task

def main():
    setup_logging(settings.LOG_JSON_FORMAT, settings.LOG_LEVEL)

    os.makedirs(settings.lf_data_dir, exist_ok=True)
    os.makedirs(os.path.join(settings.lf_data_dir, "broker", "in"), exist_ok=True)
    os.makedirs(os.path.join(settings.lf_data_dir, "broker", "processed"), exist_ok=True)
    os.makedirs(os.path.join(settings.lf_data_dir, "broker", "results"), exist_ok=True)

    celery_config = get_celery_config()
    celery_app.conf.update(celery_config)

    celery_app.register_task(rag_query_task)
    celery_app.register_task(rag_ingest_task)

    print("Starting RAG Celery worker...")
    celery_app.worker_main([
        "worker",
        "-P", "solo",
        "--hostname", "rag-worker@%h",
        "--queues", "rag",
        "--concurrency", "1",
        "--loglevel", settings.LOG_LEVEL.lower()
    ])

if __name__ == "__main__":
    main()
