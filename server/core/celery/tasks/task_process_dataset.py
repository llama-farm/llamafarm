from pathlib import Path

from celery import Task

from core.celery import app
from core.logging import FastAPIStructLogger
from services.data_service import DataService
from services.project_service import ProjectService
from .rag_tasks import rag_ingest_task

logger = FastAPIStructLogger(__name__)


@app.task(bind=True)
def process_dataset_task(self: Task, namespace: str, project: str, dataset: str):
    logger.info("Processing dataset task started")
    project_config = ProjectService.get_project(namespace, project).config

    # Get the dataset config
    dataset_config = next(
        (ds for ds in (project_config.datasets or []) if ds.name == dataset), None
    )
    if not dataset_config:
        raise ValueError(f"Dataset {dataset} not found")

    path_to_raw_dir = Path(DataService.get_data_dir(namespace, project)) / "raw"

    # Prepare file paths for ingestion
    file_paths = []
    for file_hash in dataset_config.files:
        file_path = path_to_raw_dir / file_hash
        if not file_path.exists():
            raise FileNotFoundError(f"Raw file not found: {file_path}")
        file_paths.append(str(file_path))

    if not file_paths:
        raise ValueError("No valid files found for processing")

    # Submit RAG ingestion task
    logger.info(f"Submitting RAG ingest task for {len(file_paths)} files")
    result = rag_ingest_task.delay(namespace, project, dataset, file_paths)

    # Wait for completion and get result
    ingest_result = result.get(timeout=300)  # 5 minute timeout

    return {
        "message": "Dataset processed successfully",
        "namespace": namespace,
        "project": project,
        "dataset": dataset,
        "strategy": dataset_config.data_processing_strategy,
        "files": dataset_config.files,
        "ingest_result": ingest_result,
    }
