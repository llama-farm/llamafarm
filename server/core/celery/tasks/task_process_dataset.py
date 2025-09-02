from pathlib import Path

from celery import Task

from core.celery import app
from core.logging import FastAPIStructLogger
from services.data_service import DataService
from services.project_service import ProjectService
from services.rag_subprocess import ingest_file_with_rag

logger = FastAPIStructLogger(__name__)


@app.task(bind=True)
def process_dataset_task(self: Task, namespace: str, project: str, dataset: str):
    logger.info("Processing dataset task started")
    project_config = ProjectService.get_project(namespace, project).config

    # Get the dataset config
    dataset_config = next(
        (ds for ds in project_config.datasets if ds.name == dataset), None
    )
    if not dataset_config:
        raise ValueError(f"Dataset {dataset} not found")

    # Get the RAG strategy for the dataset
    strategy_name = dataset_config.rag_strategy or "default"
    strategy = next(
        (s for s in project_config.rag.strategies if s.name == strategy_name), None
    )
    if not strategy:
        raise ValueError(f"Strategy {strategy_name} not found")

    path_to_raw_dir = Path(DataService.get_data_dir(namespace, project)) / "raw"

    # Ingest each file using the RAG strategy defined in the dataset config
    files_ingested = []
    for file_hash in dataset_config.files:
        file_path = path_to_raw_dir / file_hash
        if not file_path.exists():
            raise FileNotFoundError(f"Raw file not found: {file_path}")
        logger.info(f"Ingesting file {file_path}")
        if not ingest_file_with_rag(strategy, str(file_path)):
            raise Exception(f"Failed to ingest file {file_path}")
        files_ingested.append(file_hash)
        self.update_state(
            meta={
                "processed_files": files_ingested,
            },
        )

    return {
        "message": "Dataset processed successfully",
        "namespace": namespace,
        "project": project,
        "dataset": dataset,
        "strategy": strategy_name,
        "files": dataset_config.files,
    }
