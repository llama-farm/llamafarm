import time
from pathlib import Path

from celery import Task, signature, group

from core.celery import app
from core.logging import FastAPIStructLogger
from services.data_service import DataService
from services.project_service import ProjectService

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

    # Get the RAG strategy for the dataset
    ds_data_processing_strategy_name = dataset_config.data_processing_strategy
    strategies = getattr(project_config.rag, "data_processing_strategies", [])
    strategy = next(
        (s for s in strategies if s.name == ds_data_processing_strategy_name),
        None,
    )
    if not strategy:
        raise ValueError(f"Strategy {ds_data_processing_strategy_name} not found")

    path_to_raw_dir = Path(DataService.get_data_dir(namespace, project)) / "raw"

    # Create task signatures for all files to ingest
    project_dir = str(Path(DataService.get_data_dir(namespace, project)).parent)
    ingest_tasks = []

    for file_hash in dataset_config.files:
        file_path = path_to_raw_dir / file_hash
        if not file_path.exists():
            raise FileNotFoundError(f"Raw file not found: {file_path}")

        # Create task signature for this file
        ingest_task = signature(
            "rag.ingest_file",
            args=[
                project_dir,
                ds_data_processing_strategy_name,
                dataset_config.database,
                str(file_path),
                None,  # filename
                dataset,  # dataset_name
            ],
            app=app,
        )
        ingest_tasks.append(ingest_task)

    # Execute all ingest tasks in parallel using group
    if ingest_tasks:
        logger.info(f"Starting parallel ingestion of {len(ingest_tasks)} files")
        job = group(*ingest_tasks)
        result = job.apply_async()

        # Wait for all tasks to complete using polling to avoid result.get() error
        timeout = 300  # 5 minutes
        poll_interval = 2  # seconds
        waited = 0

        while waited < timeout:
            if result.status not in ("PENDING", "STARTED"):
                break
            time.sleep(poll_interval)
            waited += poll_interval

        if result.status == "SUCCESS":
            results = result.result
        elif result.status == "FAILURE":
            raise Exception(f"Group task failed: {result.result}")
        else:
            raise Exception(
                f"Group task timed out or failed with status: {result.status}"
            )

        # Process results
        files_ingested = []
        for i, (success, details) in enumerate(results):
            file_hash = dataset_config.files[i]
            if not success:
                logger.error(f"Failed to ingest file {file_hash}: {details}")
                raise Exception(f"Failed to ingest file {file_hash}")
            files_ingested.append(file_hash)

        logger.info(f"Successfully ingested {len(files_ingested)} files")
    else:
        files_ingested = []

    return {
        "message": "Dataset processed successfully",
        "namespace": namespace,
        "project": project,
        "dataset": dataset,
        "strategy": ds_data_processing_strategy_name,
        "files": files_ingested,
        "total_files": len(files_ingested),
    }


@app.task(bind=True)
def process_single_file_task(
    self: Task,
    namespace: str,
    project: str,
    dataset: str,
    file_hash: str,
    data_processing_strategy_name: str,
    database_name: str,
) -> dict:
    """
    Process a single file for a dataset using task chaining.
    This task chains with RAG tasks instead of blocking on result.get().
    """
    from services.data_service import DataService

    logger.info(f"Processing single file task: {file_hash}")

    # Get file path
    path_to_raw_dir = Path(DataService.get_data_dir(namespace, project)) / "raw"
    file_path = path_to_raw_dir / file_hash

    if not file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {file_path}")

    # Get project directory
    project_dir = str(Path(DataService.get_data_dir(namespace, project)).parent)

    # Create and execute the RAG ingest task
    ingest_task = signature(
        "rag.ingest_file",
        args=[
            project_dir,
            data_processing_strategy_name,
            database_name,
            str(file_path),
            None,  # filename
            dataset,  # dataset_name
        ],
        app=app,
    )

    # Execute the task and wait for result using polling to avoid result.get() error
    result = ingest_task.apply_async()

    # Poll for result completion without using result.get()
    timeout = 300  # 5 minutes
    poll_interval = 2  # seconds
    waited = 0

    while waited < timeout:
        if result.status not in ("PENDING", "STARTED"):
            break
        time.sleep(poll_interval)
        waited += poll_interval

    if result.status == "SUCCESS":
        success, details = result.result
    elif result.status == "FAILURE":
        # Re-raise the exception without using result.get()
        raise Exception(f"RAG task failed: {result.result}")
    else:
        # Timeout or other status
        raise Exception(f"RAG task timed out or failed with status: {result.status}")

    if not success:
        raise Exception(f"Failed to ingest file {file_hash}: {details}")

    return {
        "file_hash": file_hash,
        "success": success,
        "details": details,
    }


def create_dataset_processing_chain(
    namespace: str,
    project: str,
    dataset: str,
    data_processing_strategy_name: str,
    database_name: str,
    file_hashes: list[str],
) -> group:
    """
    Create a Celery task chain for processing all files in a dataset.
    Returns a group of tasks that can be executed in parallel.
    """
    tasks = []

    for file_hash in file_hashes:
        task = process_single_file_task.s(
            namespace=namespace,
            project=project,
            dataset=dataset,
            file_hash=file_hash,
            data_processing_strategy_name=data_processing_strategy_name,
            database_name=database_name,
        )
        tasks.append(task)

    return group(*tasks)
