# pyright: reportMissingImports=false

import contextlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from celery import group  # type: ignore[import-not-found,import-untyped]
from config.datamodel import Dataset
from fastapi import UploadFile
from pydantic import BaseModel

from api.errors import DatasetNotFoundError
from core.celery import app
from core.celery.rag_client import (
    build_ingest_signature,
    delete_file_from_rag,
    list_rag_documents,
)
from core.logging import FastAPIStructLogger
from services.data_service import DataService, MetadataFileContent
from services.project_service import ProjectService

logger = FastAPIStructLogger()


@dataclass
class DatasetIngestLaunchResult:
    task_id: str
    message: str
    files: list[str]


class DatasetDetails(BaseModel):
    files_metadata: list[MetadataFileContent]


class DatasetWithFileDetails(Dataset):
    details: DatasetDetails


class DatasetService:
    """Service for managing datasets within projects"""

    @classmethod
    def list_datasets(cls, namespace: str, project: str) -> list[Dataset]:
        """
        List all datasets for a given project
        """
        project_config = ProjectService.load_config(namespace, project)
        return project_config.datasets or []

    @classmethod
    def list_datasets_with_file_details(
        cls, namespace: str, project: str
    ) -> list[DatasetWithFileDetails]:
        """
        List all datasets for a given project with file details including original filenames
        and chunk counts from vector database.
        """
        datasets = cls.list_datasets(namespace, project)
        datasets_with_details: list[DatasetWithFileDetails] = []

        # Build a map of database -> {file_hash -> chunk_count} for all databases used by datasets
        project_dir = ProjectService.get_project_dir(namespace, project)
        database_chunk_counts: dict[str, dict[str, int]] = {}

        # Get unique databases from datasets
        unique_databases = {d.database for d in datasets}

        for database in unique_databases:
            try:
                # Fetch documents from the database (with a high limit to get all)
                result = list_rag_documents(
                    project_dir=str(project_dir),
                    database=database,
                    limit=10000,  # High limit to get all documents
                    offset=0,
                )
                # Build hash -> chunk_count map
                # The document id is typically the file hash
                chunk_map: dict[str, int] = {}
                for doc in result.get("documents", []):
                    doc_id = doc.get("id", "")
                    chunk_count = doc.get("chunk_count", 0)
                    if doc_id:
                        chunk_map[doc_id] = chunk_count
                database_chunk_counts[database] = chunk_map
            except Exception as e:
                logger.warning(
                    "Failed to fetch chunk counts for database",
                    database=database,
                    error=str(e),
                )
                database_chunk_counts[database] = {}

        for dataset in datasets:
            files_with_details = cls.list_dataset_files(
                namespace, project, dataset.name
            )

            # Enrich files with chunk counts from the database
            chunk_map = database_chunk_counts.get(dataset.database, {})
            for file_meta in files_with_details:
                file_meta.chunk_count = chunk_map.get(file_meta.hash)

            dataset_with_details = DatasetWithFileDetails(
                name=dataset.name,
                auto_process=dataset.auto_process,
                data_processing_strategy=dataset.data_processing_strategy,
                database=dataset.database,
                details=DatasetDetails(files_metadata=files_with_details),
            )
            datasets_with_details.append(dataset_with_details)

        return datasets_with_details

    @classmethod
    def list_dataset_files(
        cls, namespace: str, project: str, dataset: str
    ) -> list[MetadataFileContent]:
        """
        List all files in a dataset
        """
        dataset_dir = DataService.ensure_data_dir(namespace, project, dataset)
        dataset_meta_dir = os.path.join(dataset_dir, "meta")
        files: list[MetadataFileContent] = []

        try:
            file_list = os.listdir(dataset_meta_dir)
        except FileNotFoundError:
            logger.warning(
                "Dataset metadata directory not found",
                namespace=namespace,
                project=project,
                dataset=dataset,
                path=dataset_meta_dir,
            )
            return files
        except PermissionError as e:
            logger.error(
                "Permission denied reading dataset metadata directory",
                namespace=namespace,
                project=project,
                dataset=dataset,
                path=dataset_meta_dir,
                error=str(e),
            )
            return files

        for file in file_list:
            file_path = os.path.join(dataset_meta_dir, file)
            try:
                with open(file_path) as f:
                    metadata = MetadataFileContent.model_validate_json(f.read())
                    files.append(metadata)
            except OSError as e:
                logger.warning(
                    "Failed to read metadata file",
                    namespace=namespace,
                    project=project,
                    dataset=dataset,
                    file=file,
                    error=str(e),
                )
            except ValueError as e:
                # Pydantic validation errors (including JSON parsing) raise ValueError
                logger.warning(
                    "Failed to parse metadata file",
                    namespace=namespace,
                    project=project,
                    dataset=dataset,
                    file=file,
                    error=str(e),
                )

        return files

    @classmethod
    def get_dataset_config(cls, namespace: str, project: str, dataset: str) -> Dataset:
        """
        Load a dataset configuration from the project config.
        """
        project_config = ProjectService.load_config(namespace, project)
        existing_datasets = project_config.datasets or []
        dataset_obj = next(
            (ds for ds in existing_datasets if ds.name == dataset),
            None,
        )
        if dataset_obj is None:
            raise DatasetNotFoundError(dataset)
        return dataset_obj

    @classmethod
    def create_dataset(
        cls,
        namespace: str,
        project: str,
        name: str,
        data_processing_strategy: str,
        database: str,
        auto_process: bool | None = True,
    ) -> Dataset:
        """
        Create a new dataset in the project

        Raises:
            ValueError: If dataset with same name already exists or parser is not supported
        """
        project_config = ProjectService.load_config(namespace, project)
        existing_datasets = project_config.datasets or []

        # Check if dataset already exists
        for dataset in existing_datasets or []:
            if dataset.name == name:
                raise ValueError(f"Dataset {name} already exists")

        # Validate RAG strategy
        supported_data_processing_strategies = (
            cls.get_supported_data_processing_strategies(namespace, project)
        )
        supported_data_processing_strategies = (
            supported_data_processing_strategies  # Add default auto strategy
        )

        if data_processing_strategy not in supported_data_processing_strategies:
            raise ValueError(
                f"RAG data processing strategy {data_processing_strategy} not supported"
            )

        supported_databases = cls.get_supported_databases(namespace, project)
        supported_databases = supported_databases  # Add default auto strategy

        if database not in supported_databases:
            raise ValueError(f"RAG database {database} not supported")

        new_dataset = Dataset(
            name=name,
            auto_process=auto_process,
            data_processing_strategy=data_processing_strategy,
            database=database,
        )

        # Add the new dataset to the project config
        existing_datasets.append(new_dataset)
        project_config.datasets = existing_datasets
        ProjectService.save_config(namespace, project, project_config)

        return new_dataset

    @classmethod
    def delete_dataset(cls, namespace: str, project: str, name: str) -> Dataset:
        """
        Delete a dataset from the project

        Returns:
            Dataset: The deleted dataset object

        Raises:
            ValueError: If dataset with given name is not found
        """
        project_config = ProjectService.load_config(namespace, project)
        existing_datasets = project_config.datasets or []

        # Filter out the dataset to delete
        dataset_to_delete = next(
            (dataset for dataset in existing_datasets if dataset.name == name), None
        )
        if dataset_to_delete is None:
            raise ValueError(f"Dataset {name} not found")

        project_config.datasets = [
            dataset for dataset in existing_datasets if dataset.name != name
        ]
        ProjectService.save_config(namespace, project, project_config)

        return dataset_to_delete

    @classmethod
    def get_supported_data_processing_strategies(
        cls, namespace: str, project: str
    ) -> list[str]:
        """
        Get list of supported data processing strategies
        """
        project_config = ProjectService.load_config(namespace, project)
        rag_config = project_config.rag

        if rag_config is None:
            return []

        custom_data_processing_strategies: list[str] = []

        # Only support new schema - no backwards compatibility
        if (
            hasattr(rag_config, "data_processing_strategies")
            and rag_config.data_processing_strategies
        ):
            for strategy in rag_config.data_processing_strategies:
                if hasattr(strategy, "name") and strategy.name:
                    custom_data_processing_strategies.append(strategy.name)
        elif (
            isinstance(rag_config, dict) and "data_processing_strategies" in rag_config
        ):
            strategies = rag_config["data_processing_strategies"]
            if isinstance(strategies, list):
                for strategy in strategies:
                    if isinstance(strategy, dict) and "name" in strategy:
                        custom_data_processing_strategies.append(strategy["name"])

        return custom_data_processing_strategies

    @classmethod
    def get_supported_databases(cls, namespace: str, project: str) -> list[str]:
        """
        Get list of supported databases
        """
        project_config = ProjectService.load_config(namespace, project)
        rag_config = project_config.rag

        if rag_config is None:
            return []

        databases: list[str] = []

        if hasattr(rag_config, "databases") and rag_config.databases:
            for database in rag_config.databases:
                if hasattr(database, "name") and database.name:
                    databases.append(database.name)

        return databases

    @classmethod
    async def add_file_to_dataset(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        file: UploadFile,
    ) -> tuple[bool, MetadataFileContent]:
        """
        Add a file to a dataset

        Returns:
            bool: True if file was added, False if it was skipped (duplicate)
        """

        project_config = ProjectService.load_config(namespace, project)
        existing_datasets = project_config.datasets or []
        dataset_obj = next(
            (ds for ds in existing_datasets if ds.name == dataset),
            None,
        )
        if dataset_obj is None:
            raise DatasetNotFoundError(dataset)

        # Check if file already exists in dataset (duplicate detection)
        file_hash = DataService.hash_data(await file.read())
        await file.seek(0)
        existing_file: MetadataFileContent | None = None
        with contextlib.suppress(FileNotFoundError):
            existing_file = DataService.get_data_file_metadata_by_hash(
                namespace, project, dataset, file_hash
            )

        if existing_file is not None:
            logger.info(
                "File already exists in dataset, skipping",
                dataset=dataset,
                filename=existing_file.original_file_name,
                hash=existing_file.hash,
            )
            return False, existing_file

        metadata_file_content = await DataService.add_data_file(
            namespace=namespace,
            project_id=project,
            dataset=dataset,
            file=file,
        )

        return True, metadata_file_content

    @classmethod
    async def remove_file_from_dataset(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        file_hash: str,
    ) -> dict:
        """
        Remove a file from a dataset and delete its chunks from the vector store.

        Returns:
            Dictionary with deletion results including deleted_count from vector store.
        """

        project_config = ProjectService.load_config(namespace, project)
        existing_datasets = project_config.datasets or []
        dataset_obj = next(
            (ds for ds in existing_datasets if ds.name == dataset),
            None,
        )
        if dataset_obj is None:
            raise DatasetNotFoundError(dataset)

        # Delete chunks from vector store via RAG task
        project_dir = ProjectService.get_project_dir(namespace, project)
        result = await delete_file_from_rag(
            project_dir=project_dir,
            database_name=dataset_obj.database,
            file_hash=file_hash,
        )

        if result.get("status") == "error":
            raise Exception(result.get("error"))

        logger.info(
            "Deleted chunks from vector store",
            namespace=namespace,
            project=project,
            dataset=dataset,
            file_hash=file_hash[:16] + "...",
            deleted_chunks=result.get("deleted_count", 0),
        )

        # Then delete from disk
        metadata_file_content = DataService.delete_data_file(
            namespace=namespace,
            project_id=project,
            dataset=dataset,
            file_hash=file_hash,
        )

        logger.info(
            "Removed file from dataset",
            namespace=namespace,
            project=project,
            dataset=dataset,
            metadata=metadata_file_content.model_dump_json(),
        )

        return result

    @classmethod
    async def delete_file_chunks(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        file_hash: str,
    ) -> dict:
        """
        Delete chunks for a file from the vector store WITHOUT deleting the source file.
        Used for reprocessing files.

        Returns:
            Dictionary with deletion results including deleted_count from vector store.
        """
        project_config = ProjectService.load_config(namespace, project)
        existing_datasets = project_config.datasets or []
        dataset_obj = next(
            (ds for ds in existing_datasets if ds.name == dataset),
            None,
        )
        if dataset_obj is None:
            raise DatasetNotFoundError(dataset)

        project_dir = ProjectService.get_project_dir(namespace, project)
        result = await delete_file_from_rag(
            project_dir=project_dir,
            database_name=dataset_obj.database,
            file_hash=file_hash,
        )

        if result.get("status") == "error":
            raise Exception(result.get("error"))

        logger.info(
            "Deleted chunks from vector store (keeping source file)",
            namespace=namespace,
            project=project,
            dataset=dataset,
            file_hash=file_hash[:16] + "...",
            deleted_chunks=result.get("deleted_count", 0),
        )

        return result

    @classmethod
    async def delete_dataset_chunks(
        cls,
        namespace: str,
        project: str,
        dataset: str,
    ) -> dict:
        """
        Delete chunks for ALL files from the vector store WITHOUT deleting the source files.
        Used for reprocessing entire dataset.

        Returns:
            Dictionary with total_deleted_chunks, total_files_cleared, and total_files_failed counts.
        """
        project_config = ProjectService.load_config(namespace, project)
        existing_datasets = project_config.datasets or []
        dataset_obj = next(
            (ds for ds in existing_datasets if ds.name == dataset),
            None,
        )
        if dataset_obj is None:
            raise DatasetNotFoundError(dataset)

        # Get all files in the dataset
        files = cls.list_dataset_files(namespace, project, dataset)

        total_deleted = 0
        files_cleared = 0
        files_failed = 0
        project_dir = ProjectService.get_project_dir(namespace, project)

        for file_info in files:
            file_hash = file_info.hash
            if not file_hash:
                continue

            result = await delete_file_from_rag(
                project_dir=project_dir,
                database_name=dataset_obj.database,
                file_hash=file_hash,
            )

            if result.get("status") == "error":
                files_failed += 1
                logger.warning(
                    "Failed to delete chunks for file",
                    namespace=namespace,
                    project=project,
                    dataset=dataset,
                    file_hash=file_hash[:16] + "...",
                    error=result.get("error"),
                )
            else:
                total_deleted += result.get("deleted_count", 0)
                files_cleared += 1

        logger.info(
            "Deleted all chunks from vector store (keeping source files)",
            namespace=namespace,
            project=project,
            dataset=dataset,
            total_deleted=total_deleted,
            files_cleared=files_cleared,
            files_failed=files_failed,
        )

        return {
            "total_deleted_chunks": total_deleted,
            "total_files_cleared": files_cleared,
            "total_files_failed": files_failed,
        }

    @classmethod
    def _build_ingest_tasks(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        file_hashes: list[str] | None = None,
        parser_overrides: dict | None = None,
    ) -> tuple[Dataset, list[str], list]:
        dataset_config = cls.get_dataset_config(namespace, project, dataset)
        dataset_file_hashes: list[str] = []

        if file_hashes is None:
            dataset_files = cls.list_dataset_files(namespace, project, dataset)
            dataset_file_hashes = [file.hash for file in dataset_files if file.hash]
        else:
            dataset_file_hashes = [file_hash for file_hash in file_hashes if file_hash]

        project_dir = ProjectService.get_project_dir(namespace, project)
        raw_dir = Path(DataService.ensure_data_dir(namespace, project, dataset)) / "raw"
        ingest_tasks = []

        for file_hash in dataset_file_hashes:
            file_path = raw_dir / file_hash
            if not file_path.exists():
                raise FileNotFoundError(f"Raw file not found: {file_path}")

            metadata = DataService.get_data_file_metadata_by_hash(
                namespace, project, dataset, file_hash
            )
            original_filename = (
                metadata.original_file_name if metadata else file_hash
            )

            ingest_tasks.append(
                build_ingest_signature(
                    project_dir=project_dir,
                    data_processing_strategy_name=dataset_config.data_processing_strategy,
                    database_name=dataset_config.database,
                    source_path=str(file_path),
                    filename=original_filename,
                    dataset_name=dataset,
                    parser_overrides=parser_overrides,
                )
            )

        return dataset_config, dataset_file_hashes, ingest_tasks

    @classmethod
    def start_dataset_ingestion(
        cls, namespace: str, project: str, dataset: str
    ) -> DatasetIngestLaunchResult:
        """
        Kick off ingestion tasks for all files in a dataset and return the tracking task id.
        """
        dataset_config, dataset_file_hashes, ingest_tasks = cls._build_ingest_tasks(
            namespace=namespace,
            project=project,
            dataset=dataset,
        )

        if not ingest_tasks:
            task_id = str(uuid.uuid4())
            payload = {
                "message": "Dataset processed successfully",
                "namespace": namespace,
                "project": project,
                "dataset": dataset,
                "strategy": dataset_config.data_processing_strategy,
                "files": [],
                "total_files": 0,
            }
            app.backend.store_result(task_id, payload, "SUCCESS")
            return DatasetIngestLaunchResult(
                task_id=task_id, message=str(payload["message"]), files=[]
            )

        job = group(*ingest_tasks)
        result = job.apply_async()
        child_task_ids = []
        if result.results:
            child_task_ids = [
                child.id for child in result.results if hasattr(child, "id")
            ]

        app.backend.store_result(
            result.id,
            {
                "type": "group",
                "children": child_task_ids,
                "file_hashes": dataset_file_hashes,
                "namespace": namespace,
                "project": project,
                "dataset": dataset,
                "strategy": dataset_config.data_processing_strategy,
            },
            "PENDING",
        )

        logger.info(
            "Started dataset ingestion",
            dataset=dataset,
            namespace=namespace,
            project=project,
            task_id=result.id,
            file_count=len(dataset_file_hashes),
        )

        return DatasetIngestLaunchResult(
            task_id=result.id,
            message="Dataset ingestion started",
            files=dataset_file_hashes,
        )

    @classmethod
    def start_ingestion_for_hashes(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        file_hashes: list[str],
        parser_overrides: dict | None = None,
    ) -> DatasetIngestLaunchResult:
        """
        Kick off ingestion tasks for a subset of dataset files.
        """
        dataset_config, dataset_file_hashes, ingest_tasks = cls._build_ingest_tasks(
            namespace=namespace,
            project=project,
            dataset=dataset,
            file_hashes=file_hashes,
            parser_overrides=parser_overrides,
        )

        if not ingest_tasks:
            task_id = str(uuid.uuid4())
            payload = {
                "message": "Dataset processed successfully",
                "namespace": namespace,
                "project": project,
                "dataset": dataset,
                "strategy": dataset_config.data_processing_strategy,
                "files": [],
                "total_files": 0,
            }
            app.backend.store_result(task_id, payload, "SUCCESS")
            return DatasetIngestLaunchResult(
                task_id=task_id, message=str(payload["message"]), files=[]
            )

        job = group(*ingest_tasks)
        result = job.apply_async()
        child_task_ids = []
        if result.results:
            child_task_ids = [
                child.id for child in result.results if hasattr(child, "id")
            ]

        app.backend.store_result(
            result.id,
            {
                "type": "group",
                "children": child_task_ids,
                "file_hashes": dataset_file_hashes,
                "namespace": namespace,
                "project": project,
                "dataset": dataset,
                "strategy": dataset_config.data_processing_strategy,
            },
            "PENDING",
        )

        logger.info(
            "Started dataset ingestion for subset",
            dataset=dataset,
            namespace=namespace,
            project=project,
            task_id=result.id,
            file_count=len(dataset_file_hashes),
        )

        return DatasetIngestLaunchResult(
            task_id=result.id,
            message="Dataset ingestion started",
            files=dataset_file_hashes,
        )
