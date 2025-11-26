import contextlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from celery import group  # type: ignore[import-untyped]
from config.datamodel import Dataset
from fastapi import UploadFile
from pydantic import BaseModel

from api.errors import DatasetNotFoundError
from core.celery import app
from core.celery.rag_client import build_ingest_signature
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
        """
        datasets = cls.list_datasets(namespace, project)
        datasets_with_details: list[DatasetWithFileDetails] = []

        for dataset in datasets:
            files_with_details = cls.list_dataset_files(
                namespace, project, dataset.name
            )

            dataset_with_details = DatasetWithFileDetails(
                name=dataset.name,
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
            except (OSError, IOError) as e:
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
    def create_dataset(
        cls,
        namespace: str,
        project: str,
        name: str,
        data_processing_strategy: str,
        database: str,
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
    def remove_file_from_dataset(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        file_hash: str,
    ):
        """
        Remove a file from a dataset
        """

        project_config = ProjectService.load_config(namespace, project)
        existing_datasets = project_config.datasets or []
        dataset_obj = next(
            (ds for ds in existing_datasets if ds.name == dataset),
            None,
        )
        if dataset_obj is None:
            raise DatasetNotFoundError(dataset)

        DataService.delete_data_file(
            namespace=namespace,
            project_id=project,
            dataset=dataset,
            file_hash=file_hash,
        )

    @classmethod
    def start_dataset_ingestion(
        cls, namespace: str, project: str, dataset: str
    ) -> DatasetIngestLaunchResult:
        """
        Kick off ingestion tasks for all files in a dataset and return the tracking task id.
        """
        project_config = ProjectService.get_project(namespace, project).config
        dataset_config = next(
            (ds for ds in (project_config.datasets or []) if ds.name == dataset), None
        )
        if dataset_config is None:
            raise ValueError(f"Dataset {dataset} not found")

        dataset_files = cls.list_dataset_files(namespace, project, dataset)
        dataset_file_hashes = [file.hash for file in dataset_files]
        strategy_name = dataset_config.data_processing_strategy

        if not dataset_files:
            task_id = str(uuid.uuid4())
            payload = {
                "message": "Dataset processed successfully",
                "namespace": namespace,
                "project": project,
                "dataset": dataset,
                "strategy": strategy_name,
                "files": [],
                "total_files": 0,
            }
            app.backend.store_result(task_id, payload, "SUCCESS")
            return DatasetIngestLaunchResult(
                task_id=task_id, message=str(payload["message"]), files=[]
            )

        project_dir = ProjectService.get_project_dir(namespace, project)
        raw_dir = Path(DataService.ensure_data_dir(namespace, project, dataset)) / "raw"
        ingest_tasks = []
        for file_metadata in dataset_files:
            file_path = raw_dir / file_metadata.hash
            if not file_path.exists():
                raise FileNotFoundError(f"Raw file not found: {file_path}")

            ingest_tasks.append(
                build_ingest_signature(
                    project_dir=project_dir,
                    data_processing_strategy_name=strategy_name,
                    database_name=dataset_config.database,
                    source_path=str(file_path),
                    filename=file_metadata.original_file_name,
                    dataset_name=dataset,
                )
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
                "strategy": strategy_name,
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
