from config.datamodel import Dataset

from api.errors import DatasetNotFoundError, NotFoundError
from core.logging import FastAPIStructLogger
from services.data_service import MetadataFileContent
from services.project_service import ProjectService

logger = FastAPIStructLogger()


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

        # Create new dataset
        new_dataset = Dataset(
            name=name,
            data_processing_strategy=data_processing_strategy,
            database=database,
            files=[],
        )

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
    def add_file_to_dataset(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        file: MetadataFileContent,
    ):
        """
        Add a file to a dataset
        """
        project_config = ProjectService.load_config(namespace, project)
        existing_datasets = project_config.datasets or []
        dataset_to_update = next(
            (ds for ds in existing_datasets if ds.name == dataset),
            None,
        )
        if dataset_to_update is None:
            raise DatasetNotFoundError(dataset)
        dataset_to_update.files.append(file.hash)
        project_config.datasets = existing_datasets
        ProjectService.save_config(namespace, project, project_config)

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
        dataset_to_update = next(
            (ds for ds in existing_datasets if ds.name == dataset),
            None,
        )
        if dataset_to_update is None:
            raise ValueError(f"Dataset {dataset} not found")
        try:
            dataset_to_update.files.remove(file_hash)
        except ValueError as e:
            raise NotFoundError(f"File {file_hash} not found in dataset") from e
        project_config.datasets = existing_datasets
        ProjectService.save_config(namespace, project, project_config)
