from config.datamodel import Dataset

from api.errors import DatasetNotFoundError, NotFoundError
from core.logging import FastAPIStructLogger
from services.data_service import MetadataFileContent
from services.project_service import ProjectService

# TODO: populate this with default strategies from the rug sub-system
DEFAULT_RAG_STRATEGIES: list[str] = ["auto"]

logger = FastAPIStructLogger()


class DatasetService:
    """Service for managing datasets within projects"""

    @classmethod
    def list_datasets(cls, namespace: str, project: str) -> list[Dataset]:
        """
        List all datasets for a given project
        """
        project_config = ProjectService.load_config(namespace, project)
        return project_config.datasets

    @classmethod
    def create_dataset(
        cls,
        namespace: str,
        project: str,
        name: str,
        rag_strategy: str,
    ) -> Dataset:
        """
        Create a new dataset in the project

        Raises:
            ValueError: If dataset with same name already exists or parser is not supported
        """
        project_config = ProjectService.load_config(namespace, project)
        existing_datasets = project_config.datasets

        # Check if dataset already exists
        for dataset in existing_datasets:
            if dataset.name == name:
                raise ValueError(f"Dataset {name} already exists")

        # Validate RAG strategy
        supported_rag_strategies = cls.get_supported_rag_strategies(namespace, project)

        if rag_strategy not in supported_rag_strategies:
            raise ValueError(f"RAG strategy {rag_strategy} not supported")

        # Create new dataset
        new_dataset = Dataset(
            name=name,
            rag_strategy=rag_strategy,
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
        existing_datasets = project_config.datasets

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
    def get_supported_rag_strategies(cls, namespace: str, project: str) -> list[str]:
        """
        Get list of supported RAG strategies for the project
        """
        project_config = ProjectService.load_config(namespace, project)
        rag_config = project_config.rag

        custom_rag_strategies: list[str] = []
        try:
            # Support new strict schema: rag.strategies is a list of strategy objects
            strategies = getattr(rag_config, "strategies", None)
            if strategies is not None:
                for s in strategies:
                    name = getattr(s, "name", None)
                    if name is None and isinstance(s, dict):
                        name = s.get("name")
                    if isinstance(name, str):
                        custom_rag_strategies.append(name)
            elif isinstance(rag_config, dict):
                # Dict form: prefer 'strategies' item names; fallback to legacy 'rag_strategies'
                dict_strategies = rag_config.get("strategies")
                if isinstance(dict_strategies, list):
                    custom_rag_strategies.extend(
                        s["name"]
                        for s in dict_strategies
                        if isinstance(s, dict) and isinstance(s.get("name"), str)
                    )
        except Exception:
            # Be resilient and fall back to defaults only
            custom_rag_strategies = []

        return DEFAULT_RAG_STRATEGIES + custom_rag_strategies

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
        existing_datasets = project_config.datasets
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
