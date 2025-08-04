import os
import sys
import logging

from pathlib import Path
from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)

# Add the parent directory to the Python path so we can import the config package
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from config import config_types
from core.config import settings
from services.project_service import ProjectService

type Dataset = config_types.DatasetsConfig

DEFAULT_PARSERS = ["pdf", "text", "csv", "json", "markdown"]

class DatasetService:
    """Service for managing datasets within projects"""

    @classmethod
    def list_datasets(cls, namespace: str, project: str) -> List[Dataset]:
        """
        List all datasets for a given project
        """
        project_config = ProjectService.load_config(namespace, project)
        datasets_config = project_config.get("datasets", [])
        
        datasets = [
            {
                "name": dataset["name"],
                "parser": dataset["parser"],
                "files": dataset["files"],
            }
            for dataset in datasets_config
        ]
        
        return datasets

    @classmethod
    def create_dataset(cls, namespace: str, project: str, name: str, parser: str) -> Dataset:
        """
        Create a new dataset in the project
        
        Raises:
            ValueError: If dataset with same name already exists or parser is not supported
        """
        project_config = ProjectService.load_config(namespace, project)
        existing_datasets = project_config.get("datasets", [])

        # Check if dataset already exists
        for dataset in existing_datasets:
            if dataset["name"] == name:
                raise ValueError(f"Dataset {name} already exists")

        # Validate parser
        supported_parsers = cls.get_supported_parsers(namespace, project)
        
        if parser not in supported_parsers:
            raise ValueError(f"Parser {parser} not supported")

        # Create new dataset
        new_dataset = {
            "name": name,
            "parser": parser,
            "files": [],
        }
        
        existing_datasets.append(new_dataset)
        project_config.update({"datasets": existing_datasets})
        ProjectService.save_config(namespace, project, project_config)
        
        return {
            "name": name,
            "parser": parser,
            "files": [],
        }

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
        existing_datasets = project_config.get("datasets", [])
        
        # Filter out the dataset to delete
        dataset_to_delete = next((dataset for dataset in existing_datasets if dataset["name"] == name), None)
        if dataset_to_delete is None:
            raise ValueError(f"Dataset {name} not found")
        
        project_config.update({"datasets": [dataset for dataset in existing_datasets if dataset["name"] != name]})
        ProjectService.save_config(namespace, project, project_config)
        
        return {
            "name": dataset_to_delete["name"],
            "parser": dataset_to_delete["parser"],
            "files": dataset_to_delete["files"],
        }

    @classmethod
    def get_supported_parsers(cls, namespace: str, project: str) -> List[str]:
        """
        Get list of supported parsers for the project
        """
        project_config = ProjectService.load_config(namespace, project)
        custom_parsers = project_config.get("rag", {}).get("parsers", []) or []
        return DEFAULT_PARSERS + custom_parsers