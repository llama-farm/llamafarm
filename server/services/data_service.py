import hashlib
import os
from datetime import datetime

from fastapi import UploadFile
from pydantic import BaseModel

from core.logging import FastAPIStructLogger
from services.project_service import ProjectService

logger = FastAPIStructLogger()

DATA_DIR_NAME = "lf_data"


class MetadataFileContent(BaseModel):
    original_file_name: str
    resolved_file_name: str
    timestamp: float
    size: int
    mime_type: str
    hash: str


class FileExistsInAnotherDatasetError(Exception):
    pass


class DataService:
    """
    Service for managing data

    Data is stored in the project data directory in the following structure:

    data_dir/
      meta/
        <file_content_hash>.json # Metadata file
      raw/
        <file_content_hash> # File content
      index/
        by_name/
          <original_file_name> -> ../raw/<file_content_hash> # symlink to file content

    The metadata file is a json file that contains the following information:
    {
      "original_file_name": "example.pdf",
      "resolved_file_name": "example_1719852800.pdf", # resolved with timestamp
      "timestamp": "1753978291",
      "size": 1000, # in bytes
      "mime_type": "application/pdf", # mime type of the original file
      "hash": "2b3e321d021e5c625a4a003f0624801fa46faab59b530caddcd65a5c106b8a17"
    }

    File name collisions are resolved by adding an epoch timestamp to the file name.
    E.g. "example.pdf" -> "example_1719852800.pdf"
    """

    @classmethod
    def get_data_dir(cls, namespace: str, project_id: str):
        project_dir = ProjectService.get_project_dir(namespace, project_id)
        os.makedirs(os.path.join(project_dir, DATA_DIR_NAME), exist_ok=True)
        os.makedirs(os.path.join(project_dir, DATA_DIR_NAME, "meta"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, DATA_DIR_NAME, "raw"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, DATA_DIR_NAME, "stores"), exist_ok=True)
        os.makedirs(
            os.path.join(project_dir, DATA_DIR_NAME, "index", "by_name"), exist_ok=True
        )
        return os.path.join(project_dir, DATA_DIR_NAME)

    @classmethod
    def hash_data(cls, data: bytes):
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def append_collision_timestamp(cls, file_name: str):
        file_name_without_extension, extension = os.path.splitext(file_name)
        return f"{file_name_without_extension}_{datetime.now().timestamp()}{extension}"

    @classmethod
    async def add_data_file(
        cls,
        namespace: str,
        project_id: str,
        file: UploadFile,
    ) -> MetadataFileContent:
        import mimetypes

        data_dir = cls.get_data_dir(namespace, project_id)
        file_data = await file.read()
        data_hash = cls.hash_data(file_data)
        resolved_file_name = cls.append_collision_timestamp(file.filename or "unknown")

        # Detect MIME type from filename if not provided or is generic
        mime_type = file.content_type
        if not mime_type or mime_type == "application/octet-stream":
            # Try to guess from filename
            guessed_type, _ = mimetypes.guess_type(file.filename or "")
            mime_type = guessed_type if guessed_type else "application/octet-stream"

        # Create metadata file
        metadata_path = os.path.join(data_dir, "meta", f"{data_hash}.json")
        metadata_file_content = MetadataFileContent(
            original_file_name=file.filename or "unknown",
            resolved_file_name=resolved_file_name,
            timestamp=datetime.now().timestamp(),
            size=len(file_data),
            mime_type=mime_type,
            hash=data_hash,
        )
        with open(metadata_path, "w") as f:
            f.write(metadata_file_content.model_dump_json())

        # Create raw file
        data_path = os.path.join(data_dir, "raw", data_hash)
        with open(data_path, "wb") as f:
            f.write(file_data)

        # Create index file
        index_path = os.path.join(data_dir, "index", "by_name", resolved_file_name)
        os.symlink(data_path, index_path)

        logger.info(
            f"Wrote file '{file.filename}' to disk",
            metadata=metadata_file_content.model_dump(),
            data_dir=data_dir,
        )
        return metadata_file_content

    @classmethod
    def read_data_file(
        cls,
        namespace: str,
        project_id: str,
        file_content_hash: str,
    ) -> bytes:
        data_dir = cls.get_data_dir(namespace, project_id)
        data_path = os.path.join(data_dir, "raw", file_content_hash)
        with open(data_path, "rb") as f:
            return f.read()

    @classmethod
    def get_data_file_metadata_by_hash(
        cls,
        namespace: str,
        project_id: str,
        file_content_hash: str,
    ) -> MetadataFileContent:
        data_dir = cls.get_data_dir(namespace, project_id)
        metadata_path = os.path.join(data_dir, "meta", f"{file_content_hash}.json")
        with open(metadata_path) as f:
            return MetadataFileContent.model_validate_json(f.read())

    @classmethod
    def delete_data_file(
        cls,
        namespace: str,
        project_id: str,
        dataset: str,
        file: MetadataFileContent,
    ) -> MetadataFileContent:
        data_dir = cls.get_data_dir(namespace, project_id)

        # Make sure the file is not in use by another dataset
        project = ProjectService.get_project(namespace, project_id)
        other_dataset = next(
            (
                ds
                for ds in project.config.datasets
                if ds.name != dataset and file.hash in ds.files
            ),
            None,
        )
        if other_dataset:
            raise FileExistsInAnotherDatasetError(
                f"File '{file.hash}' is in use by dataset '{other_dataset.name}'"
            )

        metadata_path = os.path.join(data_dir, "meta", f"{file.hash}.json")
        os.remove(metadata_path)
        data_path = os.path.join(data_dir, "raw", file.hash)
        os.remove(data_path)
        index_path = os.path.join(data_dir, "index", "by_name", file.resolved_file_name)
        os.remove(index_path)

        logger.info(
            f"Deleted file '{file.hash}' from disk",
            metadata=file.model_dump(),
            data_dir=data_dir,
        )

        return file
