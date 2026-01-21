import hashlib
import os
from datetime import datetime

from fastapi import UploadFile
from pydantic import BaseModel

from api.errors import NotFoundError
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
    # Optional fields populated at runtime by dataset service
    chunk_count: int | None = None  # Number of chunks in vector DB (None = unknown)


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
    def ensure_data_dir(cls, namespace: str, project_id: str, dataset: str):
        project_dir = ProjectService.get_project_dir(namespace, project_id)
        datasets_dir = os.path.join(project_dir, DATA_DIR_NAME, "datasets")
        dataset_dir = os.path.join(datasets_dir, dataset)
        norm_datasets_dir = os.path.abspath(os.path.normpath(datasets_dir))
        norm_dataset_dir = os.path.abspath(os.path.normpath(dataset_dir))
        if not norm_dataset_dir.startswith(norm_datasets_dir + os.sep):
            raise ValueError(f"Invalid dataset name: {dataset!r}")
        os.makedirs(norm_dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(norm_dataset_dir, "meta"), exist_ok=True)
        os.makedirs(os.path.join(norm_dataset_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(norm_dataset_dir, "stores"), exist_ok=True)
        os.makedirs(os.path.join(norm_dataset_dir, "index", "by_name"), exist_ok=True)
        return norm_dataset_dir

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
        dataset: str,
        file: UploadFile,
    ) -> MetadataFileContent:
        import mimetypes

        data_dir = cls.ensure_data_dir(namespace, project_id, dataset)
        file_data = await file.read()
        data_hash = cls.hash_data(file_data)

        # Strip directory paths from filename (handles folder uploads)
        # Use only the basename to avoid creating nested directories
        base_filename = os.path.basename(file.filename or "unknown")
        resolved_file_name = cls.append_collision_timestamp(base_filename)

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
    def get_data_file_metadata_by_hash(
        cls,
        namespace: str,
        project_id: str,
        dataset: str,
        file_content_hash: str,
    ) -> MetadataFileContent | None:
        data_dir = cls.ensure_data_dir(namespace, project_id, dataset)
        metadata_path = os.path.join(data_dir, "meta", f"{file_content_hash}.json")
        try:
            with open(metadata_path) as f:
                return MetadataFileContent.model_validate_json(f.read())
        except FileNotFoundError:
            return None

    @classmethod
    def delete_data_file(
        cls,
        namespace: str,
        project_id: str,
        dataset: str,
        file_hash: str,
    ) -> MetadataFileContent:
        data_dir = cls.ensure_data_dir(namespace, project_id, dataset)

        metadata_path = os.path.join(data_dir, "meta", f"{file_hash}.json")
        metadata_file_content = cls.get_data_file_metadata_by_hash(
            namespace, project_id, dataset, file_hash
        )

        if metadata_file_content is None:
            raise NotFoundError(f"File {file_hash} not found")

        os.remove(metadata_path)
        data_path = os.path.join(data_dir, "raw", file_hash)
        os.remove(data_path)
        index_path = os.path.join(
            data_dir, "index", "by_name", metadata_file_content.resolved_file_name
        )
        os.remove(index_path)

        logger.info(
            f"Deleted file '{file_hash}' from disk",
            metadata=metadata_file_content.model_dump(),
            data_dir=data_dir,
        )

        return metadata_file_content
