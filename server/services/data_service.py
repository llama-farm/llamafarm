import os
import hashlib
import mimetypes

from datetime import datetime
from pydantic import BaseModel
from server.services.project_service import ProjectService

DATA_DIR_NAME = "lf_data"

class MetadataFileContent(BaseModel):
  original_file_name: str
  resolved_file_name: str
  added_at: datetime
  size: int
  mime_type: str
  hash: str

class DataService:
  """
  Service for managing data

  Data is stored in the project data directory in the following structure:
  
  data_dir/
    meta/
      <file_content_hash>.json # Metadata file
    blobs/
      <file_content_hash> # File content
    index/
      by_name/
        <original_file_name> -> ../raw/<file_content_hash> # symlink to the file content

  The metadata file is a json file that contains the following information:
  {
    "original_file_name": "example.pdf",
    "resolved_file_name": "example_1719852800.pdf", # resolved file name with timestamp to avoid collisions
    "timestamp": "1753978291",
    "size": 1000, # in bytes
    "mime_type": "application/pdf", # mime type of the original file
    "hash": "2b3e321d021e5c625a4a003f0624801fa46faab59b530caddcd65a5c106b8a17" # hash of the file content
  }

  File name collisions are resolved by adding an epoch timestamp to the file name. E.g. "example.pdf" -> "example_1719852800.pdf"
  """

  @classmethod
  def get_data_dir(cls, project_id: str):
    return os.path.join(ProjectService.get_project_dir(project_id), DATA_DIR_NAME)

  @classmethod
  def hash_data(cls, data: bytes):
    return hashlib.sha256(data).hexdigest()

  @classmethod
  def append_collision_timestamp(cls, file_name: str):
    return f"{file_name}_{datetime.now().timestamp()}"

  def add_data_bytes(
    self,
    project_id: str,
    file_name: str,
    data: bytes,
    mime_type: str = None,
  ) -> str:
    data_dir = self.get_data_dir(project_id)
    data_hash = self.hash_data(data)
    resolved_file_name = self.append_collision_timestamp(file_name)

    # Create metadata file
    metadata_path = os.path.join(data_dir, "meta", f"{data_hash}.json")
    with open(metadata_path, "w") as f:
      f.write(MetadataFileContent(
        original_file_name=file_name,
        resolved_file_name=resolved_file_name,
        timestamp=datetime.now().timestamp(),
        size=len(data),
        mime_type=mime_type or mimetypes.guess_type(file_name),
        hash=data_hash
      ).model_dump_json())

    # Create raw file
    data_path = os.path.join(data_dir, "raw", data_hash)
    with open(data_path, "wb") as f:
      f.write(data)

    # Create index file
    index_path = os.path.join(data_dir, "index", "by_name", resolved_file_name)
    os.symlink(data_path, index_path)
    return data_path


  def add_data_file(self, project_id: str, file_path: str) -> str:
    data_dir = self.get_data_dir(project_id)
    file = open(file_path, "rb")
    file_data = file.read()
    file.close()
    data_hash = self.hash_data(file_data)
    resolved_file_name = self.append_collision_timestamp(os.path.basename(file_path))
    
    # Create metadata file
    metadata_path = os.path.join(data_dir, "meta", f"{data_hash}.json")
    with open(metadata_path, "w") as f:
      f.write(MetadataFileContent(
        original_file_name=os.path.basename(file_path),
        resolved_file_name=resolved_file_name,
        timestamp=datetime.now().timestamp(),
        size=len(file_data),
        mime_type=mimetypes.guess_type(file_path)[0],
        hash=data_hash
      ).model_dump_json())

    # Create raw file
    data_path = os.path.join(data_dir, "raw", data_hash)
    with open(data_path, "wb") as f:
      f.write(file_data)
    
    # Create index file
    index_path = os.path.join(data_dir, "index", "by_name", resolved_file_name)
    os.symlink(data_path, index_path)
    return data_path

  def read_data_file(self, project_id: str, file_content_hash: str) -> bytes:
    data_dir = self.get_data_dir(project_id)
    data_path = os.path.join(data_dir, "raw", file_content_hash)
    with open(data_path, "rb") as f:
      return f.read()

  def delete_data_file(self, project_id: str, file_content_hash: str):
    data_dir = self.get_data_dir(project_id)
    data_path = os.path.join(data_dir, "raw", file_content_hash)
    os.remove(data_path)
    metadata_path = os.path.join(data_dir, "meta", f"{file_content_hash}.json")
    os.remove(metadata_path)
    index_path = os.path.join(data_dir, "index", "by_name", file_content_hash)
    os.remove(index_path)
