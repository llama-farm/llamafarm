"""Dataset models for server-specific functionality"""
from pydantic import BaseModel, Field
from config.datamodel import Dataset


class DatasetFile(BaseModel):
    """Represents a file in a dataset with metadata"""
    file_hash: str = Field(..., description="SHA256 hash of the file")
    original_filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    created_at: str = Field(..., description="Timestamp when file was added")


class DatasetWithFileDetails(Dataset):
    """Dataset with detailed file information"""
    file_details: list[DatasetFile] = Field(..., description="Detailed file information")