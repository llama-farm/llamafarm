"""
Request/response types for document understanding endpoints.
"""

from pydantic import BaseModel


class DocumentExtractRequest(BaseModel):
    """Document extraction request."""

    model: str  # HuggingFace model ID (e.g., "naver-clova-ix/donut-base-finetuned-cord-v2")
    images: list[str] | None = None  # Base64-encoded document images
    file_id: str | None = None  # File ID from /v1/files upload
    prompts: list[str] | None = None  # Optional prompts for each image
    task: str = "extraction"  # extraction, vqa, classification
