"""Base preprocessor for document preprocessing (OCR, format conversion, etc.)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class PreprocessorResult:
    """Result from preprocessing operation."""

    content: str  # Extracted text content
    metadata: dict[str, Any]  # Layout info, confidence, tables, etc.
    output_format: str  # 'text', 'markdown', 'searchable_pdf'
    output_file: Optional[str] = None  # Path to searchable PDF if created
    success: bool = True
    errors: list[str] = field(default_factory=list)


class BasePreprocessor(ABC):
    """Abstract base class for document preprocessors.

    Preprocessors run BEFORE parsers to extract/convert content:
    - OCR: Extract text from images/scanned PDFs
    - Format conversion: Convert proprietary formats to standard formats
    - Decompression: Extract archives, unwrap containers
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary for the preprocessor
        """
        self.config = config or {}

    @abstractmethod
    def can_process(self, file_path: str, metadata: dict[str, Any]) -> bool:
        """Check if this preprocessor can handle the file.

        Args:
            file_path: Path to the file
            metadata: File metadata (MIME type, size, etc.)

        Returns:
            True if preprocessor can process this file
        """
        pass

    @abstractmethod
    def preprocess(
        self, file_path: str, metadata: dict[str, Any]
    ) -> PreprocessorResult:
        """Preprocess the document.

        Args:
            file_path: Path to input file
            metadata: File metadata

        Returns:
            PreprocessorResult with extracted content and metadata
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported file extensions (e.g., ['.pdf', '.png'])
        """
        pass
