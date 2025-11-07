"""Base class for OCR preprocessors."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

from components.preprocessors.base import BasePreprocessor, PreprocessorResult
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.preprocessors.ocr.base")


class BaseOCRPreprocessor(BasePreprocessor):
    """Base class for OCR engines with shared utilities."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize OCR preprocessor.

        Args:
            config: Configuration dictionary for OCR settings
        """
        super().__init__(config)

        # Common OCR configuration
        self.languages = self.config.get("languages", ["en"])
        self.detect_layout = self.config.get("detect_layout", True)
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.output_format = self.config.get(
            "output_format", "text"
        )  # text, markdown, searchable_pdf
        self.scanned_threshold = self.config.get(
            "scanned_threshold", 50
        )  # chars/page threshold

    def can_process(self, file_path: str, metadata: dict[str, Any]) -> bool:
        """Check if file needs OCR processing.

        Args:
            file_path: Path to the file
            metadata: File metadata

        Returns:
            True if file is an image or scanned PDF
        """
        ext = Path(file_path).suffix.lower()

        # Images always need OCR
        if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            return True

        # PDFs: check if image-based (no text layer)
        if ext == ".pdf":
            return self._is_image_based_pdf(file_path)

        return False

    def _is_image_based_pdf(self, pdf_path: str) -> bool:
        """Detect if PDF is image-based (scanned) vs. text-based.

        Strategy:
        1. Check first 5 pages for text content
        2. If text < scanned_threshold chars/page on average → likely scanned

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if PDF appears to be image-based
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            pages_to_check = min(5, len(doc))
            total_chars = 0

            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text()
                total_chars += len(text.strip())

            doc.close()

            # Threshold: < scanned_threshold chars/page suggests image-based
            avg_chars = total_chars / pages_to_check if pages_to_check > 0 else 0
            is_image_based = avg_chars < self.scanned_threshold

            logger.info(
                f"PDF analysis: {pdf_path}",
                pages_checked=pages_to_check,
                avg_chars_per_page=avg_chars,
                threshold=self.scanned_threshold,
                is_image_based=is_image_based,
            )

            return is_image_based

        except Exception as e:
            logger.warning(f"Failed to analyze PDF {pdf_path}: {e}")
            # Fail open: try OCR if detection fails
            return True

    def get_supported_formats(self) -> list[str]:
        """OCR supports images and PDFs.

        Returns:
            List of supported file extensions
        """
        return [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]

    @abstractmethod
    def _run_ocr(self, image_path: str) -> dict[str, Any]:
        """Run OCR engine on image. Implemented by subclasses.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with OCR results (implementation-specific)
        """
        pass
