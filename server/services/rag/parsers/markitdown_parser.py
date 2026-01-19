"""MarkItDown parser for lightweight RAG in server."""

import logging
from pathlib import Path
from typing import Any, Optional

import requests
from markitdown import MarkItDown

logger = logging.getLogger(__name__)


class MarkItDownParser:
    """Lightweight parser using Microsoft MarkItDown with OCR fallback."""

    def __init__(self, universal_runtime_url: str = "http://127.0.0.1:11540"):
        """Initialize parser.

        Args:
            universal_runtime_url: URL of Universal Runtime for OCR fallback
        """
        self.converter = MarkItDown()
        self.ocr_endpoint = f"{universal_runtime_url}/v1/ocr"
        logger.info("Initialized MarkItDown parser with OCR fallback")

    async def parse(
        self, file_path: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Parse document to markdown with automatic OCR fallback.

        Args:
            file_path: Path to file
            metadata: Optional file metadata (filename, content_type, etc.)

        Returns:
            Markdown text preserving structure

        Raises:
            Exception: If parsing fails
        """
        metadata = metadata or {}
        logger.info(f"Parsing {file_path} with MarkItDown")

        try:
            # Parse with MarkItDown first
            result = self.converter.convert(file_path)
            text = result.text_content

            # Check if OCR fallback needed
            if self._needs_ocr(file_path, text):
                logger.info(
                    f"Low text detected ({len(text)} chars), triggering OCR fallback"
                )
                ocr_text = await self._ocr_fallback(file_path)
                if ocr_text:
                    # Use OCR text if it has more content
                    if len(ocr_text) > len(text):
                        logger.info(
                            f"OCR extracted more text ({len(ocr_text)} chars), using OCR result"
                        )
                        text = ocr_text
                    else:
                        logger.info(
                            f"Original text has more content, keeping MarkItDown result"
                        )

            logger.info(
                f"Successfully parsed {file_path}, {len(text)} chars extracted"
            )
            return text

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise

    def _needs_ocr(self, file_path: str, text: str) -> bool:
        """Check if OCR fallback is needed.

        OCR is triggered if:
        1. File is an image (PNG, JPG, etc.)
        2. Very little text extracted from PDF (< 50 chars)

        Args:
            file_path: Path to file
            text: Extracted text

        Returns:
            True if OCR fallback should be attempted
        """
        ext = Path(file_path).suffix.lower()

        # Always OCR images
        if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            logger.debug(f"Image file detected ({ext}), OCR needed")
            return True

        # OCR scanned PDFs with little text
        if ext == ".pdf" and len(text.strip()) < 50:
            logger.debug(
                f"PDF with minimal text ({len(text.strip())} chars), OCR needed"
            )
            return True

        return False

    async def _ocr_fallback(self, file_path: str) -> Optional[str]:
        """Call Universal Runtime OCR endpoint.

        Args:
            file_path: Path to file for OCR

        Returns:
            Extracted text or None if OCR fails
        """
        try:
            logger.debug(f"Calling Universal Runtime OCR for {file_path}")
            with open(file_path, "rb") as f:
                files = {"file": f}
                response = requests.post(self.ocr_endpoint, files=files, timeout=30)
                response.raise_for_status()
                result = response.json()
                text = result.get("text", "")
                logger.info(f"OCR successful, extracted {len(text)} chars")
                return text
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"OCR fallback failed: Universal Runtime not available at {self.ocr_endpoint}"
            )
            return None
        except requests.exceptions.Timeout:
            logger.warning("OCR fallback failed: Request timed out after 30s")
            return None
        except Exception as e:
            logger.warning(f"OCR fallback failed: {e}")
            return None

    def parse_sync(
        self, file_path: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Synchronous version of parse() for compatibility.

        Args:
            file_path: Path to file
            metadata: Optional file metadata

        Returns:
            Markdown text preserving structure
        """
        import asyncio

        return asyncio.run(self.parse(file_path, metadata))
