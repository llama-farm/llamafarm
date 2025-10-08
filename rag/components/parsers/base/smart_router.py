"""Smart router for intelligent file type detection and parser selection."""

import mimetypes
from pathlib import Path
from typing import Any, Optional

from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.base.smart_router")

# Try to import magic for content-based detection
try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logger.warning("python-magic not available. Install with: pip install python-magic")


class SmartRouter:
    """Intelligent router for selecting appropriate parsers based on file content."""

    def __init__(self, parser_registry=None):
        """Initialize smart router.

        Args:
            parser_registry: ParserRegistry instance
        """
        self.parser_registry = parser_registry
        self.mime_to_parser = self._build_mime_mapping()
        self.ext_to_parser = self._build_extension_mapping()

        # Initialize magic if available
        if MAGIC_AVAILABLE:
            self.magic = magic.Magic(mime=True)
        else:
            self.magic = None

    def _build_mime_mapping(self) -> dict[str, str]:
        """Build MIME type to parser mapping.

        Returns:
            Dictionary mapping MIME types to parser names
        """
        return {
            "application/pdf": "pdf",
            "text/plain": "text",
            "text/csv": "csv_excel",
            "application/vnd.ms-excel": "csv_excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "csv_excel",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/msword": "docx",
            "text/markdown": "markdown",
            "text/html": "web",
            "application/xhtml+xml": "web",
        }

    def _build_extension_mapping(self) -> dict[str, str]:
        """Build file extension to parser mapping.

        Returns:
            Dictionary mapping extensions to parser names
        """
        return {
            ".pdf": "pdf",
            ".txt": "text",
            ".text": "text",
            ".log": "text",
            ".csv": "csv_excel",
            ".xls": "csv_excel",
            ".xlsx": "csv_excel",
            ".xlsm": "csv_excel",
            ".docx": "docx",
            ".md": "markdown",
            ".markdown": "markdown",
            ".html": "web",
            ".htm": "web",
            ".xhtml": "web",
        }

    def detect_file_type(self, file_path: str) -> tuple[str, float]:
        """Detect file type using multiple methods.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (detected_type, confidence_score)
        """
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return ("unknown", 0.0)

        # Method 1: Content-based detection (most reliable)
        if self.magic and MAGIC_AVAILABLE:
            try:
                if (
                    mime_type := self.magic.from_file(file_path)
                ) and mime_type != "application/octet-stream":
                    if parser_type := self.mime_to_parser.get(mime_type):
                        return (parser_type, 0.9)
            except Exception as e:
                logger.debug(f"Magic detection failed: {e}")

        # Method 2: Content analysis (for files without clear MIME)
        if content_type := self._analyze_content(file_path):
            return (content_type, 0.8)

        # Method 3: Extension-based (fallback)
        if path.suffix:
            if parser_type := self.ext_to_parser.get(path.suffix.lower()):
                return (parser_type, 0.7)

        # Method 4: MIME type guess from extension
        if mime_type := mimetypes.guess_type(file_path)[0]:
            if parser_type := self.mime_to_parser.get(mime_type):
                return (parser_type, 0.6)

        # Default to text parser for unknown files
        return ("text", 0.3)

    def _analyze_content(self, file_path: str) -> Optional[str]:
        """Analyze file content to determine type.

        Args:
            file_path: Path to file

        Returns:
            Detected parser type or None
        """
        try:
            # Read first 4KB of file
            with open(file_path, "rb") as f:
                header = f.read(4096)

            # Check for PDF
            if header.startswith(b"%PDF"):
                return "pdf"

            # Check for Office documents (ZIP-based)
            if header.startswith(b"PK\x03\x04"):
                # Further analysis needed for Office docs
                if b"word/" in header:
                    return "docx"
                elif b"xl/" in header or b"workbook" in header:
                    return "csv_excel"
                elif b"ppt/" in header:
                    return None  # PowerPoint not supported yet

            # Check for HTML
            if b"<html" in header.lower() or b"<!doctype html" in header.lower():
                return "web"

            # Check for Markdown indicators
            if self._looks_like_markdown(header):
                return "markdown"

            # Check for CSV
            if self._looks_like_csv(header):
                return "csv_excel"

            # Check if it's likely text
            try:
                header.decode("utf-8")
                return "text"
            except UnicodeDecodeError:
                pass

        except Exception as e:
            logger.debug(f"Content analysis failed: {e}")

        return None

    def _looks_like_csv(self, content: bytes) -> bool:
        """Check if content looks like CSV.

        Args:
            content: File content bytes

        Returns:
            True if likely CSV
        """
        try:
            text = content.decode("utf-8", errors="ignore")
            lines = text.split("\n")[:10]  # Check first 10 lines

            if not lines:
                return False

            # Count delimiters
            comma_count = sum(line.count(",") for line in lines)
            tab_count = sum(line.count("\t") for line in lines)
            pipe_count = sum(line.count("|") for line in lines)

            # Check for consistent delimiter usage
            max_delim_count = max(comma_count, tab_count, pipe_count)
            if (
                max_delim_count > len(lines) * 2
            ):  # At least 2 delimiters per line average
                return True

        except Exception:
            pass

        return False

    def _looks_like_markdown(self, content: bytes) -> bool:
        """Check if content looks like Markdown.

        Args:
            content: File content bytes

        Returns:
            True if likely Markdown
        """
        try:
            text = content.decode("utf-8", errors="ignore")

            # Check for Markdown patterns
            markdown_patterns = [
                "# ",  # Headers
                "## ",
                "### ",
                "- ",  # Lists
                "* ",
                "1. ",
                "[",  # Links
                "![",  # Images
                "```",  # Code blocks
                "**",  # Bold
                "~~",  # Strikethrough
            ]

            pattern_count = sum(1 for pattern in markdown_patterns if pattern in text)
            if pattern_count >= 3:  # At least 3 different Markdown patterns
                return True

        except Exception:
            pass

        return False

    def select_parser(self, file_path: str) -> Optional[Any]:
        """Select appropriate parser for file.

        Args:
            file_path: Path to file

        Returns:
            Parser instance or None
        """
        parser_type, confidence = self.detect_file_type(file_path)

        logger.debug(
            f"Detected type '{parser_type}' with confidence {confidence} for {file_path}"
        )

        if self.parser_registry:
            return self.parser_registry.get_parser(parser_type)

        return None

    def route_files(self, file_paths: list[str]) -> dict[str, list[str]]:
        """Route multiple files to appropriate parsers.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping parser types to file lists
        """
        routing: dict[str, list[str]] = {}

        for file_path in file_paths:
            parser_type, _ = self.detect_file_type(file_path)

            if parser_type not in routing:
                routing[parser_type] = []
            routing[parser_type].append(file_path)

        return routing
