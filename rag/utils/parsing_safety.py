"""
Parsing safety utilities to prevent inappropriate parser fallbacks.

This module provides:
- Exception classes for parser failures
- Binary file detection to prevent text parser fallback
- File extension utilities
"""

from pathlib import Path

from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.utils.parsing_safety")


class ParsingError(Exception):
    """Base exception for parsing errors."""

    pass


class UnsupportedFileTypeError(ParsingError):
    """Raised when no appropriate parser is available for a file type."""

    def __init__(
        self,
        filename: str,
        extension: str,
        available_parsers: list[str] | None = None,
    ):
        self.filename = filename
        self.extension = extension
        self.available_parsers = available_parsers or []

        msg = f"No parser available for file '{filename}' (extension: {extension})"
        if available_parsers:
            msg += f". Available parsers: {', '.join(available_parsers)}"
        super().__init__(msg)


class ParserFailedError(ParsingError):
    """Raised when all available parsers failed to process a file."""

    def __init__(
        self,
        filename: str,
        tried_parsers: list[str],
        errors: list[str],
    ):
        self.filename = filename
        self.tried_parsers = tried_parsers
        self.errors = errors

        msg = f"All parsers failed for '{filename}'. Tried: {', '.join(tried_parsers)}"
        super().__init__(msg)


# Binary file extensions that should NEVER be parsed as text
# These files will produce garbage/useless chunks if processed with TextParser
BINARY_EXTENSIONS = {
    # Documents (binary formats)
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".tgz",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".ico",
    ".svg",  # SVG is text-based but often binary-encoded
    ".heic",
    ".heif",
    # Audio
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
    ".wma",
    # Video
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    # Executables and libraries
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".msi",
    ".dmg",
    ".app",
    # Email formats
    ".msg",
    ".eml",
    ".mbox",
    # Database files
    ".db",
    ".sqlite",
    ".sqlite3",
    ".mdb",
    ".accdb",
    # Other binary formats
    ".pyc",
    ".pyo",
    ".class",
    ".o",
    ".a",
    ".lib",
    ".obj",
    ".wasm",
}


def is_binary_extension(filename: str) -> bool:
    """
    Check if a file extension indicates binary content.

    Binary files should not be processed with text parsers as they
    produce garbage/useless chunks.

    Args:
        filename: Name of the file to check

    Returns:
        True if the file extension indicates binary content
    """
    ext = get_file_extension(filename)
    return ext in BINARY_EXTENSIONS


def get_file_extension(filename: str) -> str:
    """
    Get normalized (lowercase) file extension from filename.

    Args:
        filename: Name of the file

    Returns:
        Lowercase file extension including the dot (e.g., ".pdf")
    """
    return Path(filename).suffix.lower()


def validate_parser_for_file(
    filename: str,
    parser_type: str,
) -> tuple[bool, str | None]:
    """
    Validate that a parser is appropriate for a given file.

    Args:
        filename: Name of the file to check
        parser_type: Type of parser (e.g., "TextParser_Python")

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if this is a binary file being parsed with a text parser
    is_text_parser = "TextParser" in parser_type or "text" in parser_type.lower()
    if is_binary_extension(filename) and is_text_parser:
        return (
            False,
            f"Cannot use {parser_type} for binary file {filename}. "
            f"Configure an appropriate parser for {get_file_extension(filename)} files.",
        )

    return True, None

