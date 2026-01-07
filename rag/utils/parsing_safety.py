"""
Parsing safety utilities to prevent inappropriate parser fallbacks.

This module provides exception classes to ensure files are only processed
with explicitly configured parsers. No fallback logic - if no parser is
configured for a file type, it's a configuration error.
"""

from pathlib import Path


class ParsingError(Exception):
    """Base exception for parsing errors."""

    pass


class UnsupportedFileTypeError(ParsingError):
    """Raised when no parser is configured for a file type."""

    def __init__(
        self,
        filename: str,
        extension: str,
        available_parsers: list[str] | None = None,
    ):
        self.filename = filename
        self.extension = extension
        self.available_parsers = available_parsers or []

        msg = f"No parser configured for file '{filename}' (extension: {extension})"
        if available_parsers:
            msg += f". Available parsers: {', '.join(available_parsers)}"
        else:
            msg += ". No parsers are currently configured."
        msg += " Configure an appropriate parser in your data_processing_strategy."
        super().__init__(msg)


class ParserFailedError(ParsingError):
    """Raised when all configured parsers failed to process a file."""

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
        if errors:
            msg += f". Errors: {'; '.join(errors[:3])}"  # Show first 3 errors
            if len(errors) > 3:
                msg += f" (and {len(errors) - 3} more)"
        super().__init__(msg)


def get_file_extension(filename: str) -> str:
    """
    Get normalized file extension from a filename.

    Args:
        filename: Name of the file (can be full path or just filename)

    Returns:
        Lowercase file extension including the dot (e.g., '.pdf')
        Returns empty string if no extension found
    """
    return Path(filename).suffix.lower()
