"""Parser components for RAG system - Unified architecture."""

from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.__init__")

# Import the unified parser factory
from .parser_factory import ToolAwareParserFactory

# Import base components
try:
    from .base import BaseParser, LlamaIndexParser
except ImportError as e:
    logger.warning(f"Base parser components not available: {e}")
    BaseParser = None  # type: ignore
    LlamaIndexParser = None  # type: ignore


# Simplified parser factory that delegates to ToolAwareParserFactory
class ParserFactory:
    """Factory for creating parser instances - delegates to ToolAwareParserFactory."""

    @classmethod
    def create_parser(cls, name: str, config: dict | None = None):
        """Create a parser instance.

        Args:
            name: Parser name
            config: Parser configuration

        Returns:
            Parser instance
        """
        # Use the enhanced ToolAwareParserFactory
        return ToolAwareParserFactory.create_parser(parser_name=name, config=config)


# Legacy parser aliases - kept for backwards compatibility
# These now delegate to the unified factory system
def _create_legacy_parser_class(parser_name: str):
    """Create a legacy parser wrapper class."""

    class LegacyParserWrapper:
        def __init__(self, name: str | None = None, config: dict | None = None):
            self._parser = ToolAwareParserFactory.create_parser(
                parser_name=parser_name, config=config
            )

        def __getattr__(self, name):
            return getattr(self._parser, name)

    return LegacyParserWrapper


# Create legacy parser classes
LlamaIndexTextParser = _create_legacy_parser_class("TextParser_LlamaIndex")
LlamaIndexPDFParser = _create_legacy_parser_class("PDFParser_LlamaIndex")
LlamaIndexCSVExcelParser = _create_legacy_parser_class("CSVParser_Pandas")
LlamaIndexDocxParser = _create_legacy_parser_class("DocxParser_LlamaIndex")
LlamaIndexMarkdownParser = _create_legacy_parser_class("MarkdownParser_LlamaIndex")
LlamaIndexWebParser = _create_legacy_parser_class(
    "TextParser_LlamaIndex"
)  # Web parser fallback to text
LlamaIndexMsgParser = _create_legacy_parser_class("MsgParser_ExtractMsg")

# Additional legacy aliases
PlainTextParser = LlamaIndexTextParser
PDFParser = LlamaIndexPDFParser
CSVParser = LlamaIndexCSVExcelParser
DocxParser = LlamaIndexDocxParser
MarkdownParser = LlamaIndexMarkdownParser
HTMLParser = LlamaIndexWebParser
MsgParser = LlamaIndexMsgParser

# Directory parser - simplified to use ToolAwareParserFactory
from pathlib import Path
from typing import Any, Optional


class DirectoryParser:
    """Directory parser that uses individual parsers for files."""

    def __init__(
        self, name: str = "DirectoryParser", config: Optional[dict[str, Any]] = None
    ):
        self.name = name
        self.config = config or {}

        # Configuration options for controlling directory scanning
        self.include_patterns = self.config.get("include_patterns", [])
        self.exclude_patterns = self.config.get("exclude_patterns", [])
        self.parser_map = self.config.get("parser_map", {})
        self.parser_configs = self.config.get("parser_configs", {})
        self.max_files = self.config.get("max_files", 1000)

    def validate_config(self) -> bool:
        """Validate the configuration."""
        if self.max_files and self.max_files <= 0:
            logger.warning("max_files must be positive, using default of 1000")
            self.max_files = 1000
        return True

    def parse(self, source: str, **kwargs):
        """Parse all files in a directory."""
        from core.base import ProcessingResult
        import fnmatch

        source_path = Path(source)
        if not source_path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"Path not found: {source}", "source": source}],
            )

        all_documents = []
        all_errors = []

        # Use ToolAwareParserFactory to get appropriate parsers for files
        if source_path.is_file():
            files = [source_path]
        else:
            files = list(source_path.rglob("*")) if source_path.is_dir() else []

        # Filter files based on patterns and limits
        filtered_files = []
        files_processed = 0

        for file_path in files:
            if not file_path.is_file():
                continue

            # Check max_files limit
            if self.max_files and files_processed >= self.max_files:
                logger.warning(
                    f"Reached max_files limit of {self.max_files}, stopping scan"
                )
                break

            # Apply include patterns (if specified)
            if self.include_patterns:
                if not any(
                    fnmatch.fnmatch(file_path.name, pattern)
                    for pattern in self.include_patterns
                ):
                    continue

            # Apply exclude patterns
            if self.exclude_patterns:
                if any(
                    fnmatch.fnmatch(file_path.name, pattern)
                    for pattern in self.exclude_patterns
                ):
                    continue

            filtered_files.append(file_path)
            files_processed += 1

        # Process filtered files
        for file_path in filtered_files:
            try:
                parser = ToolAwareParserFactory.get_parser_for_file(
                    file_path, config=self.config
                )
                result = parser.parse(str(file_path), **kwargs)
                all_documents.extend(result.documents)
                all_errors.extend(result.errors)
            except Exception as e:
                all_errors.append(
                    {
                        "error": f"Failed to parse file: {str(e)}",
                        "source": str(file_path),
                    }
                )

        return ProcessingResult(
            documents=all_documents,
            errors=all_errors,
            metrics={
                "total_documents": len(all_documents),
                "total_errors": len(all_errors),
                "files_processed": len(filtered_files),
                "files_scanned": files_processed,
                "parser_type": self.name,
            },
        )


__all__ = [
    # Main factory
    "ToolAwareParserFactory",
    "ParserFactory",  # Simplified wrapper
    # Base components
    "BaseParser",
    "LlamaIndexParser",
    # Directory parser
    "DirectoryParser",
    # Legacy compatibility classes (for backwards compatibility)
    "LlamaIndexTextParser",
    "LlamaIndexPDFParser",
    "LlamaIndexCSVExcelParser",
    "LlamaIndexDocxParser",
    "LlamaIndexMarkdownParser",
    "LlamaIndexWebParser",
    "LlamaIndexMsgParser",
    "PlainTextParser",
    "PDFParser",
    "CSVParser",
    "DocxParser",
    "MarkdownParser",
    "HTMLParser",
    "MsgParser",
]
