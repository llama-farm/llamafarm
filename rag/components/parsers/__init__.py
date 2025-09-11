"""Parser components for RAG system - New modular architecture."""

from pathlib import Path
import importlib
import importlib.util
import logging

logger = logging.getLogger(__name__)

# Try to import base components
try:
    from .base import BaseParser, LlamaIndexParser, ParserRegistry, SmartRouter
except ImportError as e:
    logger.warning(f"Base parser components not available: {e}")
    BaseParser = None
    LlamaIndexParser = None
    ParserRegistry = None
    SmartRouter = None

# Try to import auto-generated registry
try:
    from .parser_registry import registry
except ImportError:
    logger.warning("Parser registry not available, using fallback")
    registry = None

# Import legacy parsers for backward compatibility
# These will be mapped to new parsers
legacy_parser_mapping = {
    "LlamaIndexTextParser": "text",
    "LlamaIndexPDFParser": "pdf",
    "LlamaIndexCSVExcelParser": "csv_excel",
    "LlamaIndexDocxParser": "docx",
    "LlamaIndexMarkdownParser": "markdown",
    "LlamaIndexWebParser": "web",
    "PlainTextParser": "text",
    "PDFParser": "pdf",
    "CSVParser": "csv_excel",
    "DocxParser": "docx",
    "MarkdownParser": "markdown",
    "HTMLParser": "web",
}

# Import the new parser factory
try:
    from .parser_factory import (
        ToolAwareParserFactory,
        ParserFactory as NewParserFactory,
    )
except ImportError:
    ToolAwareParserFactory = None
    NewParserFactory = None


# Create parser factory wrapper
class ParserFactory:
    """Factory for creating parser instances."""

    _parsers = {}

    @classmethod
    def register_parser(cls, name: str, parser_class):
        """Register a parser class."""
        cls._parsers[name] = parser_class

    @classmethod
    def create_parser(cls, name: str, config: dict = None):
        """Create a parser instance.

        Args:
            name: Parser name (can be legacy name or new name)
            config: Parser configuration

        Returns:
            Parser instance
        """
        # Try the new ToolAwareParserFactory first
        if ToolAwareParserFactory:
            try:
                return ToolAwareParserFactory.create_parser(
                    parser_name=name, config=config
                )
            except Exception as e:
                logger.debug(f"ToolAwareParserFactory failed for {name}: {e}")

        # Map legacy names to new names
        if name in legacy_parser_mapping:
            new_name = legacy_parser_mapping[name]
            logger.info(f"Mapping legacy parser {name} to {new_name}")
            name = new_name

        # Try to get from registry first
        if registry and hasattr(registry, "get_parser"):
            try:
                return registry.get_parser(name, config)
            except Exception as e:
                logger.warning(f"Failed to get parser from registry: {e}")

        # Fallback to registered parsers
        if name in cls._parsers:
            return cls._parsers[name](config=config)

        # Try to import directly from module
        try:
            module_path = Path(__file__).parent / name / "parser.py"
            if module_path.exists():
                spec = importlib.util.spec_from_file_location(
                    f"parser_{name}", module_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find the parser class in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and attr_name.endswith("Parser"):
                        return attr(config=config)
        except Exception as e:
            logger.warning(f"Failed to import parser {name}: {e}")

        # Try simple fallback parsers
        try:
            from .simple_text_parser import SimpleTextParser, SimplePDFParser

            if name in ["text", "markdown"]:
                return SimpleTextParser(name=name, config=config)
            elif name == "pdf":
                return SimplePDFParser(name=name, config=config)
            elif name == "csv_excel":
                # Use text parser for CSV as fallback
                return SimpleTextParser(name=name, config=config)
            elif name in ["docx", "web"]:
                # Use text parser as fallback
                return SimpleTextParser(name=name, config=config)
        except Exception as e:
            logger.warning(f"Failed to use simple parser fallback: {e}")

        raise ValueError(f"Parser '{name}' not found")


# Create compatibility aliases for legacy parsers
class LlamaIndexTextParser:
    """Compatibility wrapper for text parser."""

    def __init__(self, name: str = "text", config: dict = None):
        self._parser = ParserFactory.create_parser("text", config)

    def __getattr__(self, name):
        return getattr(self._parser, name)


class LlamaIndexPDFParser:
    """Compatibility wrapper for PDF parser."""

    def __init__(self, name: str = "pdf", config: dict = None):
        self._parser = ParserFactory.create_parser("pdf", config)

    def __getattr__(self, name):
        return getattr(self._parser, name)


class LlamaIndexCSVExcelParser:
    """Compatibility wrapper for CSV/Excel parser."""

    def __init__(self, name: str = "csv_excel", config: dict = None):
        self._parser = ParserFactory.create_parser("csv_excel", config)

    def __getattr__(self, name):
        return getattr(self._parser, name)


class LlamaIndexDocxParser:
    """Compatibility wrapper for DOCX parser."""

    def __init__(self, name: str = "docx", config: dict = None):
        self._parser = ParserFactory.create_parser("docx", config)

    def __getattr__(self, name):
        return getattr(self._parser, name)


class LlamaIndexMarkdownParser:
    """Compatibility wrapper for Markdown parser."""

    def __init__(self, name: str = "markdown", config: dict = None):
        self._parser = ParserFactory.create_parser("markdown", config)

    def __getattr__(self, name):
        return getattr(self._parser, name)


class LlamaIndexWebParser:
    """Compatibility wrapper for Web parser."""

    def __init__(self, name: str = "web", config: dict = None):
        self._parser = ParserFactory.create_parser("web", config)

    def __getattr__(self, name):
        return getattr(self._parser, name)


# Legacy parser aliases
PlainTextParser = LlamaIndexTextParser
PDFParser = LlamaIndexPDFParser
CSVParser = LlamaIndexCSVExcelParser
DocxParser = LlamaIndexDocxParser
MarkdownParser = LlamaIndexMarkdownParser
HTMLParser = LlamaIndexWebParser

# Directory parser implementation
from pathlib import Path
from typing import List, Optional, Dict, Any


class DirectoryParser:
    """Directory parser that uses individual parsers for files."""

    def __init__(
        self, name: str = "DirectoryParser", config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.config = config or {}
        self.recursive = self.config.get("recursive", True)
        self.include_patterns = self.config.get("include_patterns", ["*"])
        self.exclude_patterns = self.config.get("exclude_patterns", [])
        self.parser_map = self.config.get("parser_map", {})
        self.parser_configs = self.config.get("parser_configs", {})
        self.max_files = self.config.get("max_files", 1000)

    def validate_config(self) -> bool:
        """Validate the configuration."""
        return True  # Simple validation for now

    def parse(self, source: str, **kwargs):
        """Parse all files in a directory."""
        from core.base import ProcessingResult

        source_path = Path(source)

        if not source_path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"Directory not found: {source}", "source": source}],
            )

        if not source_path.is_dir():
            return ProcessingResult(
                documents=[],
                errors=[
                    {"error": f"Path is not a directory: {source}", "source": source}
                ],
            )

        all_documents = []
        all_errors = []
        files_processed = 0

        # Get file list
        if self.recursive:
            file_pattern = "**/*"
        else:
            file_pattern = "*"

        files = list(source_path.glob(file_pattern))

        # Filter files
        filtered_files = []
        for file_path in files:
            if not file_path.is_file():
                continue

            # Check include patterns
            included = False
            for pattern in self.include_patterns:
                if file_path.match(pattern):
                    included = True
                    break

            if not included:
                continue

            # Check exclude patterns
            excluded = False
            for pattern in self.exclude_patterns:
                if file_path.match(pattern):
                    excluded = True
                    break

            if excluded:
                continue

            filtered_files.append(file_path)

            if len(filtered_files) >= self.max_files:
                break

        # Process files
        for file_path in filtered_files:
            try:
                # Get parser for this file
                parser = self._get_parser_for_file(file_path)

                if parser:
                    result = parser.parse(str(file_path), **kwargs)
                    all_documents.extend(result.documents)
                    all_errors.extend(result.errors)
                    files_processed += 1
                else:
                    all_errors.append(
                        {"error": "No suitable parser found", "source": str(file_path)}
                    )

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
                "files_processed": files_processed,
                "files_found": len(filtered_files),
                "directory_processed": source,
                "parser_type": self.name,
            },
        )

    def _get_parser_for_file(self, file_path: Path):
        """Get the appropriate parser for a file."""
        extension = file_path.suffix.lower()

        # Check explicit parser mapping
        if extension in self.parser_map:
            parser_name = self.parser_map[extension]
            # Get the config for this specific parser
            parser_config = self.parser_configs.get(parser_name, {})

            # Try to create the parser
            try:
                return ParserFactory.create_parser(parser_name, parser_config)
            except Exception as e:
                logger.warning(f"Failed to create parser {parser_name}: {e}")

                # Fallback logic for legacy names
                for orig_name, mapped_name in legacy_parser_mapping.items():
                    if mapped_name == parser_name:
                        parser_config = self.parser_configs.get(orig_name, {})
                        if parser_config:
                            break
            try:
                return ParserFactory.create_parser(parser_name, parser_config)
            except Exception as e:
                logger.warning(f"Failed to create parser {parser_name}: {e}")

        # Try to determine parser by extension
        extension_to_parser = {
            ".pdf": "pdf",
            ".txt": "text",
            ".text": "text",
            ".md": "markdown",
            ".markdown": "markdown",
            ".csv": "csv_excel",
            ".xls": "csv_excel",
            ".xlsx": "csv_excel",
            ".docx": "docx",
            ".doc": "docx",
            ".html": "web",
            ".htm": "web",
        }

        if extension in extension_to_parser:
            parser_name = extension_to_parser[extension]
            parser_config = self.parser_configs.get(parser_name, {})
            try:
                return ParserFactory.create_parser(parser_name, parser_config)
            except Exception as e:
                logger.warning(f"Failed to create parser {parser_name}: {e}")

        # Default to text parser
        try:
            return ParserFactory.create_parser("text", {})
        except:
            return None


# Register parsers with factory
ParserFactory.register_parser("DirectoryParser", DirectoryParser)

# Register legacy names
for legacy_name in legacy_parser_mapping:
    ParserFactory.register_parser(
        legacy_name,
        lambda config, n=legacy_name: ParserFactory.create_parser(
            legacy_parser_mapping[n], config
        ),
    )

__all__ = [
    "ParserFactory",
    "BaseParser",
    "LlamaIndexParser",
    "SmartRouter",
    "ParserRegistry",
    "LlamaIndexTextParser",
    "LlamaIndexPDFParser",
    "LlamaIndexCSVExcelParser",
    "LlamaIndexDocxParser",
    "LlamaIndexMarkdownParser",
    "LlamaIndexWebParser",
    "PlainTextParser",
    "PDFParser",
    "CSVParser",
    "DocxParser",
    "MarkdownParser",
    "HTMLParser",
    "DirectoryParser",
]
