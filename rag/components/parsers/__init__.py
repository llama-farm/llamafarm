"""Parser components for RAG system - New modular architecture."""

from pathlib import Path
import importlib
import importlib.util

from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.__init__")

# Check for libmagic availability
try:
    from utils.libmagic_helper import check_libmagic

    check_libmagic()  # This will show warnings if libmagic is missing
except ImportError:
    pass  # Helper not available, continue anyway

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
    def create_parser(cls, name: str, config: dict | None = None):
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
            except ImportError as e:
                if "llama-index" in str(e).lower() or "llama_index" in str(e).lower():
                    logger.warning(
                        "LlamaIndex not installed. Install with: uv pip install llama-index"
                    )
                elif "magic" in str(e).lower():
                    logger.warning(
                        "python-magic not installed. Install with: uv pip install python-magic"
                    )
                else:
                    logger.debug(
                        f"Import error in ToolAwareParserFactory for {name}: {e}"
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
                parser_info = registry.get_parser(name)
                if parser_info:
                    # Create parser instance from registry info
                    parser_class = parser_info.get("class")
                    if parser_class:
                        # Import and instantiate the parser
                        module_name = parser_info.get(
                            "module", f"components.parsers.{name}"
                        )
                        try:
                            import importlib

                            module = importlib.import_module(module_name)
                            cls = getattr(module, parser_class)
                            return cls(config=config)
                        except Exception as e:
                            logger.warning(f"Failed to import parser {name}: {e}")
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

        # Try LlamaIndex parsers as fallback
        try:
            if name in ["text", "markdown"]:
                from .text.llamaindex_parser import TextParser_LlamaIndex

                logger.info(f"Using LlamaIndex text parser as fallback for {name}")
                return TextParser_LlamaIndex(name=name, config=config)
            elif name == "pdf":
                from .pdf.llamaindex_parser import PDFParser_LlamaIndex

                logger.info("Using LlamaIndex PDF parser as fallback")
                return PDFParser_LlamaIndex(name=name, config=config)
            elif name == "csv_excel":
                # Use pandas CSV parser as fallback
                try:
                    from .csv.pandas_parser import CSVParser_Pandas

                    logger.info("Using Pandas CSV parser as fallback")
                    return CSVParser_Pandas(name=name, config=config)
                except ImportError:
                    # If pandas not available, use Python CSV parser
                    from .csv.python_parser import CSVParser_Python

                    logger.info("Using Python CSV parser as fallback")
                    return CSVParser_Python(name=name, config=config)
            elif name in ["docx", "web"]:
                # Use text parser as fallback
                from .text.llamaindex_parser import TextParser_LlamaIndex

                logger.info(f"Using LlamaIndex text parser as fallback for {name}")
                return TextParser_LlamaIndex(name=name, config=config)
        except Exception as e:
            logger.warning(f"Failed to use LlamaIndex parser fallback: {e}")

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
                errors=[{"error": f"Path not found: {source}", "source": source}],
            )

        all_documents = []
        all_errors = []
        files_processed = 0

        # Handle single file or directory
        if source_path.is_file():
            # Single file - process it directly
            files = [source_path]
        elif source_path.is_dir():
            # Directory - get file list
            if self.recursive:
                file_pattern = "**/*"
            else:
                file_pattern = "*"

            files = list(source_path.glob(file_pattern))
        else:
            return ProcessingResult(
                documents=[],
                errors=[
                    {
                        "error": f"Path is neither file nor directory: {source}",
                        "source": source,
                    }
                ],
            )

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

        # NEW: If we have a parsers list, use it directly
        if hasattr(self.config, "get") and "parsers" in self.config:
            parsers_list = self.config.get("parsers", [])

            # Try each parser in the list to see if it can handle this file
            for parser_info in parsers_list:
                parser_type = parser_info.get("type")
                parser_config = parser_info.get("config", {})

                # Check if this parser handles this extension
                # Look for file_extensions in parser config
                file_extensions = parser_info.get("file_extensions", [])
                if extension in file_extensions:
                    try:
                        return ParserFactory.create_parser(parser_type, parser_config)
                    except Exception as e:
                        logger.warning(f"Failed to create parser {parser_type}: {e}")
                        continue

            # If no specific match, use the first parser as default for text files
            if parsers_list and extension in [".txt", ".text", ".log"]:
                parser_info = parsers_list[0]
                parser_type = parser_info.get("type")
                parser_config = parser_info.get("config", {})
                try:
                    return ParserFactory.create_parser(parser_type, parser_config)
                except Exception as e:
                    logger.warning(
                        f"Failed to create default parser {parser_type}: {e}"
                    )

        # OLD: Check explicit parser mapping
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

        # Try to determine parser by extension using parser_configs keys
        if self.parser_configs:
            # Look for a parser that matches the extension
            for parser_name, parser_config in self.parser_configs.items():
                # Check if parser name contains the extension type
                if extension == ".txt" and "Text" in parser_name:
                    try:
                        return ParserFactory.create_parser(parser_name, parser_config)
                    except Exception as e:
                        logger.warning(f"Failed to create parser {parser_name}: {e}")
                elif extension == ".pdf" and "PDF" in parser_name:
                    try:
                        return ParserFactory.create_parser(parser_name, parser_config)
                    except Exception as e:
                        logger.warning(f"Failed to create parser {parser_name}: {e}")
                elif extension in [".csv", ".tsv"] and "CSV" in parser_name:
                    try:
                        return ParserFactory.create_parser(parser_name, parser_config)
                    except Exception as e:
                        logger.warning(f"Failed to create parser {parser_name}: {e}")

        # Default to LlamaIndex text parser as fallback
        try:
            from .text.llamaindex_parser import TextParser_LlamaIndex

            logger.info("Using LlamaIndex text parser as default fallback")
            return TextParser_LlamaIndex(name="text", config={})
        except Exception as e:
            logger.warning(f"Failed to create LlamaIndex text parser: {e}")
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
