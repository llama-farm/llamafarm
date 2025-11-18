"""Enhanced Parser Factory with tool-specific parser selection."""

import importlib.util
from pathlib import Path
from typing import Any, Optional

import yaml

from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.parser_factory")


class ToolAwareParserFactory:
    """Factory for creating tool-specific parser instances."""

    # Cache for loaded parser configurations
    _parser_configs: dict[str, list[dict[str, Any]]] = {}
    # Cache for loaded parser classes
    _parser_classes: dict[str, type] = {}

    @classmethod
    def discover_parsers(cls) -> dict[str, list[dict[str, Any]]]:
        """Discover all available parsers from configuration files.

        Returns:
            Dictionary mapping parser types to their configurations
        """
        if cls._parser_configs:
            return cls._parser_configs

        parsers_dir = Path(__file__).parent
        parser_types: dict[str, list[dict[str, Any]]] = {}

        # Scan all subdirectories for config.yaml files
        for subdir in parsers_dir.iterdir():
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue

            config_file = subdir / "config.yaml"
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = yaml.safe_load(f)
                        if config and "parsers" in config:
                            parser_type = subdir.name
                            parser_types[parser_type] = config["parsers"]
                            logger.info(
                                f"Discovered {len(config['parsers'])} {parser_type} parsers"
                            )
                except Exception as e:
                    logger.error(f"Failed to load config from {config_file}: {e}")

        cls._parser_configs = parser_types
        return parser_types

    @classmethod
    def list_parsers(cls, parser_type: Optional[str] = None) -> list[str]:
        """List available parsers.

        Args:
            parser_type: Optional filter by parser type (pdf, csv, etc.)

        Returns:
            List of parser names
        """
        parsers = cls.discover_parsers()
        parser_names: list[str] = []

        for ptype, configs in parsers.items():
            if parser_type and ptype != parser_type:
                continue
            parser_names.extend(config["name"] for config in configs)

        return parser_names

    @classmethod
    def get_parser_info(cls, parser_name: str) -> Optional[dict[str, Any]]:
        """Get information about a specific parser.

        Args:
            parser_name: Name of the parser (e.g., "PDFParser_PyPDF2")

        Returns:
            Parser configuration dictionary or None
        """
        parsers = cls.discover_parsers()

        for parser_type, configs in parsers.items():
            for config in configs:
                if config["name"] == parser_name:
                    config["parser_type"] = parser_type
                    return config

        return None

    @classmethod
    def load_parser_class(cls, parser_name: str) -> Optional[type]:
        """Load a parser class dynamically.

        Args:
            parser_name: Name of the parser (e.g., "PDFParser_PyPDF2")

        Returns:
            Parser class or None
        """
        # Check cache
        if parser_name in cls._parser_classes:
            return cls._parser_classes[parser_name]

        # Get parser info
        info = cls.get_parser_info(parser_name)
        if not info:
            logger.error(f"Parser {parser_name} not found")
            return None

        # Check dependencies BEFORE trying to load
        deps = info.get("dependencies", {})
        required_deps = deps.get("required", [])
        missing_deps: list[str] = []

        for dep in required_deps:
            try:
                # Try to import the dependency
                __import__(dep.replace("-", "_"))
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            logger.warning(f"Parser {parser_name} missing dependencies: {missing_deps}")
            # Return None to trigger fallback
            return None

        parser_type = info["parser_type"]

        # Try to find the implementation file
        parsers_dir = Path(__file__).parent
        parser_dir = parsers_dir / parser_type

        # Common naming patterns for parser files
        possible_files = [
            f"{parser_name.lower()}.py",
            f"{info['tool'].lower()}_parser.py",
            f"{parser_name.split('_')[-1].lower()}_parser.py",
            "parser.py",
        ]

        for filename in possible_files:
            parser_file = parser_dir / filename
            if parser_file.exists():
                try:
                    # Load the module
                    spec = importlib.util.spec_from_file_location(
                        f"parsers.{parser_type}.{filename[:-3]}", parser_file
                    )
                    if not spec:
                        raise ValueError(f"Spec for parser {parser_type} not found")

                    module = importlib.util.module_from_spec(spec)
                    if not module:
                        raise ValueError(f"Module for parser {parser_type} not found")

                    if not spec.loader:
                        raise ValueError(f"Loader for parser {parser_type} not found")

                    spec.loader.exec_module(module)

                    # Find the parser class
                    for attr_name in dir(module):
                        if attr_name == parser_name:
                            parser_class = getattr(module, attr_name)
                            cls._parser_classes[parser_name] = parser_class
                            return parser_class

                    # If exact name not found, try to find any Parser class
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and attr_name.endswith("Parser")
                            and not attr_name.startswith("_")
                        ):
                            cls._parser_classes[parser_name] = attr
                            return attr

                except Exception as e:
                    logger.error(f"Failed to load parser from {parser_file}: {e}")

        # Fallback to dynamic discovery using blob_processor logic
        logger.debug(f"Trying dynamic discovery for parser {parser_name}")
        if parser_class := cls._dynamic_parser_discovery(parser_name):
            cls._parser_classes[parser_name] = parser_class
            return parser_class

        logger.error(f"Could not find implementation for parser {parser_name}")
        logger.warning(
            f"Parser {parser_name} will fall back to mock implementation - this may cause silent failures"
        )
        return None

    @classmethod
    def create_parser(
        cls,
        parser_name: Optional[str] = None,
        file_type: Optional[str] = None,
        tool: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Create a parser instance.

        Args:
            parser_name: Specific parser name (e.g., "PDFParser_PyPDF2")
            file_type: File type to parse (e.g., "pdf", "csv")
            tool: Preferred tool (e.g., "PyPDF2", "Pandas")
            config: Parser configuration

        Returns:
            Parser instance
        """
        # If specific parser name provided, use it
        if parser_name:
            parser_class = cls.load_parser_class(parser_name)
            if parser_class:
                return parser_class(name=parser_name, config=config)
            else:
                # Create mock parser as fallback
                logger.warning(f"Parser {parser_name} not found, creating mock parser")
                mock_class = cls.create_mock_parser(parser_name)
                return mock_class(name=parser_name, config=config)

        # If file_type and/or tool provided, find matching parser
        if file_type:
            parsers = cls.discover_parsers()

            if file_type not in parsers:
                raise ValueError(f"No parsers available for file type: {file_type}")

            available_parsers = parsers[file_type]

            # Filter by tool if specified
            if tool:
                matching = [p for p in available_parsers if p.get("tool") == tool]
                if matching:
                    available_parsers = matching
                else:
                    logger.warning(
                        f"No {file_type} parser found for tool {tool}, using default"
                    )

            # Try each available parser until one loads successfully
            for selected in available_parsers:
                parser_class = cls.load_parser_class(selected["name"])
                if parser_class:
                    # Merge default config with provided config
                    final_config = selected.get("default_config", {}).copy()
                    if config:
                        final_config.update(config)
                    logger.info(f"Using parser {selected['name']} for {file_type}")
                    return parser_class(name=selected["name"], config=final_config)
                else:
                    logger.warning(
                        f"Could not load parser {selected['name']}, trying next..."
                    )

        raise ValueError("Unable to create parser: specify parser_name or file_type")

    @classmethod
    def get_parser_for_file(
        cls,
        file_path: str | Path,
        preferred_tool: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Get the appropriate parser for a file.

        Args:
            file_path: Path to the file
            preferred_tool: Preferred parsing tool
            config: Parser configuration

        Returns:
            Parser instance
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        extension = file_path.suffix.lower()

        # Map extensions to parser types
        extension_map = {
            ".pdf": "pdf",
            ".txt": "text",
            ".csv": "csv",
            ".tsv": "csv",
            ".xlsx": "excel",
            ".xls": "excel",
            ".docx": "docx",
            ".md": "markdown",
            ".markdown": "markdown",
            ".html": "website",
            ".htm": "website",
            ".msg": "msg",
        }

        file_type = extension_map.get(extension)
        if not file_type:
            # Default to text parser for unknown types
            file_type = "text"
            logger.warning(f"Unknown file extension {extension}, using text parser")

        return cls.create_parser(
            file_type=file_type, tool=preferred_tool, config=config
        )

    @classmethod
    def check_dependencies(cls, parser_name: str) -> dict[str, bool]:
        """Check if a parser's dependencies are installed.

        Args:
            parser_name: Name of the parser

        Returns:
            Dictionary mapping dependency names to installation status
        """
        info = cls.get_parser_info(parser_name)
        if not info:
            return {}

        dependencies = info.get("dependencies", {})
        status = {}

        # Check required dependencies
        for dep in dependencies.get("required", []):
            try:
                __import__(dep.replace("-", "_"))
                status[dep] = True
            except ImportError:
                status[dep] = False

        # Check optional dependencies
        for dep in dependencies.get("optional", []):
            try:
                __import__(dep.replace("-", "_"))
                status[f"{dep} (optional)"] = True
            except ImportError:
                status[f"{dep} (optional)"] = False

        return status

    @classmethod
    def _dynamic_parser_discovery(cls, parser_type: str) -> Optional[type]:
        """
        Dynamically discover and load parser class by type name using blob_processor logic.

        This follows the convention:
        - Parser type: PDFParser_LlamaIndex
        - Module path: components.parsers.pdf.llamaindex_parser
        - Class name: PDFParser_LlamaIndex

        Args:
            parser_type: Name of the parser type (e.g., "PDFParser_LlamaIndex")

        Returns:
            Parser class or None
        """
        parser_category, implementation = cls._parse_parser_type_name(parser_type)
        potential_paths = cls._build_potential_module_paths(
            parser_category, implementation, parser_type
        )

        # Try to import from potential paths
        for module_path in potential_paths:
            if parser_class := cls._try_import_parser_from_module(
                module_path, parser_type
            ):
                return parser_class

        logger.warning(
            f"Dynamic parser discovery failed for {parser_type} - no matching implementation found"
        )
        return None

    @classmethod
    def _parse_parser_type_name(cls, parser_type: str) -> tuple[str, str]:
        """Parse parser type name into category and implementation."""
        if "_" in parser_type:
            parts = parser_type.split("_")
            # Handle both TypeParser_Implementation and Type_Parser_Implementation
            if "Parser" in parts[0]:
                # Format: PDFParser_LlamaIndex
                parser_category = parts[0].replace("Parser", "").lower()  # pdf
                implementation = "_".join(parts[1:]).lower()  # llamaindex
            else:
                # Format: PDF_Parser_LlamaIndex (less common)
                parser_category = parts[0].lower()  # pdf
                implementation = (
                    "_".join(parts[2:]).lower() if len(parts) > 2 else "default"
                )
        else:
            # No underscore, assume it's a simple parser name
            parser_category = parser_type.lower().replace("parser", "")
            implementation = "default"

        return parser_category, implementation

    @classmethod
    def _build_potential_module_paths(
        cls, parser_category: str, implementation: str, parser_type: str
    ) -> list[str]:
        """Build potential module paths to try for parser discovery."""
        from pathlib import Path

        # Build potential module paths to try
        potential_paths = [
            f"components.parsers.{parser_category}.{implementation}_parser",
            f"components.parsers.{parser_category}.{parser_category}_parser",
            f"components.parsers.{parser_type.lower()}",
            f"components.parsers.{parser_category}.parser",
        ]

        # Also check if there's a direct mapping based on file structure
        parsers_dir = Path(__file__).parent
        if parsers_dir.exists():
            # Look for matching directories
            for category_dir in parsers_dir.iterdir():
                if (
                    category_dir.is_dir()
                    and parser_category in category_dir.name.lower()
                ):
                    # Look for implementation files
                    for py_file in category_dir.glob("*_parser.py"):
                        if implementation in py_file.stem.lower():
                            module_name = (
                                f"components.parsers.{category_dir.name}.{py_file.stem}"
                            )
                            potential_paths.insert(0, module_name)

        return potential_paths

    @classmethod
    def _try_import_parser_from_module(
        cls, module_path: str, parser_type: str
    ) -> Optional[type]:
        """Try to import parser class from a specific module path."""
        import importlib

        try:
            logger.debug(f"Trying to import parser from: {module_path}")
            module = importlib.import_module(module_path)

            # Try to get the class with the exact name first
            if hasattr(module, parser_type):
                parser_class = getattr(module, parser_type)
                logger.debug(f"Successfully loaded {parser_type} from {module_path}")
                return parser_class

            # Try variations of the class name
            for attr_name in dir(module):
                if attr_name.lower() == parser_type.lower():
                    parser_class = getattr(module, attr_name)
                    logger.debug(f"Successfully loaded {attr_name} from {module_path}")
                    return parser_class

        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not load from {module_path}: {e}")

        return None

    @classmethod
    def create_mock_parser(cls, parser_type: str) -> type:
        """Create a mock parser for testing or fallback purposes."""
        logger.warning(
            f"Creating mock parser for {parser_type} - this indicates a missing parser implementation"
        )

        from components.parsers.base.base_parser import BaseParser, ParserConfig
        from core.base import Document, ProcessingResult

        class MockParser(BaseParser):
            def __init__(self, name: str | None = None, config: dict | None = None):
                self.name = name or parser_type
                self.config = config or {}

            def _load_metadata(self):
                return ParserConfig(
                    name=parser_type,
                    display_name=parser_type,
                    version="1.0",
                    supported_extensions=[],
                    mime_types=[],
                    capabilities=[],
                    dependencies={},
                    default_config={},
                )

            def parse(self, source):
                # Mock implementation
                return ProcessingResult(documents=[], errors=[])

            def can_parse(self, file_path):
                return True

            def parse_blob(self, blob_data, metadata):
                # Simple text extraction for testing
                try:
                    content = blob_data.decode("utf-8", errors="ignore")
                except Exception:
                    content = str(blob_data)[:1000]

                return [
                    Document(
                        content=content[:1000],  # Limit for testing
                        metadata={**metadata, "parser": parser_type},
                    )
                ]

        return MockParser


# Backward compatibility wrapper
class ParserFactory:
    """Backward compatible factory interface."""

    @classmethod
    def create_parser(cls, name: str, config: dict[str, Any] | None = None):
        """Create a parser instance (backward compatible)."""
        # Map legacy names
        legacy_mapping = {
            "text": ("text", "Python"),
            "pdf": ("pdf", None),
            "csv_excel": ("csv", None),
            "docx": ("docx", None),
            "markdown": ("markdown", "Python"),
            "web": ("website", None),
            "msg": ("msg", "extract-msg"),
        }

        if name in legacy_mapping:
            file_type, tool = legacy_mapping[name]
            return ToolAwareParserFactory.create_parser(
                file_type=file_type, tool=tool, config=config
            )

        # Try as parser name
        try:
            return ToolAwareParserFactory.create_parser(parser_name=name, config=config)
        except Exception as e:
            logger.warning(f"Failed to create parser {name}: {e}")
            # Try as file type
            return ToolAwareParserFactory.create_parser(file_type=name, config=config)
