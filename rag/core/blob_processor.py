"""
Blob-based document processor for LlamaFarm integration.
Handles iterative parser selection based on file patterns.
"""

import fnmatch
import sys
from pathlib import Path
from typing import Any, TypedDict

from components.extractors.base import BaseExtractor
from components.parsers.base.base_parser import BaseParser
from core.base import Document
from core.logging import RAGStructLogger

repo_root = Path(__file__).parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from config.datamodel import (
        DataProcessingStrategy,
        Extractor,
        Parser,
    )
except ImportError as e:
    raise ImportError(
        f"Could not import config module. Make sure you're running from the repo root. Error: {e}"
    ) from e

logger = RAGStructLogger("rag.core.blob_processor")


class ExtractorOutput(TypedDict):
    name: str
    count: int
    new_fields: list[str]


class BlobProcessor:
    """
    Central processor for handling blob data with pattern-based parser/extractor routing.
    Implements centralized pattern matching using fnmatch for glob-style patterns.
    """

    def __init__(self, strategy_config: DataProcessingStrategy):
        """
        Initialize the blob processor with a strategy configuration.

        Args:
            strategy_config: Dictionary containing preprocessors, parsers and extractors config
        """
        self.strategy_config = strategy_config
        self.preprocessors = self._initialize_preprocessors(
            strategy_config.preprocessors or []
        )
        self.parsers = self._initialize_parsers(strategy_config.parsers or [])
        self.extractors = self._initialize_extractors(strategy_config.extractors or [])

    def _initialize_preprocessors(
        self, preprocessor_configs: list[Any]
    ) -> list[tuple[Any, Any]]:
        """
        Initialize preprocessors from configuration and sort by priority.

        Args:
            preprocessor_configs: List of preprocessor configurations

        Returns:
            List of tuples containing (config, preprocessor_instance) sorted by priority
        """
        preprocessors: list[tuple[Any, Any]] = []
        for config in preprocessor_configs:
            if not config.type:
                continue

            preprocessor_type = config.type
            try:
                preprocessor_class = self._get_preprocessor_class(preprocessor_type)
                # Pass the config
                preprocessor_instance = preprocessor_class(config=config.config or {})
                preprocessors.append((config, preprocessor_instance))
            except Exception as e:
                logger.warning(
                    f"Failed to initialize preprocessor {preprocessor_type}: {e}"
                )

        # Sort by priority (lower numbers are higher priority)
        preprocessors.sort(key=lambda x: x[0].priority or 0)
        return preprocessors

    def _initialize_parsers(
        self, parser_configs: list[Parser]
    ) -> list[tuple[Parser, BaseParser]]:
        """
        Initialize parsers from configuration and sort by priority.

        Args:
            parser_configs: List of parser configurations

        Returns:
            List of tuples containing (config, parser_instance) sorted by priority
        """
        parsers: list[tuple[Parser, BaseParser]] = []
        for config in parser_configs:
            if not config.type:
                continue

            parser_type = config.type
            try:
                parser_class = self._get_parser_class(parser_type)
                # Pass the parser type name and config
                parser_instance = parser_class(
                    name=parser_type, config=config.config or {}
                )
                parsers.append((config, parser_instance))
            except Exception as e:
                logger.warning(f"Failed to initialize parser {parser_type}: {e}")

        # Sort by priority (lower numbers are higher priority)
        parsers.sort(key=lambda x: x[0].priority or 0)
        return parsers

    def _initialize_extractors(
        self, extractor_configs: list[Extractor]
    ) -> list[tuple[Extractor, BaseExtractor]]:
        """
        Initialize extractors from configuration and sort by priority.

        Args:
            extractor_configs: List of extractor configurations

        Returns:
            List of tuples containing (config, extractor_instance) sorted by priority
        """
        extractors = []
        for config in extractor_configs:
            try:
                extractor_type = config.type
                extractor_config = config.config or {}

                extractor_class = self._get_extractor_class(extractor_type)
                extractor_instance = extractor_class(extractor_config)
                extractors.append((config, extractor_instance))
            except Exception as e:
                logger.warning(f"Failed to initialize extractor {config.type}: {e}")

        # Sort by priority (lower numbers are higher priority)
        extractors.sort(key=lambda x: x[0].priority or 0)
        return extractors

    def _get_parser_class(self, parser_type: str) -> type:
        """
        Get parser class using the enhanced ToolAwareParserFactory.

        Args:
            parser_type: Name of the parser type (e.g., "PDFParser_LlamaIndex")

        Returns:
            Parser class

        Raises:
            ImportError: If parser requires missing dependencies
            ValueError: If parser not found in registry
        """
        from components.parsers.parser_factory import ToolAwareParserFactory

        # Use the enhanced factory to load the parser class
        if parser_class := ToolAwareParserFactory.load_parser_class(parser_type):
            return parser_class

        # Parser not found or dependencies missing - raise clear error
        parser_info = ToolAwareParserFactory.get_parser_info(parser_type)
        if parser_info:
            deps = parser_info.get("dependencies", {})
            required_deps = deps.get("required", [])
            if required_deps:
                raise ImportError(
                    f"Parser '{parser_type}' requires missing dependencies: {required_deps}\n"
                    f"Install with: uv pip install {' '.join(required_deps)}"
                )

        raise ValueError(
            f"Parser '{parser_type}' not found. "
            f"Available parsers: {list(ToolAwareParserFactory.list_parsers())}"
        )

    def _get_preprocessor_class(self, preprocessor_type: str) -> type:
        """
        Get preprocessor class using the PreprocessorFactory.

        Args:
            preprocessor_type: Name of the preprocessor type (e.g., "PaddleOCRPreprocessor")

        Returns:
            Preprocessor class

        Raises:
            ImportError: If preprocessor requires missing dependencies
            ValueError: If preprocessor not found in registry
        """
        from components.preprocessors.factory import PreprocessorFactory

        # Use the factory to load the preprocessor class
        if preprocessor_class := PreprocessorFactory.load_preprocessor_class(
            preprocessor_type
        ):
            return preprocessor_class

        # Preprocessor not found or dependencies missing - raise clear error
        preprocessor_info = PreprocessorFactory.get_preprocessor_info(preprocessor_type)
        if preprocessor_info:
            deps = preprocessor_info.get("dependencies", {})
            required_deps = deps.get("required", [])
            if required_deps:
                raise ImportError(
                    f"Preprocessor '{preprocessor_type}' requires missing dependencies: {required_deps}\n"
                    f"Install with: uv pip install {' '.join(required_deps)}"
                )

        raise ValueError(
            f"Preprocessor '{preprocessor_type}' not found. "
            f"Available preprocessors: {list(PreprocessorFactory.list_preprocessors())}"
        )

    def _get_extractor_class(self, extractor_type: str) -> type:
        """
        Dynamically discover and load extractor class by type name.

        Args:
            extractor_type: Name of the extractor type

        Returns:
            Extractor class
        """
        import importlib

        # Handle different naming conventions
        # ContentStatisticsExtractor -> statistics_extractor
        # EntityExtractor -> entity_extractor
        # KeywordExtractor -> keyword_extractor
        # Convert CamelCase to snake_case for directory name
        import re

        snake_name = re.sub("([A-Z]+)", r"_\1", extractor_type).lower().strip("_")
        snake_name = snake_name.replace("__", "_")  # Fix double underscores

        # Build potential module paths to try
        potential_paths = [
            # Try subdirectory first (most extractors are in subdirs)
            f"components.extractors.{snake_name}.{snake_name}",
            f"components.extractors.{snake_name}",
            # Try without 'extractor' suffix
            f"components.extractors.{snake_name.replace('_extractor', '')}.{snake_name.replace('_extractor', '')}_extractor",
            # Try in base module
            "components.extractors.base",
        ]

        # Special cases for known extractors
        if extractor_type == "ContentStatisticsExtractor":
            potential_paths.insert(
                0, "components.extractors.statistics_extractor.statistics_extractor"
            )
        elif extractor_type == "EntityExtractor":
            potential_paths.insert(
                0, "components.extractors.entity_extractor.entity_extractor"
            )
        elif extractor_type == "KeywordExtractor":
            potential_paths.insert(
                0, "components.extractors.keyword_extractor.keyword_extractor"
            )

        # Try to import from potential paths
        extractor_load_errors = []
        for module_path in potential_paths:
            try:
                logger.debug(f"Trying to import extractor from: {module_path}")
                module = importlib.import_module(module_path)

                # Try to get the class with the exact name first
                if hasattr(module, extractor_type):
                    extractor_class = getattr(module, extractor_type)
                    logger.debug(
                        f"Successfully loaded {extractor_type} from {module_path}"
                    )
                    return extractor_class

                # Try variations of the class name
                for attr_name in dir(module):
                    if attr_name.lower() == extractor_type.lower():
                        extractor_class = getattr(module, attr_name)
                        logger.debug(
                            f"Successfully loaded {attr_name} from {module_path}"
                        )
                        return extractor_class

            except (ImportError, AttributeError) as e:
                extractor_load_errors.append(f"Could not load from {module_path}: {e}")
                logger.debug(f"Could not load parser from {module_path}: {e}")
                continue

        # If we couldn't find the extractor, log a warning and return mock
        error_details = "; ".join(extractor_load_errors)
        logger.warning(
            f"Could not dynamically load extractor {extractor_type}. Falling back to mock extractor. Errors: {error_details}"
        )

        class MockExtractor(BaseExtractor):
            def __init__(self, config):
                super().__init__(name=extractor_type, config=config)

            def extract(self, documents):
                # Add mock metadata
                for doc in documents:
                    doc.metadata[f"extractor_{extractor_type}"] = True
                return documents

            def get_dependencies(self):
                return []

        return MockExtractor

    def _matches_patterns(self, filename: str, patterns: list[str]) -> bool:
        """
        Check if filename matches any of the glob patterns.

        Args:
            filename: Name of the file
            patterns: List of glob patterns to match against

        Returns:
            True if filename matches any pattern, False otherwise
        """
        for pattern in patterns:
            if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                return True
        return False

    def _is_excluded(self, filename: str, exclude_patterns: list[str]) -> bool:
        """
        Check if filename matches any of the exclusion patterns.

        Args:
            filename: Name of the file
            exclude_patterns: List of glob patterns to exclude

        Returns:
            True if filename should be excluded, False otherwise
        """
        return (
            self._matches_patterns(filename, exclude_patterns)
            if exclude_patterns
            else False
        )

    def process_blob(
        self, blob_data: bytes, metadata: dict[str, Any]
    ) -> list[Document]:
        """
        Process a blob of data with optional preprocessing, then parser selection.

        Pipeline: Preprocessors → Parsers → Extractors

        Args:
            blob_data: Raw bytes of the document
            metadata: Metadata including filename, content_type, etc.

        Returns:
            List of processed Document objects
        """
        import tempfile

        filename = metadata.get("filename", "unknown")
        logger.info(f"Processing blob: {filename}")
        logger.debug(f"Blob metadata: {metadata}")
        logger.debug(
            f"First 20 bytes of blob: {blob_data[:20].decode(errors='replace') if blob_data else 'empty'}"
        )

        # Phase 1: Preprocessing (OCR, format conversion, etc.)
        preprocessed = False
        for config, preprocessor in self.preprocessors:
            if not config.type:
                continue

            # Save blob to temp file for preprocessing
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(filename).suffix
            ) as tmp:
                tmp.write(blob_data)
                tmp.flush()
                tmp_path = tmp.name

            try:
                # Check if preprocessor can handle this file
                if preprocessor.can_process(tmp_path, metadata):
                    preprocessor_type = config.type
                    logger.info(f"Preprocessing {filename} with {preprocessor_type}")

                    # Run preprocessor
                    result = preprocessor.preprocess(tmp_path, metadata)

                    if result.success:
                        # Update blob_data with preprocessed content
                        blob_data = result.content.encode("utf-8")
                        metadata.update(result.metadata)
                        metadata["preprocessed"] = True
                        metadata["preprocessor"] = preprocessor_type

                        logger.info(
                            f"Preprocessing successful: {len(result.content)} chars extracted"
                        )
                        preprocessed = True

                        # If preprocessor created an output file (e.g., searchable PDF),
                        # update the filename to point to it
                        if result.output_file:
                            filename = result.output_file
                            metadata["output_file"] = result.output_file

                        break  # Only run first matching preprocessor
                    else:
                        logger.warning(
                            f"Preprocessing failed with {preprocessor_type}: {result.errors}"
                        )
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

        if preprocessed:
            logger.debug(f"Blob preprocessed, continuing to parser phase")

        # Phase 2: Parsing (chunking)
        # Find matching parsers based on file patterns
        matching_parsers = self._find_matching_parsers(filename)
        logger.debug(
            f"Found {len(matching_parsers)} matching parsers for {filename}: {[p[0].type or None for p in matching_parsers]}"
        )

        if not matching_parsers:
            logger.warning(f"No parser found for file: {filename}")
            # Try with the lowest priority text parser as ultimate fallback
            for config, parser in self.parsers:
                if config.type and config.type == "TextParser_Python":
                    matching_parsers = [(config, parser)]
                    break

        # Try parsers in priority order until one succeeds
        documents = []
        for config, parser in matching_parsers:
            if not config.type:
                logger.warning(
                    f"Parser config missing 'type': {config}. This may indicate a misconfiguration."
                )
                continue

            parser_type = config.type
            try:
                logger.debug(
                    f"Attempting to parse {filename} with {parser_type} (priority: {config.priority})"
                )
                documents = parser.parse_blob(blob_data, metadata)
                logger.debug(
                    f"{parser_type} returned {len(documents) if documents else 0} documents"
                )

                if documents:
                    # Calculate chunk statistics
                    chunk_sizes = [len(doc.content) for doc in documents]
                    avg_chunk_size = (
                        sum(chunk_sizes) // len(chunk_sizes) if chunk_sizes else 0
                    )

                    logger.info(
                        f"Successfully parsed {filename} with {parser_type} - got {len(documents)} chunks"
                    )
                    # Use debug level for detailed parser output
                    logger.debug(f"\n📄 Parser Output: {parser_type}")
                    logger.debug(f"   ├─ Chunks created: {len(documents)}")
                    logger.debug(f"   ├─ Average chunk size: {avg_chunk_size} chars")
                    logger.debug(
                        f"   └─ Chunk sizes: min={min(chunk_sizes)}, max={max(chunk_sizes)}"
                    )

                    # Apply extractors to the documents
                    documents = self._apply_extractors(documents, filename)
                    break

            except Exception as e:
                logger.warning(f"{parser_type} FAILED for {filename}: {e}")
                import traceback

                logger.warning(f"Traceback: {traceback.format_exc()}")
                continue

        if not documents:
            logger.error(f"All parsers failed for file: {filename}")
            # Create a basic document with raw text as fallback
            documents = [
                Document(
                    content=blob_data.decode("utf-8", errors="ignore"),
                    metadata={**metadata, "parser": "fallback_raw"},
                )
            ]

        return documents

    def _find_matching_parsers(self, filename: str) -> list[tuple[Parser, BaseParser]]:
        """
        Find all parsers that match the given filename based on patterns.

        Args:
            filename: Name of the file to match

        Returns:
            List of matching (config, parser) tuples sorted by priority
        """
        matching: list[tuple[Parser, BaseParser]] = []

        for config, parser in self.parsers:
            include_patterns = config.file_include_patterns or []

            # Check if file matches include patterns and not exclude patterns
            if include_patterns:
                if self._matches_patterns(filename, include_patterns):
                    matching.append((config, parser))
            # If no include patterns specified, parser accepts all files (unless excluded)
            else:
                matching.append((config, parser))

        return matching

    def _apply_extractors(
        self, documents: list[Document], filename: str
    ) -> list[Document]:
        """
        Apply matching extractors to the documents based on file patterns.

        Args:
            documents: List of documents to process
            filename: Name of the file being processed

        Returns:
            List of documents with extracted metadata
        """
        # Find matching extractors
        matching_extractors = self._find_matching_extractors(filename)

        # Apply each matching extractor
        extractor_outputs: list[ExtractorOutput] = []
        for config, extractor in matching_extractors:
            try:
                extractor_type = config.type
                logger.debug(f"Applying extractor {extractor_type} to {filename}")

                # Count metadata before extraction
                before_keys: set = set()
                for doc in documents:
                    before_keys.update(doc.metadata.keys())

                # Extractors work on the list of documents
                documents = extractor.extract(documents)

                # Count metadata after extraction
                after_keys: set = set()
                for doc in documents:
                    after_keys.update(doc.metadata.keys())
                    # Mark that this extractor was applied
                    doc.metadata[f"extractor_{extractor_type}"] = True

                # Find what was extracted
                new_keys = after_keys - before_keys - {f"extractor_{extractor_type}"}

                # Count extracted items for specific extractors
                extraction_count = 0
                extractor_type_lower = extractor_type.lower() if extractor_type else ""

                if "keyword" in extractor_type_lower:
                    for doc in documents:
                        if "keywords" in doc.metadata:
                            extraction_count += len(doc.metadata.get("keywords", []))
                elif "entity" in extractor_type_lower:
                    for doc in documents:
                        if "entities" in doc.metadata:
                            extraction_count += len(doc.metadata.get("entities", []))
                elif "link" in extractor_type_lower:
                    for doc in documents:
                        if "links" in doc.metadata:
                            extraction_count += len(doc.metadata.get("links", []))
                elif "heading" in extractor_type_lower:
                    for doc in documents:
                        if "headings" in doc.metadata:
                            extraction_count += len(doc.metadata.get("headings", []))
                elif "table" in extractor_type_lower:
                    for doc in documents:
                        if "tables" in doc.metadata:
                            extraction_count += len(doc.metadata.get("tables", []))

                if extraction_count > 0 or new_keys:
                    extractor_outputs.append(
                        {
                            "name": extractor_type,
                            "count": extraction_count,
                            "new_fields": list(new_keys),
                        }
                    )

            except Exception as e:
                logger.warning(f"Extractor {extractor_type} failed for {filename}: {e}")
                continue

        # Log extractor outputs at debug level
        if extractor_outputs and len(extractor_outputs) > 0:
            logger.debug("\n🔍 Extractors Applied:")
            for output in extractor_outputs:
                output_count = output["count"] or 0
                output_fields = output.get("new_fields", [])
                if output_count > 0:
                    logger.debug(
                        f"   ├─ {output['name']}: extracted {output_count} items"
                    )
                elif output_fields:
                    logger.debug(
                        f"   ├─ {output['name']}: added fields {output_fields}"
                    )
                else:
                    logger.debug(f"   ├─ {output['name']}: applied")

        return documents

    def _find_matching_extractors(
        self, filename: str
    ) -> list[tuple[Extractor, BaseExtractor]]:
        """
        Find all extractors that match the given filename based on patterns.

        Args:
            filename: Name of the file to match

        Returns:
            List of matching (config, extractor) tuples sorted by priority
        """
        matching: list[tuple[Extractor, BaseExtractor]] = []

        for config, extractor in self.extractors:
            include_patterns = config.file_include_patterns or []

            # Check if file matches include patterns and not exclude patterns
            if include_patterns:
                if self._matches_patterns(filename, include_patterns):
                    matching.append((config, extractor))
            # If no include patterns specified, extractor applies to all files
            else:
                matching.append((config, extractor))

        return matching

    def get_supported_extensions(self) -> list[str]:
        """
        Get list of all supported file extensions from all parsers.

        Returns:
            List of supported extensions
        """
        extensions = set()
        for config, _ in self.parsers:
            patterns = config.file_include_patterns or []
            for pattern in patterns:
                # Extract extensions from patterns like "*.pdf"
                if pattern.startswith("*."):
                    extensions.add(pattern[1:])  # Remove the "*"
        return sorted(extensions)
