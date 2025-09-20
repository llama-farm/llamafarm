"""Base classes for extractors."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
import logging

from core.base import Document

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Base class for all extractors."""

    def __init__(self, name: str = None, config: Optional[Dict[str, Any]] = None):
        self.name = str(name) if name is not None else self.__class__.__name__
        self.config = config or {}
        # Ensure logger name is always a string
        logger_name = f"{__name__}.{self.name}"
        if not isinstance(logger_name, str):
            raise TypeError(
                f"Logger name must be a string, got {type(logger_name)}: {logger_name}"
            )
        self.logger = logging.getLogger(logger_name)

    @abstractmethod
    def extract(self, documents: List[Document]) -> List[Document]:
        """
        Extract metadata from documents and enhance them.

        Args:
            documents: List of documents to process

        Returns:
            List of enhanced documents with extracted metadata
        """
        pass

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Get list of required dependencies for this extractor.

        Returns:
            List of package names required for this extractor
        """
        pass

    def validate_dependencies(self) -> bool:
        """
        Validate that all required dependencies are available.

        Returns:
            True if all dependencies are available, False otherwise
        """
        dependencies = self.get_dependencies()
        missing = []

        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)

        if missing:
            self.logger.error(f"Missing dependencies for {self.name}: {missing}")
            return False

        return True

    def configure(self, config: Dict[str, Any]) -> None:
        """Update extractor configuration."""
        self.config.update(config)

    def get_extraction_info(self) -> Dict[str, Any]:
        """Get information about what this extractor produces."""
        return {
            "name": self.name,
            "description": self.__doc__ or "No description available",
            "config": self.config,
            "dependencies": self.get_dependencies(),
        }


class ExtractorRegistry:
    """Registry for managing extractors."""

    def __init__(self):
        self._extractors: Dict[str, Type[BaseExtractor]] = {}

    def register(self, name: str, extractor_class: Type[BaseExtractor]) -> None:
        """Register an extractor class."""
        self._extractors[name] = extractor_class
        logger.info(f"Registered extractor: {name}")

    def get(self, name: str) -> Optional[Type[BaseExtractor]]:
        """Get an extractor class by name."""
        return self._extractors.get(name)

    def create(
        self, name: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseExtractor]:
        """Create an extractor instance."""
        extractor_class = self.get(name)
        if extractor_class is None:
            logger.error(f"Unknown extractor: {name}")
            return None

        try:
            extractor = extractor_class(config=config)
            if not extractor.validate_dependencies():
                logger.error(f"Failed to validate dependencies for {name}")
                return None
            return extractor
        except Exception as e:
            logger.error(f"Failed to create extractor {name}: {e}")
            return None

    def list_extractors(self) -> List[str]:
        """List all registered extractors."""
        return list(self._extractors.keys())

    def get_all_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered extractors."""
        info = {}
        for name, extractor_class in self._extractors.items():
            try:
                temp_extractor = extractor_class()
                info[name] = temp_extractor.get_extraction_info()
            except Exception as e:
                info[name] = {
                    "name": name,
                    "error": f"Failed to get info: {e}",
                    "dependencies": [],
                }
        return info


class ExtractorPipeline:
    """Pipeline for running multiple extractors in sequence."""

    def __init__(self, extractors: List[BaseExtractor]):
        self.extractors = extractors
        self.logger = logging.getLogger(f"{__name__}.ExtractorPipeline")

    def run(self, documents: List[Document]) -> List[Document]:
        """Run all extractors in sequence."""
        processed_docs = documents

        for extractor in self.extractors:
            try:
                self.logger.info(f"Running extractor: {extractor.name}")
                processed_docs = extractor.extract(processed_docs)
                self.logger.info(f"Completed extractor: {extractor.name}")
            except Exception as e:
                self.logger.error(f"Extractor {extractor.name} failed: {e}")
                # Continue with other extractors

        return processed_docs

    def add_extractor(self, extractor: BaseExtractor) -> None:
        """Add an extractor to the pipeline."""
        self.extractors.append(extractor)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        return {
            "extractors": [
                extractor.get_extraction_info() for extractor in self.extractors
            ],
            "total_extractors": len(self.extractors),
        }


def create_extractor_from_config(
    extractor_config: Dict[str, Any], registry: ExtractorRegistry
) -> Optional[BaseExtractor]:
    """
    Create an extractor instance from configuration.

    Args:
        extractor_config: Configuration containing 'type' and 'config'
        registry: Extractor registry to use

    Returns:
        Configured extractor instance or None if creation fails
    """
    extractor_type = extractor_config.get("type")
    if not extractor_type:
        logger.error("Extractor configuration missing 'type' field")
        return None

    extractor_settings = extractor_config.get("config", {})

    return registry.create(extractor_type, extractor_settings)


def create_pipeline_from_config(
    extractors_config: List[Dict[str, Any]], registry: ExtractorRegistry
) -> ExtractorPipeline:
    """
    Create an extractor pipeline from configuration.

    Args:
        extractors_config: List of extractor configurations
        registry: Extractor registry to use

    Returns:
        Configured extractor pipeline
    """
    extractors = []

    for extractor_config in extractors_config:
        extractor = create_extractor_from_config(extractor_config, registry)
        if extractor:
            extractors.append(extractor)
        else:
            logger.warning(
                f"Failed to create extractor from config: {extractor_config}"
            )

    return ExtractorPipeline(extractors)
