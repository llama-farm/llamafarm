"""Source path extractor to ensure file path information is preserved in document metadata."""

from pathlib import Path
from typing import List

from ..base import BaseExtractor
from core.base import Document

from core.logging import RAGStructLogger
logger = RAGStructLogger("rag.components.extractors.path_extractor.path_extractor")


class PathExtractor(BaseExtractor):
    """Extract and preserve source path information in document metadata.

    This extractor ensures that source file paths are properly stored in metadata
    so they can be retrieved from vector databases and displayed in search results.
    """

    def __init__(self, name: str = "path_extractor", config: dict = None):
        super().__init__(name, config)
        config = config or {}
        self.store_full_path = config.get("store_full_path", True)
        self.store_filename = config.get("store_filename", True)
        self.store_directory = config.get("store_directory", True)
        self.store_extension = config.get("store_extension", True)
        self.normalize_paths = config.get("normalize_paths", False)
        self.relative_to = config.get("relative_to", None)

    def get_dependencies(self) -> List[str]:
        """Return list of dependencies (none for path extractor)."""
        return []

    def extract(self, documents: List[Document]) -> List[Document]:
        """Extract path information from documents and store in metadata."""

        for doc in documents:
            try:
                if doc.source:
                    source_path = Path(doc.source)

                    # Store various path components in metadata for reliable retrieval
                    if "extractors" not in doc.metadata:
                        doc.metadata["extractors"] = {}

                    path_info = {}

                    # Process full path with optional normalization and relative conversion
                    if self.store_full_path:
                        full_path = source_path

                        # Make relative if configured
                        if self.relative_to:
                            try:
                                relative_base = Path(self.relative_to).resolve()
                                full_path = source_path.resolve().relative_to(
                                    relative_base
                                )
                            except (ValueError, RuntimeError):
                                # If path is not relative to base, keep absolute
                                logger.debug(
                                    f"Path {source_path} is not relative to {self.relative_to}"
                                )

                        # Normalize path separators if configured
                        path_str = str(full_path)
                        if self.normalize_paths:
                            path_str = path_str.replace("\\", "/")

                        path_info["full_path"] = path_str
                        doc.metadata["file_path"] = path_str

                    if self.store_filename:
                        path_info["filename"] = source_path.name
                        doc.metadata["file_name"] = source_path.name

                    if self.store_directory:
                        directory = source_path.parent

                        # Make directory relative if configured
                        if self.relative_to:
                            try:
                                relative_base = Path(self.relative_to).resolve()
                                directory = directory.resolve().relative_to(
                                    relative_base
                                )
                            except (ValueError, RuntimeError):
                                pass

                        dir_str = str(directory)
                        if self.normalize_paths:
                            dir_str = dir_str.replace("\\", "/")

                        path_info["directory"] = dir_str
                        doc.metadata["directory"] = dir_str

                    if self.store_extension:
                        path_info["extension"] = source_path.suffix
                        doc.metadata["file_extension"] = source_path.suffix

                    # Store in extractors section
                    doc.metadata["extractors"]["path"] = path_info

                    logger.debug(f"Extracted path information for: {source_path.name}")

                else:
                    logger.warning(f"Document {doc.id} has no source path to extract")

            except Exception as e:
                logger.error(
                    f"Failed to extract path information from document {doc.id}: {e}"
                )

        return documents
