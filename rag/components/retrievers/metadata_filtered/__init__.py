"""MetadataFiltered Component

Component for metadata filtered.
"""

from .metadata_filtered import MetadataFilteredStrategy

__all__ = ['MetadataFilteredStrategy']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "retriever"
COMPONENT_NAME = "metadata_filtered"
