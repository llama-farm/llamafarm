"""
Metadata management components for RAG system.
"""

from .metadata_config import (
    MetadataLevel,
    CoreMetadataConfig,
    MetadataSchema,
    DocumentMetadata,
    MetadataPresets
)

from .metadata_enricher import (
    MetadataEnricher,
    MetadataFilter
)

__all__ = [
    # Config classes
    "MetadataLevel",
    "CoreMetadataConfig", 
    "MetadataSchema",
    "DocumentMetadata",
    "MetadataPresets",
    
    # Enricher classes
    "MetadataEnricher",
    "MetadataFilter"
]