"""Statistics Extractor Component

This component extracts comprehensive content statistics including readability metrics,
vocabulary analysis, and structural information.
"""

from .statistics_extractor import ContentStatisticsExtractor

__all__ = ['ContentStatisticsExtractor']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "extractor"
COMPONENT_NAME = "statistics_extractor"