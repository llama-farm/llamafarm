"""Summary Extractor Component

This component generates document summaries using statistical approaches.
"""

from .summary_extractor import SummaryExtractor

__all__ = ['SummaryExtractor']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "extractor"
COMPONENT_NAME = "summary_extractor"