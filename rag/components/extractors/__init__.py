"""
Schema-driven extractors with self-contained components.

This module provides various extractors organized in a schema-driven architecture
where each extractor is a self-contained component with its own configuration.
"""

from .base import BaseExtractor, ExtractorRegistry, ExtractorPipeline

# Create a global registry instance
registry = ExtractorRegistry()

# Import and register all available extractors
try:
    from .entity_extractor.entity_extractor import EntityExtractor

    registry.register("EntityExtractor", EntityExtractor)
except ImportError:
    pass

try:
    from .keyword_extractor.keyword_extractor import KeywordExtractor

    registry.register("KeywordExtractor", KeywordExtractor)
except ImportError:
    pass

try:
    from .summary_extractor.summary_extractor import SummaryExtractor

    registry.register("SummaryExtractor", SummaryExtractor)
except ImportError:
    pass

try:
    from .pattern_extractor.pattern_extractor import PatternExtractor

    registry.register("PatternExtractor", PatternExtractor)
except ImportError:
    pass

try:
    from .link_extractor.link_extractor import LinkExtractor

    registry.register("LinkExtractor", LinkExtractor)
except ImportError:
    pass

try:
    from .heading_extractor.heading_extractor import HeadingExtractor

    registry.register("HeadingExtractor", HeadingExtractor)
except ImportError:
    pass

try:
    from .date_time_extractor.date_time_extractor import DateTimeExtractor

    registry.register("DateTimeExtractor", DateTimeExtractor)
except ImportError:
    pass

try:
    from .path_extractor.path_extractor import PathExtractor

    registry.register("PathExtractor", PathExtractor)
except ImportError:
    pass

try:
    from .statistics_extractor.statistics_extractor import StatisticsExtractor

    registry.register("StatisticsExtractor", StatisticsExtractor)
    registry.register("ContentStatisticsExtractor", StatisticsExtractor)  # Alias
except ImportError:
    pass

try:
    from .table_extractor.table_extractor import TableExtractor

    registry.register("TableExtractor", TableExtractor)
except ImportError:
    pass

__all__ = [
    "BaseExtractor",
    "ExtractorRegistry",
    "ExtractorPipeline",
    "registry",
    "EntityExtractor",
    "KeywordExtractor",
    "SummaryExtractor",
    "PatternExtractor",
    "LinkExtractor",
    "HeadingExtractor",
    "DateTimeExtractor",
    "PathExtractor",
    "StatisticsExtractor",
    "TableExtractor",
]
