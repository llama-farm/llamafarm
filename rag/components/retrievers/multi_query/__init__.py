"""MultiQuery Component

Component for multi query.
"""

from .multi_query import MultiQueryStrategy

__all__ = ['MultiQueryStrategy']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "retriever"
COMPONENT_NAME = "multi_query"
