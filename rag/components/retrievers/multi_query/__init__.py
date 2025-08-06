"""MultiQuery Component

Component for multi query.
"""

from .multi_query import MultiQuery

__all__ = ['MultiQuery']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "retriever"
COMPONENT_NAME = "multi_query"
