"""
Server-side Celery Tasks Package

This package is reserved for server-side Celery task definitions.
Currently empty - all RAG-related Celery tasks are defined in the
rag/tasks/ package and registered with the RAG worker.

If server-side tasks are added in the future, they should be:
1. Defined in submodules of this package
2. Imported here
3. Added to __all__ for proper registration
"""

__all__: list[str] = []
