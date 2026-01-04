"""
Memory Service - Unified memory operations for the Embedded Trinity Memory System.

This service provides a facade for memory operations, delegating to the
MemoryStore and Consolidator in the RAG container.
"""

import importlib.util
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Resolve RAG directory path
_current_file = Path(__file__).resolve()
_server_dir = _current_file.parent.parent  # server/
_project_root = _server_dir.parent  # llamafarm/
_rag_dir = _project_root / "rag"

# Global memory store instance (lazy initialization)
_memory_store = None
_consolidator = None
_temp_dir = None


def _import_from_rag(module_name: str, class_name: str):
    """Import a class from RAG's core package using importlib.

    This avoids conflicts with server's own core package by directly
    loading the module from the RAG directory.

    Args:
        module_name: Module name (e.g., 'memory', 'consolidator')
        class_name: Class to import from the module

    Returns:
        The imported class
    """
    import sys

    module_path = _rag_dir / "core" / f"{module_name}.py"

    if not module_path.exists():
        raise ImportError(f"RAG module not found: {module_path}")

    # Add RAG directory to sys.path so the module can import its dependencies
    rag_path = str(_rag_dir)
    if rag_path not in sys.path:
        sys.path.insert(0, rag_path)

    # Load the module directly from file
    spec = importlib.util.spec_from_file_location(
        f"rag_core_{module_name}", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise ImportError(f"Class {class_name} not found in {module_path}")

    return getattr(module, class_name)


def _get_memory_store():
    """Get or create the global MemoryStore instance."""
    global _memory_store, _temp_dir

    if _memory_store is None:
        try:
            MemoryStore = _import_from_rag("memory", "MemoryStore")

            # Use temp directory for now; in production this would be configured
            _temp_dir = tempfile.mkdtemp(prefix="llamafarm_memory_")
            config = {"base_path": _temp_dir}
            _memory_store = MemoryStore(config=config)
            logger.info(f"MemoryStore initialized at {_temp_dir}")
        except ImportError as e:
            logger.error(f"Failed to import MemoryStore: {e}")
            raise

    return _memory_store


def _get_consolidator():
    """Get or create the global Consolidator instance."""
    global _consolidator

    if _consolidator is None:
        try:
            Consolidator = _import_from_rag("consolidator", "Consolidator")

            memory_store = _get_memory_store()
            _consolidator = Consolidator(memory_store=memory_store)
            logger.info("Consolidator initialized")
        except ImportError as e:
            logger.error(f"Failed to import Consolidator: {e}")
            raise

    return _consolidator


class MemoryService:
    """Service for unified memory operations."""

    @staticmethod
    def add(
        data: Any,
        data_type: str = "text",
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict[str, Any]:
        """Add data to the appropriate memory store.

        Args:
            data: The data to store
            data_type: Type of data (text, telemetry, chat, audio, node, edge)
            metadata: Optional metadata
            timestamp: Optional timestamp
            latitude: Latitude for spatial data
            longitude: Longitude for spatial data

        Returns:
            Dictionary with success status and store information
        """
        try:
            memory_store = _get_memory_store()

            result = memory_store.add(
                data=data,
                data_type=data_type,
                metadata=metadata,
                timestamp=timestamp,
                latitude=latitude,
                longitude=longitude,
            )

            return {
                "success": True,
                "uuid": result.get("uuid"),
                "store": result.get("store"),
                "message": f"Data added to {result.get('store', 'memory')} store",
                "nodes_created": result.get("nodes_created"),
                "edges_created": result.get("edges_created"),
                "expires_at": result.get("expires_at"),
            }

        except Exception as e:
            logger.error(f"Failed to add data to memory: {e}")
            return {
                "success": False,
                "message": str(e),
            }

    @staticmethod
    def query(
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        data_types: list[str] | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_m: float | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Query unified context from all stores.

        Args:
            start_time: Start of time range
            end_time: End of time range
            data_types: Filter by data types
            latitude: Center latitude for spatial query
            longitude: Center longitude for spatial query
            radius_m: Radius in meters for spatial query
            limit: Maximum results to return

        Returns:
            Dictionary with results and total count
        """
        try:
            memory_store = _get_memory_store()

            # Build query parameters
            time_range = None
            if start_time or end_time:
                time_range = (start_time, end_time)

            spatial = None
            if latitude is not None and longitude is not None and radius_m:
                spatial = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "radius_m": radius_m,
                }

            results = memory_store.query(
                time_range=time_range,
                data_types=data_types,
                spatial=spatial,
                recent={"limit": limit} if not time_range else None,
            )

            return {
                "results": results,
                "total_count": len(results),
            }

        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            return {
                "results": [],
                "total_count": 0,
                "error": str(e),
            }

    @staticmethod
    def delete(uuid: str) -> dict[str, Any] | None:
        """Delete a record by UUID (cascade delete via LinkageTable).

        Args:
            uuid: The concept UUID to delete

        Returns:
            Dictionary with deletion results, or None if not found
        """
        try:
            memory_store = _get_memory_store()

            success = memory_store.delete(uuid)

            if success:
                return {
                    "success": True,
                    "uuid": uuid,
                    "deleted_from": ["vector", "graph", "timeseries", "linkage"],
                    "message": "Record deleted from all linked stores",
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to delete from memory: {e}")
            return {
                "success": False,
                "uuid": uuid,
                "message": str(e),
            }

    @staticmethod
    def consolidate(use_llm: bool = False, force: bool = False) -> dict[str, Any]:
        """Trigger memory consolidation cycle.

        Args:
            use_llm: Whether to use LLM for synthesis
            force: Force consolidation even if below threshold

        Returns:
            Dictionary with consolidation results
        """
        try:
            consolidator = _get_consolidator()

            result = consolidator.run_cycle(use_llm=use_llm)

            return {
                "success": True,
                "records_processed": result.get("records_processed", 0),
                "facts_extracted": result.get("facts_extracted", 0),
                "nodes_created": result.get("nodes_created", 0),
                "pruned": result.get("pruned", 0),
                "skipped": result.get("skipped", False),
                "synthesis_method": "llm" if use_llm else "rule_based",
            }

        except Exception as e:
            logger.error(f"Failed to consolidate memory: {e}")
            return {
                "success": False,
                "message": str(e),
            }

    @staticmethod
    def get_stats() -> dict[str, Any]:
        """Get storage statistics from all stores.

        Returns:
            Dictionary with statistics from each store
        """
        try:
            memory_store = _get_memory_store()

            stats = memory_store.get_stats()

            return stats

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {
                "working_memory": {"total_records": 0, "error": str(e)},
                "graph": {"total_nodes": 0, "total_edges": 0},
                "timeseries": {"total_records": 0},
                "linkage": {"total_links": 0},
            }

    @staticmethod
    def get_context(
        recent_minutes: int = 10,
        include_graph: bool = True,
        include_working_memory: bool = True,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get aggregated context from all stores.

        Args:
            recent_minutes: How far back to look
            include_graph: Whether to include graph data
            include_working_memory: Whether to include working memory
            limit: Maximum records per store

        Returns:
            Dictionary with context from each store
        """
        try:
            memory_store = _get_memory_store()

            context = memory_store.get_context(
                recent_minutes=recent_minutes,
                include_graph=include_graph,
                include_working_memory=include_working_memory,
                limit=limit,
            )

            return context

        except Exception as e:
            logger.error(f"Failed to get memory context: {e}")
            return {
                "working_memory": [],
                "graph": [],
                "timeseries": [],
                "error": str(e),
            }

    @staticmethod
    def prune(older_than_hours: int | None = None) -> dict[str, Any]:
        """Prune expired records from working memory.

        Args:
            older_than_hours: Override TTL and prune records older than this

        Returns:
            Dictionary with prune results
        """
        try:
            memory_store = _get_memory_store()

            pruned_count = memory_store.prune_working_memory()

            # Get remaining count
            stats = memory_store.get_stats()
            remaining = stats.get("working_memory", {}).get("total_records", 0)

            return {
                "success": True,
                "pruned_count": pruned_count,
                "remaining_count": remaining,
            }

        except Exception as e:
            logger.error(f"Failed to prune memory: {e}")
            return {
                "success": False,
                "pruned_count": 0,
                "message": str(e),
            }
