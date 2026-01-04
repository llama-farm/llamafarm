"""
Memory Store Service - Per-project memory store management.

Phase 10: Memory Store Service Layer

This service manages memory stores on a per-project basis, following the same
pattern as DatabaseService for RAG databases. Memory stores are:
- Configured in llamafarm.yaml under the `memory:` section
- Stored in {project_dir}/lf_data/memory/{store_name}/
- Scoped to namespace/project

Storage Layout:
    {project_dir}/
    └── lf_data/
        ├── stores/              # RAG databases (ChromaDB)
        └── memory/              # Memory stores
            └── {store_name}/
                ├── timeseries.duckdb
                ├── graph.duckdb
                ├── working_memory.duckdb
                └── linkage.duckdb
"""

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from config.datamodel import MemoryConfig, MemoryStoreConfig

from api.errors import MemoryStoreNotFoundError
from core.logging import FastAPIStructLogger
from services.project_service import ProjectService

logger = FastAPIStructLogger()

# Resolve RAG directory path
_current_file = Path(__file__).resolve()
_server_dir = _current_file.parent.parent  # server/
_project_root = _server_dir.parent  # llamafarm/
_rag_dir = _project_root / "rag"


def _import_memory_store():
    """Import MemoryStore from RAG using importlib.

    This avoids conflicts with server's own core package by directly
    loading the module from the RAG directory.
    """
    import sys

    module_path = _rag_dir / "core" / "memory.py"

    if not module_path.exists():
        raise ImportError(f"RAG memory module not found: {module_path}")

    # Add RAG directory to sys.path so the module can import its dependencies
    rag_path = str(_rag_dir)
    if rag_path not in sys.path:
        sys.path.insert(0, rag_path)

    # Load the module directly from file
    spec = importlib.util.spec_from_file_location("rag_core_memory", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "MemoryStore"):
        raise ImportError(f"MemoryStore not found in {module_path}")

    return module.MemoryStore


def _import_consolidator():
    """Import Consolidator from RAG using importlib."""
    import sys

    module_path = _rag_dir / "core" / "consolidator.py"

    if not module_path.exists():
        raise ImportError(f"RAG consolidator module not found: {module_path}")

    # Add RAG directory to sys.path
    rag_path = str(_rag_dir)
    if rag_path not in sys.path:
        sys.path.insert(0, rag_path)

    # Load the module directly from file
    spec = importlib.util.spec_from_file_location("rag_core_consolidator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Consolidator"):
        raise ImportError(f"Consolidator not found in {module_path}")

    return module.Consolidator


# Cache for memory store instances per project/store
# Key: f"{namespace}/{project}/{store_name}"
_store_cache: dict[str, Any] = {}


class MemoryStoreService:
    """Service for managing per-project memory stores."""

    @classmethod
    def _get_memory_dir(cls, namespace: str, project: str) -> Path:
        """Get the memory directory for a project."""
        project_dir = Path(ProjectService.get_project_dir(namespace, project))
        return project_dir / "lf_data" / "memory"

    @classmethod
    def _get_store_path(cls, namespace: str, project: str, store_name: str) -> Path:
        """Get the path for a specific memory store."""
        return cls._get_memory_dir(namespace, project) / store_name

    @classmethod
    def _get_cache_key(cls, namespace: str, project: str, store_name: str) -> str:
        """Get cache key for a store instance."""
        return f"{namespace}/{project}/{store_name}"

    @classmethod
    def _get_memory_config(cls, namespace: str, project: str) -> MemoryConfig | None:
        """Get memory configuration from project config."""
        project_config = ProjectService.load_config(namespace, project)
        return project_config.memory

    @classmethod
    def _get_store_config(
        cls, namespace: str, project: str, store_name: str
    ) -> MemoryStoreConfig:
        """Get configuration for a specific store.

        Raises:
            MemoryStoreNotFoundError: If store is not configured
        """
        memory_config = cls._get_memory_config(namespace, project)

        if not memory_config or not memory_config.stores:
            raise MemoryStoreNotFoundError(store_name)

        for store in memory_config.stores:
            if store.name == store_name:
                return store

        raise MemoryStoreNotFoundError(store_name)

    @classmethod
    def list_stores(cls, namespace: str, project: str) -> list[MemoryStoreConfig]:
        """List all configured memory stores for a project."""
        memory_config = cls._get_memory_config(namespace, project)
        if not memory_config:
            return []
        return memory_config.stores or []

    @classmethod
    def get_store(cls, namespace: str, project: str, store_name: str) -> Any:
        """Get or create a memory store instance.

        Args:
            namespace: Project namespace
            project: Project name
            store_name: Name of the memory store

        Returns:
            MemoryStore instance

        Raises:
            MemoryStoreNotFoundError: If store is not configured
        """
        # Check cache first
        cache_key = cls._get_cache_key(namespace, project, store_name)
        if cache_key in _store_cache:
            store = _store_cache[cache_key]
            if store.is_connected():
                return store
            # Store is disconnected, remove from cache
            del _store_cache[cache_key]

        # Validate store is configured
        store_config = cls._get_store_config(namespace, project, store_name)

        # Create store directory
        store_path = cls._get_store_path(namespace, project, store_name)
        store_path.mkdir(parents=True, exist_ok=True)

        # Build MemoryStore config
        config = {"base_path": str(store_path)}

        # Add working memory config
        if store_config.working_memory:
            config["working_memory"] = {
                "ttl_seconds": store_config.working_memory.ttl_seconds,
                "max_size": store_config.working_memory.max_records,
            }

        # Add timeseries config
        if store_config.timeseries:
            config["timeseries_store"] = {
                "retention_days": store_config.timeseries.retention_days,
            }

        # Add graph config
        if store_config.graph:
            config["graph_store"] = {
                "max_path_depth": store_config.graph.max_path_depth,
            }

        # Create MemoryStore instance
        MemoryStore = _import_memory_store()
        store = MemoryStore(config=config)

        # Cache the store
        _store_cache[cache_key] = store

        logger.info(
            "Created memory store",
            namespace=namespace,
            project=project,
            store_name=store_name,
            store_path=str(store_path),
        )

        return store

    @classmethod
    def get_default_store(cls, namespace: str, project: str) -> Any:
        """Get the default memory store for a project.

        Falls back to first configured store if no default is set.

        Raises:
            MemoryStoreNotFoundError: If no stores are configured
        """
        memory_config = cls._get_memory_config(namespace, project)

        if not memory_config or not memory_config.stores:
            raise MemoryStoreNotFoundError("No memory stores configured")

        # Use default if specified
        if memory_config.default_store:
            return cls.get_store(namespace, project, memory_config.default_store)

        # Fall back to first store
        return cls.get_store(namespace, project, memory_config.stores[0].name)

    @classmethod
    def get_store_stats(
        cls, namespace: str, project: str, store_name: str | None = None
    ) -> dict:
        """Get statistics for a memory store.

        Args:
            namespace: Project namespace
            project: Project name
            store_name: Memory store name (uses default if not specified)
        """
        if store_name:
            store = cls.get_store(namespace, project, store_name)
        else:
            store = cls.get_default_store(namespace, project)
            # Get store name for path info
            memory_config = cls._get_memory_config(namespace, project)
            store_name = (
                memory_config.default_store
                if memory_config and memory_config.default_store
                else memory_config.stores[0].name
                if memory_config.stores
                else "default"
            )
        stats = store.get_stats()

        # Add path info
        store_path = cls._get_store_path(namespace, project, store_name)
        stats["store_path"] = str(store_path)

        # Calculate total size
        total_size = 0
        if store_path.exists():
            for f in store_path.glob("*.duckdb"):
                total_size += f.stat().st_size
        stats["total_size_bytes"] = total_size

        return stats

    @classmethod
    def clear_store(cls, namespace: str, project: str, store_name: str) -> dict:
        """Clear all data from a memory store but keep the store.

        Returns:
            Dictionary with cleared counts
        """
        store = cls.get_store(namespace, project, store_name)

        # Get stats before clearing
        before_stats = store.get_stats()

        # Clear each component
        store.clear_working_memory()

        # Clear timeseries - need to add this method
        if hasattr(store.timeseries_store, "clear"):
            store.timeseries_store.clear()

        # Clear graph - need to add this method
        if hasattr(store.graph_store, "clear"):
            store.graph_store.clear()

        # Clear linkage table - need to add this method
        if hasattr(store.linkage_table, "clear"):
            store.linkage_table.clear()

        after_stats = store.get_stats()

        logger.info(
            "Cleared memory store",
            namespace=namespace,
            project=project,
            store_name=store_name,
        )

        return {
            "success": True,
            "store_name": store_name,
            "before": before_stats,
            "after": after_stats,
        }

    @classmethod
    def delete_store(
        cls, namespace: str, project: str, store_name: str, delete_data: bool = True
    ) -> dict:
        """Delete a memory store and optionally its data.

        Note: This removes the store from cache but NOT from config.
        To fully remove, also update the project's llamafarm.yaml.

        Args:
            namespace: Project namespace
            project: Project name
            store_name: Store to delete
            delete_data: Whether to delete the data files

        Returns:
            Dictionary with deletion info
        """
        cache_key = cls._get_cache_key(namespace, project, store_name)

        # Close store if cached
        if cache_key in _store_cache:
            try:
                _store_cache[cache_key].close()
            except Exception as e:
                logger.warning(f"Error closing store: {e}")
            del _store_cache[cache_key]

        store_path = cls._get_store_path(namespace, project, store_name)
        data_deleted = False

        if delete_data and store_path.exists():
            shutil.rmtree(store_path)
            data_deleted = True
            logger.info(
                "Deleted memory store data",
                namespace=namespace,
                project=project,
                store_name=store_name,
                store_path=str(store_path),
            )

        return {
            "success": True,
            "store_name": store_name,
            "data_deleted": data_deleted,
            "store_path": str(store_path),
        }

    @classmethod
    def close_all_stores(cls, namespace: str | None = None, project: str | None = None):
        """Close cached store instances.

        Args:
            namespace: Optional namespace filter
            project: Optional project filter (requires namespace)
        """
        keys_to_remove = []

        for cache_key, store in _store_cache.items():
            parts = cache_key.split("/")
            if len(parts) != 3:
                continue

            key_ns, key_proj, key_store = parts

            # Apply filters
            if namespace and key_ns != namespace:
                continue
            if project and key_proj != project:
                continue

            try:
                store.close()
            except Exception as e:
                logger.warning(f"Error closing store {cache_key}: {e}")

            keys_to_remove.append(cache_key)

        for key in keys_to_remove:
            del _store_cache[key]

        logger.info(f"Closed {len(keys_to_remove)} memory store(s)")

    @classmethod
    def get_consolidator(
        cls, namespace: str, project: str, store_name: str | None = None
    ) -> Any:
        """Get a Consolidator instance for a memory store.

        Args:
            namespace: Project namespace
            project: Project name
            store_name: Memory store name (uses default if not specified)
        """
        # Resolve default store name if not provided
        if not store_name:
            memory_config = cls._get_memory_config(namespace, project)
            store_name = (
                memory_config.default_store
                if memory_config and memory_config.default_store
                else memory_config.stores[0].name
                if memory_config.stores
                else "default"
            )

        store = cls.get_store(namespace, project, store_name)
        store_config = cls._get_store_config(namespace, project, store_name)

        # Build consolidator config
        config = {}
        if store_config.consolidation:
            config = {
                "min_records": store_config.consolidation.min_records,
                "batch_size": store_config.consolidation.batch_size,
                "prune_after_consolidate": store_config.consolidation.prune_after_consolidate,
            }

        Consolidator = _import_consolidator()
        return Consolidator(memory_store=store, config=config)
