"""TypedDatasetService - Extended dataset service for typed datasets.

Phase 21: Dataset Service Layer Updates

Extends the base DatasetService to support the new typed dataset system:
- Dataset types: knowledge, realtime, graph, timeseries, spatial, hybrid
- Integration with UnifiedDatasetStore
- Stream record operations for realtime/timeseries/spatial datasets
- Hybrid query execution
"""

import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config.datamodel import Dataset

from services.dataset_service import DatasetService
from services.project_service import ProjectService

logger = logging.getLogger(__name__)

# RAG directory for importing unified store
_current_file = Path(__file__).resolve()
_server_dir = _current_file.parent.parent  # server/
_project_root = _server_dir.parent  # llamafarm/
_rag_dir = _project_root / "rag"

# Cache for UnifiedDatasetStore instances
_store_cache: dict[str, Any] = {}


def _import_from_rag(module_name: str, class_name: str):
    """Import a class from RAG's core package."""
    module_path = _rag_dir / "core" / f"{module_name}.py"

    if not module_path.exists():
        raise ImportError(f"RAG module not found: {module_path}")

    rag_path = str(_rag_dir)
    if rag_path not in sys.path:
        sys.path.insert(0, rag_path)

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


def _get_store(namespace: str, project: str, dataset: str) -> Any:
    """Get or create a UnifiedDatasetStore for a dataset."""
    cache_key = f"{namespace}/{project}/{dataset}"

    if cache_key in _store_cache:
        store = _store_cache[cache_key]
        if store.is_connected():
            return store
        else:
            # Remove disconnected store from cache
            del _store_cache[cache_key]

    # Get project directory
    project_dir = ProjectService.get_project_dir(namespace, project)
    if not project_dir:
        raise ValueError(f"Project not found: {namespace}/{project}")

    # Load dataset config
    datasets = DatasetService.list_datasets(namespace, project)
    dataset_config = None
    for ds in datasets:
        if ds.name == dataset:
            dataset_config = ds.model_dump()
            break

    if dataset_config is None:
        raise ValueError(f"Dataset not found: {dataset}")

    # Import and create store
    UnifiedDatasetStore = _import_from_rag("unified_store", "UnifiedDatasetStore")

    store = UnifiedDatasetStore(
        dataset_config={
            "name": dataset,
            "type": dataset_config.get("type", "knowledge"),
        },
        project_dir=str(project_dir),
    )

    _store_cache[cache_key] = store
    return store


class TypedDatasetService(DatasetService):
    """Extended dataset service for typed datasets.

    Inherits all base DatasetService functionality and adds:
    - Dataset type management
    - Stream record operations
    - Hybrid query execution
    - Entity extraction
    """

    # ─────────────────────────────────────────────────────────────────────
    # Dataset Type Operations
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def get_dataset_type(cls, namespace: str, project: str, dataset: str) -> str:
        """Get the type of a dataset.

        Args:
            namespace: Namespace
            project: Project name
            dataset: Dataset name

        Returns:
            Dataset type string (knowledge, realtime, graph, etc.)
        """
        datasets = cls.list_datasets(namespace, project)
        for ds in datasets:
            if ds.name == dataset:
                return getattr(ds, "type", "knowledge") or "knowledge"
        raise ValueError(f"Dataset not found: {dataset}")

    @classmethod
    def get_enabled_stores(
        cls, namespace: str, project: str, dataset: str
    ) -> list[str]:
        """Get list of enabled stores for a dataset.

        Args:
            namespace: Namespace
            project: Project name
            dataset: Dataset name

        Returns:
            List of enabled store names
        """
        try:
            store = _get_store(namespace, project, dataset)
            return store.get_enabled_stores()
        except Exception as e:
            logger.error(f"Failed to get enabled stores: {e}")
            return []

    @classmethod
    def create_typed_dataset(
        cls,
        namespace: str,
        project: str,
        name: str,
        dataset_type: str = "knowledge",
        data_processing_strategy: str = "auto",
        database: str = "chromadb",
        config_overrides: dict[str, Any] | None = None,
    ) -> Dataset:
        """Create a new typed dataset.

        Args:
            namespace: Namespace
            project: Project name
            name: Dataset name
            dataset_type: Type (knowledge, realtime, graph, timeseries, spatial, hybrid)
            data_processing_strategy: Processing strategy name
            database: Vector database name
            config_overrides: Optional config overrides for stores

        Returns:
            Created Dataset object
        """
        # Validate dataset type
        valid_types = [
            "knowledge",
            "realtime",
            "graph",
            "timeseries",
            "spatial",
            "hybrid",
        ]
        if dataset_type not in valid_types:
            raise ValueError(
                f"Invalid dataset type: {dataset_type}. Must be one of {valid_types}"
            )

        # Create using base service
        dataset = cls.create_dataset(
            namespace=namespace,
            project=project,
            name=name,
            data_processing_strategy=data_processing_strategy,
            database=database,
        )

        # TODO: Store the type in the dataset config
        # This will be done when the Dataset model supports the 'type' field

        return dataset

    # ─────────────────────────────────────────────────────────────────────
    # Stream Record Operations
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def add_stream_record(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        data: dict[str, Any],
        data_type: str = "telemetry",
        timestamp: datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a stream record to the dataset.

        Args:
            namespace: Namespace
            project: Project name
            dataset: Dataset name
            data: Record data
            data_type: Type of data
            timestamp: Optional timestamp
            latitude: Optional latitude
            longitude: Optional longitude
            metadata: Optional metadata

        Returns:
            Dictionary with record_id and stores written to
        """
        try:
            store = _get_store(namespace, project, dataset)

            result = store.add_stream_record(
                data=data,
                data_type=data_type,
                timestamp=timestamp,
                latitude=latitude,
                longitude=longitude,
                metadata=metadata,
            )

            return {
                "success": True,
                **result,
            }

        except Exception as e:
            logger.error(f"Failed to add stream record: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    @classmethod
    def add_stream_batch(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        records: list[dict[str, Any]],
        fail_fast: bool = False,
    ) -> dict[str, Any]:
        """Add multiple stream records in a batch.

        Args:
            namespace: Namespace
            project: Project name
            dataset: Dataset name
            records: List of record dictionaries
            fail_fast: Stop on first error

        Returns:
            Dictionary with success counts and results
        """
        try:
            store = _get_store(namespace, project, dataset)

            successful = 0
            failed = 0
            results = []

            for record in records:
                try:
                    result = store.add_stream_record(
                        data=record.get("data", {}),
                        data_type=record.get("data_type", "telemetry"),
                        timestamp=record.get("timestamp"),
                        latitude=record.get("latitude"),
                        longitude=record.get("longitude"),
                        metadata=record.get("metadata"),
                    )
                    successful += 1
                    results.append({"success": True, **result})

                except Exception as e:
                    failed += 1
                    results.append({"success": False, "error": str(e)})
                    if fail_fast:
                        break

            return {
                "success": failed == 0,
                "total": len(records),
                "successful": successful,
                "failed": failed,
                "results": results,
            }

        except Exception as e:
            logger.error(f"Batch ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    # ─────────────────────────────────────────────────────────────────────
    # Query Operations
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def hybrid_query(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        query_text: str | None = None,
        graph_node_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_meters: float = 1000.0,
        mode: str = "hybrid",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Execute a hybrid query across multiple stores.

        Args:
            namespace: Namespace
            project: Project name
            dataset: Dataset name
            query_text: Text for semantic search
            graph_node_id: Node ID for graph traversal
            start_time: Start of time range
            end_time: End of time range
            latitude: Latitude for spatial query
            longitude: Longitude for spatial query
            radius_meters: Radius for spatial query
            mode: Query mode (hybrid, vector, graph, etc.)
            limit: Maximum results

        Returns:
            Dictionary with query results
        """
        try:
            store = _get_store(namespace, project, dataset)

            # Import and execute hybrid query
            hybrid_query_func = _import_from_rag("hybrid_query", "hybrid_query")

            result = hybrid_query_func(
                unified_store=store,
                query_text=query_text,
                graph_node_id=graph_node_id,
                start_time=start_time,
                end_time=end_time,
                latitude=latitude,
                longitude=longitude,
                radius_meters=radius_meters,
                mode=mode,
                limit=limit,
            )

            return {
                "success": True,
                **result,
            }

        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
            }

    @classmethod
    def query_context(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        recent_minutes: int = 10,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get recent context from the dataset.

        Args:
            namespace: Namespace
            project: Project name
            dataset: Dataset name
            recent_minutes: How far back to look
            limit: Maximum results per store

        Returns:
            Dictionary with context from all stores
        """
        try:
            store = _get_store(namespace, project, dataset)

            # Use hybrid query with context mode
            result = store.query(
                query_type="context",
                limit=limit,
            )

            return {
                "success": True,
                **result,
            }

        except Exception as e:
            logger.error(f"Context query failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    # ─────────────────────────────────────────────────────────────────────
    # Graph Operations
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def add_graph_node(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        name: str,
        node_type: str = "entity",
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a node to the graph store.

        Args:
            namespace: Namespace
            project: Project name
            dataset: Dataset name
            name: Node name
            node_type: Node type
            properties: Optional properties

        Returns:
            Dictionary with node_id if successful
        """
        try:
            store = _get_store(namespace, project, dataset)

            node_id = store.add_node(
                name=name,
                node_type=node_type,
                properties=properties,
            )

            if node_id:
                return {
                    "success": True,
                    "node_id": node_id,
                }
            else:
                return {
                    "success": False,
                    "error": "Graph store not enabled for this dataset",
                }

        except Exception as e:
            logger.error(f"Failed to add graph node: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    @classmethod
    def add_graph_edge(
        cls,
        namespace: str,
        project: str,
        dataset: str,
        source_id: str,
        target_id: str,
        relationship: str = "related_to",
        weight: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add an edge to the graph store.

        Args:
            namespace: Namespace
            project: Project name
            dataset: Dataset name
            source_id: Source node ID
            target_id: Target node ID
            relationship: Relationship type
            weight: Edge weight
            properties: Optional properties

        Returns:
            Dictionary with edge_id if successful
        """
        try:
            store = _get_store(namespace, project, dataset)

            edge_id = store.add_edge(
                source_id=source_id,
                target_id=target_id,
                relationship=relationship,
                weight=weight,
                properties=properties,
            )

            if edge_id:
                return {
                    "success": True,
                    "edge_id": edge_id,
                }
            else:
                return {
                    "success": False,
                    "error": "Graph store not enabled for this dataset",
                }

        except Exception as e:
            logger.error(f"Failed to add graph edge: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    # ─────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def get_dataset_stats(
        cls,
        namespace: str,
        project: str,
        dataset: str,
    ) -> dict[str, Any]:
        """Get statistics for a typed dataset.

        Args:
            namespace: Namespace
            project: Project name
            dataset: Dataset name

        Returns:
            Dictionary with statistics from all enabled stores
        """
        try:
            store = _get_store(namespace, project, dataset)
            stats = store.get_stats()

            return {
                "success": True,
                **stats,
            }

        except Exception as e:
            logger.error(f"Failed to get dataset stats: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def clear_dataset_stores(
        cls,
        namespace: str,
        project: str,
        dataset: str,
    ) -> dict[str, Any]:
        """Clear all data from dataset stores (not files).

        Args:
            namespace: Namespace
            project: Project name
            dataset: Dataset name

        Returns:
            Dictionary with clear results for each store
        """
        try:
            store = _get_store(namespace, project, dataset)
            result = store.clear()

            return {
                "success": True,
                "cleared_stores": result,
            }

        except Exception as e:
            logger.error(f"Failed to clear dataset stores: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    @classmethod
    def close_store(cls, namespace: str, project: str, dataset: str) -> None:
        """Close a dataset store and remove from cache.

        Args:
            namespace: Namespace
            project: Project name
            dataset: Dataset name
        """
        cache_key = f"{namespace}/{project}/{dataset}"
        if cache_key in _store_cache:
            try:
                _store_cache[cache_key].close()
            except Exception as e:
                logger.warning(f"Error closing store: {e}")
            del _store_cache[cache_key]

    @classmethod
    def close_all_stores(cls) -> None:
        """Close all cached stores."""
        for key, store in list(_store_cache.items()):
            try:
                store.close()
            except Exception as e:
                logger.warning(f"Error closing store {key}: {e}")
        _store_cache.clear()
