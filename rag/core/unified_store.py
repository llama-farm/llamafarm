"""UnifiedDatasetStore - Unified storage backend for typed datasets.

Phase 17: Unified Dataset Store

Manages all storage backends (vector, graph, timeseries, spatial, working memory)
based on dataset configuration. Supports the new dataset type system:
- knowledge: Vector + Graph (document RAG with entity extraction)
- realtime: All stores (streaming telemetry, chat, live data)
- graph: Graph only (pure knowledge graph)
- timeseries: TimeSeries + WorkingMemory (IoT, metrics)
- spatial: Spatial + WorkingMemory (geo-tracking only)
- hybrid: All capabilities enabled
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Store capability matrix for dataset types
DATASET_TYPE_CAPABILITIES = {
    "knowledge": {
        "vector": True,
        "graph": True,
        "timeseries": False,
        "spatial": False,
        "working_memory": False,
    },
    "realtime": {
        "vector": True,
        "graph": True,
        "timeseries": True,
        "spatial": True,
        "working_memory": True,
    },
    "graph": {
        "vector": False,
        "graph": True,
        "timeseries": False,
        "spatial": False,
        "working_memory": False,
    },
    "timeseries": {
        "vector": False,
        "graph": False,
        "timeseries": True,
        "spatial": False,
        "working_memory": True,
    },
    "spatial": {
        "vector": False,
        "graph": False,
        "timeseries": False,
        "spatial": True,
        "working_memory": True,
    },
    "hybrid": {
        "vector": True,
        "graph": True,
        "timeseries": True,
        "spatial": True,
        "working_memory": True,
    },
}


class UnifiedDatasetStore:
    """Unified storage backend for typed datasets.

    Manages vector, graph, timeseries, spatial, and working memory stores
    based on dataset configuration. Provides a single API for:
    - Adding documents (routed to vector + graph)
    - Adding stream records (routed to timeseries/spatial + working memory)
    - Querying across all enabled stores
    - Cross-store linking via LinkageTable
    """

    def __init__(self, dataset_config: dict, project_dir: str):
        """Initialize UnifiedDatasetStore with stores based on dataset config.

        Args:
            dataset_config: Dataset configuration dict with type and store configs
            project_dir: Base project directory for data storage
        """
        self.config = dataset_config
        self.name = dataset_config.get("name", "default")
        self.dataset_type = dataset_config.get("type", "knowledge")
        self.base_path = str(Path(project_dir) / "lf_data" / "datasets" / self.name)
        self._closed = False

        # Ensure base path exists
        Path(self.base_path).mkdir(parents=True, exist_ok=True)

        # Get capabilities for this dataset type
        self._capabilities = self._resolve_capabilities()

        # Initialize stores based on capabilities
        self._init_stores()

        logger.info(
            f"UnifiedDatasetStore '{self.name}' initialized at {self.base_path}"
        )
        logger.debug(
            f"Enabled stores: {[k for k, v in self._capabilities.items() if v]}"
        )

    def _resolve_capabilities(self) -> dict[str, bool]:
        """Resolve which stores should be enabled based on type and config."""
        # Start with type defaults
        caps = DATASET_TYPE_CAPABILITIES.get(
            self.dataset_type, DATASET_TYPE_CAPABILITIES["knowledge"]
        ).copy()

        # Override with explicit config if provided
        if "vector" in self.config:
            vec_cfg = self.config["vector"]
            caps["vector"] = vec_cfg.get("enabled", True) if vec_cfg else False

        if "graph" in self.config:
            graph_cfg = self.config["graph"]
            caps["graph"] = graph_cfg.get("enabled", True) if graph_cfg else False

        if "timeseries" in self.config:
            ts_cfg = self.config["timeseries"]
            caps["timeseries"] = ts_cfg.get("enabled", True) if ts_cfg else False

        if "spatial" in self.config:
            spatial_cfg = self.config["spatial"]
            caps["spatial"] = spatial_cfg.get("enabled", True) if spatial_cfg else False

        if "working_memory" in self.config:
            wm_cfg = self.config["working_memory"]
            caps["working_memory"] = wm_cfg.get("enabled", True) if wm_cfg else False

        return caps

    def _init_stores(self) -> None:
        """Initialize all enabled stores."""
        from components.stores.duckdb_store import (
            DuckDBStore,
            GraphStore,
            LinkageTable,
            WorkingMemory,
        )

        # Vector store (ChromaDB - future integration)
        self.vector_store = None
        if self._capabilities["vector"]:
            # TODO: Integrate ChromaDB when ready
            # For now, vector store uses the RAG pipeline's existing ChromaDB
            pass

        # Graph store
        self.graph_store = None
        if self._capabilities["graph"]:
            graph_config = self.config.get("graph", {})
            self.graph_store = GraphStore(
                config={
                    "path": f"{self.base_path}/graph.duckdb",
                    "max_path_depth": graph_config.get("max_path_depth", 10),
                }
            )

        # Timeseries store (includes timestamp-based queries)
        self.timeseries_store = None
        if self._capabilities["timeseries"]:
            ts_config = self.config.get("timeseries", {})
            self.timeseries_store = DuckDBStore(
                config={
                    "path": f"{self.base_path}/timeseries.duckdb",
                    "retention_days": ts_config.get("retention_days", 30),
                }
            )

        # Spatial store (separate from timeseries for geo-only queries)
        self.spatial_store = None
        if self._capabilities["spatial"]:
            spatial_config = self.config.get("spatial", {})
            self.spatial_store = DuckDBStore(
                config={
                    "path": f"{self.base_path}/spatial.duckdb",
                    "retention_days": spatial_config.get("retention_days", 30),
                    "index_type": spatial_config.get("index_type", "rtree"),
                }
            )

        # Working memory (short-term buffer with TTL)
        self.working_memory = None
        if self._capabilities["working_memory"]:
            wm_config = self.config.get("working_memory", {})
            self.working_memory = WorkingMemory(
                config={
                    "path": f"{self.base_path}/working.duckdb",
                    "ttl_seconds": wm_config.get("ttl_seconds", 3600),
                    "max_records": wm_config.get("max_records", 10000),
                }
            )

        # Linkage table (always created for cross-store linking)
        self.linkage_table = LinkageTable(
            config={"path": f"{self.base_path}/linkage.duckdb"}
        )

    # ─────────────────────────────────────────────────────────────────────
    # Capability Checks
    # ─────────────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        """Check if all enabled stores are connected."""
        if self._closed:
            return False

        connected = True
        if self.graph_store:
            connected = connected and self.graph_store.is_connected()
        if self.timeseries_store:
            connected = connected and self.timeseries_store.is_connected()
        if self.spatial_store:
            connected = connected and self.spatial_store.is_connected()
        if self.working_memory:
            connected = connected and self.working_memory.is_connected()
        if self.linkage_table:
            connected = connected and self.linkage_table.is_connected()

        return connected

    def get_enabled_stores(self) -> list[str]:
        """Get list of enabled store names."""
        return [k for k, v in self._capabilities.items() if v]

    def has_capability(self, capability: str) -> bool:
        """Check if a specific capability is enabled."""
        return self._capabilities.get(capability, False)

    # ─────────────────────────────────────────────────────────────────────
    # Document Operations (for knowledge/hybrid datasets)
    # ─────────────────────────────────────────────────────────────────────

    def add_document(
        self,
        content: str,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        extract_entities: bool | None = None,
    ) -> dict[str, Any]:
        """Add document to vector store and optionally extract entities to graph.

        Args:
            content: Document text content
            doc_id: Optional document ID (generated if not provided)
            metadata: Optional document metadata
            extract_entities: Override config for entity extraction

        Returns:
            Dictionary with document_id, stores written to, entity count
        """
        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}
        result = {"document_id": doc_id, "stores": []}

        # Add to vector store
        if self.vector_store:
            # TODO: Integrate with ChromaDB when ready
            result["stores"].append("vector")

        # Extract entities to graph
        graph_cfg = self.config.get("graph", {})
        should_extract = (
            extract_entities
            if extract_entities is not None
            else graph_cfg.get("entity_extraction", True)
        )

        if self.graph_store and should_extract:
            entities = self._extract_entities(content, doc_id, metadata)
            for entity in entities:
                node_id = self.graph_store.add_node(
                    name=entity["name"],
                    node_type=entity["type"],
                    properties={"source_doc": doc_id, **entity.get("properties", {})},
                )
                if node_id:
                    self.linkage_table.link(concept_uuid=doc_id, graph_node_id=node_id)

            result["entities"] = len(entities)
            result["stores"].append("graph")

        return result

    def _extract_entities(
        self, content: str, doc_id: str, metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract named entities from document content.

        Uses spaCy NER for basic entity extraction. This is a placeholder
        that will be fully implemented in Phase 18.
        """
        # Phase 18 will implement full entity extraction with spaCy
        # For now, return empty list (no extraction)
        return []

    # ─────────────────────────────────────────────────────────────────────
    # Stream Operations (for realtime/timeseries/spatial datasets)
    # ─────────────────────────────────────────────────────────────────────

    def add_stream_record(
        self,
        data: dict[str, Any],
        data_type: str = "telemetry",
        timestamp: datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add streaming data to working memory and timeseries/spatial stores.

        Args:
            data: Record data dictionary
            data_type: Type of data (telemetry, chat, audio, etc.)
            timestamp: Optional timestamp (defaults to now)
            latitude: Optional latitude for spatial data
            longitude: Optional longitude for spatial data
            metadata: Optional metadata

        Returns:
            Dictionary with record_id and stores written to
        """
        record_id = str(uuid.uuid4())
        timestamp = timestamp or datetime.now()
        metadata = metadata or {}
        result = {"record_id": record_id, "stores": []}

        # Add to working memory (short-term buffer)
        if self.working_memory:
            content = str(data) if not isinstance(data, str) else data
            wm_id = self.working_memory.add(data_type, content, metadata)
            result["working_memory_id"] = wm_id
            result["stores"].append("working_memory")

        # Add to timeseries store
        if self.timeseries_store:
            record = {
                "source": data_type,
                "ts": timestamp,
                "data": data,
                "metadata": metadata,
            }
            count = self.timeseries_store.add_records([record])
            result["timeseries_count"] = count
            result["stores"].append("timeseries")

        # Add to spatial store if location provided
        if self.spatial_store and latitude is not None and longitude is not None:
            record = {
                "source": data_type,
                "ts": timestamp,
                "data": data,
                "metadata": metadata,
                "location": {"lat": latitude, "lon": longitude},
            }
            count = self.spatial_store.add_records([record])
            result["spatial_count"] = count
            result["stores"].append("spatial")

        # Create linkage
        self.linkage_table.link(
            concept_uuid=record_id,
            timeseries_row_id=f"ts_{record_id}" if self.timeseries_store else None,
        )

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Graph Operations (for graph/knowledge/hybrid datasets)
    # ─────────────────────────────────────────────────────────────────────

    def add_node(
        self,
        name: str,
        node_type: str = "entity",
        node_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str | None:
        """Add node directly to graph store.

        Args:
            name: Node name
            node_type: Node type (entity, person, location, etc.)
            node_id: Optional node ID
            properties: Optional node properties

        Returns:
            Node ID if successful, None otherwise
        """
        if not self.graph_store:
            logger.warning("Graph store not enabled for this dataset")
            return None

        return self.graph_store.add_node(
            name=name,
            node_type=node_type,
            node_id=node_id,
            properties=properties or {},
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: str = "related_to",
        weight: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> str | None:
        """Add edge to graph store.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship: Edge relationship type
            weight: Edge weight
            properties: Optional edge properties

        Returns:
            Edge ID if successful, None otherwise
        """
        if not self.graph_store:
            logger.warning("Graph store not enabled for this dataset")
            return None

        return self.graph_store.add_edge(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            weight=weight,
            properties=properties or {},
        )

    # ─────────────────────────────────────────────────────────────────────
    # Query Operations
    # ─────────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str | None = None,
        query_type: str = "hybrid",
        time_range: dict[str, datetime] | None = None,
        spatial: dict[str, Any] | None = None,
        graph_query: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Unified query across all enabled stores.

        Args:
            query_text: Text query for vector/semantic search
            query_type: Query type (vector, graph, hybrid, etc.)
            time_range: {"start": datetime, "end": datetime}
            spatial: {"latitude": float, "longitude": float, "radius_meters": float}
            graph_query: {"node_id": str, "direction": str}
            limit: Maximum results per store

        Returns:
            Dictionary with results from each queried store
        """
        results = {"query_type": query_type, "stores_queried": []}

        # Vector/semantic search (future)
        if query_text and self.vector_store:
            # TODO: Implement when ChromaDB integrated
            results["stores_queried"].append("vector")

        # Graph query
        if graph_query and self.graph_store:
            node_id = graph_query.get("node_id")
            direction = graph_query.get("direction", "both")
            relationship = graph_query.get("relationship")

            neighbors = self.graph_store.find_neighbors(
                node_id, direction=direction, relationship=relationship
            )
            results["graph"] = neighbors[:limit]
            results["stores_queried"].append("graph")

        # Timeseries query
        if time_range and self.timeseries_store:
            ts_results = self.timeseries_store.query_time_range(
                start_time=time_range.get("start"),
                end_time=time_range.get("end"),
            )
            results["timeseries"] = ts_results[:limit]
            results["stores_queried"].append("timeseries")

        # Spatial query
        if spatial and self.spatial_store:
            spatial_results = self.spatial_store.query_spatial(
                center_lat=spatial.get("latitude"),
                center_lon=spatial.get("longitude"),
                radius_meters=spatial.get("radius_meters", 1000),
            )
            results["spatial"] = spatial_results[:limit]
            results["stores_queried"].append("spatial")

        # Working memory (recent items)
        if self.working_memory and query_type in ("recent", "hybrid"):
            wm_results = self.working_memory.get_recent(limit=limit)
            results["working_memory"] = wm_results
            results["stores_queried"].append("working_memory")

        return results

    # ─────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated statistics from all enabled stores."""
        stats = {
            "dataset_name": self.name,
            "dataset_type": self.dataset_type,
            "base_path": self.base_path,
            "enabled_stores": self.get_enabled_stores(),
            "stores": {},
        }

        if self.graph_store:
            stats["stores"]["graph"] = self.graph_store.get_stats()

        if self.timeseries_store:
            stats["stores"]["timeseries"] = self.timeseries_store.get_stats()

        if self.spatial_store:
            stats["stores"]["spatial"] = self.spatial_store.get_stats()

        if self.working_memory:
            stats["stores"]["working_memory"] = self.working_memory.get_stats()

        if self.linkage_table:
            stats["stores"]["linkage"] = self.linkage_table.get_stats()

        return stats

    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────

    def clear(self) -> dict[str, bool]:
        """Clear all data from all enabled stores."""
        result = {}

        if self.graph_store:
            self.graph_store.clear()
            result["graph"] = True

        if self.timeseries_store:
            self.timeseries_store.clear()
            result["timeseries"] = True

        if self.spatial_store:
            self.spatial_store.clear()
            result["spatial"] = True

        if self.working_memory:
            self.working_memory.clear()
            result["working_memory"] = True

        if self.linkage_table:
            self.linkage_table.clear()
            result["linkage"] = True

        return result

    def close(self) -> None:
        """Close all component stores."""
        if self._closed:
            return

        try:
            if self.graph_store:
                self.graph_store.close()
            if self.timeseries_store:
                self.timeseries_store.close()
            if self.spatial_store:
                self.spatial_store.close()
            if self.working_memory:
                self.working_memory.close()
            if self.linkage_table:
                self.linkage_table.close()

            self._closed = True
            logger.info(f"UnifiedDatasetStore '{self.name}' closed")
        except Exception as e:
            logger.warning(f"Error closing UnifiedDatasetStore: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __del__(self):
        """Destructor to ensure stores are closed."""
        self.close()
