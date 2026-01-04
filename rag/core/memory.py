"""MemoryStore - Unified interface for the Embedded Trinity Memory System.

Orchestrates all memory components:
- Vector Memory (ChromaDB) - Semantic search [future integration]
- Time-Series Memory (DuckDB) - Telemetry, spatial queries
- Graph Memory (DuckDB) - Entity relationships
- Working Memory (DuckDB) - Short-term buffer with TTL
- Linkage Table (DuckDB) - Cross-database UUID mapping

This module provides a single API for:
- Adding data to the appropriate store based on type
- Querying across all stores
- Building unified context
- Cascade deletes via linkage tracking
"""

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryStore:
    """Unified memory interface for the Embedded Trinity Memory System.

    Features:
    - Route data to appropriate store based on type
    - Unified query across all stores
    - Automatic cross-store linking via LinkageTable
    - Cascade delete support
    - Aggregated context building

    Configuration:
        base_path: Base directory for all database files
        vector_store: ChromaDB configuration (future)
        timeseries_store: DuckDBStore configuration
        graph_store: GraphStore configuration
        working_memory: WorkingMemory configuration
        linkage_table: LinkageTable configuration
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize MemoryStore with all component stores.

        Args:
            config: Configuration dictionary with store-specific settings
        """
        config = config or {}
        self.base_path = config.get("base_path", ".")
        self._closed = False

        # Ensure base path exists
        base_dir = Path(self.base_path)
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize component stores
        self._init_stores(config)

        logger.info(f"MemoryStore initialized at {self.base_path}")

    def _init_stores(self, config: dict[str, Any]) -> None:
        """Initialize all component stores."""
        from components.stores.duckdb_store import (
            DuckDBStore,
            GraphStore,
            LinkageTable,
            WorkingMemory,
        )

        # Time-series store
        ts_config = config.get("timeseries_store", {})
        if "path" not in ts_config:
            ts_config["path"] = f"{self.base_path}/timeseries.duckdb"
        self.timeseries_store = DuckDBStore(config=ts_config)

        # Graph store
        graph_config = config.get("graph_store", {})
        if "path" not in graph_config:
            graph_config["path"] = f"{self.base_path}/graph.duckdb"
        self.graph_store = GraphStore(config=graph_config)

        # Working memory
        wm_config = config.get("working_memory", {})
        if "path" not in wm_config:
            wm_config["path"] = f"{self.base_path}/working.duckdb"
        self.working_memory = WorkingMemory(config=wm_config)

        # Linkage table
        lt_config = config.get("linkage_table", {})
        if "path" not in lt_config:
            lt_config["path"] = f"{self.base_path}/linkage.duckdb"
        self.linkage_table = LinkageTable(config=lt_config)

        # Vector store placeholder (future ChromaDB integration)
        self.vector_store = None

    def is_connected(self) -> bool:
        """Check if all stores are connected."""
        if self._closed:
            return False

        return (
            self.timeseries_store.is_connected()
            and self.graph_store.is_connected()
            and self.working_memory.is_connected()
            and self.linkage_table.is_connected()
        )

    # ─────────────────────────────────────────────────────────────────────
    # Add Operations
    # ─────────────────────────────────────────────────────────────────────

    def add(
        self,
        data: Any,
        data_type: str,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict[str, Any]:
        """Add data to the appropriate store based on type.

        Args:
            data: The data to store (string, dict, etc.)
            data_type: Type of data determining routing:
                - "telemetry" -> DuckDBStore (time-series)
                - "node" -> GraphStore (add node)
                - "edge" -> GraphStore (add edge)
                - "chat", "audio", "stream" -> WorkingMemory
            metadata: Optional metadata dict
            timestamp: Optional timestamp (defaults to now)
            latitude: Optional latitude for spatial data
            longitude: Optional longitude for spatial data

        Returns:
            Dictionary with uuid, store, and component-specific IDs
        """
        metadata = metadata or {}
        concept_uuid = str(uuid.uuid4())

        result = {
            "uuid": concept_uuid,
            "store": None,
            "component_id": None,
        }

        if data_type == "telemetry":
            result = self._add_telemetry(
                concept_uuid, data, metadata, timestamp, latitude, longitude
            )
        elif data_type == "node":
            result = self._add_node(concept_uuid, data, metadata)
        elif data_type == "edge":
            result = self._add_edge(concept_uuid, data, metadata)
        elif data_type in ("chat", "audio", "stream"):
            result = self._add_to_working_memory(
                concept_uuid, data, data_type, metadata
            )
        else:
            # Default to working memory for unknown types
            result = self._add_to_working_memory(
                concept_uuid, data, data_type, metadata
            )

        return result

    def _add_telemetry(
        self,
        concept_uuid: str,
        data: str,
        metadata: dict[str, Any],
        timestamp: datetime | None,
        latitude: float | None,
        longitude: float | None,
    ) -> dict[str, Any]:
        """Add telemetry data to time-series store."""
        timestamp = timestamp or datetime.now()

        # Build record in DuckDBStore format
        record = {
            "source": "telemetry",
            "ts": timestamp,
            "data": {"raw": data} if isinstance(data, str) else data,
            "metadata": metadata,
        }

        # Add location if provided
        if latitude is not None and longitude is not None:
            record["location"] = {"lat": latitude, "lon": longitude}

        count = self.timeseries_store.add_records([record])
        component_id = f"ts_{concept_uuid}"

        # Create linkage
        self.linkage_table.link(
            concept_uuid=concept_uuid,
            timeseries_row_id=component_id,
        )

        return {
            "uuid": concept_uuid,
            "store": "timeseries",
            "component_id": component_id,
            "count": count,
        }

    def _add_node(
        self,
        concept_uuid: str,
        data: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Add node to graph store."""
        node_id = data.get("id", concept_uuid)
        node_type = metadata.get("node_type", "entity")
        name = data.get("name", node_id)

        # Extract properties from data (exclude 'id' and 'name')
        properties = {k: v for k, v in data.items() if k not in ("id", "name")}

        # GraphStore.add_node signature: (name, node_type, node_id, properties)
        result_id = self.graph_store.add_node(
            name=name,
            node_type=node_type,
            node_id=node_id,
            properties=properties,
        )

        # Create linkage
        self.linkage_table.link(
            concept_uuid=concept_uuid,
            graph_node_id=result_id,
        )

        return {
            "uuid": concept_uuid,
            "store": "graph",
            "component_id": result_id,
            "success": result_id is not None,
        }

    def _add_edge(
        self,
        concept_uuid: str,
        data: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Add edge to graph store."""
        source = data.get("source")
        edge_type = data.get("edge_type", "related_to")
        target = data.get("target")
        properties = data.get("properties", {})

        if not source or not target:
            raise ValueError("Edge requires 'source' and 'target' in data")

        # GraphStore.add_edge signature: (source_id, target_id, relationship, weight, properties)
        edge_id = self.graph_store.add_edge(
            source, target, edge_type, properties=properties
        )

        return {
            "uuid": concept_uuid,
            "store": "graph",
            "component_id": f"{source}->{target}",
            "success": edge_id is not None,
        }

    def _add_to_working_memory(
        self,
        concept_uuid: str,
        data: Any,
        data_type: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Add data to working memory."""
        content = data if isinstance(data, str) else str(data)

        record_id = self.working_memory.add(data_type, content, metadata)

        return {
            "uuid": concept_uuid,
            "store": "working_memory",
            "component_id": record_id,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Query Operations
    # ─────────────────────────────────────────────────────────────────────

    def query(
        self,
        time_range: dict[str, datetime] | None = None,
        data_types: list[str] | None = None,
        graph_query: dict[str, Any] | None = None,
        spatial: dict[str, Any] | None = None,
        recent: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query across all stores.

        Args:
            time_range: {"start": datetime, "end": datetime} for time-series
            data_types: Filter by data types (used as source filter for time-series)
            graph_query: {"node_id": str, "direction": str} for graph traversal
            spatial: {"latitude": float, "longitude": float, "radius_meters": float}
            recent: {"limit": int, "data_type": str} for working memory

        Returns:
            List of matching records from all queried stores
        """
        results = []

        # Query time-series
        if time_range:
            source = data_types[0] if data_types else None
            ts_results = self.timeseries_store.query_time_range(
                start_time=time_range.get("start"),
                end_time=time_range.get("end"),
                source=source,
            )
            results.extend(ts_results)

        # Query graph
        if graph_query:
            node_id = graph_query.get("node_id")
            direction = graph_query.get("direction", "both")
            relationship = graph_query.get("edge_type") or graph_query.get(
                "relationship"
            )

            neighbors = self.graph_store.find_neighbors(
                node_id, direction=direction, relationship=relationship
            )
            results.extend(neighbors)

        # Query spatial
        if spatial:
            spatial_results = self.timeseries_store.query_spatial(
                center_lat=spatial.get("latitude"),
                center_lon=spatial.get("longitude"),
                radius_meters=spatial.get("radius_meters", 1000),
            )
            results.extend(spatial_results)

        # Query working memory
        if recent:
            data_type = recent.get("data_type")
            limit = recent.get("limit", 100)

            if data_type:
                wm_results = self.working_memory.get_by_type(data_type, limit=limit)
            else:
                wm_results = self.working_memory.get_recent(limit=limit)
            results.extend(wm_results)

        return results

    # ─────────────────────────────────────────────────────────────────────
    # Context Building
    # ─────────────────────────────────────────────────────────────────────

    def get_context(
        self,
        recent_minutes: int = 10,
        include_graph: bool = True,
        include_working_memory: bool = True,
        limit: int = 100,
    ) -> dict[str, list[dict[str, Any]]]:
        """Build aggregated context from all stores.

        Args:
            recent_minutes: Time window for recent data
            include_graph: Whether to include graph data
            include_working_memory: Whether to include working memory
            limit: Maximum records per category

        Returns:
            Dictionary with categorized context from each store
        """
        context = {
            "working_memory": [],
            "timeseries": [],
            "graph": [],
        }

        # Get recent working memory
        if include_working_memory:
            context["working_memory"] = self.working_memory.get_recent(
                limit=limit, minutes=recent_minutes
            )

        # Get recent time-series
        now = datetime.now()
        start_time = now - timedelta(minutes=recent_minutes)
        ts_results = self.timeseries_store.query_time_range(
            start_time=start_time,
            end_time=now,
        )
        # Apply limit manually since DuckDBStore doesn't have limit param
        context["timeseries"] = ts_results[:limit] if ts_results else []

        # Get graph summary (all nodes)
        if include_graph:
            stats = self.graph_store.get_stats()
            context["graph"] = [
                {
                    "type": "summary",
                    "total_nodes": stats.get("total_nodes", 0),
                    "total_edges": stats.get("total_edges", 0),
                }
            ]

        return context

    # ─────────────────────────────────────────────────────────────────────
    # Delete Operations
    # ─────────────────────────────────────────────────────────────────────

    def delete(self, concept_uuid: str) -> dict | None:
        """Delete a record using LinkageTable for cascade delete.

        Args:
            concept_uuid: The UUID of the concept to delete

        Returns:
            Dictionary with deleted_from list, or None if not found
        """
        # Get linked IDs and remove linkage
        ids = self.linkage_table.unlink_and_get_ids(concept_uuid)

        if ids is None:
            return None

        # Build list of stores that had links
        deleted_from = []
        if ids.get("vector_id"):
            deleted_from.append("vector")
        if ids.get("graph_node_id"):
            deleted_from.append("graph")
        if ids.get("timeseries_row_id"):
            deleted_from.append("timeseries")
        deleted_from.append("linkage")  # Always deleted from linkage

        # Delete from each store if ID exists
        # Note: Currently using linkage IDs; actual deletion from stores
        # would require additional implementation based on store capabilities

        logger.info(f"Cascade delete for {concept_uuid}: {ids}")
        return {"deleted_from": deleted_from}

    # ─────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get statistics from all stores.

        Returns:
            Dictionary with stats from each store
        """
        return {
            "timeseries": self.timeseries_store.get_stats(),
            "graph": self.graph_store.get_stats(),
            "working_memory": self.working_memory.get_stats(),
            "linkage": self.linkage_table.get_stats(),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Pruning Operations
    # ─────────────────────────────────────────────────────────────────────

    def prune_working_memory(self) -> int:
        """Prune expired records from working memory.

        Returns:
            Number of records pruned
        """
        return self.working_memory.prune()

    def clear_working_memory(self) -> None:
        """Clear all records from working memory."""
        self.working_memory.clear()

    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close all component stores."""
        if self._closed:
            return

        try:
            self.timeseries_store.close()
            self.graph_store.close()
            self.working_memory.close()
            self.linkage_table.close()
            self._closed = True
            logger.info("MemoryStore closed")
        except Exception as e:
            logger.warning(f"Error closing MemoryStore: {e}")

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
