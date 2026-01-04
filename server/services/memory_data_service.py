"""
Memory Data Service - CRUD operations for per-project memory stores.

Phase 11: Memory Data Service (CRUD Operations)

This service provides data operations on memory stores:
- add: Route data to correct store component
- query: Perform unified retrieval
- get_context: Return aggregated context
- delete: Cascade delete via UUID
- clear_table: Clear specific table (working_memory, timeseries, graph, linkage, all)
- consolidate: Trigger consolidation
- prune: Remove expired records
- get_stats: Return detailed statistics
"""

from datetime import datetime
from typing import Any

from core.logging import FastAPIStructLogger
from services.memory_store_service import MemoryStoreService

logger = FastAPIStructLogger()


class MemoryDataService:
    """Service for CRUD operations on per-project memory stores."""

    @classmethod
    def _get_store(
        cls,
        namespace: str,
        project: str,
        store_name: str | None = None,
    ) -> Any:
        """Resolve and return the appropriate memory store.

        Args:
            namespace: Project namespace
            project: Project name
            store_name: Memory store name (uses default if not specified)

        Returns:
            MemoryStore instance
        """
        if store_name:
            return MemoryStoreService.get_store(namespace, project, store_name)
        return MemoryStoreService.get_default_store(namespace, project)

    @classmethod
    def add(
        cls,
        namespace: str,
        project: str,
        data: Any,
        data_type: str = "text",
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """Add data to the appropriate memory store component.

        Args:
            namespace: Project namespace
            project: Project name
            data: The data to store
            data_type: Type of data (text, telemetry, chat, audio, node, edge)
            metadata: Optional metadata
            timestamp: Optional timestamp
            latitude: Latitude for spatial data
            longitude: Longitude for spatial data
            store_name: Memory store name (uses default if not specified)

        Returns:
            Dictionary with success status and store information
        """
        try:
            store = cls._get_store(namespace, project, store_name)

            result = store.add(
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
                "component_id": result.get("component_id"),
                "message": f"Data added to {result.get('store', 'memory')} store",
            }

        except Exception as e:
            logger.error(
                "Failed to add data to memory",
                namespace=namespace,
                project=project,
                store_name=store_name,
                error=str(e),
            )
            return {
                "success": False,
                "message": str(e),
            }

    @classmethod
    def query(
        cls,
        namespace: str,
        project: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        data_types: list[str] | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_m: float | None = None,
        graph_query: dict[str, Any] | None = None,
        recent_limit: int | None = None,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """Query unified context from memory store.

        Args:
            namespace: Project namespace
            project: Project name
            start_time: Start of time range
            end_time: End of time range
            data_types: Filter by data types
            latitude: Center latitude for spatial query
            longitude: Center longitude for spatial query
            radius_m: Radius in meters for spatial query
            graph_query: {"node_id": str, "direction": str} for graph traversal
            recent_limit: Limit for recent working memory query
            store_name: Memory store name (uses default if not specified)

        Returns:
            Dictionary with results and total count
        """
        try:
            store = cls._get_store(namespace, project, store_name)

            # Build query parameters
            time_range = None
            if start_time or end_time:
                time_range = {"start": start_time, "end": end_time}

            spatial = None
            if latitude is not None and longitude is not None and radius_m:
                spatial = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "radius_meters": radius_m,
                }

            recent = None
            if recent_limit:
                recent = {"limit": recent_limit}
                if data_types:
                    recent["data_type"] = data_types[0]

            results = store.query(
                time_range=time_range,
                data_types=data_types,
                graph_query=graph_query,
                spatial=spatial,
                recent=recent,
            )

            return {
                "success": True,
                "results": results,
                "total_count": len(results),
            }

        except Exception as e:
            logger.error(
                "Failed to query memory",
                namespace=namespace,
                project=project,
                store_name=store_name,
                error=str(e),
            )
            return {
                "success": False,
                "results": [],
                "total_count": 0,
                "error": str(e),
            }

    @classmethod
    def get_context(
        cls,
        namespace: str,
        project: str,
        recent_minutes: int = 10,
        include_graph: bool = True,
        include_working_memory: bool = True,
        limit: int = 100,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated context from all store components.

        Args:
            namespace: Project namespace
            project: Project name
            recent_minutes: How far back to look
            include_graph: Whether to include graph data
            include_working_memory: Whether to include working memory
            limit: Maximum records per store
            store_name: Memory store name (uses default if not specified)

        Returns:
            Dictionary with context from each store component
        """
        try:
            store = cls._get_store(namespace, project, store_name)

            context = store.get_context(
                recent_minutes=recent_minutes,
                include_graph=include_graph,
                include_working_memory=include_working_memory,
                limit=limit,
            )

            return {
                "success": True,
                **context,
            }

        except Exception as e:
            logger.error(
                "Failed to get memory context",
                namespace=namespace,
                project=project,
                store_name=store_name,
                error=str(e),
            )
            return {
                "success": False,
                "working_memory": [],
                "graph": [],
                "timeseries": [],
                "error": str(e),
            }

    @classmethod
    def delete(
        cls,
        namespace: str,
        project: str,
        uuid: str,
        store_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Delete a record by UUID (cascade delete via LinkageTable).

        Args:
            namespace: Project namespace
            project: Project name
            uuid: The concept UUID to delete
            store_name: Memory store name (uses default if not specified)

        Returns:
            Dictionary with deletion results, or None if not found
        """
        try:
            store = cls._get_store(namespace, project, store_name)

            result = store.delete(uuid)

            if result:
                deleted_from = result.get("deleted_from", [])
                return {
                    "success": True,
                    "uuid": uuid,
                    "deleted_from": deleted_from,
                    "message": f"Record deleted from {len(deleted_from)} store(s)",
                }
            else:
                return None

        except Exception as e:
            logger.error(
                "Failed to delete from memory",
                namespace=namespace,
                project=project,
                uuid=uuid,
                store_name=store_name,
                error=str(e),
            )
            return {
                "success": False,
                "uuid": uuid,
                "message": str(e),
            }

    @classmethod
    def clear_table(
        cls,
        namespace: str,
        project: str,
        table: str,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """Clear specific table or all tables.

        Args:
            namespace: Project namespace
            project: Project name
            table: Table to clear: working_memory, timeseries, graph, linkage, or all
            store_name: Memory store name (uses default if not specified)

        Returns:
            Dictionary with clear results
        """
        valid_tables = {"working_memory", "timeseries", "graph", "linkage", "all"}
        if table not in valid_tables:
            return {
                "success": False,
                "message": f"Invalid table: {table}. Valid options: {valid_tables}",
            }

        try:
            store = cls._get_store(namespace, project, store_name)

            cleared = {}

            if table == "working_memory" or table == "all":
                store.clear_working_memory()
                cleared["working_memory"] = True

            if table == "timeseries" or table == "all":
                if hasattr(store.timeseries_store, "clear"):
                    count = store.timeseries_store.clear()
                    cleared["timeseries"] = count
                else:
                    cleared["timeseries"] = 0

            if table == "graph" or table == "all":
                if hasattr(store.graph_store, "clear"):
                    result = store.graph_store.clear()
                    cleared["graph"] = result
                else:
                    cleared["graph"] = {"nodes_deleted": 0, "edges_deleted": 0}

            if table == "linkage" or table == "all":
                if hasattr(store.linkage_table, "clear"):
                    count = store.linkage_table.clear()
                    cleared["linkage"] = count
                else:
                    cleared["linkage"] = 0

            logger.info(
                "Cleared memory table",
                namespace=namespace,
                project=project,
                table=table,
                store_name=store_name,
                cleared=cleared,
            )

            return {
                "success": True,
                "table": table,
                "cleared": cleared,
            }

        except Exception as e:
            logger.error(
                "Failed to clear memory table",
                namespace=namespace,
                project=project,
                table=table,
                store_name=store_name,
                error=str(e),
            )
            return {
                "success": False,
                "table": table,
                "message": str(e),
            }

    @classmethod
    def consolidate(
        cls,
        namespace: str,
        project: str,
        use_llm: bool = False,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """Trigger memory consolidation cycle.

        Args:
            namespace: Project namespace
            project: Project name
            use_llm: Whether to use LLM for synthesis
            store_name: Memory store name (uses default if not specified)

        Returns:
            Dictionary with consolidation results
        """
        try:
            consolidator = MemoryStoreService.get_consolidator(
                namespace, project, store_name
            )

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
            logger.error(
                "Failed to consolidate memory",
                namespace=namespace,
                project=project,
                store_name=store_name,
                error=str(e),
            )
            return {
                "success": False,
                "message": str(e),
            }

    @classmethod
    def prune(
        cls,
        namespace: str,
        project: str,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """Prune expired records from working memory.

        Args:
            namespace: Project namespace
            project: Project name
            store_name: Memory store name (uses default if not specified)

        Returns:
            Dictionary with prune results
        """
        try:
            store = cls._get_store(namespace, project, store_name)

            pruned_count = store.prune_working_memory()

            # Get remaining count
            stats = store.get_stats()
            remaining = stats.get("working_memory", {}).get("total_records", 0)

            return {
                "success": True,
                "pruned_count": pruned_count,
                "remaining_count": remaining,
            }

        except Exception as e:
            logger.error(
                "Failed to prune memory",
                namespace=namespace,
                project=project,
                store_name=store_name,
                error=str(e),
            )
            return {
                "success": False,
                "pruned_count": 0,
                "message": str(e),
            }

    @classmethod
    def get_stats(
        cls,
        namespace: str,
        project: str,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """Get detailed storage statistics.

        Args:
            namespace: Project namespace
            project: Project name
            store_name: Memory store name (uses default if not specified)

        Returns:
            Dictionary with statistics from each store component
        """
        try:
            stats = MemoryStoreService.get_store_stats(namespace, project, store_name)

            return {
                "success": True,
                **stats,
            }

        except Exception as e:
            logger.error(
                "Failed to get memory stats",
                namespace=namespace,
                project=project,
                store_name=store_name,
                error=str(e),
            )
            return {
                "success": False,
                "working_memory": {"total_records": 0, "error": str(e)},
                "graph": {"node_count": 0, "edge_count": 0},
                "timeseries": {"record_count": 0},
                "linkage": {"total_links": 0},
            }
