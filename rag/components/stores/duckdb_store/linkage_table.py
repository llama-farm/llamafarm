"""Linkage Table - Cross-database record linking for cascade operations.

Part of the Embedded Trinity Memory System:
- Vector Memory (ChromaDB) - Semantic search
- Time-Series Memory (DuckDB) - See duckdb_store.py
- Graph Memory (DuckDB) - See graph_store.py
- Working Memory (DuckDB) - See working_memory.py
- Linkage Table (DuckDB) - This module (cross-DB linking)

This module handles:
- UUID -> {vector_id, graph_node_id, timeseries_row_id} mapping
- Reverse lookups (find UUID from any component ID)
- Cascade delete support (unlink and get all IDs)
- Cross-database consistency
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


class LinkageTable:
    """Cross-database record linking table.

    Features:
    - Map concept UUIDs to IDs in each store (vector, graph, timeseries)
    - Reverse lookup: find concept UUID from any component ID
    - Support cascade deletes by returning all linked IDs
    - Track creation timestamps

    Configuration options:
        path: Path to the DuckDB database file
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Linkage Table.

        Args:
            config: Configuration dictionary with:
                - path: Database file path (required)
        """
        config = config or {}
        self.db_path = config.get("path", ":memory:")

        self._conn: duckdb.DuckDBPyConnection | None = None
        self._closed = False

        # Ensure directory exists for file-based databases
        if self.db_path != ":memory:":
            db_dir = Path(self.db_path).parent
            if not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise ValueError(
                        f"Cannot create directory for database path: {self.db_path}. Error: {e}"
                    ) from e

        # Initialize connection and schema
        self._connect()
        self._create_schema()

    def _connect(self) -> None:
        """Establish database connection."""
        try:
            self._conn = duckdb.connect(self.db_path)
            self._closed = False
            logger.info(f"Connected to Linkage Table at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise

    def _create_schema(self) -> None:
        """Create the linkage table schema."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS linkage (
                uuid VARCHAR PRIMARY KEY,
                vector_id VARCHAR,
                graph_node_id VARCHAR,
                timeseries_row_id VARCHAR,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Create indexes for reverse lookups
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_linkage_vector ON linkage(vector_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_linkage_graph ON linkage(graph_node_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_linkage_timeseries ON linkage(timeseries_row_id)"
        )

        logger.info("Created linkage table schema")

    def is_connected(self) -> bool:
        """Check if the database connection is active."""
        return self._conn is not None and not self._closed

    def execute(self, sql: str, params: list | None = None) -> list[tuple]:
        """Execute a raw SQL query.

        Args:
            sql: SQL query string
            params: Query parameters

        Returns:
            List of result tuples
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            if params:
                result = self._conn.execute(sql, params)
            else:
                result = self._conn.execute(sql)
            return result.fetchall()
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────
    # Link Operations
    # ─────────────────────────────────────────────────────────────────────

    def link(
        self,
        concept_uuid: str | None = None,
        vector_id: str | None = None,
        graph_node_id: str | None = None,
        timeseries_row_id: str | None = None,
    ) -> str:
        """Create or update a linkage mapping.

        Args:
            concept_uuid: Optional custom UUID (auto-generated if not provided)
            vector_id: ID in vector store (ChromaDB)
            graph_node_id: Node ID in graph store
            timeseries_row_id: Row ID in timeseries store

        Returns:
            The concept UUID (provided or generated)
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        # Generate UUID if not provided
        if concept_uuid is None:
            concept_uuid = str(uuid.uuid4())

        now = datetime.now()

        try:
            # Check if UUID already exists
            existing = self._conn.execute(
                "SELECT uuid FROM linkage WHERE uuid = ?", [concept_uuid]
            ).fetchone()

            if existing:
                # Update existing record - merge IDs (don't overwrite with None)
                current = self.get_links(concept_uuid)

                # Use new values if provided, otherwise keep existing
                final_vector = (
                    vector_id if vector_id is not None else current.get("vector_id")
                )
                final_graph = (
                    graph_node_id
                    if graph_node_id is not None
                    else current.get("graph_node_id")
                )
                final_ts = (
                    timeseries_row_id
                    if timeseries_row_id is not None
                    else current.get("timeseries_row_id")
                )

                self._conn.execute(
                    """
                    UPDATE linkage
                    SET vector_id = ?, graph_node_id = ?, timeseries_row_id = ?
                    WHERE uuid = ?
                    """,
                    [final_vector, final_graph, final_ts, concept_uuid],
                )
                logger.debug(f"Updated linkage: {concept_uuid}")
            else:
                # Insert new record
                self._conn.execute(
                    """
                    INSERT INTO linkage (uuid, vector_id, graph_node_id, timeseries_row_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [concept_uuid, vector_id, graph_node_id, timeseries_row_id, now],
                )
                logger.debug(f"Created linkage: {concept_uuid}")

            return concept_uuid
        except Exception as e:
            logger.error(f"Failed to create/update linkage: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────
    # Query Operations
    # ─────────────────────────────────────────────────────────────────────

    def get_links(self, concept_uuid: str) -> dict[str, Any] | None:
        """Get all linked IDs for a concept UUID.

        Args:
            concept_uuid: The concept UUID to look up

        Returns:
            Dictionary with uuid, vector_id, graph_node_id, timeseries_row_id
            or None if not found
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            result = self._conn.execute(
                """
                SELECT uuid, vector_id, graph_node_id, timeseries_row_id, created_at
                FROM linkage
                WHERE uuid = ?
                """,
                [concept_uuid],
            ).fetchone()

            if result is None:
                return None

            return {
                "uuid": result[0],
                "vector_id": result[1],
                "graph_node_id": result[2],
                "timeseries_row_id": result[3],
                "created_at": result[4],
            }
        except Exception as e:
            logger.error(f"Failed to get links: {e}")
            raise

    def find_by_any_id(
        self,
        vector_id: str | None = None,
        graph_node_id: str | None = None,
        timeseries_row_id: str | None = None,
    ) -> str | None:
        """Find concept UUID from any component ID.

        Args:
            vector_id: Vector store ID to search for
            graph_node_id: Graph node ID to search for
            timeseries_row_id: Timeseries row ID to search for

        Returns:
            The concept UUID if found, None otherwise
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            if vector_id is not None:
                result = self._conn.execute(
                    "SELECT uuid FROM linkage WHERE vector_id = ?", [vector_id]
                ).fetchone()
            elif graph_node_id is not None:
                result = self._conn.execute(
                    "SELECT uuid FROM linkage WHERE graph_node_id = ?", [graph_node_id]
                ).fetchone()
            elif timeseries_row_id is not None:
                result = self._conn.execute(
                    "SELECT uuid FROM linkage WHERE timeseries_row_id = ?",
                    [timeseries_row_id],
                ).fetchone()
            else:
                return None

            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to find by ID: {e}")
            raise

    def list_all(self) -> list[dict[str, Any]]:
        """List all linkage records.

        Returns:
            List of linkage dictionaries
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            result = self._conn.execute(
                """
                SELECT uuid, vector_id, graph_node_id, timeseries_row_id, created_at
                FROM linkage
                ORDER BY created_at DESC
                """
            ).fetchall()

            return [
                {
                    "uuid": row[0],
                    "vector_id": row[1],
                    "graph_node_id": row[2],
                    "timeseries_row_id": row[3],
                    "created_at": row[4],
                }
                for row in result
            ]
        except Exception as e:
            logger.error(f"Failed to list all: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────
    # Unlink Operations
    # ─────────────────────────────────────────────────────────────────────

    def unlink(self, concept_uuid: str) -> bool:
        """Remove a linkage mapping.

        Args:
            concept_uuid: The concept UUID to remove

        Returns:
            True if removed, False if not found
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            # Check if exists first
            existing = self._conn.execute(
                "SELECT uuid FROM linkage WHERE uuid = ?", [concept_uuid]
            ).fetchone()

            if existing is None:
                return False

            self._conn.execute("DELETE FROM linkage WHERE uuid = ?", [concept_uuid])
            logger.debug(f"Unlinked: {concept_uuid}")
            return True
        except Exception as e:
            logger.error(f"Failed to unlink: {e}")
            raise

    def unlink_and_get_ids(self, concept_uuid: str) -> dict[str, str] | None:
        """Remove a linkage and return the component IDs for cascade delete.

        Args:
            concept_uuid: The concept UUID to remove

        Returns:
            Dictionary with vector_id, graph_node_id, timeseries_row_id
            or None if not found
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            # Get current links first
            links = self.get_links(concept_uuid)

            if links is None:
                return None

            # Delete the record
            self._conn.execute("DELETE FROM linkage WHERE uuid = ?", [concept_uuid])
            logger.debug(f"Unlinked and retrieved IDs: {concept_uuid}")

            return {
                "vector_id": links.get("vector_id"),
                "graph_node_id": links.get("graph_node_id"),
                "timeseries_row_id": links.get("timeseries_row_id"),
            }
        except Exception as e:
            logger.error(f"Failed to unlink and get IDs: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────

    def clear(self) -> int:
        """Clear all linkage records.

        Returns:
            Number of records deleted
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            # Get count before deletion
            count = self._conn.execute("SELECT COUNT(*) FROM linkage").fetchone()[0]

            # Delete all records
            self._conn.execute("DELETE FROM linkage")

            logger.info(f"Cleared {count} linkage records")
            return count

        except Exception as e:
            logger.error(f"Clear error: {e}")
            return 0

    def get_stats(self) -> dict[str, int]:
        """Get linkage table statistics.

        Returns:
            Dictionary with counts by link type
        """
        if not self.is_connected():
            return {"error": "Connection closed"}

        try:
            total = self._conn.execute("SELECT COUNT(*) FROM linkage").fetchone()[0]

            with_vector = self._conn.execute(
                "SELECT COUNT(*) FROM linkage WHERE vector_id IS NOT NULL"
            ).fetchone()[0]

            with_graph = self._conn.execute(
                "SELECT COUNT(*) FROM linkage WHERE graph_node_id IS NOT NULL"
            ).fetchone()[0]

            with_timeseries = self._conn.execute(
                "SELECT COUNT(*) FROM linkage WHERE timeseries_row_id IS NOT NULL"
            ).fetchone()[0]

            return {
                "total_links": total,
                "links_with_vector": with_vector,
                "links_with_graph": with_graph,
                "links_with_timeseries": with_timeseries,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._closed:
            try:
                self._conn.close()
                self._closed = True
                logger.info("Linkage table connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()
