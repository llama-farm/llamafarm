"""Graph Store for entity relationships and knowledge graph.

Part of the Embedded Trinity Memory System:
- Vector Memory (ChromaDB) - Semantic search
- Time-Series Memory (DuckDB) - See duckdb_store.py
- Graph Memory (DuckDB) - This module

This store handles:
- Entity nodes with types and properties
- Relationships between entities
- Path finding and graph traversal
- Integration with the unified MemoryStore

Note: Uses pure SQL for graph operations. DuckPGQ extension is optional
and provides enhanced graph query capabilities when available.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


class GraphStore:
    """DuckDB-based graph store for entity relationships.

    Features:
    - Node and edge storage with properties
    - Graph traversal (neighbors, paths)
    - Relationship filtering
    - Integration with DuckDBStore for shared connection

    Configuration options:
        path: Path to the DuckDB database file
        use_duckpgq: Whether to use DuckPGQ extension (default: False)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Graph store.

        Args:
            config: Configuration dictionary with:
                - path: Database file path (required)
                - use_duckpgq: Enable DuckPGQ extension (default: False)
        """
        config = config or {}
        self.db_path = config.get("path", ":memory:")
        self.use_duckpgq = config.get("use_duckpgq", False)

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
        self._load_extensions()
        self._create_schema()

    def _connect(self) -> None:
        """Establish database connection."""
        try:
            self._conn = duckdb.connect(self.db_path)
            self._closed = False
            logger.info(f"Connected to DuckDB graph store at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise

    def _load_extensions(self) -> None:
        """Load optional DuckPGQ extension."""
        if self.use_duckpgq:
            try:
                self._conn.execute("INSTALL duckpgq")
                self._conn.execute("LOAD duckpgq")
                logger.info("Loaded DuckPGQ extension")
            except Exception as e:
                logger.warning(f"Could not load DuckPGQ extension: {e}")
                self.use_duckpgq = False

    def _create_schema(self) -> None:
        """Create the graph schema (nodes and edges tables)."""
        # Nodes table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                node_type VARCHAR NOT NULL,
                properties JSON,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Edges table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id VARCHAR PRIMARY KEY,
                source_id VARCHAR NOT NULL,
                target_id VARCHAR NOT NULL,
                relationship VARCHAR NOT NULL,
                weight DOUBLE DEFAULT 1.0,
                properties JSON,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Create indexes for efficient lookups
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(relationship)"
        )

        logger.info("Created graph schema")

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
    # Node Operations
    # ─────────────────────────────────────────────────────────────────────

    def add_node(
        self,
        name: str,
        node_type: str,
        node_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Add a node to the graph.

        Args:
            name: Node name/label
            node_type: Type of node (e.g., "person", "location", "event")
            node_id: Optional custom ID (auto-generated if not provided)
            properties: Additional properties as dict

        Returns:
            Node ID
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        node_id = node_id or str(uuid.uuid4())
        props_json = json.dumps(properties or {})

        try:
            self._conn.execute(
                """
                INSERT INTO nodes (id, name, node_type, properties)
                VALUES (?, ?, ?, ?)
                """,
                [node_id, name, node_type, props_json],
            )
            logger.debug(f"Added node: {name} ({node_type})")
            return node_id
        except Exception as e:
            logger.error(f"Failed to add node: {e}")
            raise

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            Node dict or None if not found
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            result = self._conn.execute(
                "SELECT id, name, node_type, properties, created_at FROM nodes WHERE id = ?",
                [node_id],
            )
            row = result.fetchone()

            if not row:
                return None

            properties = {}
            if row[3]:
                try:
                    properties = json.loads(row[3])
                except (json.JSONDecodeError, TypeError):
                    pass

            return {
                "id": row[0],
                "name": row[1],
                "node_type": row[2],
                "properties": properties,
                "created_at": row[4],
            }
        except Exception as e:
            logger.error(f"Failed to get node: {e}")
            return None

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its connected edges.

        Args:
            node_id: Node ID to delete

        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            # Delete connected edges first
            self._conn.execute(
                "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                [node_id, node_id],
            )

            # Delete the node
            self._conn.execute(
                "DELETE FROM nodes WHERE id = ?",
                [node_id],
            )

            return True
        except Exception as e:
            logger.error(f"Failed to delete node: {e}")
            return False

    def find_nodes(
        self,
        node_type: str | None = None,
        name_contains: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find nodes matching criteria.

        Args:
            node_type: Filter by node type
            name_contains: Filter by name substring
            limit: Maximum results

        Returns:
            List of matching node dicts
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        sql = "SELECT id, name, node_type, properties, created_at FROM nodes WHERE 1=1"
        params = []

        if node_type:
            sql += " AND node_type = ?"
            params.append(node_type)

        if name_contains:
            # Escape LIKE special characters to prevent pattern injection
            escaped = (
                name_contains.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            sql += " AND name LIKE ?"
            params.append(f"%{escaped}%")

        sql += " LIMIT ?"
        params.append(limit)

        try:
            result = self._conn.execute(sql, params)
            rows = result.fetchall()

            nodes = []
            for row in rows:
                properties = {}
                if row[3]:
                    try:
                        properties = json.loads(row[3])
                    except (json.JSONDecodeError, TypeError):
                        pass

                nodes.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "node_type": row[2],
                        "properties": properties,
                        "created_at": row[4],
                    }
                )

            return nodes
        except Exception as e:
            logger.error(f"Failed to find nodes: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────
    # Edge Operations
    # ─────────────────────────────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        weight: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> str | None:
        """Add an edge between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship: Relationship type
            weight: Edge weight (default: 1.0)
            properties: Additional properties

        Returns:
            Edge ID or None if failed
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        edge_id = str(uuid.uuid4())
        props_json = json.dumps(properties or {})

        try:
            self._conn.execute(
                """
                INSERT INTO edges (id, source_id, target_id, relationship, weight, properties)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [edge_id, source_id, target_id, relationship, weight, props_json],
            )
            logger.debug(f"Added edge: {source_id} -[{relationship}]-> {target_id}")
            return edge_id
        except Exception as e:
            logger.error(f"Failed to add edge: {e}")
            return None

    def get_edges(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        relationship: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get edges matching criteria.

        Args:
            source_id: Filter by source node
            target_id: Filter by target node
            relationship: Filter by relationship type

        Returns:
            List of matching edge dicts
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        sql = """
            SELECT id, source_id, target_id, relationship, weight, properties, created_at
            FROM edges WHERE 1=1
        """
        params = []

        if source_id:
            sql += " AND source_id = ?"
            params.append(source_id)

        if target_id:
            sql += " AND target_id = ?"
            params.append(target_id)

        if relationship:
            sql += " AND relationship = ?"
            params.append(relationship)

        try:
            result = self._conn.execute(sql, params)
            rows = result.fetchall()

            edges = []
            for row in rows:
                properties = {}
                if row[5]:
                    try:
                        properties = json.loads(row[5])
                    except (json.JSONDecodeError, TypeError):
                        pass

                edges.append(
                    {
                        "id": row[0],
                        "source_id": row[1],
                        "target_id": row[2],
                        "relationship": row[3],
                        "weight": row[4],
                        "properties": properties,
                        "created_at": row[6],
                    }
                )

            return edges
        except Exception as e:
            logger.error(f"Failed to get edges: {e}")
            return []

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by ID.

        Args:
            edge_id: Edge ID to delete

        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            self._conn.execute("DELETE FROM edges WHERE id = ?", [edge_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete edge: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────
    # Graph Traversal
    # ─────────────────────────────────────────────────────────────────────

    def find_neighbors(
        self,
        node_id: str,
        direction: str = "outgoing",
        relationship: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find neighboring nodes.

        Args:
            node_id: Starting node ID
            direction: "outgoing", "incoming", or "both"
            relationship: Optional relationship filter

        Returns:
            List of neighbor node dicts
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        neighbor_ids = set()

        try:
            # Outgoing edges (node_id -> neighbor)
            if direction in ("outgoing", "both"):
                sql = "SELECT target_id FROM edges WHERE source_id = ?"
                params = [node_id]
                if relationship:
                    sql += " AND relationship = ?"
                    params.append(relationship)

                result = self._conn.execute(sql, params)
                for row in result.fetchall():
                    neighbor_ids.add(row[0])

            # Incoming edges (neighbor -> node_id)
            if direction in ("incoming", "both"):
                sql = "SELECT source_id FROM edges WHERE target_id = ?"
                params = [node_id]
                if relationship:
                    sql += " AND relationship = ?"
                    params.append(relationship)

                result = self._conn.execute(sql, params)
                for row in result.fetchall():
                    neighbor_ids.add(row[0])

            # Get full node info for neighbors
            neighbors = []
            for nid in neighbor_ids:
                node = self.get_node(nid)
                if node:
                    neighbors.append(node)

            return neighbors

        except Exception as e:
            logger.error(f"Failed to find neighbors: {e}")
            return []

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        relationship: str | None = None,
    ) -> list[list[str]]:
        """Find paths between two nodes using BFS.

        Args:
            start_id: Starting node ID
            end_id: Target node ID
            max_depth: Maximum path length
            relationship: Optional relationship filter

        Returns:
            List of paths (each path is list of node IDs)
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        if start_id == end_id:
            return [[start_id]]

        # BFS to find paths
        queue = [[start_id]]  # Queue of paths
        found_paths = []
        visited_at_depth = {}  # Track when nodes were first visited

        try:
            while queue:
                path = queue.pop(0)
                current = path[-1]
                current_depth = len(path)

                if current_depth > max_depth:
                    continue

                # Check if we've visited this node at a shorter depth
                if (
                    current in visited_at_depth
                    and visited_at_depth[current] < current_depth
                ):
                    continue
                visited_at_depth[current] = current_depth

                # Get outgoing edges
                sql = "SELECT target_id FROM edges WHERE source_id = ?"
                params = [current]
                if relationship:
                    sql += " AND relationship = ?"
                    params.append(relationship)

                result = self._conn.execute(sql, params)

                for row in result.fetchall():
                    next_node = row[0]

                    # Avoid cycles within the same path
                    if next_node in path:
                        continue

                    new_path = path + [next_node]

                    if next_node == end_id:
                        found_paths.append(new_path)
                    elif len(new_path) < max_depth:
                        queue.append(new_path)

            # Sort by path length (shortest first)
            found_paths.sort(key=len)
            return found_paths

        except Exception as e:
            logger.error(f"Failed to find path: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────

    def clear(self) -> dict[str, int]:
        """Clear all nodes and edges from the graph.

        Returns:
            Dictionary with counts of deleted nodes and edges
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            # Get counts before deletion
            node_count = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            edge_count = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

            # Delete all edges first (referential integrity)
            self._conn.execute("DELETE FROM edges")
            # Delete all nodes
            self._conn.execute("DELETE FROM nodes")

            logger.info(f"Cleared graph: {node_count} nodes, {edge_count} edges")
            return {"nodes_deleted": node_count, "edges_deleted": edge_count}

        except Exception as e:
            logger.error(f"Clear error: {e}")
            return {"nodes_deleted": 0, "edges_deleted": 0}

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary with node/edge counts and type distributions
        """
        if not self.is_connected():
            return {"error": "Connection closed"}

        try:
            node_count = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            edge_count = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

            # Get node type distribution
            type_result = self._conn.execute(
                "SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type"
            )
            node_types = {row[0]: row[1] for row in type_result.fetchall()}

            # Get relationship type distribution
            rel_result = self._conn.execute(
                "SELECT relationship, COUNT(*) FROM edges GROUP BY relationship"
            )
            relationship_types = {row[0]: row[1] for row in rel_result.fetchall()}

            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "node_types": node_types,
                "relationship_types": relationship_types,
                "db_path": self.db_path,
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
                logger.info("Graph store connection closed")
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
