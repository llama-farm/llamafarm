"""Working Memory - Short-term buffer with TTL.

Part of the Embedded Trinity Memory System:
- Vector Memory (ChromaDB) - Semantic search
- Time-Series Memory (DuckDB) - See duckdb_store.py
- Graph Memory (DuckDB) - See graph_store.py
- Working Memory (DuckDB) - This module (short-term buffer)

This module handles:
- Incoming data streams (chat, telemetry, audio)
- TTL-based expiration
- Auto-pruning when buffer exceeds max size
- Recent context retrieval for agents
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Short-term memory buffer with TTL-based expiration.

    Features:
    - Store incoming streams temporarily
    - Automatic expiration via TTL
    - Query recent records by type and time
    - Auto-prune when max size exceeded
    - Integration with the unified MemoryStore

    Configuration options:
        path: Path to the DuckDB database file
        ttl_seconds: Time-to-live for records (default: 3600 = 1 hour)
        max_size: Maximum records before auto-prune (default: 10000)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Working Memory.

        Args:
            config: Configuration dictionary with:
                - path: Database file path (required)
                - ttl_seconds: Record TTL in seconds (default: 3600)
                - max_size: Max records before prune (default: 10000)
        """
        config = config or {}
        self.db_path = config.get("path", ":memory:")
        self.ttl_seconds = config.get("ttl_seconds", 3600)  # 1 hour default
        self.max_size = config.get("max_size", 10000)

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
            logger.info(f"Connected to Working Memory at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise

    def _create_schema(self) -> None:
        """Create the working memory table schema."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS working_memory (
                id VARCHAR PRIMARY KEY,
                data_type VARCHAR NOT NULL,
                content TEXT NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT NOW(),
                expires_at TIMESTAMP NOT NULL
            )
        """)

        # Create indexes for efficient queries
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wm_type ON working_memory(data_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wm_expires ON working_memory(expires_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wm_created ON working_memory(created_at)"
        )

        logger.info("Created working memory schema")

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
    # Add Operations
    # ─────────────────────────────────────────────────────────────────────

    def add(
        self,
        data_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a record to working memory.

        Args:
            data_type: Type of data (e.g., "chat", "telemetry", "audio")
            content: Content string
            metadata: Optional metadata dict

        Returns:
            Record ID
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        record_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(seconds=self.ttl_seconds)
        metadata_json = json.dumps(metadata or {})

        try:
            self._conn.execute(
                """
                INSERT INTO working_memory (id, data_type, content, metadata, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [record_id, data_type, content, metadata_json, now, expires_at],
            )
            logger.debug(f"Added record to working memory: {data_type}")

            # Check if auto-prune needed
            self._check_auto_prune()

            return record_id
        except Exception as e:
            logger.error(f"Failed to add record: {e}")
            raise

    def add_batch(self, records: list[dict[str, Any]]) -> int:
        """Add multiple records efficiently.

        Args:
            records: List of record dicts with data_type, content, metadata

        Returns:
            Number of records added
        """
        if not records:
            return 0

        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        now = datetime.now()
        expires_at = now + timedelta(seconds=self.ttl_seconds)
        added = 0

        for record in records:
            try:
                record_id = str(uuid.uuid4())
                data_type = record.get("data_type", "unknown")
                content = record.get("content", "")
                metadata = json.dumps(record.get("metadata", {}))

                self._conn.execute(
                    """
                    INSERT INTO working_memory (id, data_type, content, metadata, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [record_id, data_type, content, metadata, now, expires_at],
                )
                added += 1
            except Exception as e:
                logger.warning(f"Failed to add batch record: {e}")
                continue

        # Check if auto-prune needed
        self._check_auto_prune()

        logger.debug(f"Added {added} records in batch")
        return added

    def _check_auto_prune(self) -> None:
        """Check if auto-prune is needed and run if so."""
        try:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM working_memory"
            ).fetchone()[0]

            if count > self.max_size:
                logger.info(f"Auto-prune triggered: {count} > {self.max_size}")
                self.prune()
        except Exception as e:
            logger.warning(f"Auto-prune check failed: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Query Operations
    # ─────────────────────────────────────────────────────────────────────

    def get_recent(
        self,
        limit: int = 100,
        minutes: int | None = None,
        seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent non-expired records.

        Args:
            limit: Maximum records to return
            minutes: Optional time window in minutes
            seconds: Optional time window in seconds

        Returns:
            List of record dicts
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        now = datetime.now()
        sql = """
            SELECT id, data_type, content, metadata, created_at, expires_at
            FROM working_memory
            WHERE expires_at > ?
        """
        params = [now]

        if minutes is not None:
            cutoff = now - timedelta(minutes=minutes)
            sql += " AND created_at >= ?"
            params.append(cutoff)
        elif seconds is not None:
            if seconds == 0:
                # Special case: 0 seconds means nothing
                return []
            cutoff = now - timedelta(seconds=seconds)
            sql += " AND created_at >= ?"
            params.append(cutoff)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        try:
            result = self._conn.execute(sql, params)
            rows = result.fetchall()
            return self._rows_to_dicts(rows)
        except Exception as e:
            logger.error(f"Failed to get recent records: {e}")
            return []

    def get_by_type(
        self,
        data_type: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get records by data type.

        Args:
            data_type: Type filter (e.g., "chat", "telemetry")
            limit: Maximum records to return

        Returns:
            List of matching record dicts
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        now = datetime.now()
        sql = """
            SELECT id, data_type, content, metadata, created_at, expires_at
            FROM working_memory
            WHERE data_type = ? AND expires_at > ?
            ORDER BY created_at DESC
            LIMIT ?
        """

        try:
            result = self._conn.execute(sql, [data_type, now, limit])
            rows = result.fetchall()
            return self._rows_to_dicts(rows)
        except Exception as e:
            logger.error(f"Failed to get records by type: {e}")
            return []

    def _rows_to_dicts(self, rows: list[tuple]) -> list[dict[str, Any]]:
        """Convert SQL rows to dictionaries."""
        records = []
        for row in rows:
            metadata = {}
            if row[3]:
                try:
                    metadata = json.loads(row[3])
                except (json.JSONDecodeError, TypeError):
                    pass

            records.append(
                {
                    "id": row[0],
                    "data_type": row[1],
                    "content": row[2],
                    "metadata": metadata,
                    "created_at": row[4],
                    "expires_at": row[5],
                }
            )
        return records

    # ─────────────────────────────────────────────────────────────────────
    # Prune Operations
    # ─────────────────────────────────────────────────────────────────────

    def prune(self) -> int:
        """Remove expired records.

        Returns:
            Number of records removed
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        now = datetime.now()

        try:
            # Count before delete
            count_result = self._conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE expires_at <= ?",
                [now],
            )
            count = count_result.fetchone()[0]

            # Delete expired
            self._conn.execute(
                "DELETE FROM working_memory WHERE expires_at <= ?",
                [now],
            )

            logger.info(f"Pruned {count} expired records")
            return count
        except Exception as e:
            logger.error(f"Failed to prune: {e}")
            return 0

    def clear(self) -> None:
        """Clear all records from working memory."""
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            self._conn.execute("DELETE FROM working_memory")
            logger.info("Cleared working memory")
        except Exception as e:
            logger.error(f"Failed to clear: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get working memory statistics.

        Returns:
            Dictionary with buffer stats
        """
        if not self.is_connected():
            return {"error": "Connection closed"}

        try:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM working_memory"
            ).fetchone()[0]

            # Get type counts
            type_result = self._conn.execute(
                "SELECT data_type, COUNT(*) FROM working_memory GROUP BY data_type"
            )
            type_counts = {row[0]: row[1] for row in type_result.fetchall()}

            # Get oldest and newest
            oldest = self._conn.execute(
                "SELECT MIN(created_at) FROM working_memory"
            ).fetchone()[0]
            newest = self._conn.execute(
                "SELECT MAX(created_at) FROM working_memory"
            ).fetchone()[0]

            return {
                "total_records": total,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "type_counts": type_counts,
                "oldest_record": oldest.isoformat() if oldest else None,
                "newest_record": newest.isoformat() if newest else None,
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
                logger.info("Working memory connection closed")
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
