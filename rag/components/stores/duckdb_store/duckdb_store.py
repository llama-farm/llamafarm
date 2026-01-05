"""DuckDB Store for time-series and spatial data.

Part of the Embedded Trinity Memory System:
- Vector Memory (ChromaDB) - Semantic search
- Time-Series Memory (DuckDB) - This module
- Graph Memory (DuckDB + DuckPGQ) - See graph_store.py

This store handles:
- High-velocity time-series data (biometrics, logs, telemetry)
- Spatial/geo queries using the spatial extension
- Fast aggregations and window functions
- Integration with the unified MemoryStore

Phase 26: Performance & Polish
- Added connection pooling for concurrent access
- Added batch insert optimizations
"""

import json
import logging
import queue
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Simple connection pool for DuckDB connections.

    Manages a pool of reusable database connections for concurrent access.
    Thread-safe with connection checkout/return semantics.
    """

    def __init__(
        self,
        db_path: str,
        pool_size: int = 5,
        timeout_seconds: float = 30.0,
    ):
        """Initialize connection pool.

        Args:
            db_path: Path to DuckDB database file
            pool_size: Maximum number of pooled connections
            timeout_seconds: Timeout for acquiring a connection
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout_seconds = timeout_seconds
        self._pool: queue.Queue[duckdb.DuckDBPyConnection] = queue.Queue(
            maxsize=pool_size
        )
        self._created_count = 0
        self._lock = threading.Lock()
        self._closed = False

        # Pre-create connections
        for _ in range(min(2, pool_size)):
            self._add_connection()

    def _add_connection(self) -> None:
        """Add a new connection to the pool."""
        with self._lock:
            if self._created_count < self.pool_size and not self._closed:
                conn = duckdb.connect(self.db_path)
                self._pool.put(conn)
                self._created_count += 1
                logger.debug(
                    f"Created connection #{self._created_count} for {self.db_path}"
                )

    @contextmanager
    def get_connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Get a connection from the pool.

        Yields:
            DuckDB connection

        Raises:
            RuntimeError: If pool is closed or timeout occurs
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        conn = None
        try:
            # Try to get from pool
            try:
                conn = self._pool.get(timeout=self.timeout_seconds)
            except queue.Empty as err:
                # Pool empty, try to create new connection
                with self._lock:
                    if self._created_count < self.pool_size:
                        conn = duckdb.connect(self.db_path)
                        self._created_count += 1
                    else:
                        raise RuntimeError("Connection pool exhausted") from err

            yield conn

        finally:
            # Return connection to pool
            if conn and not self._closed:
                try:
                    self._pool.put_nowait(conn)
                except queue.Full:
                    conn.close()

    def close(self) -> None:
        """Close all pooled connections."""
        self._closed = True
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break
        logger.debug(f"Closed connection pool for {self.db_path}")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self.pool_size,
            "created_count": self._created_count,
            "available": self._pool.qsize(),
            "closed": self._closed,
        }


class DuckDBStore:
    """DuckDB-based store for time-series and spatial data.

    Features:
    - Time-series data storage with efficient querying
    - Spatial queries using DuckDB spatial extension
    - Rolling aggregations with window functions
    - Batch insert performance with configurable batch sizes
    - Optional connection pooling for concurrent access

    Configuration options:
        path: Path to the DuckDB database file
        table_name: Name of the time-series table (default: "telemetry")
        extensions: List of extensions to load (e.g., ["spatial", "vss"])
        enable_spatial: Whether to enable spatial columns and queries
        use_pool: Whether to use connection pooling (default: False)
        pool_size: Size of connection pool (default: 5)
        batch_size: Batch size for bulk inserts (default: 1000)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize DuckDB store.

        Args:
            config: Configuration dictionary with:
                - path: Database file path (required)
                - table_name: Table name (default: "telemetry")
                - extensions: Extensions to load (default: [])
                - enable_spatial: Enable spatial features (default: False)
                - use_pool: Enable connection pooling (default: False)
                - pool_size: Connection pool size (default: 5)
                - batch_size: Batch size for inserts (default: 1000)
        """
        config = config or {}
        self.db_path = config.get("path", ":memory:")
        self.table_name = config.get("table_name", "telemetry")
        self.extensions = config.get("extensions", [])
        self.enable_spatial = config.get("enable_spatial", False)
        self.use_pool = config.get("use_pool", False)
        self.pool_size = config.get("pool_size", 5)
        self.batch_size = config.get("batch_size", 1000)

        self._conn: duckdb.DuckDBPyConnection | None = None
        self._pool: ConnectionPool | None = None
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
            if self.use_pool and self.db_path != ":memory:":
                self._pool = ConnectionPool(
                    db_path=self.db_path,
                    pool_size=self.pool_size,
                )
                # Get a connection for initialization
                with self._pool.get_connection() as conn:
                    self._conn = conn
                logger.info(
                    f"Connected to DuckDB at {self.db_path} (pooled, size={self.pool_size})"
                )
            else:
                self._conn = duckdb.connect(self.db_path)
                logger.info(f"Connected to DuckDB at {self.db_path}")
            self._closed = False
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise

    @contextmanager
    def _get_conn(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Get a connection (from pool or direct).

        Yields:
            DuckDB connection
        """
        if self._pool:
            with self._pool.get_connection() as conn:
                yield conn
        elif self._conn:
            yield self._conn
        else:
            raise RuntimeError("No database connection available")

    def _load_extensions(self) -> None:
        """Load requested DuckDB extensions."""
        for ext in self.extensions:
            try:
                self._conn.execute(f"INSTALL {ext}")
                self._conn.execute(f"LOAD {ext}")
                logger.info(f"Loaded extension: {ext}")
            except Exception as e:
                # Some extensions may not be available on all platforms
                logger.warning(f"Could not load extension {ext}: {e}")

    def _create_schema(self) -> None:
        """Create the time-series table schema."""
        # Base schema for time-series data
        schema_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id VARCHAR PRIMARY KEY,
                ts TIMESTAMP NOT NULL DEFAULT NOW(),
                source VARCHAR NOT NULL,
                data JSON,
                metadata JSON,
                created_at TIMESTAMP DEFAULT NOW()
        """

        # Add spatial column if enabled
        if self.enable_spatial and "spatial" in self.extensions:
            schema_sql += ",\n                location GEOMETRY"

        schema_sql += "\n            )"

        self._conn.execute(schema_sql)

        # Create index on timestamp for efficient time-range queries
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_ts
            ON {self.table_name}(ts)
            """
        )

        # Create index on source for efficient filtering
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source
            ON {self.table_name}(source)
            """
        )

        logger.info(f"Created schema for table: {self.table_name}")

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

    def add_records(self, records: list[dict[str, Any]]) -> int:
        """Add time-series records to the store.

        Uses batch inserts for better performance when inserting many records.

        Args:
            records: List of record dictionaries with:
                - source: Source identifier (required)
                - ts: Timestamp (optional, defaults to now)
                - data: Data payload as dict
                - metadata: Additional metadata as dict
                - location: {lat, lon} for spatial data (optional)

        Returns:
            Number of records inserted
        """
        if not records:
            return 0

        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        # Use batch insert for large record sets
        if len(records) > self.batch_size:
            return self._batch_insert(records)

        inserted = 0
        spatial_enabled = self.enable_spatial and "spatial" in self.extensions

        with self._get_conn() as conn:
            for record in records:
                try:
                    record_id = str(uuid.uuid4())
                    source = record.get("source", "unknown")
                    ts = record.get("ts", datetime.now())
                    data = json.dumps(record.get("data", {}))
                    metadata = json.dumps(record.get("metadata", {}))

                    if spatial_enabled and "location" in record:
                        loc = record["location"]
                        lat = loc.get("lat", 0)
                        lon = loc.get("lon", 0)

                        conn.execute(
                            f"""
                            INSERT INTO {self.table_name} (id, ts, source, data, metadata, location)
                            VALUES (?, ?, ?, ?, ?, ST_Point(?, ?))
                            """,
                            [record_id, ts, source, data, metadata, lon, lat],
                        )
                    else:
                        if spatial_enabled:
                            conn.execute(
                                f"""
                                INSERT INTO {self.table_name} (id, ts, source, data, metadata, location)
                                VALUES (?, ?, ?, ?, ?, NULL)
                                """,
                                [record_id, ts, source, data, metadata],
                            )
                        else:
                            conn.execute(
                                f"""
                                INSERT INTO {self.table_name} (id, ts, source, data, metadata)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                [record_id, ts, source, data, metadata],
                            )

                    inserted += 1

                except Exception as e:
                    logger.warning(f"Failed to insert record: {e}")
                    continue

        logger.debug(f"Inserted {inserted} records")
        return inserted

    def _batch_insert(self, records: list[dict[str, Any]]) -> int:
        """Batch insert records for better performance.

        Args:
            records: List of records to insert

        Returns:
            Number of records inserted
        """
        spatial_enabled = self.enable_spatial and "spatial" in self.extensions
        total_inserted = 0

        # Process in batches
        for i in range(0, len(records), self.batch_size):
            batch = records[i : i + self.batch_size]

            # Prepare batch data
            batch_data = []
            for record in batch:
                record_id = str(uuid.uuid4())
                source = record.get("source", "unknown")
                ts = record.get("ts", datetime.now())
                data = json.dumps(record.get("data", {}))
                metadata = json.dumps(record.get("metadata", {}))

                if spatial_enabled:
                    loc = record.get("location", {})
                    lat = loc.get("lat") if loc else None
                    lon = loc.get("lon") if loc else None
                    batch_data.append((record_id, ts, source, data, metadata, lon, lat))
                else:
                    batch_data.append((record_id, ts, source, data, metadata))

            try:
                with self._get_conn() as conn:
                    if spatial_enabled:
                        # Use COPY for batch insert (faster)
                        conn.executemany(
                            f"""
                            INSERT INTO {self.table_name} (id, ts, source, data, metadata, location)
                            VALUES (?, ?, ?, ?, ?, CASE WHEN ? IS NOT NULL AND ? IS NOT NULL
                                THEN ST_Point(?, ?) ELSE NULL END)
                            """,
                            [
                                (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[5], r[6])
                                for r in batch_data
                            ],
                        )
                    else:
                        conn.executemany(
                            f"""
                            INSERT INTO {self.table_name} (id, ts, source, data, metadata)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            batch_data,
                        )
                    total_inserted += len(batch_data)

            except Exception as e:
                logger.warning(
                    f"Batch insert failed: {e}, falling back to individual inserts"
                )
                # Fall back to individual inserts
                for record in batch:
                    try:
                        self.add_records([record])
                        total_inserted += 1
                    except Exception:
                        continue

        logger.debug(f"Batch inserted {total_inserted} records")
        return total_inserted

    def query_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        source: str | None = None,
        aggregation: str | None = None,
        window_size: int = 10,
    ) -> list[dict[str, Any]]:
        """Query records within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            source: Optional source filter
            aggregation: Optional aggregation type ("rolling_avg", "sum", etc.)
            window_size: Window size for rolling aggregations

        Returns:
            List of matching records as dictionaries
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        # Build query
        params = [start_time, end_time]

        if aggregation == "rolling_avg":
            sql = f"""
                SELECT
                    id,
                    ts,
                    source,
                    data,
                    metadata,
                    AVG(CAST(JSON_EXTRACT_STRING(data, 'value') AS DOUBLE)) OVER (
                        PARTITION BY source
                        ORDER BY ts
                        ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW
                    ) as rolling_avg
                FROM {self.table_name}
                WHERE ts >= ? AND ts <= ?
            """
        else:
            sql = f"""
                SELECT id, ts, source, data, metadata
                FROM {self.table_name}
                WHERE ts >= ? AND ts <= ?
            """

        if source:
            sql += " AND source = ?"
            params.append(source)

        sql += " ORDER BY ts DESC"

        try:
            result = self._conn.execute(sql, params)
            rows = result.fetchall()
            columns = [desc[0] for desc in result.description]

            records = []
            for row in rows:
                record = dict(zip(columns, row))
                # Parse JSON fields
                if "data" in record and record["data"]:
                    try:
                        record["data"] = json.loads(record["data"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                if "metadata" in record and record["metadata"]:
                    try:
                        record["metadata"] = json.loads(record["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                records.append(record)

            return records

        except Exception as e:
            logger.error(f"Query error: {e}")
            return []

    def query_spatial(
        self,
        center_lat: float,
        center_lon: float,
        radius_meters: float,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Query records within a spatial radius.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_meters: Radius in meters
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of matching records with distance_meters field
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        if not self.enable_spatial or "spatial" not in self.extensions:
            logger.warning("Spatial queries not enabled")
            return []

        # Convert radius from meters to degrees (approximate)
        # 1 degree ~ 111km at equator
        radius_degrees = radius_meters / 111000.0

        params = [center_lon, center_lat, center_lon, center_lat, radius_degrees]

        sql = f"""
            SELECT
                id,
                ts,
                source,
                data,
                metadata,
                ST_Distance(
                    location,
                    ST_Point(?, ?)
                ) * 111000.0 as distance_meters
            FROM {self.table_name}
            WHERE location IS NOT NULL
            AND ST_DWithin(
                location,
                ST_Point(?, ?),
                ?
            )
        """

        if start_time and end_time:
            sql += " AND ts >= ? AND ts <= ?"
            params.extend([start_time, end_time])

        sql += " ORDER BY distance_meters ASC"

        try:
            result = self._conn.execute(sql, params)
            rows = result.fetchall()
            columns = [desc[0] for desc in result.description]

            records = []
            for row in rows:
                record = dict(zip(columns, row))
                # Parse JSON fields
                if "data" in record and record["data"]:
                    try:
                        record["data"] = json.loads(record["data"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                if "metadata" in record and record["metadata"]:
                    try:
                        record["metadata"] = json.loads(record["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                records.append(record)

            return records

        except Exception as e:
            logger.error(f"Spatial query error: {e}")
            return []

    def delete_records(self, record_ids: list[str]) -> int:
        """Delete records by ID.

        Args:
            record_ids: List of record IDs to delete

        Returns:
            Number of records deleted
        """
        if not record_ids:
            return 0

        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            # Get count before deletion
            placeholders = ", ".join(["?" for _ in record_ids])
            count_result = self._conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE id IN ({placeholders})",
                record_ids,
            )
            count = count_result.fetchone()[0]

            # Delete records
            self._conn.execute(
                f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",
                record_ids,
            )

            logger.debug(f"Deleted {count} records")
            return count

        except Exception as e:
            logger.error(f"Delete error: {e}")
            return 0

    def delete_older_than(self, cutoff: datetime) -> int:
        """Delete records older than a timestamp.

        Args:
            cutoff: Delete records with ts < cutoff

        Returns:
            Number of records deleted
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            # Get count before deletion
            count_result = self._conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE ts < ?",
                [cutoff],
            )
            count = count_result.fetchone()[0]

            # Delete records
            self._conn.execute(
                f"DELETE FROM {self.table_name} WHERE ts < ?",
                [cutoff],
            )

            logger.debug(f"Deleted {count} records older than {cutoff}")
            return count

        except Exception as e:
            logger.error(f"Delete error: {e}")
            return 0

    def clear(self) -> int:
        """Clear all records from the table.

        Returns:
            Number of records deleted
        """
        if not self.is_connected():
            raise RuntimeError("Database connection is closed")

        try:
            # Get count before deletion
            count = self._conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()[0]

            # Delete all records
            self._conn.execute(f"DELETE FROM {self.table_name}")

            logger.info(f"Cleared {count} records from {self.table_name}")
            return count

        except Exception as e:
            logger.error(f"Clear error: {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with record counts and storage info
        """
        if not self.is_connected():
            return {"error": "Connection closed"}

        try:
            count = self._conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()[0]

            oldest = self._conn.execute(
                f"SELECT MIN(ts) FROM {self.table_name}"
            ).fetchone()[0]

            newest = self._conn.execute(
                f"SELECT MAX(ts) FROM {self.table_name}"
            ).fetchone()[0]

            sources = self._conn.execute(
                f"SELECT COUNT(DISTINCT source) FROM {self.table_name}"
            ).fetchone()[0]

            return {
                "table_name": self.table_name,
                "record_count": count,
                "oldest_record": oldest.isoformat() if oldest else None,
                "newest_record": newest.isoformat() if newest else None,
                "unique_sources": sources,
                "db_path": self.db_path,
            }

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"error": str(e)}

    def close(self) -> None:
        """Close the database connection."""
        if self._closed:
            return

        try:
            if self._pool:
                self._pool.close()
                self._pool = None
            elif self._conn:
                self._conn.close()

            self._closed = True
            logger.info("DuckDB connection closed")
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
