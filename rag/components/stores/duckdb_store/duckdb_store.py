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
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


class DuckDBStore:
    """DuckDB-based store for time-series and spatial data.

    Features:
    - Time-series data storage with efficient querying
    - Spatial queries using DuckDB spatial extension
    - Rolling aggregations with window functions
    - Batch insert performance
    - Connection pooling for concurrent access

    Configuration options:
        path: Path to the DuckDB database file
        table_name: Name of the time-series table (default: "telemetry")
        extensions: List of extensions to load (e.g., ["spatial", "vss"])
        enable_spatial: Whether to enable spatial columns and queries
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize DuckDB store.

        Args:
            config: Configuration dictionary with:
                - path: Database file path (required)
                - table_name: Table name (default: "telemetry")
                - extensions: Extensions to load (default: [])
                - enable_spatial: Enable spatial features (default: False)
        """
        config = config or {}
        self.db_path = config.get("path", ":memory:")
        self.table_name = config.get("table_name", "telemetry")
        self.extensions = config.get("extensions", [])
        self.enable_spatial = config.get("enable_spatial", False)

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
            logger.info(f"Connected to DuckDB at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise

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

        inserted = 0
        spatial_enabled = self.enable_spatial and "spatial" in self.extensions

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

                    self._conn.execute(
                        f"""
                        INSERT INTO {self.table_name} (id, ts, source, data, metadata, location)
                        VALUES (?, ?, ?, ?, ?, ST_Point(?, ?))
                        """,
                        [record_id, ts, source, data, metadata, lon, lat],
                    )
                else:
                    if spatial_enabled:
                        self._conn.execute(
                            f"""
                            INSERT INTO {self.table_name} (id, ts, source, data, metadata, location)
                            VALUES (?, ?, ?, ?, ?, NULL)
                            """,
                            [record_id, ts, source, data, metadata],
                        )
                    else:
                        self._conn.execute(
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
        if self._conn and not self._closed:
            try:
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
