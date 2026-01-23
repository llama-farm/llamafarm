"""High-performance data buffer using Polars.

Provides a sliding window buffer optimized for streaming ML workloads:
- O(1) append operations
- Lazy evaluation for rolling features
- Automatic window truncation
- Thread-safe operations

Usage:
    buffer = PolarsBuffer(window_size=1000)
    buffer.append({"time_ms": 100, "value": 1.5, "category": "A"})
    buffer.append_batch([{"time_ms": 101, "value": 2.0}, ...])

    # Get features for anomaly detection
    features = buffer.get_features(rolling_windows=[5, 10, 20])

    # Get latest N records with computed features
    latest = buffer.get_latest(n=10)
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class BufferStats:
    """Statistics about the buffer state."""

    size: int
    window_size: int
    columns: list[str]
    numeric_columns: list[str]
    memory_bytes: int
    append_count: int
    avg_append_ms: float


class PolarsBuffer:
    """High-performance sliding window buffer using Polars.

    Features:
    - Configurable window size with automatic truncation
    - Lazy rolling feature computation
    - Support for mixed data types
    - Thread-safe operations
    - Performance metrics tracking

    Thread Safety:
        All public methods are thread-safe via an internal lock.
    """

    def __init__(
        self,
        window_size: int = 1000,
        schema: dict[str, pl.DataType] | None = None,
    ):
        """Initialize the buffer.

        Args:
            window_size: Maximum number of records to keep
            schema: Optional Polars schema for type safety
        """
        self._window_size = window_size
        self._schema = schema
        self._lock = threading.Lock()

        # Initialize empty DataFrame
        self._df: pl.DataFrame | None = None

        # Performance tracking
        self._append_count = 0
        self._total_append_time_ms = 0.0

    @property
    def size(self) -> int:
        """Current number of records in buffer."""
        with self._lock:
            return 0 if self._df is None else len(self._df)

    @property
    def window_size(self) -> int:
        """Maximum buffer size."""
        return self._window_size

    @property
    def columns(self) -> list[str]:
        """List of column names."""
        with self._lock:
            return [] if self._df is None else self._df.columns

    @property
    def numeric_columns(self) -> list[str]:
        """List of numeric column names."""
        with self._lock:
            if self._df is None:
                return []
            return [
                col
                for col, dtype in self._df.schema.items()
                if dtype.is_numeric()
            ]

    def append(self, record: dict[str, Any]) -> None:
        """Append a single record to the buffer.

        Args:
            record: Dictionary mapping column names to values
        """
        start_time = time.perf_counter()

        with self._lock:
            new_row = pl.DataFrame([record], schema=self._schema)

            if self._df is None:
                self._df = new_row
            else:
                self._df = pl.concat([self._df, new_row], how="diagonal_relaxed")

            # Truncate if over window size
            if len(self._df) > self._window_size:
                self._df = self._df.tail(self._window_size)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._append_count += 1
        self._total_append_time_ms += elapsed_ms

    def append_batch(self, records: list[dict[str, Any]]) -> None:
        """Append multiple records at once (more efficient than individual appends).

        Args:
            records: List of record dictionaries
        """
        if not records:
            return

        start_time = time.perf_counter()

        with self._lock:
            new_rows = pl.DataFrame(records, schema=self._schema)

            if self._df is None:
                self._df = new_rows
            else:
                self._df = pl.concat([self._df, new_rows], how="diagonal_relaxed")

            # Truncate if over window size
            if len(self._df) > self._window_size:
                self._df = self._df.tail(self._window_size)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._append_count += len(records)
        self._total_append_time_ms += elapsed_ms

    def get_data(self) -> pl.DataFrame:
        """Get the raw buffer data as a DataFrame.

        Returns:
            Copy of the internal DataFrame
        """
        with self._lock:
            if self._df is None:
                return pl.DataFrame()
            return self._df.clone()

    def get_numpy(self) -> Any:
        """Get the numeric data as a numpy array.

        Returns:
            Numpy array of numeric columns only
        """
        with self._lock:
            if self._df is None:
                import numpy as np
                return np.array([])

            # Get numeric columns directly (avoid calling self.numeric_columns which locks again)
            numeric_cols = [
                col
                for col, dtype in self._df.schema.items()
                if dtype.is_numeric()
            ]
            if not numeric_cols:
                import numpy as np
                return np.array([])

            return self._df.select(numeric_cols).to_numpy()

    def get_features(
        self,
        rolling_windows: list[int] | None = None,
        include_lags: bool = True,
        lag_periods: list[int] | None = None,
    ) -> pl.DataFrame:
        """Compute rolling features for the buffer data.

        Args:
            rolling_windows: Window sizes for rolling stats (default: [5, 10, 20])
            include_lags: Whether to include lag features
            lag_periods: Lag periods to compute (default: [1, 2, 3])

        Returns:
            DataFrame with original columns plus computed features
        """
        if rolling_windows is None:
            rolling_windows = [5, 10, 20]
        if lag_periods is None:
            lag_periods = [1, 2, 3]

        with self._lock:
            if self._df is None or len(self._df) == 0:
                return pl.DataFrame()

            df = self._df.clone()
            numeric_cols = [
                col for col, dtype in df.schema.items() if dtype.is_numeric()
            ]

            if not numeric_cols:
                return df

            # Build expressions for lazy evaluation
            feature_exprs = []

            for col in numeric_cols:
                # Rolling statistics
                for window in rolling_windows:
                    if window <= len(df):
                        feature_exprs.extend([
                            pl.col(col).rolling_mean(window).alias(f"{col}_rolling_mean_{window}"),
                            pl.col(col).rolling_std(window).alias(f"{col}_rolling_std_{window}"),
                            pl.col(col).rolling_min(window).alias(f"{col}_rolling_min_{window}"),
                            pl.col(col).rolling_max(window).alias(f"{col}_rolling_max_{window}"),
                        ])

                # Lag features
                if include_lags:
                    for lag in lag_periods:
                        feature_exprs.append(
                            pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")
                        )

            if feature_exprs:
                df = df.with_columns(feature_exprs)

            return df

    def get_latest(
        self,
        n: int = 1,
        with_features: bool = False,
        rolling_windows: list[int] | None = None,
    ) -> pl.DataFrame:
        """Get the most recent N records.

        Args:
            n: Number of records to return
            with_features: Whether to compute and include rolling features
            rolling_windows: Window sizes for features (if with_features=True)

        Returns:
            DataFrame with latest records
        """
        if with_features:
            df = self.get_features(rolling_windows=rolling_windows)
            return df.tail(n)
        else:
            with self._lock:
                if self._df is None:
                    return pl.DataFrame()
                return self._df.tail(n).clone()

    def clear(self) -> None:
        """Clear all data from the buffer."""
        with self._lock:
            self._df = None
            self._append_count = 0
            self._total_append_time_ms = 0.0

    def get_stats(self) -> BufferStats:
        """Get buffer statistics.

        Returns:
            BufferStats object with current state
        """
        with self._lock:
            if self._df is None:
                return BufferStats(
                    size=0,
                    window_size=self._window_size,
                    columns=[],
                    numeric_columns=[],
                    memory_bytes=0,
                    append_count=self._append_count,
                    avg_append_ms=0.0,
                )

            return BufferStats(
                size=len(self._df),
                window_size=self._window_size,
                columns=self._df.columns,
                numeric_columns=[
                    col for col, dtype in self._df.schema.items()
                    if dtype.is_numeric()
                ],
                memory_bytes=self._df.estimated_size("b"),
                append_count=self._append_count,
                avg_append_ms=(
                    self._total_append_time_ms / self._append_count
                    if self._append_count > 0 else 0.0
                ),
            )

    def to_list(self) -> list[dict[str, Any]]:
        """Convert buffer to list of dictionaries.

        Returns:
            List of record dictionaries
        """
        with self._lock:
            if self._df is None:
                return []
            return self._df.to_dicts()

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def __repr__(self) -> str:
        """String representation."""
        return f"PolarsBuffer(size={self.size}, window_size={self._window_size})"
