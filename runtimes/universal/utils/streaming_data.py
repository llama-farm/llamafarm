"""Streaming data utilities for large dataset training.

Provides memory-efficient data loading and processing for anomaly detection
training with datasets too large to fit in memory.

Supported formats:
- CSV (.csv)
- JSON Lines (.jsonl, .ndjson)
- Parquet (.parquet) - if pyarrow is installed

Usage:
    loader = StreamingDataLoader()
    file_ref = await loader.upload_file("data.csv")
    async for batch in file_ref.iter_batches(batch_size=1000):
        # Process batch (numpy array)
        ...
    await loader.cleanup_file(file_ref.file_id)
"""

import asyncio
import csv
import json
import logging
import os
import shutil
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default temp directory for uploaded files
_LF_DATA_DIR = Path(os.environ.get("LF_DATA_DIR", Path.home() / ".llamafarm"))
STREAMING_TEMP_DIR = _LF_DATA_DIR / "temp" / "streaming"


@dataclass
class FileReference:
    """Reference to an uploaded file for streaming training.

    Attributes:
        file_id: Unique identifier for this file
        file_path: Path to the file (may be in temp directory)
        file_type: File format (csv, jsonl, parquet)
        row_count: Number of rows in the file
        column_count: Number of columns/features
        column_names: List of column names (if available)
        original_path: Original path where file was uploaded from
    """

    file_id: str
    file_path: Path
    file_type: str
    row_count: int
    column_count: int
    column_names: list[str] = field(default_factory=list)
    original_path: Path | None = None

    async def iter_batches(
        self,
        batch_size: int = 1000,
        skip_header: bool = True,
    ) -> AsyncIterator[np.ndarray]:
        """Iterate over the file in batches.

        Yields numpy arrays of shape (batch_size, column_count).

        Args:
            batch_size: Number of rows per batch
            skip_header: Skip header row for CSV files

        Yields:
            Numpy arrays of batch data
        """
        if self.file_type == "csv":
            async for batch in self._iter_csv_batches(batch_size, skip_header):
                yield batch
        elif self.file_type in ("jsonl", "ndjson"):
            async for batch in self._iter_jsonl_batches(batch_size):
                yield batch
        elif self.file_type == "parquet":
            async for batch in self._iter_parquet_batches(batch_size):
                yield batch
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

    async def _iter_csv_batches(
        self,
        batch_size: int,
        skip_header: bool,
    ) -> AsyncIterator[np.ndarray]:
        """Iterate CSV file in batches."""
        loop = asyncio.get_running_loop()

        def read_batches():
            """Synchronous batch reader (run in executor)."""
            batches = []
            with open(self.file_path, newline="") as f:
                reader = csv.reader(f)
                if skip_header:
                    next(reader, None)  # Skip header

                batch_data = []
                for row in reader:
                    batch_data.append([float(x) for x in row])
                    if len(batch_data) >= batch_size:
                        batches.append(np.array(batch_data, dtype=np.float32))
                        batch_data = []

                # Don't forget the last batch
                if batch_data:
                    batches.append(np.array(batch_data, dtype=np.float32))

            return batches

        # Run blocking I/O in thread pool
        batches = await loop.run_in_executor(None, read_batches)
        for batch in batches:
            yield batch

    async def _iter_jsonl_batches(
        self,
        batch_size: int,
    ) -> AsyncIterator[np.ndarray]:
        """Iterate JSON Lines file in batches."""
        loop = asyncio.get_running_loop()

        def read_batches():
            """Synchronous batch reader."""
            batches = []
            with open(self.file_path) as f:
                batch_data = []
                for line in f:
                    row = json.loads(line.strip())
                    # Extract values in consistent order
                    if self.column_names:
                        values = [float(row.get(col, 0)) for col in self.column_names]
                    else:
                        values = [float(v) for v in row.values()]
                    batch_data.append(values)
                    if len(batch_data) >= batch_size:
                        batches.append(np.array(batch_data, dtype=np.float32))
                        batch_data = []

                if batch_data:
                    batches.append(np.array(batch_data, dtype=np.float32))

            return batches

        batches = await loop.run_in_executor(None, read_batches)
        for batch in batches:
            yield batch

    async def _iter_parquet_batches(
        self,
        batch_size: int,
    ) -> AsyncIterator[np.ndarray]:
        """Iterate Parquet file in batches."""
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for Parquet support. "
                "Install with: uv add pyarrow"
            ) from e

        loop = asyncio.get_running_loop()

        def read_batches():
            """Synchronous batch reader."""
            batches = []
            parquet_file = pq.ParquetFile(self.file_path)

            for batch in parquet_file.iter_batches(batch_size=batch_size):
                # Convert to numpy
                df = batch.to_pandas()
                batches.append(df.values.astype(np.float32))

            return batches

        batches = await loop.run_in_executor(None, read_batches)
        for batch in batches:
            yield batch


class StreamingDataLoader:
    """Manager for streaming data uploads and iteration.

    Handles file uploads, validation, and cleanup for large dataset training.
    """

    def __init__(self, temp_dir: Path | None = None):
        """Initialize the streaming data loader.

        Args:
            temp_dir: Directory for temporary files. Defaults to ~/.llamafarm/temp/streaming
        """
        self.temp_dir = Path(temp_dir) if temp_dir else STREAMING_TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._file_refs: dict[str, FileReference] = {}

    async def upload_file(
        self,
        file_path: str | Path,
        copy_to_temp: bool = True,
    ) -> FileReference:
        """Upload/register a file for streaming training.

        Args:
            file_path: Path to the data file
            copy_to_temp: Whether to copy file to temp directory (default: True)
                If False, uses the original file path directly.

        Returns:
            FileReference for use with model.fit_from_file()
        """
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")

        # Determine file type
        suffix = source_path.suffix.lower()
        file_type = {
            ".csv": "csv",
            ".jsonl": "jsonl",
            ".ndjson": "jsonl",
            ".parquet": "parquet",
        }.get(suffix)

        if file_type is None:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: .csv, .jsonl, .ndjson, .parquet"
            )

        # Generate unique ID
        file_id = str(uuid.uuid4())

        # Copy to temp if requested
        if copy_to_temp:
            temp_path = self.temp_dir / f"{file_id}{suffix}"
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, shutil.copy2, source_path, temp_path)
            final_path = temp_path
        else:
            final_path = source_path

        # Analyze file to get metadata
        row_count, column_count, column_names = await self._analyze_file(
            final_path, file_type
        )

        file_ref = FileReference(
            file_id=file_id,
            file_path=final_path,
            file_type=file_type,
            row_count=row_count,
            column_count=column_count,
            column_names=column_names,
            original_path=source_path if copy_to_temp else None,
        )

        self._file_refs[file_id] = file_ref
        logger.info(
            f"Uploaded file {source_path.name}: {row_count} rows, {column_count} columns"
        )

        return file_ref

    async def _analyze_file(
        self,
        file_path: Path,
        file_type: str,
    ) -> tuple[int, int, list[str]]:
        """Analyze file to get row count, column count, and column names.

        Args:
            file_path: Path to the file
            file_type: File format (csv, jsonl, parquet)

        Returns:
            Tuple of (row_count, column_count, column_names)
        """
        loop = asyncio.get_running_loop()

        def analyze_csv():
            with open(file_path, newline="") as f:
                reader = csv.reader(f)
                # First row is header
                header = next(reader, None)
                if header is None:
                    return 0, 0, []

                column_names = header
                column_count = len(header)

                # Count remaining rows
                row_count = sum(1 for _ in reader)

                return row_count, column_count, column_names

        def analyze_jsonl():
            with open(file_path) as f:
                first_line = f.readline().strip()
                if not first_line:
                    return 0, 0, []

                first_row = json.loads(first_line)
                column_names = list(first_row.keys())
                column_count = len(column_names)

                # Count remaining rows (including first)
                row_count = 1 + sum(1 for _ in f)

                return row_count, column_count, column_names

        def analyze_parquet():
            try:
                import pyarrow.parquet as pq
            except ImportError as e:
                raise ImportError("pyarrow required for Parquet support") from e

            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata
            schema = parquet_file.schema

            row_count = metadata.num_rows
            column_count = len(schema)
            column_names = schema.names

            return row_count, column_count, column_names

        if file_type == "csv":
            return await loop.run_in_executor(None, analyze_csv)
        elif file_type in ("jsonl", "ndjson"):
            return await loop.run_in_executor(None, analyze_jsonl)
        elif file_type == "parquet":
            return await loop.run_in_executor(None, analyze_parquet)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    async def cleanup_file(self, file_id: str) -> None:
        """Remove a temporary file and its reference.

        Args:
            file_id: The file ID returned from upload_file()
        """
        if file_id not in self._file_refs:
            logger.warning(f"File {file_id} not found in registry")
            return

        file_ref = self._file_refs[file_id]

        # Only delete if it's in our temp directory
        if file_ref.original_path is not None and file_ref.file_path.exists():
            try:
                file_ref.file_path.unlink()
                logger.info(f"Cleaned up temp file: {file_ref.file_path}")
            except OSError as e:
                logger.warning(f"Failed to delete temp file: {e}")

        del self._file_refs[file_id]

    def get_file_ref(self, file_id: str) -> FileReference | None:
        """Get a file reference by ID.

        Args:
            file_id: The file ID

        Returns:
            FileReference or None if not found
        """
        return self._file_refs.get(file_id)

    async def cleanup_all(self) -> None:
        """Clean up all temporary files."""
        for file_id in list(self._file_refs.keys()):
            await self.cleanup_file(file_id)
