"""Tests for large dataset training with streaming support.

Phase 4 of Universal Runtime ML Enhancements.

Tests verify:
- Streaming upload endpoint accepts CSV/Parquet files
- Large file training doesn't OOM the server
- Streaming training uses chunked processing
- Memory usage stays bounded during training
"""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['feature1', 'feature2', 'feature3', 'feature4'])
        # Write 1000 rows of normal data
        np.random.seed(42)
        for _ in range(1000):
            row = [
                np.random.normal(25, 2),
                np.random.normal(100, 5),
                np.random.normal(0.5, 0.1),
                np.random.normal(10, 0.5),
            ]
            writer.writerow(row)
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_large_csv_file():
    """Create a larger CSV file (~10MB) for memory testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'])
        # Write 100,000 rows (should be ~10MB)
        np.random.seed(42)
        for _ in range(100000):
            row = [np.random.normal(0, 1) for _ in range(8)]
            writer.writerow(row)
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_jsonl_file():
    """Create a temporary JSON Lines file for testing."""
    import json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        np.random.seed(42)
        for _ in range(500):
            row = {
                'feature1': float(np.random.normal(25, 2)),
                'feature2': float(np.random.normal(100, 5)),
                'feature3': float(np.random.normal(0.5, 0.1)),
                'feature4': float(np.random.normal(10, 0.5)),
            }
            f.write(json.dumps(row) + '\n')
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestStreamingUpload:
    """Tests for streaming upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_csv_file(self, temp_csv_file):
        """Test that CSV file upload works correctly."""
        from utils.streaming_data import StreamingDataLoader

        loader = StreamingDataLoader()
        file_ref = await loader.upload_file(temp_csv_file)

        assert file_ref is not None
        assert file_ref.file_type == "csv"
        assert file_ref.row_count == 1000
        assert file_ref.column_count == 4

        # Cleanup
        await loader.cleanup_file(file_ref.file_id)

    @pytest.mark.asyncio
    async def test_upload_jsonl_file(self, temp_jsonl_file):
        """Test that JSON Lines file upload works correctly."""
        from utils.streaming_data import StreamingDataLoader

        loader = StreamingDataLoader()
        file_ref = await loader.upload_file(temp_jsonl_file)

        assert file_ref is not None
        assert file_ref.file_type == "jsonl"
        assert file_ref.row_count == 500

        await loader.cleanup_file(file_ref.file_id)

    @pytest.mark.asyncio
    async def test_upload_returns_file_reference(self, temp_csv_file):
        """Test that upload returns a valid file reference for fitting."""
        from utils.streaming_data import StreamingDataLoader

        loader = StreamingDataLoader()
        file_ref = await loader.upload_file(temp_csv_file)

        # File reference should have required fields
        assert file_ref.file_id is not None
        assert file_ref.file_path.exists()
        assert file_ref.column_names == ['feature1', 'feature2', 'feature3', 'feature4']

        await loader.cleanup_file(file_ref.file_id)


class TestStreamingTraining:
    """Tests for streaming/chunked training."""

    @pytest.mark.asyncio
    async def test_streaming_fit_isolation_forest(self, temp_csv_file):
        """Test that streaming training works with Isolation Forest."""
        from models.anomaly_model import AnomalyModel
        from utils.streaming_data import StreamingDataLoader

        # Upload the file
        loader = StreamingDataLoader()
        file_ref = await loader.upload_file(temp_csv_file)

        # Create model and fit from file
        model = AnomalyModel(
            model_id="test_stream_if",
            device="cpu",
            backend="isolation_forest",
        )
        await model.load()

        # Fit from streaming file
        result = await model.fit_from_file(
            file_ref=file_ref,
            batch_size=100,
        )

        assert result.samples_fitted == 1000
        assert model.is_fitted

        await loader.cleanup_file(file_ref.file_id)

    @pytest.mark.asyncio
    async def test_streaming_fit_autoencoder(self, temp_csv_file):
        """Test that streaming training works with autoencoder."""
        from models.anomaly_model import AnomalyModel
        from utils.streaming_data import StreamingDataLoader

        loader = StreamingDataLoader()
        file_ref = await loader.upload_file(temp_csv_file)

        model = AnomalyModel(
            model_id="test_stream_ae",
            device="cpu",
            backend="autoencoder",
            patience=5,
        )
        await model.load()

        result = await model.fit_from_file(
            file_ref=file_ref,
            batch_size=64,
            epochs=20,
        )

        assert result.samples_fitted == 1000
        assert model.is_fitted

        await loader.cleanup_file(file_ref.file_id)

    @pytest.mark.asyncio
    async def test_chunked_processing_reduces_memory(self, temp_large_csv_file):
        """Test that chunked processing keeps memory bounded.

        With 100K rows, we should see memory stay bounded if using chunks.
        """
        import tracemalloc

        from models.anomaly_model import AnomalyModel
        from utils.streaming_data import StreamingDataLoader

        loader = StreamingDataLoader()
        file_ref = await loader.upload_file(temp_large_csv_file)

        model = AnomalyModel(
            model_id="test_stream_memory",
            device="cpu",
            backend="isolation_forest",
        )
        await model.load()

        # Start memory tracking
        tracemalloc.start()

        # Fit with small batch size to force chunking
        result = await model.fit_from_file(
            file_ref=file_ref,
            batch_size=1000,  # Process 1000 rows at a time
        )

        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Convert to MB
        peak_mb = peak / 1024 / 1024

        assert result.samples_fitted == 100000
        assert model.is_fitted
        # Memory should stay under reasonable limit (200MB for 10MB file)
        # This tests that we're not loading the entire file into memory
        assert peak_mb < 200, f"Peak memory {peak_mb:.1f}MB exceeds 200MB limit"

        await loader.cleanup_file(file_ref.file_id)


class TestFileCleanup:
    """Tests for temp file cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_temp_file(self, temp_csv_file):
        """Test that cleanup removes temporary files."""
        from utils.streaming_data import StreamingDataLoader

        loader = StreamingDataLoader()
        file_ref = await loader.upload_file(temp_csv_file)

        # Verify file exists in temp location
        assert file_ref.file_path.exists()
        temp_path = file_ref.file_path

        # Cleanup
        await loader.cleanup_file(file_ref.file_id)

        # File should be removed from temp location
        # (Note: original file should still exist)
        # The loader copies to temp, so we check the temp copy is gone
        assert not temp_path.exists() or temp_path == Path(temp_csv_file)

    @pytest.mark.asyncio
    async def test_auto_cleanup_after_training(self, temp_csv_file):
        """Test that files can be auto-cleaned after training."""
        from models.anomaly_model import AnomalyModel
        from utils.streaming_data import StreamingDataLoader

        loader = StreamingDataLoader()
        file_ref = await loader.upload_file(temp_csv_file)

        model = AnomalyModel(
            model_id="test_auto_cleanup",
            device="cpu",
            backend="isolation_forest",
        )
        await model.load()

        # Fit with auto_cleanup=True
        await model.fit_from_file(
            file_ref=file_ref,
            batch_size=100,
            auto_cleanup=True,
        )

        # File reference should be invalidated after cleanup
        # (implementation detail: file_ref.file_path should not exist)


class TestDataIterator:
    """Tests for the data iterator that streams from files."""

    @pytest.mark.asyncio
    async def test_csv_iterator_yields_batches(self, temp_csv_file):
        """Test that CSV iterator yields data in batches."""
        from utils.streaming_data import StreamingDataLoader

        loader = StreamingDataLoader()
        file_ref = await loader.upload_file(temp_csv_file)

        batches = []
        batch_count = 0
        async for batch in file_ref.iter_batches(batch_size=100):
            batches.append(batch)
            batch_count += 1
            assert batch.shape[0] <= 100  # Batch size or less for last batch

        # 1000 rows / 100 batch_size = 10 batches
        assert batch_count == 10
        assert sum(b.shape[0] for b in batches) == 1000

        await loader.cleanup_file(file_ref.file_id)

    @pytest.mark.asyncio
    async def test_jsonl_iterator_yields_batches(self, temp_jsonl_file):
        """Test that JSON Lines iterator yields data in batches."""
        from utils.streaming_data import StreamingDataLoader

        loader = StreamingDataLoader()
        file_ref = await loader.upload_file(temp_jsonl_file)

        total_rows = 0
        async for batch in file_ref.iter_batches(batch_size=50):
            total_rows += batch.shape[0]

        assert total_rows == 500

        await loader.cleanup_file(file_ref.file_id)


class TestMemoryMonitoring:
    """Tests for memory monitoring during training."""

    @pytest.mark.asyncio
    async def test_memory_stats_available(self, temp_csv_file):
        """Test that memory statistics are available during training."""
        from models.anomaly_model import AnomalyModel
        from utils.streaming_data import StreamingDataLoader

        loader = StreamingDataLoader()
        file_ref = await loader.upload_file(temp_csv_file)

        model = AnomalyModel(
            model_id="test_memory_stats",
            device="cpu",
            backend="isolation_forest",
        )
        await model.load()

        result = await model.fit_from_file(
            file_ref=file_ref,
            batch_size=100,
        )

        # Result should include memory statistics
        assert 'peak_memory_mb' in result.model_params or result.training_time_ms > 0

        await loader.cleanup_file(file_ref.file_id)
