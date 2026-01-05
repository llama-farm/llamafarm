"""Tests for Streaming Data API types.

Phase 19: Streaming Data Endpoint - Type validation tests
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from api.routers.streaming.types import (
    DatasetStreamStats,
    StreamBatchRequest,
    StreamBatchResponse,
    StreamEvent,
    StreamRecord,
    StreamRecordRequest,
    StreamRecordResponse,
    StreamRecordResult,
    StreamStatusResponse,
)


class TestStreamRecord:
    """Tests for StreamRecord model."""

    def test_minimal_record(self):
        """Test minimal record with only required fields."""
        record = StreamRecord(data={"temperature": 72.5})
        assert record.data == {"temperature": 72.5}
        assert record.data_type == "telemetry"  # default
        assert record.timestamp is None
        assert record.latitude is None
        assert record.longitude is None

    def test_full_record(self):
        """Test record with all fields."""
        now = datetime.utcnow()
        record = StreamRecord(
            data={"temperature": 72.5, "humidity": 45.2},
            data_type="sensor_reading",
            timestamp=now,
            latitude=35.78,
            longitude=-78.64,
            altitude=100.5,
            metadata={"unit": "fahrenheit"},
            source_id="sensor-001",
        )
        assert record.data_type == "sensor_reading"
        assert record.timestamp == now
        assert record.latitude == 35.78
        assert record.longitude == -78.64
        assert record.altitude == 100.5
        assert record.metadata == {"unit": "fahrenheit"}
        assert record.source_id == "sensor-001"

    def test_record_requires_data(self):
        """Test that data field is required."""
        with pytest.raises(ValidationError):
            StreamRecord()


class TestStreamRecordRequest:
    """Tests for StreamRecordRequest model."""

    def test_valid_request(self):
        """Test valid request."""
        request = StreamRecordRequest(
            dataset="telemetry",
            record=StreamRecord(data={"temp": 72}),
        )
        assert request.dataset == "telemetry"
        assert request.record.data == {"temp": 72}

    def test_requires_dataset(self):
        """Test that dataset is required."""
        with pytest.raises(ValidationError):
            StreamRecordRequest(record=StreamRecord(data={"temp": 72}))


class TestStreamBatchRequest:
    """Tests for StreamBatchRequest model."""

    def test_valid_batch(self):
        """Test valid batch request."""
        request = StreamBatchRequest(
            dataset="telemetry",
            records=[
                StreamRecord(data={"temp": 72}),
                StreamRecord(data={"temp": 68}),
            ],
        )
        assert len(request.records) == 2
        assert request.fail_fast is False  # default

    def test_fail_fast_option(self):
        """Test fail_fast option."""
        request = StreamBatchRequest(
            dataset="telemetry",
            records=[StreamRecord(data={"temp": 72})],
            fail_fast=True,
        )
        assert request.fail_fast is True

    def test_empty_batch_rejected(self):
        """Test that empty batch is rejected."""
        with pytest.raises(ValidationError):
            StreamBatchRequest(dataset="telemetry", records=[])

    def test_batch_size_limit(self):
        """Test that batch size over 1000 is rejected."""
        with pytest.raises(ValidationError):
            StreamBatchRequest(
                dataset="telemetry",
                records=[StreamRecord(data={"i": i}) for i in range(1001)],
            )


class TestStreamRecordResult:
    """Tests for StreamRecordResult model."""

    def test_success_result(self):
        """Test successful result."""
        result = StreamRecordResult(
            success=True,
            record_id="rec-123",
            stores=["timeseries", "working_memory"],
            timestamp=datetime.utcnow(),
        )
        assert result.success is True
        assert result.record_id == "rec-123"
        assert "timeseries" in result.stores

    def test_failure_result(self):
        """Test failure result."""
        result = StreamRecordResult(
            success=False,
            error="Database connection failed",
        )
        assert result.success is False
        assert result.error == "Database connection failed"


class TestStreamRecordResponse:
    """Tests for StreamRecordResponse model."""

    def test_response(self):
        """Test response model."""
        response = StreamRecordResponse(
            success=True,
            record_id="rec-123",
            dataset="telemetry",
            stores=["timeseries"],
            message="Record ingested successfully",
        )
        assert response.success is True
        assert response.dataset == "telemetry"


class TestStreamBatchResponse:
    """Tests for StreamBatchResponse model."""

    def test_successful_batch(self):
        """Test successful batch response."""
        response = StreamBatchResponse(
            success=True,
            dataset="telemetry",
            total_records=10,
            successful=10,
            failed=0,
        )
        assert response.success is True
        assert response.successful == 10
        assert response.failed == 0

    def test_partial_failure_batch(self):
        """Test partial failure batch response."""
        response = StreamBatchResponse(
            success=False,
            dataset="telemetry",
            total_records=10,
            successful=8,
            failed=2,
            results=[
                StreamRecordResult(success=False, error="Error 1"),
                StreamRecordResult(success=False, error="Error 2"),
            ],
        )
        assert response.success is False
        assert response.failed == 2
        assert len(response.results) == 2


class TestDatasetStreamStats:
    """Tests for DatasetStreamStats model."""

    def test_stats(self):
        """Test stream statistics model."""
        stats = DatasetStreamStats(
            dataset_name="telemetry",
            dataset_type="realtime",
            total_records=1000,
            records_today=100,
            records_last_hour=10,
            stores_enabled=["timeseries", "spatial", "working_memory"],
            oldest_record=datetime(2024, 1, 1),
            newest_record=datetime.utcnow(),
        )
        assert stats.dataset_name == "telemetry"
        assert stats.total_records == 1000
        assert len(stats.stores_enabled) == 3


class TestStreamStatusResponse:
    """Tests for StreamStatusResponse model."""

    def test_active_status(self):
        """Test active status response."""
        response = StreamStatusResponse(
            active=True,
            dataset="telemetry",
            stats=DatasetStreamStats(
                dataset_name="telemetry",
                dataset_type="realtime",
            ),
        )
        assert response.active is True
        assert response.stats is not None


class TestStreamEvent:
    """Tests for StreamEvent model (SSE)."""

    def test_heartbeat_event(self):
        """Test heartbeat event."""
        event = StreamEvent(
            event_type="heartbeat",
            dataset="telemetry",
            data={"event_count": 42},
        )
        assert event.event_type == "heartbeat"
        assert event.data["event_count"] == 42

    def test_record_added_event(self):
        """Test record_added event."""
        event = StreamEvent(
            event_type="record_added",
            dataset="telemetry",
            data={"record_id": "rec-123", "stores": ["timeseries"]},
        )
        assert event.event_type == "record_added"

    def test_error_event(self):
        """Test error event."""
        event = StreamEvent(
            event_type="error",
            dataset="telemetry",
            data={"error": "Connection lost"},
        )
        assert event.event_type == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
