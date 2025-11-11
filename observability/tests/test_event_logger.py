"""
Unit tests for EventLogger.

Tests thread safety, file I/O, and event structure.
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from observability.event_logger import EventLogger


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for testing."""
    data_dir = tmp_path / "llamafarm_test"
    data_dir.mkdir()

    # Set LF_DATA_DIR for test
    old_env = os.environ.get("LF_DATA_DIR")
    os.environ["LF_DATA_DIR"] = str(data_dir)

    yield data_dir

    # Restore old env
    if old_env:
        os.environ["LF_DATA_DIR"] = old_env
    else:
        del os.environ["LF_DATA_DIR"]


@pytest.fixture
def mock_config():
    """Create a mock config object for testing."""
    config = MagicMock()
    config.namespace = "default"
    config.name = "test-project"
    config.model_dump.return_value = {
        "name": "test-project",
        "namespace": "default",
        "version": "v1",
    }
    return config


def test_event_logger_basic(temp_data_dir, mock_config):
    """Test basic event logging."""
    logger = EventLogger(
        event_type="inference",
        request_id="req_test123",
        namespace="default",
        project="test-project",
        config=mock_config,
    )

    logger.log_event("step1", {"data": "value1"})
    logger.log_event("step2", {"data": "value2", "count": 42})
    logger.complete_event()

    # Verify event file exists
    event_logs_dir = (
        temp_data_dir / "projects" / "default" / "test-project" / "event_logs"
    )
    assert event_logs_dir.exists()

    # Find the event file (new format: evt_timestamp_type_random)
    event_files = list(event_logs_dir.glob("evt_*_inference_*.json"))
    assert len(event_files) == 1

    # Read and verify event structure
    with open(event_files[0]) as f:
        event = json.load(f)

    assert event["event_type"] == "inference"
    assert event["request_id"] == "req_test123"
    assert "timestamp" in event  # Required by EventLogService
    assert (
        event["timestamp"] == event["events"][0]["timestamp"]
    )  # Uses first event's timestamp
    assert event["namespace"] == "default"
    assert event["project"] == "test-project"
    assert event["config_hash"].startswith("sha256_")  # Hash is computed internally
    assert event["status"] == "completed"
    assert event["error"] is None
    assert len(event["events"]) == 2

    # Verify sub-events
    assert event["events"][0]["event_name"] == "step1"
    assert event["events"][0]["data"]["data"] == "value1"
    assert event["events"][1]["event_name"] == "step2"
    assert event["events"][1]["data"]["count"] == 42


def test_event_logger_fail(temp_data_dir, mock_config):
    """Test failed event logging."""
    logger = EventLogger(
        event_type="rag_processing",
        request_id="req_fail",
        namespace="default",
        project="test-project",
        config=mock_config,
    )

    logger.log_event("parse_start", {"file": "test.pdf"})
    logger.fail_event("Parse error: invalid format")

    # Find the event file (new format: evt_timestamp_type_random)
    event_logs_dir = (
        temp_data_dir / "projects" / "default" / "test-project" / "event_logs"
    )
    event_files = list(event_logs_dir.glob("evt_*_rag_processing_*.json"))
    assert len(event_files) == 1

    # Read and verify
    with open(event_files[0]) as f:
        event = json.load(f)

    assert event["status"] == "failed"
    assert event["error"] == "Parse error: invalid format"


def test_event_logger_metadata(temp_data_dir, mock_config):
    """Test metadata addition."""
    logger = EventLogger(
        event_type="inference",
        request_id="req_metadata",
        namespace="default",
        project="test-project",
        config=mock_config,
    )

    logger.add_metadata("client_ip", "127.0.0.1")
    logger.add_metadata("user_agent", "TestClient/1.0")
    logger.log_event("request_received", {"endpoint": "/chat"})
    logger.complete_event()

    # Find and read event
    event_logs_dir = (
        temp_data_dir / "projects" / "default" / "test-project" / "event_logs"
    )
    event_files = list(event_logs_dir.glob("evt_*_inference_*.json"))

    with open(event_files[0]) as f:
        event = json.load(f)

    assert event["metadata"]["client_ip"] == "127.0.0.1"
    assert event["metadata"]["user_agent"] == "TestClient/1.0"


def test_event_logger_thread_safety(temp_data_dir, mock_config):
    """Test thread-safe logging with parallel operations."""
    logger = EventLogger(
        event_type="inference",
        request_id="req_parallel",
        namespace="default",
        project="test-project",
        config=mock_config,
    )

    # Log events from multiple threads
    def log_events(prefix: str, count: int):
        for i in range(count):
            logger.log_event(f"{prefix}_event_{i}", {"value": i})
            time.sleep(0.001)  # Small delay to simulate real work

    threads = [
        threading.Thread(target=log_events, args=("thread1", 10)),
        threading.Thread(target=log_events, args=("thread2", 10)),
        threading.Thread(target=log_events, args=("thread3", 10)),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    logger.complete_event()

    # Find and read event
    event_logs_dir = (
        temp_data_dir / "projects" / "default" / "test-project" / "event_logs"
    )
    event_files = list(event_logs_dir.glob("evt_*_inference_*.json"))

    with open(event_files[0]) as f:
        event = json.load(f)

    # Should have 30 events total (3 threads × 10 events each)
    assert len(event["events"]) == 30

    # Verify events are ordered by timestamp
    timestamps = [e["timestamp"] for e in event["events"]]
    assert timestamps == sorted(timestamps)


def test_event_logger_multiple_events(temp_data_dir, mock_config):
    """Test multiple separate events."""
    # Create first event
    logger1 = EventLogger(
        event_type="inference",
        request_id="req_1",
        namespace="default",
        project="test-project",
        config=mock_config,
    )
    logger1.log_event("step1", {"data": 1})
    logger1.complete_event()

    # Create second event
    logger2 = EventLogger(
        event_type="rag_processing",
        request_id="req_2",
        namespace="default",
        project="test-project",
        config=mock_config,
    )
    logger2.log_event("step1", {"data": 2})
    logger2.complete_event()

    # Verify both events exist
    event_logs_dir = (
        temp_data_dir / "projects" / "default" / "test-project" / "event_logs"
    )
    event_files = list(event_logs_dir.glob("evt_*.json"))
    assert len(event_files) == 2


def test_event_logger_timestamps(temp_data_dir, mock_config):
    """Test that timestamps and durations are calculated correctly."""
    logger = EventLogger(
        event_type="inference",
        request_id="req_time",
        namespace="default",
        project="test-project",
        config=mock_config,
    )

    logger.log_event("start", {"data": "begin"})
    time.sleep(0.1)  # 100ms delay
    logger.log_event("end", {"data": "finish"})
    logger.complete_event()

    # Find and read event
    event_logs_dir = (
        temp_data_dir / "projects" / "default" / "test-project" / "event_logs"
    )
    event_files = list(event_logs_dir.glob("evt_*_inference_*.json"))

    with open(event_files[0]) as f:
        event = json.load(f)

    # Verify duration increases
    assert event["events"][0]["duration_ms"] >= 0
    assert event["events"][1]["duration_ms"] >= 100  # At least 100ms
    assert event["events"][1]["duration_ms"] > event["events"][0]["duration_ms"]


def test_event_logger_no_events_timestamp(temp_data_dir, mock_config):
    """Test that timestamp field exists even when no events are logged."""
    logger = EventLogger(
        event_type="inference",
        request_id="req_no_events",
        namespace="default",
        project="test-project",
        config=mock_config,
    )

    # Complete without logging any events
    logger.complete_event()

    # Find and read event
    event_logs_dir = (
        temp_data_dir / "projects" / "default" / "test-project" / "event_logs"
    )
    event_files = list(event_logs_dir.glob("evt_*_inference_*.json"))

    with open(event_files[0]) as f:
        event = json.load(f)

    # Verify timestamp field exists (should use start_time as fallback)
    assert "timestamp" in event
    assert event["timestamp"] is not None
    assert len(event["events"]) == 0  # No events logged


def test_event_logger_eventlogservice_integration(temp_data_dir, mock_config):
    """Test that EventLogService can read events written by EventLogger."""
    # Create and log an event
    logger = EventLogger(
        event_type="inference",
        request_id="req_integration",
        namespace="default",
        project="test-project",
        config=mock_config,
    )
    logger.log_event("step1", {"data": "value1"})
    logger.complete_event()

    # Now try to read it using EventLogService
    try:
        from server.services.event_log_service import EventLogService

        events, total = EventLogService.list_events(
            namespace="default",
            project="test-project",
            event_type="inference",
        )

        assert total == 1
        assert len(events) == 1
        assert events[0].event_type == "inference"
        assert events[0].request_id == "req_integration"
        assert events[0].timestamp is not None  # Verify timestamp was read successfully

        # Also test get_event
        event_detail = EventLogService.get_event(
            namespace="default",
            project="test-project",
            event_id=events[0].event_id,
        )

        assert event_detail is not None
        assert event_detail.event_id == events[0].event_id
        assert (
            event_detail.timestamp is not None
        )  # Verify timestamp was read successfully
        assert len(event_detail.events) == 1
        assert event_detail.events[0].event_name == "step1"

    except ImportError:
        # Skip if server module not available (e.g., in isolated test environment)
        pytest.skip("EventLogService not available in test environment")
