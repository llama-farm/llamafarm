"""
Universal event logger for LlamaFarm operations.

Thread-safe event logging for inference, RAG processing, and all other operations.
Used by server, RAG worker, and future runtimes.
"""

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from observability.config_versioning import hash_config, save_config_snapshot


class EventLogger:
    """
    Thread-safe event logger for parallel operations.

    Simple interface - just throw dicts at it:
    - log_event(event_name, data) - No JSON conversion needed!
    - complete_event() - Single write to disk
    - fail_event(error) - Write with error status

    All complexity (threading, JSON, I/O, timestamps) handled internally.
    """

    def __init__(
        self,
        event_type: str,
        request_id: str,
        namespace: str,
        project: str,
        config: Any,
    ):
        """
        Initialize a new event logger.

        Args:
            event_type: Type of event (e.g., "inference", "rag_processing")
            request_id: Unique request identifier
            namespace: Project namespace
            project: Project name
            config: LlamaFarmConfig object (hash will be computed internally)
        """
        self.event_type = event_type
        self.request_id = request_id
        self.namespace = namespace
        self.project = project

        # Compute config hash and save snapshot
        self.config_hash = hash_config(config)
        save_config_snapshot(config, self.config_hash, namespace, project)

        # Event storage (internal buffer)
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()  # Thread safety
        self._start_time = datetime.now(timezone.utc)
        self._last_event_time = self._start_time  # Track last event for delta duration
        self._metadata: dict[str, Any] = {}
        self._summary_data: dict[str, Any] = {}  # Summary data from final event

    def log_event(self, event_name: str, data: dict[str, Any]) -> None:
        """
        Log a sub-event. Thread-safe, no formatting required.

        Just throw any dict at it - logger handles the rest!

        Args:
            event_name: Name of the event (e.g., "rag_query_start")
            data: Event data as plain dict (logger handles JSON serialization)

        Example:
            logger.log_event("chunk_retrieval", {
                "chunks": 5,
                "avg_score": 0.88,
                "sources": ["paper.pdf", "notes.txt"],
                "metadata": {"strategy": "hybrid"}
            })
        """
        with self._lock:  # Thread-safe
            now = datetime.now(timezone.utc)
            # Calculate duration as time since last event (not from start)
            duration_ms = (now - self._last_event_time).total_seconds() * 1000
            self._last_event_time = now  # Update for next event

            # Simple event structure - logger adds timestamp and duration
            # IMPORTANT: Copy the data dict to prevent mutations from affecting the log
            event = {
                "timestamp": now.isoformat(),
                "event_name": event_name,
                "duration_ms": round(duration_ms, 2),
                "data": dict(data),  # Shallow copy to prevent mutations
            }

            self._events.append(event)

            # If this is a final event (processing_complete, etc.), save summary data
            if event_name.endswith("_complete") or event_name == "processing_complete":
                # Calculate total elapsed time from logger's start time
                total_elapsed_ms = (now - self._start_time).total_seconds() * 1000

                # Copy data and add/override total_elapsed_time_ms
                self._summary_data = dict(data)
                self._summary_data["total_elapsed_time_ms"] = round(total_elapsed_ms, 2)

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the event (e.g., client_ip, user_agent).

        Thread-safe.

        Args:
            key: Metadata key
            value: Metadata value (any JSON-serializable type)
        """
        with self._lock:
            self._metadata[key] = value

    def complete_event(self) -> None:
        """
        Write event to disk. All JSON serialization happens here.

        Thread-safe. Single write operation.
        """
        with self._lock:
            self._write_to_disk(status="completed", error=None)

    def fail_event(self, error: str) -> None:
        """
        Write failed event to disk.

        Thread-safe.

        Args:
            error: Error message or exception string
        """
        with self._lock:
            self._write_to_disk(status="failed", error=error)

    def _write_to_disk(self, status: str, error: Optional[str]) -> None:
        """
        Internal method - handles all JSON serialization and I/O.

        Caller never deals with JSON!

        Args:
            status: Event status ("completed" or "failed")
            error: Error message if status is "failed"
        """
        # Generate event ID: timestamp-type-random (for easier time sorting)
        timestamp = self._start_time.strftime("%Y%m%d_%H%M%S")
        random_id = uuid.uuid4().hex[:6]
        event_id = f"evt_{timestamp}_{self.event_type}_{random_id}"

        # Determine timestamp: use first event's timestamp if available, otherwise use start time
        if self._events:
            # Parse ISO timestamp from first event
            event_timestamp = datetime.fromisoformat(self._events[0]["timestamp"])
        else:
            # Fallback to start time if no events logged
            event_timestamp = self._start_time

        # Build complete event structure
        full_event = {
            "event_id": event_id,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "timestamp": event_timestamp.isoformat(),  # Top-level timestamp for event listing
            "namespace": self.namespace,
            "project": self.project,
            "config_hash": self.config_hash,
            "events": self._events,  # All sub-events
            "status": status,
            "error": error,
            "metadata": self._metadata,
        }

        # Merge summary data from final event (e.g., processing_complete) into top level
        if self._summary_data:
            full_event |= self._summary_data

        # Calculate time_to_first_token_ms if llm_first_token event exists
        first_token_event = next(
            (e for e in self._events if e["event_name"] == "llm_first_token"),
            None
        )
        if first_token_event:
            first_token_time = datetime.fromisoformat(first_token_event["timestamp"])
            time_to_first_token_ms = (first_token_time - self._start_time).total_seconds() * 1000
            full_event["time_to_first_token_ms"] = round(time_to_first_token_ms, 2)

        # Set timestamp AFTER merge to ensure it's never overwritten by summary_data
        # This field is required by EventLogService.list_events() and EventLogService.get_event()
        full_event["timestamp"] = (
            event_timestamp.isoformat()
        )  # ISO format for EventLogService

        # Get project path with security validation (follows ProjectService pattern)
        from .path_utils import get_project_path, validate_file_path

        project_dir = get_project_path(self.namespace, self.project)
        event_logs_dir = os.path.join(project_dir, "event_logs")
        os.makedirs(event_logs_dir, exist_ok=True)

        event_file = os.path.join(event_logs_dir, f"{event_id}.json")

        # Security: Validate the event file path stays within event_logs directory
        validate_file_path(event_file, event_logs_dir, "event")

        # Atomic write using tempfile + os.replace()
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=event_logs_dir,
            suffix=".tmp",
        ) as tmp:
            json.dump(full_event, tmp, indent=2)
            tmp_path = tmp.name

        # Atomic move (POSIX systems)
        os.replace(tmp_path, event_file)
