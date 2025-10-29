"""
Service layer for reading event logs from filesystem.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import EventDetail, EventSummary, SubEvent


class EventLogService:
    """Service for reading event logs from the filesystem."""

    @staticmethod
    def _get_event_logs_dir(namespace: str, project: str) -> Path:
        """
        Get the event logs directory for a project.

        Args:
            namespace: Project namespace
            project: Project name

        Returns:
            Path to event logs directory
        """
        data_dir = os.getenv("LF_DATA_DIR", str(Path.home() / ".llamafarm"))
        return Path(data_dir) / "projects" / namespace / project / "event_logs"

    @staticmethod
    def list_events(
        namespace: str,
        project: str,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[list[EventSummary], int]:
        """
        List event logs with optional filtering.

        Args:
            namespace: Project namespace
            project: Project name
            event_type: Filter by event type (e.g., "inference", "rag_processing")
            start_time: Filter events after this timestamp
            end_time: Filter events before this timestamp
            limit: Maximum number of events to return
            offset: Number of events to skip

        Returns:
            Tuple of (list of EventSummary, total count)
        """
        event_logs_dir = EventLogService._get_event_logs_dir(namespace, project)

        if not event_logs_dir.exists():
            return [], 0

        # Find all event log files
        pattern = f"evt_{event_type}_*.json" if event_type else "evt_*.json"
        event_files = sorted(
            event_logs_dir.glob(pattern),
            key=lambda f: f.stat().st_mtime,
            reverse=True,  # Most recent first
        )

        # Parse and filter events
        summaries = []
        for event_file in event_files:
            try:
                with open(event_file) as f:
                    event_data = json.load(f)

                # Parse timestamp
                timestamp = datetime.fromisoformat(event_data["timestamp"])

                # Apply time filters
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue

                # Calculate total duration
                duration_ms = None
                if event_data.get("events"):
                    last_event = event_data["events"][-1]
                    duration_ms = last_event.get("duration_ms")

                summary = EventSummary(
                    event_id=event_data["event_id"],
                    event_type=event_data["event_type"],
                    request_id=event_data["request_id"],
                    timestamp=timestamp,
                    namespace=event_data["namespace"],
                    project=event_data["project"],
                    status=event_data["status"],
                    duration_ms=duration_ms,
                    config_hash=event_data["config_hash"],
                )
                summaries.append(summary)

            except Exception:
                # Skip malformed event files
                continue

        total = len(summaries)

        # Apply pagination
        paginated = summaries[offset : offset + limit]

        return paginated, total

    @staticmethod
    def get_event(namespace: str, project: str, event_id: str) -> Optional[EventDetail]:
        """
        Get a single event by ID.

        Args:
            namespace: Project namespace
            project: Project name
            event_id: Event identifier

        Returns:
            EventDetail or None if not found
        """
        event_logs_dir = EventLogService._get_event_logs_dir(namespace, project)
        event_file = event_logs_dir / f"{event_id}.json"

        if not event_file.exists():
            return None

        try:
            with open(event_file) as f:
                event_data = json.load(f)

            # Parse sub-events
            sub_events = [
                SubEvent(
                    timestamp=datetime.fromisoformat(se["timestamp"]),
                    event_name=se["event_name"],
                    duration_ms=se["duration_ms"],
                    data=se["data"],
                )
                for se in event_data["events"]
            ]

            return EventDetail(
                event_id=event_data["event_id"],
                event_type=event_data["event_type"],
                request_id=event_data["request_id"],
                timestamp=datetime.fromisoformat(event_data["timestamp"]),
                namespace=event_data["namespace"],
                project=event_data["project"],
                config_hash=event_data["config_hash"],
                events=sub_events,
                status=event_data["status"],
                error=event_data.get("error"),
                metadata=event_data.get("metadata", {}),
            )

        except Exception:
            return None
