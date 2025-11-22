"""
Service layer for reading event logs from filesystem.
"""

import json
from datetime import datetime
from pathlib import Path

from api.routers.event_logs.models import EventDetail, EventSummary, SubEvent


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
        from observability.path_utils import get_project_path

        project_dir = get_project_path(namespace, project)
        return Path(project_dir) / "event_logs"

    @staticmethod
    def list_events(
        namespace: str,
        project: str,
        event_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
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
        # Event files are named: evt_{timestamp}_{event_type}_{random_id}.json
        pattern = f"evt_*_{event_type}_*.json" if event_type else "evt_*.json"
        event_files = sorted(
            event_logs_dir.glob(pattern),
            key=lambda f: f.stat().st_mtime,
            reverse=True,  # Most recent first
        )

        # Parse and filter events
        from observability.path_utils import validate_file_path

        summaries = []
        for event_file in event_files:
            try:
                # Security: Validate each event file path stays within event_logs directory
                validate_file_path(str(event_file), str(event_logs_dir), "event")

                with open(event_file) as f:
                    event_data = json.load(f)

                # Parse timestamp
                timestamp = datetime.fromisoformat(event_data["timestamp"])

                # Apply time filters (ensure both are timezone-aware for comparison)
                if start_time:
                    # Make start_time timezone-aware if it's naive
                    if start_time.tzinfo is None:
                        from datetime import timezone
                        start_time = start_time.replace(tzinfo=timezone.utc)
                    if timestamp < start_time:
                        continue
                if end_time:
                    # Make end_time timezone-aware if it's naive
                    if end_time.tzinfo is None:
                        from datetime import timezone
                        end_time = end_time.replace(tzinfo=timezone.utc)
                    if timestamp > end_time:
                        continue

                # Get total duration (use total_elapsed_time_ms if available, not just last event)
                duration_ms = event_data.get("total_elapsed_time_ms")
                if duration_ms is None and event_data.get("events"):
                    # Fallback to last event duration if total not available
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
    def get_event(namespace: str, project: str, event_id: str) -> EventDetail | None:
        """
        Get a single event by ID.

        Args:
            namespace: Project namespace
            project: Project name
            event_id: Event identifier

        Returns:
            EventDetail or None if not found
        """
        from observability.path_utils import validate_file_path

        event_logs_dir = EventLogService._get_event_logs_dir(namespace, project)
        event_file = event_logs_dir / f"{event_id}.json"

        # Security: Validate the event file path stays within event_logs directory
        try:
            validate_file_path(str(event_file), str(event_logs_dir), "event")
        except ValueError:
            return None

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
                total_elapsed_time_ms=event_data.get("total_elapsed_time_ms"),
                time_to_first_token_ms=event_data.get("time_to_first_token_ms"),
            )

        except Exception:
            return None
