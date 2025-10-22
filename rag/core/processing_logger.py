"""
Processing logger for RAG pipeline.
Saves detailed processing logs to project folder for debugging and audit.
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class ProcessingLogger:
    """
    Handles logging of RAG processing events to both console and file.
    Creates detailed logs in the project's lf_data/logs directory.
    """

    def __init__(self, project_dir: str, dataset_name: Optional[str] = None):
        """
        Initialize the processing logger.

        Args:
            project_dir: Path to the project directory (where lf_data is located)
            dataset_name: Optional dataset name for specific logging
        """
        self.project_dir = Path(project_dir)
        self.dataset_name = dataset_name

        # Create logs directory in lf_data
        self.logs_dir = self.project_dir / "lf_data" / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for this session
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create main processing log file
        log_filename = f"processing_{self.session_timestamp}"
        if dataset_name:
            # Sanitize dataset_name to prevent path traversal
            import re

            safe_dataset_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", dataset_name)
            # Remove any leading/trailing underscores
            if safe_dataset_name := safe_dataset_name.strip("_"):
                log_filename += f"_{safe_dataset_name}"
        log_filename += ".log"

        self.log_file = self.logs_dir / log_filename

        # Also create a JSON log for structured data
        self.json_log_file = self.logs_dir / log_filename.replace(".log", ".json")

        # Initialize JSON log structure
        self.processing_events: list[dict[str, Any]] = []

        # Batching configuration
        self.batch_size = 10  # Write JSON after this many events
        self.batch_timeout = 30  # Write JSON after this many seconds
        self.last_write_time = time.time()
        self._write_lock = threading.Lock()
        self._dirty = False  # Track if we have unsaved events

        # Setup file handler for standard logging
        self.logger = logging.getLogger(f"ProcessingLogger_{self.session_timestamp}")
        self.logger.setLevel(logging.DEBUG)

        # File handler for detailed logs with UTF-8 encoding
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Add handler if not already present
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

        # Log initialization
        self.logger.info(f"Processing logger initialized for project: {project_dir}")
        if dataset_name:
            self.logger.info(f"Dataset: {dataset_name}")

    def _should_write_json(self) -> bool:
        """Check if we should write the JSON log based on batch size or timeout."""
        current_time = time.time()
        return len(self.processing_events) >= self.batch_size or (
            self._dirty and current_time - self.last_write_time >= self.batch_timeout
        )

    def _maybe_write_json(self):
        """Write JSON log if batching conditions are met."""
        if self._should_write_json():
            self._save_json_log()

    def _add_event(self, event: dict[str, Any]):
        """Add an event and potentially trigger a batched write."""
        with self._write_lock:
            self.processing_events.append(event)
            self._dirty = True
            self._maybe_write_json()

    def log_file_processing(self, file_path: str, status: str, details: dict[str, Any]):
        """
        Log a file processing event.

        Args:
            file_path: Path to the file being processed
            status: Processing status (e.g., "processed", "skipped", "failed")
            details: Additional details about the processing
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "file_processing",
            "file": file_path,
            "status": status,
            "details": details,
        }

        # Log to standard logger
        self.logger.info(f"File: {file_path} | Status: {status}")
        if details:
            for key, value in details.items():
                self.logger.debug(f"  {key}: {value}")

        # Add to events with batched writing
        self._add_event(event)

    def log_chunk_processing(
        self, chunk_id: str, status: str, metadata: dict[str, Any]
    ):
        """
        Log a chunk processing event.

        Args:
            chunk_id: ID of the chunk
            status: Processing status
            metadata: Chunk metadata
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "chunk_processing",
            "chunk_id": chunk_id,
            "status": status,
            "metadata": metadata,
        }

        # Log to standard logger
        self.logger.debug(f"Chunk: {chunk_id} | Status: {status}")

        # Add to events with batched writing
        self._add_event(event)

    def log_duplicate_detection(self, file_hash: str, chunk_count: int, action: str):
        """
        Log duplicate detection event.

        Args:
            file_hash: Hash of the file
            chunk_count: Number of chunks affected
            action: Action taken (skipped, overwritten, etc.)
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "duplicate_detection",
            "file_hash": file_hash,
            "chunk_count": chunk_count,
            "action": action,
        }

        self.logger.warning(
            f"Duplicate detected: {file_hash[:8]}... | Chunks: {chunk_count} | Action: {action}"
        )

        # Add to events with batched writing
        self._add_event(event)

    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[dict[str, Any]] = None,
    ):
        """
        Log an error event.

        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context about the error
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_type": error_type,
            "message": error_message,
            "context": context or {},
        }

        self.logger.error(f"{error_type}: {error_message}")
        if context:
            try:
                context_str = json.dumps(context, indent=2, default=str)
            except (TypeError, ValueError) as e:
                # Fallback for non-serializable objects
                context_str = str(context)
                self.logger.warning(f"Could not JSON serialize context: {e}")
            self.logger.error(f"Context: {context_str}")

        # Add to events with batched writing
        self._add_event(event)

    def log_summary(self, summary: dict[str, Any]):
        """
        Log a processing summary.

        Args:
            summary: Summary statistics and information
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "summary",
            "data": summary,
        }

        self.logger.info("=" * 60)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        for key, value in summary.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 60)

        # Add to events and force immediate write for summaries
        with self._write_lock:
            self.processing_events.append(event)
            self._save_json_log()  # Force write for summaries

    def _save_json_log(self):
        """Save the current events to JSON log file."""
        try:
            with open(self.json_log_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "session": {
                            "timestamp": self.session_timestamp,
                            "project_dir": str(self.project_dir),
                            "dataset": self.dataset_name,
                        },
                        "events": self.processing_events,
                    },
                    f,
                    indent=2,
                    default=str,
                    ensure_ascii=False,
                )
            # Update tracking variables
            self.last_write_time = time.time()
            self._dirty = False
        except Exception as e:
            self.logger.error(f"Failed to save JSON log: {e}")

    def flush(self):
        """Force write any pending events to disk."""
        with self._write_lock:
            if self._dirty:
                self._save_json_log()

    def __del__(self):
        """Ensure any pending events are written when the logger is destroyed."""
        try:
            self.flush()
        except Exception:
            pass  # Ignore errors during cleanup

    def get_log_files(self) -> dict[str, str]:
        """
        Get paths to log files.

        Returns:
            Dictionary with log file paths
        """
        return {"text_log": str(self.log_file), "json_log": str(self.json_log_file)}

    @classmethod
    def get_latest_logs(
        cls, project_dir: str, dataset_name: Optional[str] = None
    ) -> list:
        """
        Get the latest log files for a project/dataset.

        Args:
            project_dir: Project directory
            dataset_name: Optional dataset name filter

        Returns:
            List of log file paths sorted by recency
        """
        logs_dir = Path(project_dir) / "lf_data" / "logs"
        if not logs_dir.exists():
            return []

        pattern = "processing_*.log"
        if dataset_name:
            pattern = f"processing_*_{dataset_name}.log"

        log_files = list(logs_dir.glob(pattern))
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return [str(f) for f in log_files[:10]]  # Return last 10 logs
