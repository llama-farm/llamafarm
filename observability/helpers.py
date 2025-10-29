"""
Helper utilities for observability.

Provides context managers and helper functions to simplify event logging.
"""

import uuid
from contextlib import contextmanager
from typing import Any, Optional

from .config_versioning import hash_config, save_config_snapshot
from .event_logger import EventLogger


@contextmanager
def event_logging_context(
    event_type: str,
    namespace: str,
    project: str,
    config: Any,
    request_id: Optional[str] = None,
):
    """
    Context manager for event logging lifecycle.

    Automatically handles:
    - Config hashing and snapshot saving
    - EventLogger initialization
    - Ensures complete_event() or fail_event() is called

    Args:
        event_type: Type of event (e.g., "inference", "rag_processing")
        namespace: Project namespace
        project: Project name
        config: LlamaFarmConfig object
        request_id: Optional request ID (auto-generated if not provided)

    Yields:
        EventLogger instance

    Example:
        with event_logging_context("inference", "default", "my-project", config) as logger:
            logger.log_event("step1", {"data": "value"})
            # Automatically calls complete_event() on success
            # Automatically calls fail_event() on exception
    """
    # Hash config and save snapshot
    config_hash = hash_config(config)
    save_config_snapshot(config, config_hash, namespace, project)

    # Generate request ID if not provided
    if request_id is None:
        request_id = f"req_{uuid.uuid4().hex[:12]}"

    # Create event logger
    logger = EventLogger(
        event_type=event_type,
        request_id=request_id,
        namespace=namespace,
        project=project,
        config_hash=config_hash,
    )

    try:
        yield logger
        logger.complete_event()
    except Exception as e:
        logger.fail_event(str(e))
        raise
