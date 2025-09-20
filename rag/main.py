#!/usr/bin/env python3
"""
RAG Service Celery Worker Entry Point

This module serves as the main entry point for the RAG container when running
as a Celery worker service. It connects to the Celery broker and handles
RAG-related tasks from the server.
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from celery_app import app

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    """Main entry point for RAG Celery worker."""
    logger.info("Starting RAG Celery worker service")

    # Log environment info
    logger.info(
        "RAG worker configuration",
        extra={
            "data_dir": os.environ.get("LF_DATA_DIR", str(Path.home() / ".llamafarm")),
            "python_path": sys.path[:3],  # First few paths for debugging
        },
    )

    # Start the Celery worker
    # Use 'solo' pool for single-threaded execution to avoid issues with ML libraries
    # that don't play well with multiprocessing
    try:
        app.worker_main(
            argv=["worker", "--loglevel=info", "--pool=solo", "--concurrency=1"]
        )
    except KeyboardInterrupt:
        logger.info("RAG worker shutting down due to keyboard interrupt")
    except Exception as e:
        logger.error("RAG worker failed to start", extra={"error": str(e)})
        raise


if __name__ == "__main__":
    main()
