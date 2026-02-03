"""PyApp entry point for llamafarm-rag.

Adds the rag package directory to sys.path so that bare imports
(from core.xxx, from celery_app import ...) work alongside prefixed
imports (from rag.core.xxx import ...).
"""
import os
import sys

# Allow bare imports (from core.xxx, from celery_app, etc.)
sys.path.insert(0, os.path.dirname(__file__))

# Signal PyApp mode for runtime detection
os.environ["LLAMAFARM_PYAPP"] = "1"

# Import main module — this executes module-level setup
# (logging, PID file, Celery worker configuration)
from rag import main  # noqa: F401

if __name__ == '__main__':
    main.main()
