#!/usr/bin/env python3
"""
RAG Service Celery Worker Entry Point

This module serves as the main entry point for the RAG container when running
as a Celery worker service. It connects to the Celery broker and handles
RAG-related tasks from the server.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from celery_app import app, run_worker


def main():
    """Main entry point for RAG Celery worker."""
    print("Starting RAG Celery worker service")
    
    # Log environment info
    print(f"Data directory: {os.environ.get('LF_DATA_DIR', str(Path.home() / '.llamafarm'))}")
    print(f"Python path: {sys.path[:3]}")  # First few paths for debugging
    
    # Log registered tasks for debugging
    try:
        registered_tasks = list(app.tasks.keys())
        rag_tasks = [task for task in registered_tasks if task.startswith("rag.")]
        print(f"Total registered tasks: {len(registered_tasks)}")
        print(f"RAG tasks found: {len(rag_tasks)}")
        
        if rag_tasks:
            print("RAG tasks:")
            for task in sorted(rag_tasks):
                print(f"  - {task}")
    except Exception as e:
        print(f"Could not list registered tasks: {e}")
    
    # Start the worker
    run_worker()


if __name__ == "__main__":
    main()