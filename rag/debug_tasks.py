#!/usr/bin/env python3
"""
Debug script to check what Celery tasks are registered in the RAG worker.

Usage:
    cd rag/
    uv run python debug_tasks.py
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("=== RAG Celery Task Discovery Debug ===\n")

    try:
        # Import the celery app
        print("1. Importing celery_app...")
        from celery_app import app

        print("   ✓ Successfully imported celery_app")

        # Check if tasks are registered
        print(f"\n2. Checking registered tasks...")
        registered_tasks = list(app.tasks.keys())
        print(f"   Total registered tasks: {len(registered_tasks)}")

        # Filter RAG tasks
        rag_tasks = [task for task in registered_tasks if task.startswith("rag.")]
        print(f"   RAG tasks found: {len(rag_tasks)}")

        # List all tasks
        print(f"\n3. All registered tasks:")
        for task in sorted(registered_tasks):
            marker = "🎯" if task.startswith("rag.") else "  "
            print(f"   {marker} {task}")

        # Show RAG tasks specifically
        if rag_tasks:
            print(f"\n4. RAG tasks specifically:")
            for task in sorted(rag_tasks):
                print(f"   ✓ {task}")
        else:
            print(f"\n4. ❌ No RAG tasks found!")

        # Try to import task modules directly
        print(f"\n5. Testing direct task module imports...")

        try:
            import tasks.search_tasks

            print("   ✓ tasks.search_tasks imported successfully")
        except Exception as e:
            print(f"   ❌ tasks.search_tasks failed: {e}")

        try:
            import tasks.ingest_tasks

            print("   ✓ tasks.ingest_tasks imported successfully")
        except Exception as e:
            print(f"   ❌ tasks.ingest_tasks failed: {e}")

        try:
            import tasks.query_tasks

            print("   ✓ tasks.query_tasks imported successfully")
        except Exception as e:
            print(f"   ❌ tasks.query_tasks failed: {e}")

        # Check task configuration
        print(f"\n6. Celery configuration:")
        print(f"   App name: {app.main}")
        print(f"   Broker URL: {app.conf.broker_url}")
        print(f"   Result backend: {app.conf.result_backend}")

        if hasattr(app.conf, "imports"):
            print(f"   Configured imports: {app.conf.imports}")
        else:
            print("   No imports configured")

        if hasattr(app.conf, "task_routes"):
            print(f"   Task routes: {app.conf.task_routes}")
        else:
            print("   No task routes configured")

    except Exception as e:
        print(f"❌ Error during task discovery: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
