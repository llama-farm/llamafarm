#!/usr/bin/env python3
"""Memory API Demo - Unified Memory System Client.

Demonstrates using the Memory API endpoints programmatically.

This demo shows how to:
1. Add data to different memory stores via API
2. Query unified context
3. Get storage statistics
4. Trigger consolidation
5. Get aggregated context

Prerequisites:
    - LlamaFarm server running (cd server && uv run uvicorn api.main:app)
    - requests library installed

Usage:
    python demo_memory_api.py
"""

import json
import sys
from datetime import datetime

# Try to import requests
try:
    import requests
except ImportError:
    print("Error: requests library not installed")
    print("Install with: pip install requests")
    sys.exit(1)


# Configuration
BASE_URL = "http://localhost:8000"
API_PREFIX = "/v1/memory"


def print_section(title: str) -> None:
    print(f"\n\033[0;34m{'─' * 50}\033[0m")
    print(f"\033[0;34m{title}\033[0m")
    print(f"\033[0;34m{'─' * 50}\033[0m")


def print_success(msg: str) -> None:
    print(f"\033[0;32m✓\033[0m {msg}")


def print_info(msg: str) -> None:
    print(f"\033[0;36m→\033[0m {msg}")


def print_warning(msg: str) -> None:
    print(f"\033[1;33m!\033[0m {msg}")


def print_error(msg: str) -> None:
    print(f"\033[0;31m✗\033[0m {msg}")


def check_server() -> bool:
    """Check if the server is running."""
    try:
        resp = requests.get(f"{BASE_URL}/info", timeout=2)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def add_data(data: dict, data_type: str, **kwargs) -> dict:
    """Add data to memory via API."""
    payload = {
        "data": data,
        "data_type": data_type,
        **kwargs,
    }
    resp = requests.post(f"{BASE_URL}{API_PREFIX}/add", json=payload)
    return resp.json()


def query_memory(**params) -> dict:
    """Query memory via API."""
    resp = requests.get(f"{BASE_URL}{API_PREFIX}/query", params=params)
    return resp.json()


def get_stats() -> dict:
    """Get memory statistics via API."""
    resp = requests.get(f"{BASE_URL}{API_PREFIX}/stats")
    return resp.json()


def get_context(**params) -> dict:
    """Get aggregated context via API."""
    resp = requests.get(f"{BASE_URL}{API_PREFIX}/context", params=params)
    return resp.json()


def consolidate(use_llm: bool = False) -> dict:
    """Trigger consolidation via API."""
    resp = requests.post(
        f"{BASE_URL}{API_PREFIX}/consolidate",
        json={"use_llm": use_llm},
    )
    return resp.json()


def main() -> int:
    print("\n\033[0;36m╔════════════════════════════════════════════════════════╗\033[0m")
    print("\033[0;36m║       Memory API Demo - Python Client                  ║\033[0m")
    print("\033[0;36m╚════════════════════════════════════════════════════════╝\033[0m")

    # Check server
    print_section("Checking Server")
    if not check_server():
        print_warning(f"Server not running at {BASE_URL}")
        print_warning("Start server: cd server && uv run uvicorn api.main:app")
        print("")
        print("Running in demo mode (showing what the API calls would look like)...")
        print("")

        # Show example payloads
        print_section("Example: Add Text Data")
        example = {
            "data": "Medical protocol for shock: Monitor vitals.",
            "data_type": "text",
            "metadata": {"category": "medical"},
        }
        print_info(f"POST {API_PREFIX}/add")
        print(json.dumps(example, indent=2))

        print_section("Example: Add Telemetry")
        example = {
            "data": {"heart_rate": 120, "status": "distress"},
            "data_type": "telemetry",
            "latitude": 35.78,
            "longitude": -78.64,
        }
        print_info(f"POST {API_PREFIX}/add")
        print(json.dumps(example, indent=2))

        print_section("Example: Query Memory")
        print_info(f"GET {API_PREFIX}/query?data_types=chat,telemetry&limit=10")

        print_section("Example: Get Statistics")
        print_info(f"GET {API_PREFIX}/stats")

        print("\n\033[0;32m═══════════════════════════════════════════════════════\033[0m")
        print("\033[0;32m  Demo mode completed. Start server to run live demo.  \033[0m")
        print("\033[0;32m═══════════════════════════════════════════════════════\033[0m")
        return 0

    print_success(f"Server running at {BASE_URL}")

    # Track created UUIDs
    uuids = []

    # Step 1: Add text data
    print_section("Step 1: Add Text Data")
    result = add_data(
        data="Medical protocol for shock: Monitor vitals, elevate legs, keep warm.",
        data_type="text",
        metadata={"category": "medical", "priority": "high"},
    )
    if result.get("success"):
        print_success(f"Added text data, UUID: {result.get('uuid')}")
        print_info(f"Store: {result.get('store')}")
        if result.get("uuid"):
            uuids.append(result["uuid"])
    else:
        print_error(f"Failed: {result.get('message')}")

    # Step 2: Add telemetry data
    print_section("Step 2: Add Telemetry Data")
    result = add_data(
        data={"heart_rate": 120, "blood_pressure": "90/60", "status": "distress"},
        data_type="telemetry",
        latitude=35.7800,
        longitude=-78.6400,
        metadata={"soldier_id": "alpha-1"},
    )
    if result.get("success"):
        print_success(f"Added telemetry data, UUID: {result.get('uuid')}")
        print_info(f"Store: {result.get('store')}")
        if result.get("uuid"):
            uuids.append(result["uuid"])
    else:
        print_error(f"Failed: {result.get('message')}")

    # Step 3: Add chat data
    print_section("Step 3: Add Chat Data")
    result = add_data(
        data="Alpha-1: Help! Man down at checkpoint Delta!",
        data_type="chat",
        metadata={"channel": "tactical", "priority": "critical"},
    )
    if result.get("success"):
        print_success(f"Added chat data, UUID: {result.get('uuid')}")
        print_info(f"Store: {result.get('store')}")
        if result.get("uuid"):
            uuids.append(result["uuid"])
    else:
        print_error(f"Failed: {result.get('message')}")

    # Step 4: Query memory
    print_section("Step 4: Query Unified Context")
    result = query_memory(limit=10)
    print_success(f"Found {result.get('total_count', 0)} records")
    for record in result.get("results", [])[:3]:
        content = str(record.get("content", ""))[:50]
        print_info(f"  [{record.get('store')}] {content}...")

    # Step 5: Get stats
    print_section("Step 5: Get Storage Statistics")
    stats = get_stats()
    for store_name, store_stats in stats.items():
        if isinstance(store_stats, dict):
            print_info(f"  {store_name}:")
            for key, value in store_stats.items():
                print(f"      {key}: {value}")

    # Step 6: Get context
    print_section("Step 6: Get Aggregated Context")
    context = get_context(recent_minutes=10)
    if context.get("working_memory"):
        print_info(f"  Working memory: {len(context['working_memory'])} items")
    if context.get("graph"):
        print_info(f"  Graph: {len(context['graph'])} items")
    if context.get("timeseries"):
        print_info(f"  Timeseries: {len(context['timeseries'])} items")
    if context.get("summary"):
        print_info(f"  Summary: {context['summary']}")

    # Step 7: Consolidate
    print_section("Step 7: Trigger Consolidation")
    result = consolidate(use_llm=False)
    if result.get("success"):
        print_success("Consolidation completed")
        print_info(f"  Records processed: {result.get('records_processed', 0)}")
        print_info(f"  Facts extracted: {result.get('facts_extracted', 0)}")
        print_info(f"  Nodes created: {result.get('nodes_created', 0)}")
        if result.get("skipped"):
            print_warning("  (Skipped - below threshold)")
    else:
        print_error(f"Failed: {result.get('message')}")

    print("\n\033[0;32m═══════════════════════════════════════════════════════\033[0m")
    print("\033[0;32m  Memory API demo completed successfully!              \033[0m")
    print(f"\033[0;32m  Created {len(uuids)} records                                   \033[0m")
    print("\033[0;32m═══════════════════════════════════════════════════════\033[0m")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.exceptions.RequestException as e:
        print_error(f"Request failed: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
