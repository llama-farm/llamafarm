#!/usr/bin/env python3
"""Working Memory Demo - Short-Term Buffer with TTL.

Demonstrates:
1. Adding streaming data (chat, telemetry, audio)
2. TTL-based automatic expiration
3. Querying recent records
4. Filtering by data type
5. Auto-pruning when buffer exceeds max size

This is part of the Embedded Trinity Memory System.
"""

import sys
import tempfile
import time

# Add rag to path
sys.path.insert(0, ".")


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


def main() -> int:
    from components.stores.duckdb_store import WorkingMemory

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = f"{temp_dir}/demo_working.duckdb"

        # ─────────────────────────────────────────────────
        # Step 1: Initialize Working Memory
        # ─────────────────────────────────────────────────
        print_section("Step 1: Initialize Working Memory")

        # Short TTL for demo purposes (3 seconds)
        config = {
            "path": db_path,
            "ttl_seconds": 3,  # Records expire after 3 seconds
            "max_size": 100,  # Max 100 records
        }

        memory = WorkingMemory(config=config)
        print_success(f"Working Memory initialized at: {db_path}")
        print_info(f"TTL: {config['ttl_seconds']} seconds")
        print_info(f"Max size: {config['max_size']} records")

        # ─────────────────────────────────────────────────
        # Step 2: Simulate Streaming Data
        # ─────────────────────────────────────────────────
        print_section("Step 2: Simulate Streaming Data")

        # Chat messages
        print_info("Adding chat messages...")
        memory.add("chat", "User: What's the status?", {"user_id": "operator_1"})
        memory.add("chat", "System: All units operational.", {"user_id": "system"})
        memory.add("chat", "User: Any alerts?", {"user_id": "operator_1"})
        print_success("Added 3 chat messages")

        # Telemetry data
        print_info("Adding telemetry data...")
        memory.add("telemetry", '{"heart_rate": 72, "blood_oxygen": 98}', {
            "device": "watch",
            "soldier_id": "sgt_johnson",
        })
        memory.add("telemetry", '{"heart_rate": 85, "blood_oxygen": 97}', {
            "device": "watch",
            "soldier_id": "cpl_smith",
        })
        print_success("Added 2 telemetry records")

        # Audio transcriptions
        print_info("Adding audio transcriptions...")
        memory.add("audio", "Radio check, alpha team.", {
            "channel": "alpha",
            "transcription_quality": 0.95,
        })
        print_success("Added 1 audio transcription")

        # ─────────────────────────────────────────────────
        # Step 3: Query Recent Records
        # ─────────────────────────────────────────────────
        print_section("Step 3: Query Recent Records")

        recent = memory.get_recent(limit=10)
        print_success(f"Found {len(recent)} recent records:")
        for r in recent[:5]:
            content = r["content"][:40] + "..." if len(r["content"]) > 40 else r["content"]
            print(f"    [{r['data_type']}] {content}")

        # ─────────────────────────────────────────────────
        # Step 4: Filter by Type
        # ─────────────────────────────────────────────────
        print_section("Step 4: Filter by Type")

        chats = memory.get_by_type("chat", limit=10)
        print_success(f"Found {len(chats)} chat records")

        telemetry = memory.get_by_type("telemetry", limit=10)
        print_success(f"Found {len(telemetry)} telemetry records")

        # ─────────────────────────────────────────────────
        # Step 5: Demonstrate TTL Expiration
        # ─────────────────────────────────────────────────
        print_section("Step 5: Demonstrate TTL Expiration")

        stats_before = memory.get_stats()
        print_info(f"Records before waiting: {stats_before['total_records']}")

        print_warning("Waiting 4 seconds for records to expire...")
        time.sleep(4)

        # Prune expired records
        pruned = memory.prune()
        print_success(f"Pruned {pruned} expired records")

        stats_after = memory.get_stats()
        print_success(f"Records after pruning: {stats_after['total_records']}")

        # ─────────────────────────────────────────────────
        # Step 6: Batch Ingestion
        # ─────────────────────────────────────────────────
        print_section("Step 6: Batch Ingestion")

        # Simulate batch of sensor readings
        batch = []
        for i in range(20):
            batch.append({
                "data_type": "sensor",
                "content": f'{{"reading": {i * 10}, "sensor_id": "s{i % 5}"}}',
                "metadata": {"batch_id": "demo_batch"},
            })

        added = memory.add_batch(batch)
        print_success(f"Added {added} records in batch")

        stats = memory.get_stats()
        print_success(f"Total records now: {stats['total_records']}")
        print_info(f"Type distribution: {stats['type_counts']}")

        # ─────────────────────────────────────────────────
        # Step 7: Statistics
        # ─────────────────────────────────────────────────
        print_section("Step 7: Statistics")

        stats = memory.get_stats()
        print_success(f"Total records: {stats['total_records']}")
        print_success(f"Max size: {stats['max_size']}")
        print_success(f"TTL: {stats['ttl_seconds']} seconds")
        print_success(f"Type counts: {stats['type_counts']}")
        if stats.get("oldest_record"):
            print_success(f"Oldest record: {stats['oldest_record']}")
        if stats.get("newest_record"):
            print_success(f"Newest record: {stats['newest_record']}")

        # ─────────────────────────────────────────────────
        # Step 8: Clear All
        # ─────────────────────────────────────────────────
        print_section("Step 8: Clear All")

        print_warning("Clearing all records from working memory...")
        memory.clear()

        stats = memory.get_stats()
        print_success(f"Records after clear: {stats['total_records']}")

        # ─────────────────────────────────────────────────
        # Cleanup
        # ─────────────────────────────────────────────────
        memory.close()
        print_success("\nWorking Memory closed")

    print("")
    print(f"\033[0;32m{'═' * 50}\033[0m")
    print("\033[0;32m  Demo completed successfully!\033[0m")
    print(f"\033[0;32m{'═' * 50}\033[0m")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(f"\033[0;31m✗ Import error: {e}\033[0m")
        print(
            "\033[0;33m  Run from the 'rag' directory: cd rag && uv run python ../examples/database/demo_working_memory.py\033[0m"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\033[0;31m✗ Error: {e}\033[0m")
        import traceback

        traceback.print_exc()
        sys.exit(1)
