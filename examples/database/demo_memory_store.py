#!/usr/bin/env python3
"""MemoryStore Demo - Unified Memory Interface.

Demonstrates:
1. Adding data to different stores via unified API
2. Querying across all stores
3. Getting unified context
4. Cascade delete
5. Statistics

This is the main entry point for the Embedded Trinity Memory System.
"""

import sys
import tempfile
from datetime import datetime, timedelta

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
    from core.memory import MemoryStore

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        # ─────────────────────────────────────────────────
        # Step 1: Initialize MemoryStore
        # ─────────────────────────────────────────────────
        print_section("Step 1: Initialize MemoryStore")

        config = {"base_path": temp_dir}
        print_info(f"Creating MemoryStore at: {temp_dir}")

        store = MemoryStore(config=config)
        print_success("MemoryStore initialized with all component stores")

        # ─────────────────────────────────────────────────
        # Step 2: Add Telemetry Data (→ Time-Series)
        # ─────────────────────────────────────────────────
        print_section("Step 2: Add Telemetry Data (→ Time-Series)")

        print_info("Adding biometric telemetry...")
        result1 = store.add(
            data='{"heart_rate": 72, "blood_oxygen": 98}',
            data_type="telemetry",
            metadata={"device": "watch", "soldier_id": "sgt_johnson"},
            latitude=35.7796,
            longitude=-78.6382,
        )
        print_success(f"Added telemetry: {result1['uuid'][:30]}... → {result1['store']}")

        result2 = store.add(
            data='{"heart_rate": 85, "blood_oxygen": 96}',
            data_type="telemetry",
            metadata={"device": "watch", "soldier_id": "cpl_smith"},
            latitude=35.7800,
            longitude=-78.6400,
        )
        print_success(f"Added telemetry: {result2['uuid'][:30]}... → {result2['store']}")

        # ─────────────────────────────────────────────────
        # Step 3: Add Graph Data (→ Graph Store)
        # ─────────────────────────────────────────────────
        print_section("Step 3: Add Graph Data (→ Graph Store)")

        print_info("Adding personnel nodes...")
        store.add(
            data={"id": "person:sgt_johnson", "name": "Sgt. Johnson", "rank": "Sergeant"},
            data_type="node",
            metadata={"node_type": "person"},
        )
        store.add(
            data={"id": "person:cpl_smith", "name": "Cpl. Smith", "rank": "Corporal"},
            data_type="node",
            metadata={"node_type": "person"},
        )
        store.add(
            data={"id": "location:checkpoint_delta", "name": "Checkpoint Delta"},
            data_type="node",
            metadata={"node_type": "location"},
        )
        print_success("Added 3 nodes to graph")

        print_info("Adding relationships...")
        store.add(
            data={
                "source": "person:sgt_johnson",
                "edge_type": "located_at",
                "target": "location:checkpoint_delta",
            },
            data_type="edge",
        )
        store.add(
            data={
                "source": "person:cpl_smith",
                "edge_type": "reports_to",
                "target": "person:sgt_johnson",
            },
            data_type="edge",
        )
        print_success("Added 2 edges to graph")

        # ─────────────────────────────────────────────────
        # Step 4: Add Chat/Stream Data (→ Working Memory)
        # ─────────────────────────────────────────────────
        print_section("Step 4: Add Chat/Stream Data (→ Working Memory)")

        print_info("Adding chat messages...")
        store.add(
            data="Operator: All units, report status.",
            data_type="chat",
            metadata={"channel": "alpha"},
        )
        store.add(
            data="Sgt. Johnson: Alpha team in position at Checkpoint Delta.",
            data_type="chat",
            metadata={"channel": "alpha"},
        )
        store.add(
            data="Cpl. Smith: Bravo team ready, awaiting orders.",
            data_type="chat",
            metadata={"channel": "alpha"},
        )
        print_success("Added 3 chat messages to working memory")

        # ─────────────────────────────────────────────────
        # Step 5: Query Time-Series Data
        # ─────────────────────────────────────────────────
        print_section("Step 5: Query Time-Series Data")

        now = datetime.now()
        results = store.query(
            time_range={"start": now - timedelta(minutes=5), "end": now},
            data_types=["telemetry"],
        )
        print_success(f"Found {len(results)} telemetry records in last 5 minutes")
        for r in results[:2]:
            data = r.get("data", {})
            data_str = str(data) if data else ""
            print_info(f"  {data_str[:50]}...")

        # ─────────────────────────────────────────────────
        # Step 6: Query Graph Data
        # ─────────────────────────────────────────────────
        print_section("Step 6: Query Graph Data")

        results = store.query(
            graph_query={"node_id": "person:sgt_johnson", "direction": "both"},
        )
        print_success(f"Found {len(results)} relationships for Sgt. Johnson")
        for r in results:
            print_info(f"  → {r.get('id', 'unknown')}")

        # ─────────────────────────────────────────────────
        # Step 7: Query Working Memory
        # ─────────────────────────────────────────────────
        print_section("Step 7: Query Working Memory")

        results = store.query(
            recent={"limit": 10, "data_type": "chat"},
        )
        print_success(f"Found {len(results)} recent chat messages")
        for r in results[:3]:
            content = r.get("content", "")
            print_info(f"  [{r.get('data_type')}] {content[:40]}...")

        # ─────────────────────────────────────────────────
        # Step 8: Get Unified Context
        # ─────────────────────────────────────────────────
        print_section("Step 8: Get Unified Context")

        context = store.get_context(
            recent_minutes=10,
            include_graph=True,
            include_working_memory=True,
            limit=5,
        )
        print_success("Retrieved unified context:")
        print_info(f"  Working memory entries: {len(context.get('working_memory', []))}")
        print_info(f"  Graph entries: {len(context.get('graph', []))}")
        print_info(f"  Timeseries entries: {len(context.get('timeseries', []))}")

        # ─────────────────────────────────────────────────
        # Step 9: Statistics
        # ─────────────────────────────────────────────────
        print_section("Step 9: Statistics")

        stats = store.get_stats()
        print_success("Storage statistics:")
        ts_stats = stats.get("timeseries", {})
        print_info(f"  Timeseries records: {ts_stats.get('total_records', 0)}")
        graph_stats = stats.get("graph", {})
        print_info(f"  Graph nodes: {graph_stats.get('total_nodes', 0)}")
        print_info(f"  Graph edges: {graph_stats.get('total_edges', 0)}")
        wm_stats = stats.get("working_memory", {})
        print_info(f"  Working memory records: {wm_stats.get('total_records', 0)}")
        link_stats = stats.get("linkage", {})
        print_info(f"  Linkage entries: {link_stats.get('total_links', 0)}")

        # ─────────────────────────────────────────────────
        # Step 10: Cascade Delete
        # ─────────────────────────────────────────────────
        print_section("Step 10: Cascade Delete")

        print_warning(f"Deleting telemetry record: {result1['uuid'][:30]}...")
        deleted = store.delete(result1["uuid"])
        print_success(f"Deleted: {deleted}")

        # Verify
        links = store.linkage_table.get_links(result1["uuid"])
        print_success(f"Linkage entry removed: {links is None}")

        # ─────────────────────────────────────────────────
        # Cleanup
        # ─────────────────────────────────────────────────
        store.close()
        print_success("\nMemoryStore closed")

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
            "\033[0;33m  Run from the 'rag' directory: cd rag && uv run python ../examples/database/demo_memory_store.py\033[0m"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\033[0;31m✗ Error: {e}\033[0m")
        import traceback

        traceback.print_exc()
        sys.exit(1)
