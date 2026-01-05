#!/usr/bin/env python3
"""Consolidator Demo - Memory Synthesis Agent.

Demonstrates:
1. Reading raw data from WorkingMemory
2. Synthesizing facts (rule-based, no LLM required)
3. Creating graph nodes from extracted facts
4. Pruning processed records
5. Running full consolidation cycle

This is the "hippocampus" of the Embedded Trinity Memory System.
"""

import sys
import tempfile

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
    from core.consolidator import Consolidator
    from core.memory import MemoryStore

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        # ─────────────────────────────────────────────────
        # Step 1: Initialize MemoryStore and Consolidator
        # ─────────────────────────────────────────────────
        print_section("Step 1: Initialize MemoryStore and Consolidator")

        memory_config = {"base_path": temp_dir}
        memory = MemoryStore(config=memory_config)
        print_success("MemoryStore initialized")

        consolidator_config = {
            "buffer_threshold": 5,
            "retention_days": 7,
        }
        consolidator = Consolidator(memory_store=memory, config=consolidator_config)
        print_success("Consolidator initialized")
        print_info(f"Buffer threshold: {consolidator.buffer_threshold}")

        # ─────────────────────────────────────────────────
        # Step 2: Populate Working Memory with Raw Data
        # ─────────────────────────────────────────────────
        print_section("Step 2: Populate Working Memory with Raw Data")

        # Simulate incoming data streams
        chat_messages = [
            "Operator: Alpha team, report status.",
            "Sgt. Johnson: Alpha team in position at Checkpoint Delta.",
            "Cpl. Smith: Bravo team moving to sector 7.",
            "Sgt. Johnson: Contact! Two hostiles at grid reference 35.7800, -78.6400.",
            "Medic: Casualty reported. Requesting medevac at Checkpoint Delta.",
            "Command: Medevac dispatched. ETA 10 minutes.",
            "Sgt. Johnson: Hostiles neutralized. Area secure.",
            "Cpl. Smith: Bravo team providing perimeter security.",
        ]

        print_info("Adding chat messages to working memory...")
        for msg in chat_messages:
            memory.add(data=msg, data_type="chat", metadata={"channel": "tactical"})
        print_success(f"Added {len(chat_messages)} chat messages")

        # ─────────────────────────────────────────────────
        # Step 3: Get Pending Records
        # ─────────────────────────────────────────────────
        print_section("Step 3: Get Pending Records")

        pending = consolidator.get_pending_records(limit=20)
        print_success(f"Found {len(pending)} pending records")

        print_info("Sample records:")
        for r in pending[:3]:
            content = r.get("content", "")[:40]
            print(f"    [{r.get('data_type')}] {content}...")

        # ─────────────────────────────────────────────────
        # Step 4: Synthesize Facts (Rule-Based)
        # ─────────────────────────────────────────────────
        print_section("Step 4: Synthesize Facts (Rule-Based)")

        print_info("Running rule-based fact extraction...")
        result = consolidator.synthesize(pending, use_llm=False)

        print_success(f"Extracted {len(result.get('facts', []))} facts")
        print_success(f"Summary: {result.get('summary', 'N/A')[:60]}...")

        if result.get("facts"):
            print_info("Sample facts:")
            for fact in result["facts"][:3]:
                print(f"    {fact}")

        # ─────────────────────────────────────────────────
        # Step 5: Create Graph Nodes from Facts
        # ─────────────────────────────────────────────────
        print_section("Step 5: Create Graph Nodes from Facts")

        facts = result.get("facts", [])
        if facts:
            nodes_created = consolidator.create_graph_nodes(facts)
            print_success(f"Created {nodes_created} graph nodes/edges from facts")

            # Check graph stats
            graph_stats = memory.graph_store.get_stats()
            print_info(f"Graph now has {graph_stats.get('total_nodes', 0)} nodes")
            print_info(f"Graph now has {graph_stats.get('total_edges', 0)} edges")
        else:
            print_warning("No facts to convert to graph nodes")

        # ─────────────────────────────────────────────────
        # Step 6: Run Full Consolidation Cycle
        # ─────────────────────────────────────────────────
        print_section("Step 6: Run Full Consolidation Cycle")

        # Add more data to trigger threshold
        print_info("Adding more records to trigger consolidation...")
        for i in range(5):
            memory.add(data=f"Additional message {i}", data_type="chat")

        cycle_result = consolidator.run_cycle(use_llm=False)

        print_success(f"Records processed: {cycle_result.get('records_processed', 0)}")
        print_success(f"Facts extracted: {cycle_result.get('facts_extracted', 0)}")
        print_success(f"Graph nodes created: {cycle_result.get('nodes_created', 0)}")

        if cycle_result.get("skipped"):
            print_warning("Cycle skipped (below threshold)")

        # ─────────────────────────────────────────────────
        # Step 7: Final Statistics
        # ─────────────────────────────────────────────────
        print_section("Step 7: Final Statistics")

        stats = memory.get_stats()

        print_success("Final storage statistics:")
        wm = stats.get("working_memory", {})
        print_info(f"  Working memory: {wm.get('total_records', 0)} records")

        graph = stats.get("graph", {})
        print_info(f"  Graph nodes: {graph.get('total_nodes', 0)}")
        print_info(f"  Graph edges: {graph.get('total_edges', 0)}")

        linkage = stats.get("linkage", {})
        print_info(f"  Linkage entries: {linkage.get('total_links', 0)}")

        # ─────────────────────────────────────────────────
        # Cleanup
        # ─────────────────────────────────────────────────
        memory.close()
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
            "\033[0;33m  Run from the 'rag' directory: cd rag && uv run python ../examples/database/demo_consolidator.py\033[0m"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\033[0;31m✗ Error: {e}\033[0m")
        import traceback

        traceback.print_exc()
        sys.exit(1)
