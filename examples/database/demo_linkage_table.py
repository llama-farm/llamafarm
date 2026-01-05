#!/usr/bin/env python3
"""Linkage Table Demo - Cross-Database Record Linking.

Demonstrates:
1. Creating linked records across all 3 stores
2. Retrieving by UUID
3. Finding UUID from any component ID
4. Cascade delete simulation

This is part of the Embedded Trinity Memory System.
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
    from components.stores.duckdb_store import LinkageTable

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = f"{temp_dir}/demo_linkage.duckdb"

        # ─────────────────────────────────────────────────
        # Step 1: Initialize Linkage Table
        # ─────────────────────────────────────────────────
        print_section("Step 1: Initialize Linkage Table")

        config = {"path": db_path}

        print_info(f"Creating Linkage Table at: {db_path}")
        table = LinkageTable(config=config)
        print_success("Linkage Table initialized")

        # ─────────────────────────────────────────────────
        # Step 2: Create Linked Records (Simulated)
        # ─────────────────────────────────────────────────
        print_section("Step 2: Create Linked Records")

        # Scenario: A "Rescue Event" has data in all three stores
        print_info("Creating 'Rescue Event' concept with links to all stores...")

        rescue_uuid = table.link(
            concept_uuid="rescue_event_001",
            vector_id="chroma_doc_rescue_summary_001",
            graph_node_id="graph_node_rescue_event_001",
            timeseries_row_id="duckdb_telemetry_batch_001",
        )
        print_success(f"Created concept: {rescue_uuid}")

        # Soldier profile - links to vector (embeddings) and graph (relationships)
        print_info("Creating 'Soldier Profile' concept...")
        soldier_uuid = table.link(
            concept_uuid="soldier_sgt_johnson",
            vector_id="chroma_doc_soldier_profile_001",
            graph_node_id="graph_node_person_sgt_johnson",
        )
        print_success(f"Created concept: {soldier_uuid}")

        # Location - links to graph and timeseries (geo data)
        print_info("Creating 'Checkpoint Delta' location concept...")
        location_uuid = table.link(
            concept_uuid="location_checkpoint_delta",
            graph_node_id="graph_node_location_delta",
            timeseries_row_id="duckdb_geo_checkpoint_delta",
        )
        print_success(f"Created concept: {location_uuid}")

        # Medical protocol - only in vector store
        print_info("Creating 'Medical Protocol' concept (vector only)...")
        protocol_uuid = table.link(
            vector_id="chroma_doc_medevac_protocol_001",
        )
        print_success(f"Created concept: {protocol_uuid}")

        # Telemetry data - only in timeseries
        print_info("Creating 'Biometric Alert' concept (timeseries only)...")
        alert_uuid = table.link(
            timeseries_row_id="duckdb_alert_hr_spike_001",
        )
        print_success(f"Created concept: {alert_uuid}")

        # ─────────────────────────────────────────────────
        # Step 3: Retrieve by UUID
        # ─────────────────────────────────────────────────
        print_section("Step 3: Retrieve by UUID")

        print_info("Getting links for 'rescue_event_001'...")
        links = table.get_links(rescue_uuid)

        print_success("Links found:")
        print(f"    UUID:          {links['uuid']}")
        print(f"    Vector ID:     {links['vector_id']}")
        print(f"    Graph Node ID: {links['graph_node_id']}")
        print(f"    Timeseries ID: {links['timeseries_row_id']}")

        print_info("\nGetting links for 'soldier_sgt_johnson'...")
        links = table.get_links(soldier_uuid)
        print_success("Links found:")
        print(f"    UUID:          {links['uuid']}")
        print(f"    Vector ID:     {links['vector_id']}")
        print(f"    Graph Node ID: {links['graph_node_id']}")
        print(f"    Timeseries ID: {links['timeseries_row_id'] or '(none)'}")

        # ─────────────────────────────────────────────────
        # Step 4: Find UUID from Any Component ID
        # ─────────────────────────────────────────────────
        print_section("Step 4: Find UUID from Any Component ID")

        print_info("Finding UUID from vector ID 'chroma_doc_soldier_profile_001'...")
        found_uuid = table.find_by_any_id(vector_id="chroma_doc_soldier_profile_001")
        print_success(f"Found: {found_uuid}")

        print_info("Finding UUID from graph node 'graph_node_location_delta'...")
        found_uuid = table.find_by_any_id(graph_node_id="graph_node_location_delta")
        print_success(f"Found: {found_uuid}")

        print_info("Finding UUID from timeseries 'duckdb_alert_hr_spike_001'...")
        found_uuid = table.find_by_any_id(timeseries_row_id="duckdb_alert_hr_spike_001")
        print_success(f"Found: {found_uuid}")

        # ─────────────────────────────────────────────────
        # Step 5: List All Links
        # ─────────────────────────────────────────────────
        print_section("Step 5: List All Links")

        all_links = table.list_all()
        print_success(f"Total links: {len(all_links)}")
        print_info("Summary:")
        for link in all_links:
            components = []
            if link["vector_id"]:
                components.append("vector")
            if link["graph_node_id"]:
                components.append("graph")
            if link["timeseries_row_id"]:
                components.append("timeseries")
            print(f"    {link['uuid'][:30]}... -> [{', '.join(components)}]")

        # ─────────────────────────────────────────────────
        # Step 6: Statistics
        # ─────────────────────────────────────────────────
        print_section("Step 6: Statistics")

        stats = table.get_stats()
        print_success(f"Total links: {stats['total_links']}")
        print_success(f"Links with vector store: {stats['links_with_vector']}")
        print_success(f"Links with graph store: {stats['links_with_graph']}")
        print_success(f"Links with timeseries: {stats['links_with_timeseries']}")

        # ─────────────────────────────────────────────────
        # Step 7: Cascade Delete Simulation
        # ─────────────────────────────────────────────────
        print_section("Step 7: Cascade Delete Simulation")

        print_warning("Simulating cascade delete of 'rescue_event_001'...")
        print_info("In production, this would delete from all linked stores")

        # Get IDs before unlinking
        ids = table.unlink_and_get_ids(rescue_uuid)

        if ids:
            print_success("Retrieved IDs for cascade delete:")
            print(f"    Would delete from ChromaDB: {ids.get('vector_id', '(none)')}")
            print(f"    Would delete from GraphStore: {ids.get('graph_node_id', '(none)')}")
            print(f"    Would delete from DuckDB: {ids.get('timeseries_row_id', '(none)')}")

            # Verify unlinked
            verify = table.get_links(rescue_uuid)
            print_success(f"Link removed: {verify is None}")

        # Check final count
        stats = table.get_stats()
        print_success(f"Remaining links: {stats['total_links']}")

        # ─────────────────────────────────────────────────
        # Cleanup
        # ─────────────────────────────────────────────────
        table.close()
        print_success("\nLinkage Table closed")

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
            "\033[0;33m  Run from the 'rag' directory: cd rag && uv run python ../examples/database/demo_linkage_table.py\033[0m"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\033[0;31m✗ Error: {e}\033[0m")
        import traceback

        traceback.print_exc()
        sys.exit(1)
