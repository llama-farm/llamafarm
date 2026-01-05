#!/usr/bin/env python3
"""Graph Store Demo - Entity Relationships and Path Finding.

Demonstrates:
1. Adding nodes (entities) with properties
2. Creating edges (relationships) between nodes
3. Finding neighbors and traversing relationships
4. Path finding between entities
5. Querying by node type and edge type

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


def main() -> int:
    from components.stores.duckdb_store import GraphStore

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = f"{temp_dir}/demo_graph.duckdb"

        # ─────────────────────────────────────────────────
        # Step 1: Initialize Graph Store
        # ─────────────────────────────────────────────────
        print_section("Step 1: Initialize Graph Store")

        config = {"path": db_path}
        graph = GraphStore(config=config)
        print_success(f"Graph Store initialized at: {db_path}")

        # ─────────────────────────────────────────────────
        # Step 2: Build a Military Operation Knowledge Graph
        # ─────────────────────────────────────────────────
        print_section("Step 2: Build Knowledge Graph")

        # Add personnel
        print_info("Adding personnel nodes...")
        graph.add_node("person:sgt_johnson", "person", {
            "name": "Sgt. Johnson",
            "rank": "Sergeant",
            "unit": "Alpha",
            "specialty": "medic",
        })
        graph.add_node("person:cpl_smith", "person", {
            "name": "Cpl. Smith",
            "rank": "Corporal",
            "unit": "Alpha",
            "specialty": "communications",
        })
        graph.add_node("person:lt_chen", "person", {
            "name": "Lt. Chen",
            "rank": "Lieutenant",
            "unit": "Alpha",
            "specialty": "command",
        })
        print_success("Added 3 personnel nodes")

        # Add locations
        print_info("Adding location nodes...")
        graph.add_node("location:checkpoint_alpha", "location", {
            "name": "Checkpoint Alpha",
            "coordinates": "35.7796,-78.6382",
            "type": "checkpoint",
        })
        graph.add_node("location:checkpoint_delta", "location", {
            "name": "Checkpoint Delta",
            "coordinates": "35.7850,-78.6400",
            "type": "checkpoint",
        })
        graph.add_node("location:base_camp", "location", {
            "name": "Base Camp",
            "coordinates": "35.7700,-78.6300",
            "type": "base",
        })
        print_success("Added 3 location nodes")

        # Add events
        print_info("Adding event nodes...")
        graph.add_node("event:rescue_001", "event", {
            "type": "rescue",
            "timestamp": "2024-01-15T14:30:00Z",
            "status": "completed",
        })
        print_success("Added 1 event node")

        # ─────────────────────────────────────────────────
        # Step 3: Create Relationships
        # ─────────────────────────────────────────────────
        print_section("Step 3: Create Relationships")

        # Command structure
        print_info("Creating command relationships...")
        graph.add_edge("person:lt_chen", "commands", "person:sgt_johnson")
        graph.add_edge("person:lt_chen", "commands", "person:cpl_smith")
        print_success("Lt. Chen commands Sgt. Johnson and Cpl. Smith")

        # Location assignments
        print_info("Creating location assignments...")
        graph.add_edge("person:sgt_johnson", "located_at", "location:checkpoint_delta")
        graph.add_edge("person:cpl_smith", "located_at", "location:checkpoint_alpha")
        graph.add_edge("person:lt_chen", "located_at", "location:base_camp")
        print_success("Personnel assigned to locations")

        # Event participation
        print_info("Creating event participation...")
        graph.add_edge("person:sgt_johnson", "participated_in", "event:rescue_001")
        graph.add_edge("event:rescue_001", "occurred_at", "location:checkpoint_delta")
        print_success("Event relationships created")

        # Location connectivity
        print_info("Creating route connections...")
        graph.add_edge("location:base_camp", "route_to", "location:checkpoint_alpha")
        graph.add_edge("location:checkpoint_alpha", "route_to", "location:checkpoint_delta")
        print_success("Routes established")

        # ─────────────────────────────────────────────────
        # Step 4: Query Neighbors
        # ─────────────────────────────────────────────────
        print_section("Step 4: Query Neighbors")

        # Who does Lt. Chen command?
        print_info("Finding who Lt. Chen commands...")
        subordinates = graph.find_neighbors(
            "person:lt_chen",
            relationship="commands",
            direction="outgoing",
        )
        print_success(f"Lt. Chen commands {len(subordinates)} personnel:")
        for node in subordinates:
            props = node.get("properties", {})
            print(f"    - {props.get('name', node['id'])}")

        # Where is Sgt. Johnson?
        print_info("\nFinding Sgt. Johnson's location...")
        locations = graph.find_neighbors(
            "person:sgt_johnson",
            relationship="located_at",
            direction="outgoing",
        )
        if locations:
            loc = locations[0]
            props = loc.get("properties", {})
            print_success(f"Sgt. Johnson is at: {props.get('name', loc['id'])}")

        # ─────────────────────────────────────────────────
        # Step 5: Path Finding
        # ─────────────────────────────────────────────────
        print_section("Step 5: Path Finding")

        # Find route from base camp to checkpoint delta
        print_info("Finding route from Base Camp to Checkpoint Delta...")
        path = graph.find_path(
            "location:base_camp",
            "location:checkpoint_delta",
            max_depth=5,
        )

        if path:
            print_success(f"Route found with {len(path)} stops:")
            for node in path:
                props = node.get("properties", {})
                print(f"    → {props.get('name', node['id'])}")
        else:
            print_info("No route found")

        # ─────────────────────────────────────────────────
        # Step 6: Query Edges
        # ─────────────────────────────────────────────────
        print_section("Step 6: Query Edges")

        print_info("Finding all edges of type 'located_at'...")
        assignments = graph.get_edges(relationship="located_at")
        print_success(f"Found {len(assignments)} location assignments")
        for edge in assignments:
            print(f"    - {edge['source_id']} -> {edge['target_id']}")

        # ─────────────────────────────────────────────────
        # Step 7: Statistics
        # ─────────────────────────────────────────────────
        print_section("Step 7: Statistics")

        stats = graph.get_stats()
        print_success(f"Total nodes: {stats['node_count']}")
        print_success(f"Total edges: {stats['edge_count']}")
        print_success(f"Node types: {stats.get('node_types', {})}")
        print_success(f"Relationship types: {stats.get('relationship_types', {})}")

        # ─────────────────────────────────────────────────
        # Cleanup
        # ─────────────────────────────────────────────────
        graph.close()
        print_success("\nGraph Store closed")

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
            "\033[0;33m  Run from the 'rag' directory: cd rag && uv run python ../examples/database/demo_graph_store.py\033[0m"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\033[0;31m✗ Error: {e}\033[0m")
        import traceback

        traceback.print_exc()
        sys.exit(1)
