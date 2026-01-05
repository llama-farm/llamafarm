#!/usr/bin/env python3
"""CLI commands for typed dataset operations.

Phase 22: CLI Updates

Provides commands for:
- Creating typed datasets (knowledge, realtime, graph, timeseries, spatial, hybrid)
- Adding stream records
- Executing hybrid queries
- Viewing dataset statistics
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


def get_unified_store(
    project_path: str, dataset_name: str, dataset_type: str = "realtime"
):
    """Get a UnifiedDatasetStore instance."""
    from core.unified_store import UnifiedDatasetStore

    return UnifiedDatasetStore(
        dataset_config={"name": dataset_name, "type": dataset_type},
        project_dir=project_path,
    )


def cmd_create_dataset(args):
    """Create a new typed dataset."""
    console.print(Panel(f"[bold blue]Creating Dataset: {args.name}[/bold blue]"))

    valid_types = ["knowledge", "realtime", "graph", "timeseries", "spatial", "hybrid"]
    if args.type not in valid_types:
        console.print(f"[red]Invalid dataset type: {args.type}[/red]")
        console.print(f"Valid types: {', '.join(valid_types)}")
        return 1

    try:
        store = get_unified_store(args.project_path, args.name, args.type)

        # Print info
        table = Table(title="Dataset Created", show_header=False)
        table.add_row("Name", args.name)
        table.add_row("Type", args.type)
        table.add_row("Path", store.base_path)
        table.add_row("Enabled Stores", ", ".join(store.get_enabled_stores()))

        console.print(table)
        store.close()
        return 0

    except Exception as e:
        console.print(f"[red]Error creating dataset: {e}[/red]")
        return 1


def cmd_add_record(args):
    """Add a stream record to a dataset."""
    console.print(Panel(f"[bold blue]Adding Record to: {args.dataset}[/bold blue]"))

    try:
        store = get_unified_store(args.project_path, args.dataset, args.type)

        # Parse data
        data = json.loads(args.data) if args.data else {}

        result = store.add_stream_record(
            data=data,
            data_type=args.data_type,
            latitude=args.latitude,
            longitude=args.longitude,
            metadata=json.loads(args.metadata) if args.metadata else None,
        )

        # Print result
        table = Table(title="Record Added", show_header=False)
        table.add_row("Record ID", result.get("record_id", "N/A"))
        table.add_row("Stores", ", ".join(result.get("stores", [])))

        console.print(table)
        store.close()
        return 0

    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Error adding record: {e}[/red]")
        return 1


def cmd_query(args):
    """Execute a hybrid query on a dataset."""
    console.print(Panel(f"[bold blue]Querying Dataset: {args.dataset}[/bold blue]"))

    try:
        store = get_unified_store(args.project_path, args.dataset, args.type)

        from core.hybrid_query import HybridQueryExecutor, HybridQueryRequest, QueryMode

        executor = HybridQueryExecutor(store)

        request = HybridQueryRequest(
            query_text=args.query_text,
            graph_node_id=args.graph_node,
            latitude=args.latitude,
            longitude=args.longitude,
            radius_meters=args.radius or 1000.0,
            mode=QueryMode(args.mode),
            limit=args.limit,
        )

        response = executor.execute(request)

        # Print results
        console.print(f"\n[bold]Results ({response.total_count} total)[/bold]")
        console.print(f"Stores queried: {', '.join(response.stores_queried)}")
        console.print(f"Execution time: {response.execution_time_ms:.2f}ms\n")

        for i, result in enumerate(response.results, 1):
            tree = Tree(
                f"[bold cyan]Result {i}[/bold cyan] (score: {result.score:.3f})"
            )
            tree.add(f"Source: {result.source_store}")
            tree.add(f"Content: {result.content}")
            if result.metadata:
                tree.add(f"Metadata: {result.metadata}")
            console.print(tree)

        store.close()
        return 0

    except Exception as e:
        console.print(f"[red]Error querying dataset: {e}[/red]")
        return 1


def cmd_stats(args):
    """Show dataset statistics."""
    console.print(Panel(f"[bold blue]Dataset Statistics: {args.dataset}[/bold blue]"))

    try:
        store = get_unified_store(args.project_path, args.dataset, args.type)

        stats = store.get_stats()

        # Main info
        table = Table(title="Dataset Info", show_header=False)
        table.add_row("Name", stats.get("dataset_name", "N/A"))
        table.add_row("Type", stats.get("dataset_type", "N/A"))
        table.add_row("Path", stats.get("base_path", "N/A"))
        table.add_row("Enabled Stores", ", ".join(stats.get("enabled_stores", [])))
        console.print(table)

        # Store-specific stats
        stores = stats.get("stores", {})
        if stores:
            console.print("\n[bold]Store Statistics:[/bold]")
            for store_name, store_stats in stores.items():
                console.print(f"\n[cyan]{store_name.upper()}:[/cyan]")
                if isinstance(store_stats, dict):
                    for key, value in store_stats.items():
                        console.print(f"  {key}: {value}")
                else:
                    console.print(f"  {store_stats}")

        store.close()
        return 0

    except Exception as e:
        console.print(f"[red]Error getting stats: {e}[/red]")
        return 1


def cmd_add_node(args):
    """Add a node to the graph store."""
    console.print(Panel(f"[bold blue]Adding Node to: {args.dataset}[/bold blue]"))

    try:
        store = get_unified_store(args.project_path, args.dataset, args.type)

        properties = json.loads(args.properties) if args.properties else {}

        node_id = store.add_node(
            name=args.name,
            node_type=args.node_type,
            properties=properties,
        )

        if node_id:
            console.print(f"[green]Node created with ID: {node_id}[/green]")
        else:
            console.print("[yellow]Graph store not enabled for this dataset[/yellow]")

        store.close()
        return 0

    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Error adding node: {e}[/red]")
        return 1


def cmd_add_edge(args):
    """Add an edge to the graph store."""
    console.print(Panel(f"[bold blue]Adding Edge to: {args.dataset}[/bold blue]"))

    try:
        store = get_unified_store(args.project_path, args.dataset, args.type)

        properties = json.loads(args.properties) if args.properties else {}

        edge_id = store.add_edge(
            source_id=args.source,
            target_id=args.target,
            relationship=args.relationship,
            weight=args.weight,
            properties=properties,
        )

        if edge_id:
            console.print(f"[green]Edge created with ID: {edge_id}[/green]")
        else:
            console.print("[yellow]Graph store not enabled for this dataset[/yellow]")

        store.close()
        return 0

    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Error adding edge: {e}[/red]")
        return 1


def cmd_clear(args):
    """Clear all data from dataset stores."""
    console.print(Panel(f"[bold red]Clearing Dataset: {args.dataset}[/bold red]"))

    if not args.confirm:
        console.print("[yellow]Use --confirm to actually clear the data[/yellow]")
        return 1

    try:
        store = get_unified_store(args.project_path, args.dataset, args.type)

        result = store.clear()

        console.print("[green]Cleared stores:[/green]")
        for store_name, success in result.items():
            status = "✅" if success else "❌"
            console.print(f"  {status} {store_name}")

        store.close()
        return 0

    except Exception as e:
        console.print(f"[red]Error clearing dataset: {e}[/red]")
        return 1


def cmd_entity_extract(args):
    """Extract entities from text to graph."""
    console.print(Panel("[bold blue]Entity Extraction[/bold blue]"))

    try:
        from components.extractors.entity_extractor import EntityExtractor
        from core.base import Document

        extractor = EntityExtractor(
            config={
                "entity_types": args.entity_types.split(",")
                if args.entity_types
                else None,
                "min_entity_length": args.min_length,
            }
        )

        doc = Document(id="cli-input", content=args.text)
        entities = extractor.extract_entities(doc)

        console.print(f"\n[bold]Found {len(entities)} entities:[/bold]\n")

        table = Table(title="Extracted Entities")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Confidence", style="green")
        table.add_column("Method", style="dim")

        for entity in entities:
            table.add_row(
                entity.name,
                entity.entity_type,
                f"{entity.confidence:.2f}",
                entity.method,
            )

        console.print(table)
        return 0

    except Exception as e:
        console.print(f"[red]Error extracting entities: {e}[/red]")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="LlamaFarm Typed Dataset CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a realtime dataset
  python dataset_cli.py create --name telemetry --type realtime

  # Add a stream record
  python dataset_cli.py add-record --dataset telemetry --data '{"temp": 72}'

  # Query the dataset
  python dataset_cli.py query --dataset telemetry --mode context --limit 10

  # View statistics
  python dataset_cli.py stats --dataset telemetry

  # Extract entities from text
  python dataset_cli.py entity-extract --text "John works at Apple in San Francisco."
        """,
    )

    parser.add_argument(
        "--project-path",
        default=".",
        help="Project directory path (default: current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # create command
    create_parser = subparsers.add_parser("create", help="Create a new typed dataset")
    create_parser.add_argument("--name", required=True, help="Dataset name")
    create_parser.add_argument(
        "--type",
        default="realtime",
        choices=["knowledge", "realtime", "graph", "timeseries", "spatial", "hybrid"],
        help="Dataset type",
    )

    # add-record command
    record_parser = subparsers.add_parser("add-record", help="Add a stream record")
    record_parser.add_argument("--dataset", required=True, help="Dataset name")
    record_parser.add_argument("--type", default="realtime", help="Dataset type")
    record_parser.add_argument("--data", help="JSON data payload")
    record_parser.add_argument("--data-type", default="telemetry", help="Data type")
    record_parser.add_argument("--latitude", type=float, help="Latitude")
    record_parser.add_argument("--longitude", type=float, help="Longitude")
    record_parser.add_argument("--metadata", help="JSON metadata")

    # query command
    query_parser = subparsers.add_parser("query", help="Execute a hybrid query")
    query_parser.add_argument("--dataset", required=True, help="Dataset name")
    query_parser.add_argument("--type", default="realtime", help="Dataset type")
    query_parser.add_argument("--query-text", help="Text for semantic search")
    query_parser.add_argument("--graph-node", help="Node ID for graph traversal")
    query_parser.add_argument(
        "--latitude", type=float, help="Latitude for spatial query"
    )
    query_parser.add_argument(
        "--longitude", type=float, help="Longitude for spatial query"
    )
    query_parser.add_argument("--radius", type=float, help="Radius in meters")
    query_parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "vector", "graph", "timeseries", "spatial", "context"],
        help="Query mode",
    )
    query_parser.add_argument("--limit", type=int, default=10, help="Max results")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("--dataset", required=True, help="Dataset name")
    stats_parser.add_argument("--type", default="realtime", help="Dataset type")

    # add-node command
    node_parser = subparsers.add_parser("add-node", help="Add a graph node")
    node_parser.add_argument("--dataset", required=True, help="Dataset name")
    node_parser.add_argument("--type", default="knowledge", help="Dataset type")
    node_parser.add_argument("--name", required=True, help="Node name")
    node_parser.add_argument("--node-type", default="entity", help="Node type")
    node_parser.add_argument("--properties", help="JSON properties")

    # add-edge command
    edge_parser = subparsers.add_parser("add-edge", help="Add a graph edge")
    edge_parser.add_argument("--dataset", required=True, help="Dataset name")
    edge_parser.add_argument("--type", default="knowledge", help="Dataset type")
    edge_parser.add_argument("--source", required=True, help="Source node ID")
    edge_parser.add_argument("--target", required=True, help="Target node ID")
    edge_parser.add_argument(
        "--relationship", default="related_to", help="Relationship type"
    )
    edge_parser.add_argument("--weight", type=float, default=1.0, help="Edge weight")
    edge_parser.add_argument("--properties", help="JSON properties")

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear dataset stores")
    clear_parser.add_argument("--dataset", required=True, help="Dataset name")
    clear_parser.add_argument("--type", default="realtime", help="Dataset type")
    clear_parser.add_argument("--confirm", action="store_true", help="Confirm deletion")

    # entity-extract command
    extract_parser = subparsers.add_parser(
        "entity-extract", help="Extract entities from text"
    )
    extract_parser.add_argument(
        "--text", required=True, help="Text to extract entities from"
    )
    extract_parser.add_argument("--entity-types", help="Comma-separated entity types")
    extract_parser.add_argument(
        "--min-length", type=int, default=2, help="Min entity length"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    command_map = {
        "create": cmd_create_dataset,
        "add-record": cmd_add_record,
        "query": cmd_query,
        "stats": cmd_stats,
        "add-node": cmd_add_node,
        "add-edge": cmd_add_edge,
        "clear": cmd_clear,
        "entity-extract": cmd_entity_extract,
    }

    if args.command in command_map:
        return command_map[args.command](args)
    else:
        console.print(f"[red]Unknown command: {args.command}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
