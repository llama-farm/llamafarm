#!/usr/bin/env python3
"""
CLI Component Schema Compiler

This tool scans all components and generates master schemas for discovery,
validation, and easy extension of the RAG system.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def discover_components(base_path: str, component_type: str) -> List[Dict[str, Any]]:
    """Discover all components of a given type."""
    components = []
    components_path = Path(base_path) / f"{component_type}s_new"
    
    if not components_path.exists():
        console.print(f"[yellow]No {component_type}s_new directory found at {components_path}[/yellow]")
        return components
    
    for component_dir in components_path.iterdir():
        if component_dir.is_dir() and not component_dir.name.startswith('.'):
            schema_file = component_dir / "schema.json"
            defaults_file = component_dir / "defaults.json"
            
            if schema_file.exists():
                try:
                    with open(schema_file, 'r') as f:
                        schema = json.load(f)
                    
                    defaults = {}
                    if defaults_file.exists():
                        with open(defaults_file, 'r') as f:
                            defaults = json.load(f)
                    
                    component_info = {
                        "name": schema.get("name", component_dir.name),
                        "type": component_type,
                        "path": str(component_dir),
                        "schema": schema,
                        "defaults": defaults,
                        "has_implementation": (component_dir / f"{component_dir.name}.py").exists()
                    }
                    components.append(component_info)
                    
                except json.JSONDecodeError as e:
                    console.print(f"[red]Error parsing {schema_file}: {e}[/red]")
                except Exception as e:
                    console.print(f"[red]Error processing {component_dir}: {e}[/red]")
    
    return components

def generate_master_schema(components: List[Dict[str, Any]], component_type: str) -> Dict[str, Any]:
    """Generate master schema for a component type."""
    master_schema = {
        "type": component_type,
        "version": "1.0.0",
        "generated_by": "cli_compile.py",
        "components": {},
        "defaults_registry": {},
        "use_case_index": {},
        "dependency_graph": {}
    }
    
    for component in components:
        comp_name = component["name"]
        schema = component["schema"]
        defaults = component["defaults"]
        
        # Add component schema
        master_schema["components"][comp_name] = {
            "schema": schema,
            "path": component["path"],
            "has_implementation": component["has_implementation"]
        }
        
        # Add defaults
        if defaults:
            master_schema["defaults_registry"][comp_name] = defaults
        
        # Build use case index
        use_cases = schema.get("use_cases", [])
        for use_case in use_cases:
            if use_case not in master_schema["use_case_index"]:
                master_schema["use_case_index"][use_case] = []
            master_schema["use_case_index"][use_case].append(comp_name)
        
        # Build dependency graph
        deps = schema.get("dependencies", {})
        if deps.get("required") or deps.get("optional"):
            master_schema["dependency_graph"][comp_name] = deps
    
    return master_schema

def generate_strategy_recommendations(extractors: List[Dict], parsers: List[Dict]) -> Dict[str, Any]:
    """Generate recommended component combinations for different strategies."""
    strategies = {
        "research_papers": {
            "name": "Research Paper Analysis",
            "description": "Optimized for academic and research content",
            "components": {
                "parser": "text_parser",
                "extractors": ["entity_extractor", "summary_extractor", "statistics_extractor"],
                "embedder": "ollama_embedder",
                "store": "chroma_store"
            },
            "configs": {
                "entity_extractor": "research_papers",
                "summary_extractor": "detailed_summary",
                "statistics_extractor": "research_analysis"
            }
        },
        "customer_support": {
            "name": "Customer Support System",
            "description": "Optimized for support tickets and knowledge base",
            "components": {
                "parser": ["csv_parser", "text_parser"],
                "extractors": ["entity_extractor", "pattern_extractor", "summary_extractor"],
                "embedder": "ollama_embedder", 
                "store": "chroma_store"
            },
            "configs": {
                "entity_extractor": "general_purpose",
                "pattern_extractor": "document_processing",
                "summary_extractor": "brief_summary"
            }
        },
        "news_analysis": {
            "name": "News Article Analysis",
            "description": "Optimized for news and media content",
            "components": {
                "parser": "html_parser",
                "extractors": ["entity_extractor", "summary_extractor", "link_extractor"],
                "embedder": "ollama_embedder",
                "store": "chroma_store"
            },
            "configs": {
                "entity_extractor": "news_analysis",
                "summary_extractor": "news_summary",
                "link_extractor": "web_analysis"
            }
        },
        "business_reports": {
            "name": "Business Report Analysis",
            "description": "Optimized for business and financial documents",
            "components": {
                "parser": ["csv_parser", "text_parser"],
                "extractors": ["entity_extractor", "summary_extractor", "statistics_extractor", "table_extractor"],
                "embedder": "ollama_embedder",
                "store": "chroma_store"
            },
            "configs": {
                "entity_extractor": "business_reports",
                "summary_extractor": "detailed_summary",
                "statistics_extractor": "financial_analysis",
                "table_extractor": "financial_tables"
            }
        },
        "code_documentation": {
            "name": "Code Documentation Analysis", 
            "description": "Optimized for technical documentation",
            "components": {
                "parser": "markdown_parser",
                "extractors": ["link_extractor", "heading_extractor", "pattern_extractor"],
                "embedder": "ollama_embedder",
                "store": "chroma_store"
            },
            "configs": {
                "link_extractor": "web_analysis",
                "heading_extractor": "technical_docs",
                "pattern_extractor": "web_analysis"
            }
        }
    }
    
    return {
        "version": "1.0.0",
        "strategies": strategies,
        "use_case_mapping": {
            "Document summarization": ["research_papers", "business_reports"],
            "Customer service": ["customer_support"],
            "Content analysis": ["news_analysis"],
            "Technical documentation": ["code_documentation"],
            "Entity extraction": ["research_papers", "customer_support", "news_analysis", "business_reports"],
            "Pattern recognition": ["customer_support", "code_documentation"]
        }
    }

def print_component_summary(components: List[Dict[str, Any]], component_type: str):
    """Print a summary table of discovered components."""
    if not components:
        console.print(f"[yellow]No {component_type}s found[/yellow]")
        return
    
    table = Table(title=f"Discovered {component_type.title()}s", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Use Cases", style="dim")
    table.add_column("Defaults", style="green")
    table.add_column("Implementation", style="yellow")
    
    for comp in components:
        schema = comp["schema"]
        defaults = comp["defaults"]
        
        name = schema.get("name", "Unknown")
        description = schema.get("description", "No description")[:50] + "..."
        use_cases = len(schema.get("use_cases", []))
        default_configs = len(defaults)
        has_impl = "✅" if comp["has_implementation"] else "❌"
        
        table.add_row(
            name,
            description,
            f"{use_cases} cases",
            f"{default_configs} configs", 
            has_impl
        )
    
    console.print(table)

def compile_components(base_path: str = ".", output_dir: str = "./schemas", verbose: bool = False):
    """Compile RAG component schemas programmatically."""
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    console.print(Panel("[bold blue]RAG Component Schema Compiler[/bold blue]", expand=False))
    console.print(f"🔍 Scanning components in: {base_path.absolute()}")
    console.print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Discover all component types
    component_types = ["extractor", "parser", "embedder", "store", "retriever"]
    all_components = {}
    
    for comp_type in component_types:
        console.print(f"\n🔎 Discovering {comp_type}s...")
        components = discover_components(str(base_path / "components"), comp_type)
        all_components[comp_type] = components
        
        if verbose:
            print_component_summary(components, comp_type)
        else:
            console.print(f"   Found {len(components)} {comp_type}(s)")
        
        if components:
            # Generate master schema
            master_schema = generate_master_schema(components, comp_type)
            schema_file = output_dir / f"{comp_type}s.json"
            
            with open(schema_file, 'w') as f:
                json.dump(master_schema, f, indent=2)
            
            console.print(f"   ✅ Generated {schema_file}")
    
    # Generate strategy recommendations
    console.print(f"\n🎯 Generating strategy recommendations...")
    strategies = generate_strategy_recommendations(
        all_components.get("extractor", []),
        all_components.get("parser", [])
    )
    
    strategy_file = output_dir / "strategies.json"
    with open(strategy_file, 'w') as f:
        json.dump(strategies, f, indent=2)
    
    console.print(f"   ✅ Generated {strategy_file}")
    
    # Summary
    console.print(f"\n📊 [bold green]Compilation Complete![/bold green]")
    total_components = sum(len(comps) for comps in all_components.values())
    console.print(f"   📦 Total components: {total_components}")
    console.print(f"   🎯 Strategy recommendations: {len(strategies['strategies'])}")
    console.print(f"   📁 Schema files generated in: {output_dir.absolute()}")
    
    return all_components

def main():
    parser = argparse.ArgumentParser(description="Compile RAG component schemas")
    parser.add_argument("--base-path", default=".", help="Base path to search for components")
    parser.add_argument("--output-dir", default="./schemas", help="Output directory for compiled schemas")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    compile_components(args.base_path, args.output_dir, args.verbose)
    
    if args.verbose:
        console.print(f"\n💡 [bold cyan]Next Steps:[/bold cyan]")
        console.print("   1. Review generated schemas in the output directory")
        console.print("   2. Use schemas for component discovery and validation")
        console.print("   3. Extend system by adding new component folders")
        console.print("   4. Re-run compilation after adding new components")

if __name__ == "__main__":
    main()