#!/usr/bin/env python3
"""
System validation script to ensure all components are working correctly.
Tests strategies, demos, and CLI functionality.
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def validate_system():
    """Validate the entire RAG system."""
    console.print("\n[bold cyan]🦙 RAG System Validation[/bold cyan]")
    console.print("=" * 60)
    
    results = []
    
    # Test 1: Check strategies are available
    console.print("\n[bold]1. Checking strategy availability...[/bold]")
    strategies = [
        "research_papers_demo",
        "customer_support_demo", 
        "code_documentation_demo",
        "news_analysis_demo",
        "business_reports_demo"
    ]
    
    for strategy in strategies:
        cmd = f"python cli.py strategies show {strategy}"
        success = run_command(cmd, f"Check {strategy}")
        results.append((f"Strategy: {strategy}", success))
        console.print(f"  {'✅' if success else '❌'} {strategy}")
    
    # Test 2: Check CLI commands
    console.print("\n[bold]2. Testing CLI commands...[/bold]")
    cli_tests = [
        ("List strategies", "python cli.py strategies list"),
        ("Show help", "python cli.py -h"),
        ("Show ingest help", "python cli.py ingest -h"),
        ("Show search help", "python cli.py search -h"),
    ]
    
    for test_name, cmd in cli_tests:
        success = run_command(cmd, test_name)
        results.append((test_name, success))
        console.print(f"  {'✅' if success else '❌'} {test_name}")
    
    # Test 3: Check component imports
    console.print("\n[bold]3. Testing component imports...[/bold]")
    import_tests = [
        ("Core factories", "from core.factories import ComponentFactory, RetrievalStrategyFactory"),
        ("Strategy manager", "from core.strategies.manager import StrategyManager"),
        ("ChromaStore", "from components.stores.chroma_store.chroma_store import ChromaStore"),
        ("CLI", "import cli"),
    ]
    
    for test_name, import_stmt in import_tests:
        try:
            exec(import_stmt)
            success = True
        except ImportError:
            success = False
        results.append((test_name, success))
        console.print(f"  {'✅' if success else '❌'} {test_name}")
    
    # Test 4: Check demo files exist
    console.print("\n[bold]4. Checking demo files...[/bold]")
    demo_files = [
        "demos/demo1_research_papers.py",
        "demos/demo2_customer_support.py",
        "demos/demo3_code_documentation.py",
        "demos/demo4_news_analysis.py",
        "demos/demo5_business_reports.py",
        "demos/demo_strategies.yaml",
        "demos/strategy_demo_utils.py"
    ]
    
    for demo_file in demo_files:
        exists = Path(demo_file).exists()
        results.append((f"File: {demo_file}", exists))
        console.print(f"  {'✅' if exists else '❌'} {demo_file}")
    
    # Test 5: Test strategy configuration loading
    console.print("\n[bold]5. Testing strategy configuration loading...[/bold]")
    try:
        from core.strategies.manager import StrategyManager
        manager = StrategyManager(load_demos=True)
        strategies_loaded = len(manager.get_available_strategies()) > 0
        results.append(("Strategy loading", strategies_loaded))
        console.print(f"  {'✅' if strategies_loaded else '❌'} Strategies loaded: {len(manager.get_available_strategies())}")
    except Exception as e:
        results.append(("Strategy loading", False))
        console.print(f"  ❌ Strategy loading failed: {e}")
    
    # Test 6: Check HybridUniversalStrategy metadata handling
    console.print("\n[bold]6. Testing HybridUniversalStrategy metadata...[/bold]")
    try:
        from components.stores.chroma_store.chroma_store import ChromaStore
        import json
        
        # Test metadata parsing
        config = {
            "collection_name": "test_validation"
        }
        store = ChromaStore(name="test_store", config=config)
        test_metadata = {"nested": {"key": "value"}}
        
        # Simulate what ChromaDB does
        serialized = json.dumps(test_metadata["nested"])
        parsed = store._parse_metadata({"nested": serialized})
        
        success = parsed["nested"] == test_metadata["nested"]
        results.append(("Metadata parsing", success))
        console.print(f"  {'✅' if success else '❌'} Metadata parsing works correctly")
        
        # Cleanup
        store.clear()
    except Exception as e:
        results.append(("Metadata parsing", False))
        console.print(f"  ❌ Metadata parsing failed: {e}")
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Validation Summary:[/bold]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test", style="cyan")
    table.add_column("Status", justify="center")
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "[green]✅ PASS[/green]" if success else "[red]❌ FAIL[/red]"
        table.add_row(test_name, status)
        if success:
            passed += 1
        else:
            failed += 1
    
    console.print(table)
    
    console.print(f"\n[bold]Total:[/bold] {passed} passed, {failed} failed")
    
    if failed == 0:
        console.print("\n[bold green]🎉 All validation tests passed![/bold green]")
        console.print("[green]The RAG system is fully operational.[/green]")
    else:
        console.print(f"\n[bold yellow]⚠️  {failed} tests failed[/bold yellow]")
        console.print("[yellow]Some components may need attention.[/yellow]")
    
    return failed == 0


if __name__ == "__main__":
    success = validate_system()
    sys.exit(0 if success else 1)