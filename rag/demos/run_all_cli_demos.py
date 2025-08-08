#!/usr/bin/env python3
"""
Master CLI Demo Runner
Runs all 5 demos using CLI commands to showcase the complete platform capabilities.
This demonstrates how ALL demos work together using the unified CLI interface.
"""

import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Setup rich console
console = Console()


def print_banner():
    """Print the demo banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🦙 RAG Platform CLI Demonstration Suite 🦙           ║
║                                                              ║
║     Showcasing Platform Capabilities Through CLI            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")


def run_demo(demo_script: str, demo_name: str, demo_number: int, total_demos: int = 6) -> bool:
    """Run a single demo script."""
    console.print(f"\n{'='*70}", style="cyan")
    console.print(f"[bold cyan]Demo {demo_number}/{total_demos}: {demo_name}[/bold cyan]", justify="center")
    console.print(f"{'='*70}", style="cyan")
    
    demo_path = Path(__file__).parent / demo_script
    
    if not demo_path.exists():
        console.print(f"❌ [red]Demo script not found: {demo_script}[/red]")
        return False
    
    try:
        # Run the demo
        result = subprocess.run(
            [sys.executable, str(demo_path)],
            capture_output=False,  # Let output show directly
            text=True,
            cwd=Path(__file__).parent.parent  # Run from rag directory
        )
        
        if result.returncode == 0:
            console.print(f"\n✅ [bold green]Demo {demo_number} completed successfully![/bold green]")
            return True
        else:
            console.print(f"\n⚠️ [yellow]Demo {demo_number} completed with warnings[/yellow]")
            return True
            
    except KeyboardInterrupt:
        console.print(f"\n⏭️ [yellow]Skipping to next demo...[/yellow]")
        return True
    except Exception as e:
        console.print(f"\n❌ [red]Demo {demo_number} failed: {e}[/red]")
        return False


def show_demo_menu():
    """Show the demo selection menu."""
    menu_table = Table(show_header=True, header_style="bold magenta", title="Available Demos")
    menu_table.add_column("#", style="cyan", width=3)
    menu_table.add_column("Demo Name", style="yellow")
    menu_table.add_column("Description", style="white")
    menu_table.add_column("Strategy", style="green")
    
    demos = [
        ("1", "Research Papers", "Academic paper analysis with extractors", "research_papers_demo"),
        ("2", "Customer Support", "Ticket processing and knowledge base", "customer_support_demo"),
        ("3", "Code Documentation", "Technical docs and API references", "code_documentation_demo"),
        ("4", "News Analysis", "News articles with entity extraction", "news_analysis_demo"),
        ("5", "Business Reports", "Financial docs with pattern extraction", "business_reports_demo"),
        ("6", "Document Management", "Vector DB operations and lifecycle", "CLI-only")
    ]
    
    for num, name, desc, strategy in demos:
        menu_table.add_row(num, name, desc, strategy)
    
    console.print(menu_table)


def main():
    """Main demo runner."""
    print_banner()
    
    console.print("\n[bold green]Welcome to the RAG Platform CLI Demonstration Suite![/bold green]")
    console.print("\nThis suite demonstrates how the entire platform works through CLI commands.")
    console.print("Each demo uses the [bold cyan]strategy-first approach[/bold cyan] with configurations")
    console.print("from [bold]demo_strategies.yaml[/bold], showing different use cases.\n")
    
    # Show available demos
    show_demo_menu()
    
    # Ask user for choice
    console.print("\n[bold cyan]Select demo mode:[/bold cyan]")
    console.print("  [bold]A[/bold] - Run all demos in sequence")
    console.print("  [bold]1-6[/bold] - Run specific demo")
    console.print("  [bold]Q[/bold] - Quit")
    
    choice = console.input("\n[bold yellow]Your choice: [/bold yellow]").strip().upper()
    
    # Demo configurations
    demos = [
        ("demo1_research_papers_cli.py", "Research Paper Analysis", 1),
        ("demo2_customer_support_cli.py", "Customer Support System", 2),
        ("demo3_code_documentation_cli.py", "Code Documentation", 3),
        ("demo4_news_analysis.py", "News Article Analysis", 4),
        ("demo5_business_reports.py", "Business Reports Analysis", 5),
        ("demo6_document_management.py", "Document Management & Vector DB", 6)
    ]
    
    if choice == 'Q':
        console.print("\n👋 [yellow]Exiting demo suite[/yellow]")
        return
    
    elif choice == 'A':
        # Run all demos
        console.print("\n[bold cyan]Running all 6 demos in sequence...[/bold cyan]")
        console.print("[dim]💡 Press Ctrl+C to skip to the next demo[/dim]\n")
        
        success_count = 0
        
        for idx, (demo_script, demo_name, demo_num) in enumerate(demos):
            # Run the demo
            if run_demo(demo_script, demo_name, demo_num):
                success_count += 1
            
            # Pause between demos with enter prompt
            if idx < len(demos) - 1:  # Not after the last demo
                console.print("\n" + "="*70, style="dim")
                console.print("[bold yellow]Demo completed![/bold yellow]", justify="center")
                console.print("="*70, style="dim")
                console.input("\n[bold cyan]Press Enter to continue to the next demo...[/bold cyan] ")
                console.print()  # Add blank line after continue
        
        # Final summary
        console.print("\n" + "="*70, style="cyan")
        console.print("[bold cyan]Demo Suite Complete![/bold cyan]", justify="center")
        console.print("="*70, style="cyan")
        
        console.print(f"\n✅ Successfully ran {success_count}/{len(demos)} demos")
        
        if success_count == len(demos):
            console.print("\n🎉 [bold green]All demos completed successfully![/bold green]")
        
    elif choice in ['1', '2', '3', '4', '5', '6']:
        # Run specific demo
        demo_idx = int(choice) - 1
        demo_script, demo_name, demo_num = demos[demo_idx]
        
        console.print(f"\n[bold cyan]Running Demo {demo_num}: {demo_name}[/bold cyan]")
        run_demo(demo_script, demo_name, demo_num)
        
    else:
        console.print("\n❌ [red]Invalid choice. Please run again.[/red]")
        return
    
    # Show next steps
    console.print("\n" + "="*70, style="cyan")
    console.print("[bold cyan]Next Steps[/bold cyan]", justify="center")
    console.print("="*70, style="cyan")
    
    next_steps = Panel(
        """[bold]Try these CLI commands yourself:[/bold]

[yellow]⚠️ IMPORTANT: Run these commands from the 'rag' directory, not from 'demos'![/yellow]
[dim]cd /path/to/llamafarm-1/rag[/dim]

1. Search any collection:
   [dim]python cli.py --strategy-file demos/demo_strategies.yaml search "transformer architecture" --strategy research_papers_demo[/dim]

2. Ingest new documents:
   [dim]python cli.py --strategy-file demos/demo_strategies.yaml ingest new_tickets.csv --strategy customer_support_demo[/dim]

3. Get collection info:
   [dim]python cli.py --strategy-file demos/demo_strategies.yaml info --strategy code_documentation_demo[/dim]

4. Use verbose mode for details:
   [dim]python cli.py --strategy-file demos/demo_strategies.yaml --verbose search "AI breakthrough" --strategy news_analysis_demo[/dim]

5. Use quiet mode for scripting:
   [dim]python cli.py --strategy-file demos/demo_strategies.yaml --quiet search "revenue growth" --strategy business_reports_demo[/dim]

[bold]All configurations are in:[/bold] demos/demo_strategies.yaml
[bold]Databases saved to:[/bold] demos/vectordb/
""",
        title="🚀 Continue Exploring",
        border_style="green"
    )
    
    console.print(next_steps)
    
    console.print("\n[bold green]Thank you for exploring the RAG Platform CLI capabilities![/bold green]")
    console.print("🦙 Happy llamazing adventures with your RAG system! 🦙\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n👋 Demo suite interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ Demo suite failed: {str(e)}", style="red")
        sys.exit(1)