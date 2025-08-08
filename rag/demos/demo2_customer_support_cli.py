#!/usr/bin/env python3
"""
Demo 2: Customer Support System (CLI Version)
Demonstrates customer support ticket analysis using CLI commands.
Shows how the platform can be used for real-world business applications through CLI.
"""

import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup rich console for beautiful output
console = Console()


def run_cli_command(command: str, verbose: bool = False, quiet: bool = False, show_output: bool = True) -> tuple[int, str, str]:
    """
    Run a CLI command and return the result.
    
    Args:
        command: The CLI command to run
        verbose: Whether to add --verbose flag
        quiet: Whether to add --quiet flag
        show_output: Whether to display the command being run
    
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    # Add flags if requested
    parts = command.split()
    cli_index = next((i for i, part in enumerate(parts) if 'cli.py' in part), -1)
    
    if cli_index >= 0:
        flags_to_add = []
        if verbose and '--verbose' not in command:
            flags_to_add.append('--verbose')
        if quiet and '--quiet' not in command:
            flags_to_add.append('--quiet')
        
        for flag in flags_to_add:
            parts.insert(cli_index + 1, flag)
        
        command = ' '.join(parts)
    
    if show_output:
        console.print(f"\n[bold cyan]Running command:[/bold cyan]")
        console.print(f"[dim]$ {command}[/dim]")
    
    # Run the command
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent  # Run from rag directory
    )
    
    return result.returncode, result.stdout, result.stderr


def print_section_header(title: str, emoji: str = "🎧"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def demonstrate_customer_support_cli():
    """Demonstrate customer support system using CLI commands exclusively."""
    
    print_section_header("🦙 Demo 2: Customer Support System (CLI Version)", "🎧")
    
    console.print("\n[bold green]This demo showcases customer support capabilities via CLI:[/bold green]")
    console.print("• [bold cyan]Ticket processing and analysis through CLI[/bold cyan]")
    console.print("• Priority detection and sentiment analysis")
    console.print("• Knowledge base integration")
    console.print("• Fast similarity search for support queries")
    console.print("• Verbose output showing ticket metadata")
    
    # Test CLI availability
    print_section_header("System Verification", "🔧")
    
    console.print("🔍 [bold cyan]Verifying CLI and strategy availability...[/bold cyan]")
    
    returncode, stdout, stderr = run_cli_command("python cli.py test --quiet", show_output=False)
    if returncode != 0:
        console.print("⚠️ [yellow]Some tests failed, but continuing...[/yellow]")
    else:
        console.print("✅ [bold green]System tests passed![/bold green]")
    
    # Initialize customer support collection
    print_section_header("Initialize Customer Support Database", "🗄️")
    
    console.print("🚀 [bold cyan]Setting up customer support knowledge base...[/bold cyan]")
    
    # Check if collection exists
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml info --strategy customer_support_demo",
        quiet=True
    )
    
    if "document_count" in stdout and "0" not in stdout:
        console.print("[dim]💡 Customer support database already initialized[/dim]")
    
    # Ingest customer support data
    print_section_header("Customer Data Ingestion", "📥")
    
    console.print("[bold cyan]🔄 Processing customer support tickets and knowledge base...[/bold cyan]")
    console.print("[dim]💡 Using customer_support_demo strategy from demo_strategies.yaml[/dim]")
    
    # Show what files we're ingesting
    support_files = [
        "demos/static_samples/customer_support/support_tickets.csv",
        "demos/static_samples/customer_support/knowledge_base.txt"
    ]
    
    files_table = Table(show_header=True, header_style="bold yellow")
    files_table.add_column("File Type", style="cyan")
    files_table.add_column("Path", style="white")
    files_table.add_column("Description", style="dim")
    
    files_table.add_row(
        "Support Tickets",
        "support_tickets.csv",
        "Customer tickets with priorities and categories"
    )
    files_table.add_row(
        "Knowledge Base",
        "knowledge_base.txt",
        "Support documentation and solutions"
    )
    
    console.print(files_table)
    
    # Ingest with verbose output to show processing
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml --verbose ingest --strategy customer_support_demo demos/static_samples/customer_support/"
    )
    
    if returncode == 0:
        console.print("✅ [bold green]Customer data successfully processed![/bold green]")
        
        # Extract and show statistics from output
        for line in stdout.split('\n'):
            if any(keyword in line for keyword in ["Documents processed:", "Extractors applied:", "Priority:"]):
                console.print(f"   {line.strip()}")
    
    # Show collection statistics
    print_section_header("Support Database Analytics", "📊")
    
    console.print("📈 [bold cyan]Analyzing customer support database...[/bold cyan]")
    
    returncode, stdout, stderr = run_cli_command(
        "python cli.py info --strategy customer_support_demo"
    )
    
    if returncode == 0:
        # Parse the output to create a nice display
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Support Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        for line in stdout.split('\n'):
            if ':' in line and not line.startswith('python'):
                key, value = line.split(':', 1)
                stats_table.add_row(key.strip(), value.strip())
        
        console.print(stats_table)
    
    # Demonstrate customer support queries
    print_section_header("Customer Query Resolution", "🔍")
    
    support_queries = [
        "How do I reset my password?",
        "Application crashes when uploading large files",
        "Billing error double charged credit card",
        "Cannot connect to server timeout error",
        "Feature request dark mode"
    ]
    
    console.print("🎯 [bold cyan]Testing real customer support scenarios:[/bold cyan]")
    console.print("[dim]💡 Each query uses semantic search to find relevant tickets and solutions[/dim]")
    
    for i, query in enumerate(support_queries, 1):
        query_type = ["Password Issue", "Technical Bug", "Billing Problem", "Connection Issue", "Feature Request"][i-1]
        
        console.print(f"\n[bold cyan]Support Query #{i} - {query_type}:[/bold cyan]")
        console.print(f"Customer asks: [yellow]\"{query}\"[/yellow]")
        
        # Run search with appropriate verbosity
        returncode, stdout, stderr = run_cli_command(
            f'python cli.py --strategy-file demos/demo_strategies.yaml search --strategy customer_support_demo "{query}" --top-k 3',
            verbose=(i <= 2)  # Show verbose for first 2 queries
        )
        
        if returncode == 0:
            # For non-verbose queries, extract key information
            if i > 2:
                console.print("[dim]Found relevant tickets and knowledge base articles[/dim]")
                # Show a summary of results
                if "Result #" in stdout:
                    lines = stdout.split('\n')
                    for line in lines:
                        if "Source:" in line or "Priority:" in line or "Similarity:" in line:
                            console.print(f"   {line.strip()}")
            else:
                # Show full verbose output for first 2 queries
                console.print(stdout)
        
        time.sleep(1)  # Brief pause between queries
    
    # Demonstrate filtering and advanced search
    print_section_header("Advanced Search Capabilities", "🔬")
    
    console.print("🎯 [bold cyan]Demonstrating advanced CLI search features:[/bold cyan]")
    
    # High priority tickets search
    console.print("\n[bold]1. Finding high-priority tickets:[/bold]")
    returncode, stdout, stderr = run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --verbose search --strategy customer_support_demo "urgent critical system down" --top-k 5'
    )
    
    if "high" in stdout.lower() or "critical" in stdout.lower():
        console.print("✅ [green]Successfully identified high-priority tickets[/green]")
    
    # Knowledge base search
    console.print("\n[bold]2. Searching knowledge base for solutions:[/bold]")
    returncode, stdout, stderr = run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml search --strategy customer_support_demo "troubleshooting steps reset" --top-k 3'
    )
    
    if "knowledge" in stdout.lower() or "solution" in stdout.lower():
        console.print("✅ [green]Found relevant knowledge base articles[/green]")
    
    # Demonstrate quiet mode for integration
    print_section_header("Integration Mode", "🔌")
    
    console.print("🤖 [bold cyan]Quiet mode for system integration:[/bold cyan]")
    console.print("[dim]💡 Useful for integrating with ticketing systems or chatbots[/dim]")
    
    returncode, stdout, stderr = run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --quiet search --strategy customer_support_demo "password reset"',
        quiet=True
    )
    
    console.print("\n[bold]Structured output for parsing:[/bold]")
    # Show first 500 chars of quiet output
    output_preview = stdout[:500] if stdout else "No output"
    console.print(Panel(output_preview, title="Quiet Mode Output", border_style="dim"))
    
    # Show CLI integration examples
    print_section_header("Integration Examples", "🔗")
    
    console.print("📝 [bold cyan]Example CLI integrations for support systems:[/bold cyan]")
    
    integration_table = Table(show_header=True, header_style="bold cyan")
    integration_table.add_column("Use Case", style="yellow")
    integration_table.add_column("CLI Command", style="white")
    
    integration_table.add_row(
        "Ticket Auto-Response",
        "cli.py search --quiet --strategy-file demos/demo_strategies.yaml --strategy customer_support_demo '<ticket_content>'"
    )
    integration_table.add_row(
        "Priority Detection",
        "cli.py search --verbose --strategy-file demos/demo_strategies.yaml --strategy customer_support_demo '<ticket>'"
    )
    integration_table.add_row(
        "Knowledge Base Lookup",
        "cli.py search --strategy-file demos/demo_strategies.yaml --strategy customer_support_demo '<customer_question>'"
    )
    integration_table.add_row(
        "Batch Processing",
        "for ticket in *.txt; do cli.py ingest \"$ticket\"; done"
    )
    integration_table.add_row(
        "Daily Stats",
        "cli.py --strategy-file demos/demo_strategies.yaml --strategy customer_support_demo info"
    )
    
    console.print(integration_table)
    
    # Summary
    print_section_header("Customer Support CLI Demo Summary", "🎯")
    
    console.print("🚀 [bold green]Customer Support System via CLI Complete![/bold green]")
    
    console.print("\n[bold]What this demo demonstrated:[/bold]")
    console.print("✅ [bold cyan]Complete support workflow using CLI[/bold cyan]")
    console.print("✅ CSV ticket ingestion with metadata extraction")
    console.print("✅ Knowledge base integration")
    console.print("✅ Priority and sentiment detection")
    console.print("✅ Semantic search for similar issues")
    console.print("✅ Verbose and quiet modes for different use cases")
    
    console.print(f"\n[bold]Business Value Demonstrated:[/bold]")
    console.print("💰 Faster ticket resolution through similarity search")
    console.print("📈 Automatic priority detection for SLA compliance")
    console.print("🤖 Ready for chatbot/automation integration")
    console.print("📊 Analytics and reporting capabilities")
    console.print("🔄 Continuous learning from new tickets")
    
    console.print(f"\n[bold]CLI Commands for Support Teams:[/bold]")
    console.print("📥 `cli.py ingest --strategy-file demos/demo_strategies.yaml --strategy customer_support_demo tickets.csv`")
    console.print("🔍 `cli.py search --strategy-file demos/demo_strategies.yaml --strategy customer_support_demo '<issue>'`")
    console.print("📊 `cli.py info --strategy-file demos/demo_strategies.yaml --strategy customer_support_demo`")
    console.print("🔇 `cli.py --quiet` for integration with other systems")
    console.print("📝 `cli.py --verbose` for detailed analysis")
    
    console.print(f"\n📁 Support database saved to: [bold]./demos/vectordb/customer_support[/bold]")
    console.print("🔄 Continue using these commands to manage your support system:")
    console.print("[dim]$ python cli.py search --strategy-file demos/demo_strategies.yaml --strategy customer_support_demo 'customer issue'[/dim]")
    console.print("[dim]$ python cli.py ingest --strategy-file demos/demo_strategies.yaml --strategy customer_support_demo new_tickets.csv[/dim]")
    
    # Clean up the database after demo
    console.print("\n🧹 [bold cyan]Cleaning up demo database...[/bold cyan]")
    console.print("[dim]💡 Removing demo data to keep your system clean[/dim]")
    
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml manage --strategy customer_support_demo delete --all"
    )
    
    if returncode == 0:
        console.print("✅ [bold green]Demo database cleaned up successfully![/bold green]")
    else:
        console.print("⚠️ [bold yellow]Database cleanup had issues, trying direct cleanup...[/bold yellow]")
        # Fallback to direct cleanup if manage command fails
        import shutil
        from pathlib import Path
        
        db_path = Path("./demos/vectordb/customer_support")
        try:
            if db_path.exists():
                shutil.rmtree(db_path)
                console.print("✅ [bold green]Fallback cleanup successful![/bold green]")
            else:
                console.print("ℹ️ [bold blue]Database directory not found (already clean)[/bold blue]")
        except Exception as e:
            console.print(f"⚠️ [bold yellow]All cleanup methods failed: {e}[/bold yellow]")
            console.print("[dim]💡 You can manually delete ./demos/vectordb/customer_support/ if needed[/dim]")


if __name__ == "__main__":
    try:
        demonstrate_customer_support_cli()
    except KeyboardInterrupt:
        console.print("\n\n👋 Support demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ Support demo failed: {str(e)}", style="red")
        console.print("Make sure you're running from the rag directory")
        sys.exit(1)