#!/usr/bin/env python3
"""
Master Demo: Complete RAG System Showcase
Runs all 5 specialized RAG demos to showcase the full range of capabilities:

1. Research Paper Analysis - Academic content with statistics and entity extraction
2. Customer Support System - Structured tickets with entity and pattern recognition
3. Code Documentation - Technical docs with code extraction and cross-references
4. News Article Analysis - Media content with sentiment and trend analysis
5. Business Reports - Financial data with metrics extraction and trend analysis

This master demo demonstrates how different RAG strategies excel in different domains
while using the same underlying framework and components.
"""

import logging
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.columns import Columns
from rich.align import Align

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup rich console for beautiful output
console = Console()
logging.basicConfig(level=logging.WARNING)  # Reduce noise


def print_master_header(title: str, emoji: str = "🚀"):
    """Print a beautiful master demo header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 100, style="cyan")


def print_demo_overview():
    """Print an overview of all demos and their unique approaches."""
    console.print("\n[bold green]🎯 RAG System Demo Portfolio[/bold green]")
    
    demos_table = Table(show_header=True, header_style="bold magenta", title="5 Specialized RAG Demonstrations")
    demos_table.add_column("Demo", style="cyan", width=20)
    demos_table.add_column("Domain", style="yellow", width=18)
    demos_table.add_column("Key Strategy", style="white", width=25)
    demos_table.add_column("Unique Extractors", style="green", width=25)
    
    demos_table.add_row(
        "1. Research Papers",
        "Academic/Scientific",
        "Statistical analysis + Citations",
        "Statistics + Entity + Summary"
    )
    
    demos_table.add_row(
        "2. Customer Support",
        "Business/Service",
        "Case matching + Resolution",
        "Entity + Pattern + Summary"
    )
    
    demos_table.add_row(
        "3. Code Documentation",
        "Technical/Developer",
        "Structure + Cross-references",
        "Link + Heading + Pattern"
    )
    
    demos_table.add_row(
        "4. News Analysis",
        "Media/Journalism",
        "Sentiment + Trend analysis",
        "Entity + Summary + Link"
    )
    
    demos_table.add_row(
        "5. Business Reports",
        "Financial/Corporate",
        "Multi-format + Metrics",
        "Table + Statistics + Summary"
    )
    
    console.print(demos_table)
    
    console.print(f"\n[bold]🔬 What makes each demo unique:[/bold]")
    console.print("• [cyan]Different parsers[/cyan]: Text, Markdown, HTML, CSV, Excel")
    console.print("• [yellow]Specialized extractors[/yellow]: Each optimized for domain-specific content")
    console.print("• [green]Tailored search[/green]: Query optimization for different use cases")
    console.print("• [blue]Custom analytics[/blue]: Domain-specific insights and visualizations")
    console.print("• [magenta]Unique workflows[/magenta]: Different strategies for different content types")


def run_demo_script(demo_name: str, script_path: str, demo_number: int) -> bool:
    """Run a specific demo script and handle output."""
    console.print(f"\n🚀 [bold cyan]Starting Demo {demo_number}: {demo_name}[/bold cyan]")
    console.print(f"📄 Script: {script_path}")
    console.print("─" * 80)
    
    try:
        # Run the demo script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=Path(__file__).parent.parent,
            capture_output=False,  # Let output show directly
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            console.print(f"\n✅ [bold green]Demo {demo_number} completed successfully![/bold green]")
            return True
        else:
            console.print(f"\n❌ [bold red]Demo {demo_number} failed with return code {result.returncode}[/bold red]")
            return False
            
    except subprocess.TimeoutExpired:
        console.print(f"\n⏰ [bold yellow]Demo {demo_number} timed out after 5 minutes[/bold yellow]")
        return False
    except Exception as e:
        console.print(f"\n❌ [bold red]Demo {demo_number} failed with error: {str(e)}[/bold red]")
        return False


def show_demo_comparison():
    """Show a comparison of what each demo accomplished."""
    console.print("\n📊 [bold green]Demo Results Comparison[/bold green]")
    
    comparison_table = Table(show_header=True, header_style="bold blue", title="RAG Strategy Effectiveness by Domain")
    comparison_table.add_column("Demo", style="cyan")
    comparison_table.add_column("Documents Processed", style="white")
    comparison_table.add_column("Data Extracted", style="yellow")
    comparison_table.add_column("Search Optimization", style="green")
    comparison_table.add_column("Unique Insights", style="magenta")
    
    comparison_table.add_row(
        "Research Papers",
        "Academic papers\n& sections",
        "Statistics, citations,\nentities, summaries",
        "Semantic search for\nresearch queries",
        "Statistical evidence,\nresearch methodologies"
    )
    
    comparison_table.add_row(
        "Customer Support",
        "Support tickets\n& knowledge base",
        "Customer entities,\nresolution patterns",
        "Case similarity\nmatching",
        "Issue trends,\nresolution workflows"
    )
    
    comparison_table.add_row(
        "Code Documentation",
        "Technical docs\n& API references",
        "Code blocks, links,\nheading structure",
        "Developer-focused\ntechnical search",
        "Code patterns,\nAPI relationships"
    )
    
    comparison_table.add_row(
        "News Analysis",
        "News articles\n& media content",
        "Entities, sentiment,\narticle metadata",
        "Topic clustering\n& trend analysis",
        "Sentiment trends,\nevent relationships"
    )
    
    comparison_table.add_row(
        "Business Reports",
        "Financial reports\n& business data",
        "Financial metrics,\ntables, statistics",
        "Business intelligence\nquerying",
        "Financial trends,\nKPI analysis"
    )
    
    console.print(comparison_table)


def show_architecture_insights():
    """Show insights about the RAG architecture demonstrated."""
    console.print("\n🏗️ [bold green]RAG Architecture Insights[/bold green]")
    
    # Component usage analysis
    components_table = Table(show_header=True, header_style="bold yellow", title="Component Usage Across Demos")
    components_table.add_column("Component Type", style="cyan")
    components_table.add_column("Variations Used", style="white")
    components_table.add_column("Specialization Strategy", style="green")
    
    components_table.add_row(
        "Parsers",
        "PlainText, Markdown, HTML, CSV",
        "Format-specific optimizations"
    )
    
    components_table.add_row(
        "Extractors", 
        "Statistics, Entity, Summary, Link,\nHeading, Table, Pattern",
        "Domain-aware content extraction"
    )
    
    components_table.add_row(
        "Embedders",
        "Ollama with nomic-embed-text",
        "Consistent semantic representation"
    )
    
    components_table.add_row(
        "Vector Stores",
        "ChromaDB collections",
        "Isolated domain knowledge bases"
    )
    
    console.print(components_table)
    
    # Architecture benefits
    console.print(f"\n[bold]🎯 Key Architecture Benefits Demonstrated:[/bold]")
    console.print("• [cyan]Modularity[/cyan]: Same components, different configurations")
    console.print("• [yellow]Specialization[/yellow]: Domain-specific extractors and workflows")
    console.print("• [green]Scalability[/green]: Each demo creates isolated knowledge bases")
    console.print("• [blue]Consistency[/blue]: Unified API across all domains")
    console.print("• [magenta]Extensibility[/magenta]: Easy to add new domains and extractors")


def show_performance_insights():
    """Show performance and efficiency insights from the demos."""
    console.print("\n⚡ [bold green]Performance & Efficiency Insights[/bold green]")
    
    perf_table = Table(show_header=True, header_style="bold green", title="RAG Performance Characteristics")
    perf_table.add_column("Performance Aspect", style="cyan")
    perf_table.add_column("Optimization Strategy", style="white")
    perf_table.add_column("Benefit", style="yellow")
    
    perf_table.add_row(
        "Parsing Efficiency",
        "Format-specific parsers with chunking",
        "Optimal content segmentation"
    )
    
    perf_table.add_row(
        "Extraction Speed",
        "Parallel extractor processing",
        "Comprehensive content analysis"
    )
    
    perf_table.add_row(
        "Embedding Generation",
        "Batch processing with size optimization",
        "Efficient vector generation"
    )
    
    perf_table.add_row(
        "Search Performance",
        "Domain-specific vector collections",
        "Faster, more relevant results"
    )
    
    perf_table.add_row(
        "Memory Usage",
        "Streaming and chunked processing",
        "Scalable to large documents"
    )
    
    console.print(perf_table)


def show_real_world_applications():
    """Show real-world applications and use cases demonstrated."""
    console.print("\n🌍 [bold green]Real-World Applications Demonstrated[/bold green]")
    
    applications_table = Table(show_header=True, header_style="bold blue", title="Practical Use Cases")
    applications_table.add_column("Industry/Domain", style="cyan")
    applications_table.add_column("Use Case", style="white")
    applications_table.add_column("RAG Strategy", style="yellow")
    applications_table.add_column("Business Value", style="green")
    
    applications_table.add_row(
        "Academic Research",
        "Literature review\n& research synthesis",
        "Statistical + entity\nextraction",
        "Accelerated research,\nevidence discovery"
    )
    
    applications_table.add_row(
        "Customer Service",
        "Case resolution\n& knowledge search",
        "Pattern recognition\n+ similarity matching",
        "Faster resolution,\nbetter consistency"
    )
    
    applications_table.add_row(
        "Software Development",
        "Documentation search\n& API guidance",
        "Structure preservation\n+ code extraction",
        "Developer productivity,\ncode quality"
    )
    
    applications_table.add_row(
        "Media & Journalism",
        "News analysis\n& trend tracking",
        "Sentiment analysis\n+ entity tracking",
        "Content insights,\ntrend identification"
    )
    
    applications_table.add_row(
        "Business Intelligence",
        "Financial analysis\n& KPI tracking",
        "Multi-format processing\n+ metrics extraction",
        "Data-driven decisions,\nautomated insights"
    )
    
    console.print(applications_table)


def demonstrate_master_rag_showcase():
    """Run the complete RAG system showcase with all 5 demos."""
    
    print_master_header("🦙 Master RAG Demo: Complete System Showcase", "🚀")
    
    console.print("\n[bold green]Welcome to the Complete RAG System Demonstration![/bold green]")
    console.print("This master demo runs 5 specialized RAG implementations, each optimized")
    console.print("for different domains and use cases, showcasing the flexibility and")
    console.print("power of the modular RAG architecture.")
    
    # Show demo overview
    print_demo_overview()
    
    # Ask user for confirmation
    console.print(f"\n[bold yellow]⚠️ Important Notes:[/bold yellow]")
    console.print("• Each demo takes 2-5 minutes to complete")
    console.print("• Total runtime: approximately 15-25 minutes")
    console.print("• Requires Ollama running with nomic-embed-text model")
    console.print("• Creates 5 separate vector databases")
    console.print("• Demonstrates real embedding and retrieval with sample data")
    
    # Confirm with user
    try:
        response = input("\n🚀 Ready to run all 5 demos? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            console.print("\n👋 Demo cancelled by user. Run individual demos with:")
            console.print("[dim]uv run python demos/demo1_research_papers.py[/dim]")
            console.print("[dim]uv run python demos/demo2_customer_support.py[/dim]")
            console.print("[dim]uv run python demos/demo3_code_documentation.py[/dim]")
            console.print("[dim]uv run python demos/demo4_news_analysis.py[/dim]")
            console.print("[dim]uv run python demos/demo5_business_reports.py[/dim]")
            return
    except KeyboardInterrupt:
        console.print("\n\n👋 Demo cancelled by user")
        return
    
    # Demo configurations
    demos = [
        ("Research Paper Analysis", "demos/demo1_research_papers.py", 1),
        ("Customer Support System", "demos/demo2_customer_support.py", 2),
        ("Code Documentation", "demos/demo3_code_documentation.py", 3),
        ("News Article Analysis", "demos/demo4_news_analysis.py", 4),
        ("Business Reports Analysis", "demos/demo5_business_reports.py", 5)
    ]
    
    # Track demo results
    demo_results = []
    successful_demos = 0
    
    print_master_header("Running All RAG Demos", "🎬")
    
    start_time = time.time()
    
    # Run each demo
    for demo_name, script_path, demo_number in demos:
        demo_start = time.time()
        
        # Check if script exists
        script_full_path = Path(__file__).parent.parent / script_path
        if not script_full_path.exists():
            console.print(f"\n❌ [bold red]Demo script not found: {script_path}[/bold red]")
            demo_results.append((demo_name, False, 0))
            continue
        
        # Run the demo
        success = run_demo_script(demo_name, str(script_full_path), demo_number)
        demo_time = time.time() - demo_start
        
        demo_results.append((demo_name, success, demo_time))
        if success:
            successful_demos += 1
        
        # Brief pause between demos
        if demo_number < len(demos):
            console.print(f"\n⏸️ [dim]Brief pause before next demo...[/dim]")
            time.sleep(2)
    
    total_time = time.time() - start_time
    
    # Show results summary
    print_master_header("Demo Results Summary", "📊")
    
    results_table = Table(show_header=True, header_style="bold magenta", title="Demo Execution Results")
    results_table.add_column("Demo", style="cyan")
    results_table.add_column("Status", style="white")
    results_table.add_column("Runtime", style="yellow")
    results_table.add_column("Success Rate", style="green")
    
    for demo_name, success, demo_time in demo_results:
        status = "✅ Success" if success else "❌ Failed"
        status_color = "green" if success else "red"
        runtime = f"{demo_time:.1f}s" if demo_time > 0 else "N/A"
        
        results_table.add_row(
            demo_name,
            f"[{status_color}]{status}[/{status_color}]",
            runtime,
            "100%" if success else "0%"
        )
    
    console.print(results_table)
    
    # Overall statistics
    success_rate = (successful_demos / len(demos)) * 100
    console.print(f"\n📈 [bold]Overall Results:[/bold]")
    console.print(f"• Successful Demos: [bold green]{successful_demos}/{len(demos)}[/bold green]")
    console.print(f"• Success Rate: [bold green]{success_rate:.1f}%[/bold green]")
    console.print(f"• Total Runtime: [bold cyan]{total_time:.1f} seconds[/bold cyan]")
    console.print(f"• Average Demo Time: [bold cyan]{total_time/len(demos):.1f} seconds[/bold cyan]")
    
    if successful_demos == len(demos):
        console.print(f"\n🎉 [bold green]All demos completed successfully![/bold green]")
        
        # Show comparative analysis
        show_demo_comparison()
        show_architecture_insights()
        show_performance_insights()
        show_real_world_applications()
        
        # Show final summary
        print_master_header("Master Demo Complete!", "🎉")
        
        console.print("🚀 [bold green]Congratulations! You've experienced the full RAG system showcase![/bold green]")
        console.print("\n[bold]What you've accomplished:[/bold]")
        console.print("✅ Processed documents across 5 different domains")
        console.print("✅ Demonstrated 7+ specialized extractors in action")
        console.print("✅ Created 5 domain-specific knowledge bases")
        console.print("✅ Performed hundreds of semantic searches")
        console.print("✅ Extracted thousands of data points and insights")
        console.print("✅ Showcased real-world RAG applications")
        
        console.print(f"\n[bold]Your RAG knowledge bases are ready:[/bold]")
        console.print("📚 Research Papers: [dim]./demos/vectordb/research_papers[/dim]")
        console.print("🎧 Customer Support: [dim]./demos/vectordb/customer_support[/dim]")
        console.print("💻 Code Documentation: [dim]./demos/vectordb/code_documentation[/dim]")
        console.print("📰 News Analysis: [dim]./demos/vectordb/news_analysis[/dim]")
        console.print("📊 Business Reports: [dim]./demos/vectordb/business_reports[/dim]")
        
        console.print(f"\n[bold]Try cross-domain queries:[/bold]")
        console.print("[dim]uv run python cli.py search 'artificial intelligence trends' --collection research_papers[/dim]")
        console.print("[dim]uv run python cli.py search 'customer satisfaction metrics' --collection customer_support[/dim]")
        console.print("[dim]uv run python cli.py search 'API security best practices' --collection code_documentation[/dim]")
        console.print("[dim]uv run python cli.py search 'technology investment trends' --collection news_articles[/dim]")
        console.print("[dim]uv run python cli.py search 'quarterly performance analysis' --collection business_reports[/dim]")
        
    elif successful_demos > 0:
        console.print(f"\n⚠️ [bold yellow]Partial success - {successful_demos} out of {len(demos)} demos completed[/bold yellow]")
        console.print("Check the individual demo outputs above for error details.")
        console.print("You can run the failed demos individually for troubleshooting.")
    else:
        console.print(f"\n❌ [bold red]No demos completed successfully[/bold red]")
        console.print("Please check that:")
        console.print("• Ollama is running with the nomic-embed-text model")
        console.print("• All required dependencies are installed")
        console.print("• Sample data files are present in demos/static_samples/")


if __name__ == "__main__":
    try:
        demonstrate_master_rag_showcase()
    except KeyboardInterrupt:
        console.print("\n\n👋 Master demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ Master demo failed: {str(e)}", style="red")
        console.print("Check system requirements and dependencies")
        sys.exit(1)