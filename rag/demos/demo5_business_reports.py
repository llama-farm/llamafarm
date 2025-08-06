#!/usr/bin/env python3
"""
Demo 5: Business Reports Analysis System
Demonstrates RAG capabilities for business and financial documents using:
- Mixed format parsing (Excel, PDF, CSV) for business data
- Table extraction for financial metrics and KPIs
- Statistics extraction for numerical business data
- Pattern recognition for business trends and insights
- Multi-format document processing and analysis
- Executive summary generation and key insights extraction
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.text import Text

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import RAG components
from core.base import Document, Pipeline
from components.parsers.csv_parser import CSVParser
from components.parsers.text_parser import PlainTextParser  # For .xlsx files saved as text
from components.extractors.table_extractor import TableExtractor
from components.extractors.statistics_extractor import ContentStatisticsExtractor
from components.extractors.summary_extractor import SummaryExtractor
from components.embedders.ollama_embedder import OllamaEmbedder
from components.stores.chroma_store import ChromaStore

# Setup rich console for beautiful output
console = Console()
logging.basicConfig(level=logging.WARNING)  # Reduce noise


def print_section_header(title: str, emoji: str = "📊"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def extract_financial_metrics(content: str) -> Dict[str, Any]:
    """Extract financial metrics and KPIs from business report content."""
    import re
    
    metrics = {}
    
    # Common financial patterns
    patterns = {
        'revenue': r'(?:revenue|sales|income)[:\s]*\$?([0-9,.]+(?:\s*(?:billion|million|thousand|B|M|K))?)',
        'profit': r'(?:profit|earnings|income)[:\s]*\$?([0-9,.]+(?:\s*(?:billion|million|thousand|B|M|K))?)',
        'growth': r'(?:growth|increase)[:\s]*([0-9.]+)%',
        'margin': r'(?:margin)[:\s]*([0-9.]+)%',
        'customers': r'(?:customers|users)[:\s]*([0-9,.]+(?:\s*(?:billion|million|thousand|B|M|K))?)',
        'employees': r'(?:employees|staff)[:\s]*([0-9,.]+(?:\s*(?:billion|million|thousand|B|M|K))?)'
    }
    
    for metric_type, pattern in patterns.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            metrics[metric_type] = matches[:3]  # Keep top 3 matches
    
    # Extract percentages
    percentage_matches = re.findall(r'([0-9.]+)%', content)
    if percentage_matches:
        metrics['percentages'] = [float(p) for p in percentage_matches[:10]]
    
    # Extract dollar amounts
    dollar_matches = re.findall(r'\$([0-9,.]+(?:\s*(?:billion|million|thousand|B|M|K))?)', content)
    if dollar_matches:
        metrics['dollar_amounts'] = dollar_matches[:10]
    
    return metrics


def analyze_business_trends(content: str) -> Dict[str, Any]:
    """Analyze business trends and sentiment in the content."""
    positive_indicators = [
        'growth', 'increase', 'improvement', 'success', 'achievement', 'expansion',
        'breakthrough', 'record', 'milestone', 'outperformed', 'exceeded', 'strong'
    ]
    
    negative_indicators = [
        'decline', 'decrease', 'drop', 'loss', 'challenge', 'concern', 'issue',
        'problem', 'weakness', 'risk', 'threat', 'underperformed', 'missed'
    ]
    
    neutral_indicators = [
        'stable', 'maintained', 'consistent', 'unchanged', 'steady', 'flat',
        'analysis', 'report', 'data', 'information', 'metrics', 'statistics'
    ]
    
    content_lower = content.lower()
    
    positive_count = sum(1 for word in positive_indicators if word in content_lower)
    negative_count = sum(1 for word in negative_indicators if word in content_lower)
    neutral_count = sum(1 for word in neutral_indicators if word in content_lower)
    
    total_indicators = positive_count + negative_count + neutral_count
    
    if total_indicators == 0:
        return {'trend': 'neutral', 'confidence': 0.0, 'indicators': {'positive': 0, 'negative': 0, 'neutral': 0}}
    
    positive_ratio = positive_count / total_indicators
    negative_ratio = negative_count / total_indicators
    neutral_ratio = neutral_count / total_indicators
    
    if positive_ratio > negative_ratio and positive_ratio > neutral_ratio:
        trend = 'positive'
        confidence = positive_ratio
    elif negative_ratio > positive_ratio and negative_ratio > neutral_ratio:
        trend = 'negative'
        confidence = negative_ratio
    else:
        trend = 'neutral'
        confidence = neutral_ratio
    
    return {
        'trend': trend,
        'confidence': confidence,
        'indicators': {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count
        }
    }


def print_business_report_analysis(doc: Document):
    """Print analysis of a business report with extracted information."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Report Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    # Basic document info
    table.add_row("Report ID", doc.id)
    table.add_row("Source", Path(doc.source).name if doc.source else "Business Report")
    table.add_row("Content Length", f"{len(doc.content):,} characters")
    table.add_row("Word Count", f"{len(doc.content.split()):,} words")
    
    # Business-specific metadata
    if doc.metadata:
        if 'report_type' in doc.metadata:
            table.add_row("Report Type", doc.metadata['report_type'])
        
        if 'time_period' in doc.metadata:
            table.add_row("Time Period", doc.metadata['time_period'])
        
        if 'company' in doc.metadata:
            table.add_row("Company", doc.metadata['company'])
        
        # Business trend analysis
        if 'business_trend' in doc.metadata:
            trend_data = doc.metadata['business_trend']
            trend_color = {
                'positive': 'green',
                'negative': 'red',
                'neutral': 'yellow'
            }.get(trend_data['trend'], 'white')
            
            table.add_row("Business Trend", f"[{trend_color}]{trend_data['trend'].title()}[/{trend_color}] ({trend_data['confidence']:.2f})")
        
        # Financial metrics
        if 'financial_metrics' in doc.metadata:
            metrics = doc.metadata['financial_metrics']
            metric_summary = []
            if 'revenue' in metrics:
                metric_summary.append(f"Revenue: {metrics['revenue'][0] if metrics['revenue'] else 'N/A'}")
            if 'growth' in metrics:
                metric_summary.append(f"Growth: {metrics['growth'][0] if metrics['growth'] else 'N/A'}%")
            if metric_summary:
                table.add_row("Key Metrics", " | ".join(metric_summary))
        
        # Extractor results
        if 'tables' in doc.metadata:
            tables = len(doc.metadata['tables'])
            table.add_row("Data Tables", f"{tables} tables extracted")
        
        if 'statistics' in doc.metadata:
            stats = len(doc.metadata.get('extractors', {}).get('statistics', {}))
            table.add_row("Statistics", f"{stats} numerical data points")
        
        if 'summary' in doc.metadata:
            summary = doc.metadata.get('extractors', {}).get('summary', {})[:80] + "..." if len(doc.metadata.get('extractors', {}).get('summary', {})) > 80 else doc.metadata.get('extractors', {}).get('summary', {})
            table.add_row("Executive Summary", summary)
    
    console.print(table)


def print_business_insights(documents: List[Document]):
    """Print aggregated insights from business reports."""
    console.print("\n📈 [bold green]Business Intelligence Dashboard[/bold green]")
    
    # Aggregate business metrics
    report_types = {}
    trends = {'positive': 0, 'negative': 0, 'neutral': 0}
    all_financial_metrics = {}
    total_tables = 0
    total_statistics = 0
    
    for doc in documents:
        # Aggregate report types
        if 'report_type' in doc.metadata:
            report_type = doc.metadata['report_type']
            report_types[report_type] = report_types.get(report_type, 0) + 1
        
        # Aggregate trends
        if 'business_trend' in doc.metadata:
            trend = doc.metadata['business_trend']['trend']
            trends[trend] = trends.get(trend, 0) + 1
        
        # Aggregate financial metrics
        if 'financial_metrics' in doc.metadata:
            metrics = doc.metadata['financial_metrics']
            for metric_type, values in metrics.items():
                if metric_type not in all_financial_metrics:
                    all_financial_metrics[metric_type] = []
                all_financial_metrics[metric_type].extend(values[:2])  # Limit per document
        
        # Count tables and statistics
        if 'tables' in doc.metadata:
            total_tables += len(doc.metadata['tables'])
        if 'statistics' in doc.metadata:
            total_statistics += len(doc.metadata.get('extractors', {}).get('statistics', {}))
    
    # Display business overview
    if report_types or any(trends.values()):
        overview_table = Table(title="📊 Business Reports Overview", show_header=True, header_style="bold yellow")
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", style="white")
        overview_table.add_column("Details", style="dim")
        
        # Report types
        for report_type, count in sorted(report_types.items(), key=lambda x: x[1], reverse=True):
            overview_table.add_row("Report Type", report_type, f"{count} documents")
        
        # Data extraction summary
        overview_table.add_row("Data Tables", str(total_tables), "Financial and operational data")
        overview_table.add_row("Statistics", str(total_statistics), "Numerical insights extracted")
        
        console.print(overview_table)
    
    # Display business trend analysis
    if any(trends.values()):
        trend_table = Table(title="📈 Business Trend Analysis", show_header=True, header_style="bold green")
        trend_table.add_column("Trend", style="cyan")
        trend_table.add_column("Reports", style="white")
        trend_table.add_column("Percentage", style="yellow")
        
        total_reports = sum(trends.values())
        for trend, count in trends.items():
            if count > 0:
                percentage = (count / total_reports) * 100 if total_reports > 0 else 0
                color = {'positive': 'green', 'negative': 'red', 'neutral': 'yellow'}.get(trend, 'white')
                trend_table.add_row(
                    f"[{color}]{trend.title()}[/{color}]",
                    str(count),
                    f"{percentage:.1f}%"
                )
        
        console.print(trend_table)
    
    # Display key financial metrics
    if all_financial_metrics:
        metrics_table = Table(title="💰 Key Financial Indicators", show_header=True, header_style="bold blue")
        metrics_table.add_column("Metric Type", style="cyan")
        metrics_table.add_column("Sample Values", style="white")
        metrics_table.add_column("Frequency", style="yellow")
        
        for metric_type, values in all_financial_metrics.items():
            if values:
                sample_values = ", ".join(str(v) for v in values[:3])  # Show first 3 values
                if len(values) > 3:
                    sample_values += f" (+{len(values) - 3} more)"
                metrics_table.add_row(metric_type.title(), sample_values, str(len(values)))
        
        console.print(metrics_table)


def print_business_search_results(query: str, results: List[Document]):
    """Print search results optimized for business reports."""
    console.print(f"\n🔍 Business Query: [bold yellow]'{query}'[/bold yellow]")
    console.print(f"📊 Found {len(results)} relevant business reports")
    
    for i, doc in enumerate(results[:3], 1):
        score = doc.metadata.get('search_score', 'N/A')
        score_text = f"Relevance: {score:.4f}" if isinstance(score, float) else f"Relevance: {score}"
        
        # Show business-specific metadata
        business_info = []
        
        if 'report_type' in doc.metadata:
            business_info.append(f"📋 {doc.metadata['report_type']}")
        
        if 'time_period' in doc.metadata:
            business_info.append(f"📅 {doc.metadata['time_period']}")
        
        if 'company' in doc.metadata:
            business_info.append(f"🏢 {doc.metadata['company']}")
        
        if 'business_trend' in doc.metadata:
            trend_data = doc.metadata['business_trend']
            if isinstance(trend_data, dict):
                trend = trend_data.get('trend', 'neutral')
            else:
                trend = str(trend_data)
            emoji = {'positive': '📈', 'negative': '📉', 'neutral': '➡️'}.get(trend, '➡️')
            business_info.append(f"{emoji} {trend.title()}")
        
        if 'tables' in doc.metadata and len(doc.metadata['tables']) > 0:
            business_info.append(f"📊 {len(doc.metadata['tables'])} tables")
        
        business_metadata = " | ".join(business_info) if business_info else "No metadata"
        
        # Content preview focusing on business insights
        content_preview = doc.content[:400] + "..." if len(doc.content) > 400 else doc.content
        
        result_text = f"""[bold]Source:[/bold] {Path(doc.source).name if doc.source else "Business Report"}
[bold]{score_text}[/bold]
[bold]Business Info:[/bold] {business_metadata}

{content_preview}"""
        
        console.print(Panel(
            result_text,
            title=f"Business Report #{i}",
            title_align="left",
            border_style="green" if i == 1 else "blue",
            expand=False
        ))


def demonstrate_business_reports_rag():
    """Demonstrate RAG system optimized for business report analysis."""
    
    print_section_header("🦙 Demo 5: Business Reports Analysis System", "📊")
    
    console.print("\n[bold green]This demo showcases:[/bold green]")
    console.print("• Multi-format business document processing (Excel, PDF, CSV)")
    console.print("• Financial metrics and KPI extraction from reports")
    console.print("• Table extraction for structured business data")
    console.print("• Business trend analysis and sentiment detection")
    console.print("• Executive summary generation and key insights")
    console.print("• Cross-format business intelligence aggregation")
    
    # Initialize components
    print_section_header("Business Analysis System Initialization", "⚙️")
    
    console.print("🔧 Initializing multi-format business parsers...")
    # CSV parser for structured business data
    csv_parser = CSVParser(name="csv_parser", config={
        "chunk_by_rows": True,
        "rows_per_chunk": 5,  # Group related supplier data
        "extract_headers": True,
        "preserve_structure": True
    })
    
    # Text parser for Excel/PDF content (saved as text)
    text_parser = PlainTextParser(name="business_text_parser", config={
        "chunk_size": 3000,    # Larger chunks for comprehensive business content
        "overlap": 500,        # More overlap for business context
        "preserve_line_breaks": True,
        "detect_structure": True
    })
    
    console.print("📈 Setting up business intelligence extractors...")
    # Table extractor for financial data
    table_extractor = TableExtractor(name="business_table_extractor", config={"min_columns": 2, "min_rows": 2, "detect_headers": True})
    
    # Statistics extractor for numerical business data
    statistics_extractor = ContentStatisticsExtractor("stats_extractor", {
        "extract_percentages": True,      # Growth rates, margins
        "extract_currency": True,         # Revenue, costs, profits
        "extract_ratios": True,          # Financial ratios
        "extract_trends": True           # YoY comparisons
    })
    
    # Summary extractor for executive summaries
    summary_extractor = SummaryExtractor(name="business_summary_extractor", config={"summary_sentences": 3, "min_sentence_length": 10, "max_sentence_length": 500, "include_key_phrases": True, "include_statistics": True})
    
    console.print("🧠 Initializing business embedder...")
    embedder = OllamaEmbedder("business_embedder", {
        "model": "nomic-embed-text",
        "batch_size": 2  # Smaller batches for large business documents
    })
    
    console.print("🗄️ Setting up business intelligence store...")
    vector_store = ChromaStore("business_reports_store", {
        "collection_name": "business_reports",
        "persist_directory": "./demos/vectordb/business_reports"
    })
    
    console.print("✅ Business analysis system initialized!")
    
    # Process business reports
    print_section_header("Business Report Processing", "📋")
    
    business_files = [
        ("demos/static_samples/business_reports/quarterly_financial_report.xlsx", "financial", "text"),
        ("demos/static_samples/business_reports/market_analysis_2024.pdf", "market", "text"),
        ("demos/static_samples/business_reports/supply_chain_metrics.csv", "operational", "csv")
    ]
    
    all_documents = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        parse_task = progress.add_task("Processing business reports...", total=len(business_files))
        
        for file_path, report_type, file_format in business_files:
            if not Path(file_path).exists():
                console.print(f"⚠️ Business report not found: {file_path}", style="yellow")
                progress.advance(parse_task)
                continue
            
            # Choose parser based on file format
            if file_format == "csv":
                parser = csv_parser
                console.print(f"📊 Processing CSV data: [bold]{Path(file_path).name}[/bold]")
            else:
                parser = text_parser
                console.print(f"📄 Processing report: [bold]{Path(file_path).name}[/bold]")
            
            result = parser.parse(file_path)
            # Handle both ProcessingResult and List[Document] return types
            if hasattr(result, 'documents'):
                documents = result.documents
            else:
                documents = result
            
            # Add business metadata and perform analysis
            for doc in documents:
                doc.metadata['report_type'] = report_type
                
                # Extract financial metrics
                financial_metrics = extract_financial_metrics(doc.content)
                if financial_metrics:
                    doc.metadata['financial_metrics'] = financial_metrics
                
                # Analyze business trends
                business_trend = analyze_business_trends(doc.content)
                doc.metadata['business_trend'] = business_trend
                
                # Add time period information
                if 'Q3 2024' in doc.content or '2024' in doc.content:
                    doc.metadata['time_period'] = '2024'
                
                # Extract company name
                if 'TechVenture' in doc.content:
                    doc.metadata['company'] = 'TechVenture Corp'
                elif 'Global Technology' in doc.content:
                    doc.metadata['company'] = 'Global Technology Research'
            
            all_documents.extend(documents)
            console.print(f"📊 Processed {len(documents)} business sections")
            progress.advance(parse_task)
    
    console.print(f"\n✅ Total business sections processed: [bold green]{len(all_documents)}[/bold green]")
    
    # Apply business extractors
    print_section_header("Business Intelligence Extraction", "🔍")
    
    console.print("📊 Analyzing business content with specialized extractors...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        extract_task = progress.add_task("Extracting business intelligence...", total=len(all_documents) * 3)
        
        # Apply table extractor
        all_documents = table_extractor.extract(all_documents)
        progress.advance(extract_task, len(all_documents))
        
        # Apply statistics extractor
        all_documents = statistics_extractor.extract(all_documents)
        progress.advance(extract_task, len(all_documents))
        
        # Apply summary extractor
        all_documents = summary_extractor.extract(all_documents)
        progress.advance(extract_task, len(all_documents))
    
    console.print("✅ Business intelligence extraction complete!")
    
    # Show business insights
    print_business_insights(all_documents)
    
    # Show detailed analysis of business reports
    console.print("\n📊 [bold green]Sample Business Report Analysis[/bold green]")
    for i, doc in enumerate([doc for doc in all_documents if len(doc.content) > 500][:3], 1):
        console.print(f"\n📋 Business Report #{i}:")
        print_business_report_analysis(doc)
    
    # Generate embeddings
    print_section_header("Business Content Embedding", "🧠")
    
    console.print("🔄 Generating embeddings for business reports...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        embed_task = progress.add_task("Generating embeddings...", total=len(all_documents))
        
        for doc in all_documents:
            if not doc.embeddings:
                embeddings = embedder.embed([doc.content])
                doc.embeddings = embeddings[0] if embeddings else []
            progress.advance(embed_task)
    
    console.print(f"✅ Generated embeddings for {len(all_documents)} business sections")
    
    # Store in vector database
    print_section_header("Business Intelligence Storage", "🗄️")
    
    console.print("💾 Building searchable business intelligence database...")
    success = vector_store.add_documents(all_documents)
    if success:
        console.print(f"✅ Stored {len(all_documents)} business sections in database")
    else:
        console.print("❌ Failed to store business reports")
        return
    
    # Demonstrate business queries
    print_section_header("Business Intelligence Query Demonstration", "🔍")
    
    business_queries = [
        "What were the key financial performance metrics for Q3 2024?",
        "How did AI and machine learning revenue perform year-over-year?",
        "What are the major technology market trends and growth projections?",
        "Which suppliers have the highest performance scores and lowest risk?",
        "What investment opportunities exist in quantum computing and AI?",
        "How is the competitive landscape evolving in cloud computing?",
        "What are the key risk factors affecting technology companies?",
        "Which regions show the strongest growth in technology adoption?",
        "What sustainability initiatives are driving technology investment?"
    ]
    
    console.print("🎯 Running business intelligence queries to demonstrate analytical capabilities:")
    
    for i, query in enumerate(business_queries, 1):
        console.print(f"\n[bold cyan]Business Query #{i}:[/bold cyan]")
        
        # Generate query embedding
        console.print("🧠 Analyzing business query...")
        query_embeddings = embedder.embed([query])
        query_embedding = query_embeddings[0] if query_embeddings else []
        
        # Search for relevant business intelligence
        console.print("🔍 Searching business intelligence database...")
        results = vector_store.search(query_embedding=query_embedding, top_k=3)
        
        # Show business-focused results
        print_business_search_results(query, results)
        
        if i < len(business_queries):
            console.print("\n" + "─" * 50)
            time.sleep(1.5)  # Pause for readability
    
    # Show business system statistics
    print_section_header("Business Intelligence Analytics", "📊")
    
    info = vector_store.get_collection_info()
    if info:
        # Calculate business-specific metrics
        financial_reports = sum(1 for doc in all_documents if doc.metadata.get('report_type') == 'financial')
        market_reports = sum(1 for doc in all_documents if doc.metadata.get('report_type') == 'market')
        operational_reports = sum(1 for doc in all_documents if doc.metadata.get('report_type') == 'operational')
        total_tables = sum(len(doc.metadata.get('tables', [])) for doc in all_documents)
        total_statistics = sum(len(doc.metadata.get('statistics', [])) for doc in all_documents)
        positive_trends = sum(1 for doc in all_documents if doc.metadata.get('business_trend', {}).get('trend') == 'positive')
        
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Business Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Business Collection", info.get("name", "N/A"))
        stats_table.add_row("Total Sections", str(info.get("document_count", "N/A")))
        stats_table.add_row("Financial Reports", str(financial_reports))
        stats_table.add_row("Market Analysis", str(market_reports))
        stats_table.add_row("Operational Data", str(operational_reports))
        stats_table.add_row("Data Tables", str(total_tables))
        stats_table.add_row("Statistics Extracted", str(total_statistics))
        stats_table.add_row("Positive Trends", str(positive_trends))
        stats_table.add_row("Embedding Model", embedder.model)
        
        console.print(stats_table)
    
    # Business system summary
    print_section_header("Business Intelligence Summary", "🎉")
    
    console.print("🚀 [bold green]Business Reports Analysis Complete![/bold green]")
    console.print("\n[bold]What this demo demonstrated:[/bold]")
    console.print("✅ Multi-format business document processing")
    console.print("✅ Financial metrics and KPI extraction")
    console.print("✅ Table extraction for structured business data")
    console.print("✅ Business trend analysis and sentiment detection")
    console.print("✅ Executive summary generation")
    console.print("✅ Cross-format business intelligence aggregation")
    
    console.print(f"\n[bold]Why this approach is powerful for business analysis:[/bold]")
    console.print("💰 Automatic financial metrics extraction")
    console.print("📊 Structured data processing and analysis")
    console.print("📈 Trend identification and sentiment analysis")
    console.print("🔍 Cross-report insights and correlations")
    console.print("⚡ Rapid business intelligence generation")
    
    console.print(f"\n📁 Business intelligence database saved to: [bold]./demos/vectordb/business_reports[/bold]")
    console.print("🔄 You can now query this business database using the CLI:")
    console.print("[dim]uv run python cli.py search 'quarterly revenue growth trends' --collection business_reports[/dim]")

    # Clean up database to prevent duplicate accumulation
    print_section_header("Database Cleanup", "🧹")
    console.print("🗑️  Cleaning up vector database to prevent duplicate accumulation...")
    try:
        # Delete the collection to clean up
        vector_store.delete_collection()
        console.print("✅ [green]Database cleaned successfully![/green]")
        console.print("[dim]The database has been reset to prevent duplicate data accumulation in future runs.[/dim]")
    except Exception as e:
        console.print(f"⚠️  [yellow]Note: Could not clean database: {e}[/yellow]")
        console.print("[dim]You may want to manually clean the vector database directory.[/dim]")



if __name__ == "__main__":
    try:
        demonstrate_business_reports_rag()
    except KeyboardInterrupt:
        console.print("\n\n👋 Business demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ Business demo failed: {str(e)}", style="red")
        console.print("Check that Ollama is running with the nomic-embed-text model")
        sys.exit(1)