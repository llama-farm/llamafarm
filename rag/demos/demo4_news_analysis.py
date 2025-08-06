#!/usr/bin/env python3
"""
Demo 4: News Article Analysis System
Demonstrates RAG capabilities for news and media content using:
- HTML parser for web-based news articles
- Entity extraction for people, organizations, locations, events
- Summary extraction for article abstracts and key points
- Link extraction for source verification and related articles
- Sentiment and topic analysis for news categorization
- Timeline and trend analysis capabilities
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
from components.parsers.html_parser.html_parser import HtmlParser as HTMLParser
from components.extractors.entity_extractor.entity_extractor import EntityExtractor
from components.extractors.summary_extractor.summary_extractor import SummaryExtractor
from components.extractors.link_extractor.link_extractor import LinkExtractor
from components.extractors.path_extractor.path_extractor import PathExtractor
from components.embedders.ollama_embedder.ollama_embedder import OllamaEmbedder
from components.stores.chroma_store.chroma_store import ChromaStore

# Import demo utilities for metadata display
from demos.utils import (
    display_document_with_metadata,
    display_embedding_process,
    display_search_results_with_metadata,
    add_processing_timestamp,
    generate_document_id,
    display_demo_separator
)

# Setup rich console for beautiful output
console = Console()
logging.basicConfig(level=logging.WARNING)  # Reduce noise


def print_section_header(title: str, emoji: str = "📰"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def extract_article_metadata(content: str) -> Dict[str, Any]:
    """Extract metadata from HTML news article content."""
    import re
    from datetime import datetime
    
    metadata = {}
    
    # Extract title
    title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
    if title_match:
        metadata['title'] = title_match.group(1).strip()
    
    # Extract meta tags
    meta_patterns = {
        'author': r'<meta name="author" content="(.*?)"',
        'published': r'<meta name="published" content="(.*?)"',
        'category': r'<meta name="category" content="(.*?)"',
        'keywords': r'<meta name="keywords" content="(.*?)"',
        'description': r'<meta name="description" content="(.*?)"'
    }
    
    for key, pattern in meta_patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metadata[key] = match.group(1).strip()
    
    # Extract reading time
    reading_time_match = re.search(r'(\d+)\s*min\s*read', content, re.IGNORECASE)
    if reading_time_match:
        metadata['reading_time'] = int(reading_time_match.group(1))
    
    # Extract article tags
    tag_matches = re.findall(r'<span class="tag">(.*?)</span>', content)
    if tag_matches:
        metadata['tags'] = tag_matches
    
    return metadata


def analyze_article_sentiment(content: str) -> Dict[str, Any]:
    """Simple sentiment analysis of news article content."""
    positive_words = [
        'breakthrough', 'success', 'achievement', 'innovative', 'revolutionary', 
        'excellent', 'outstanding', 'remarkable', 'significant', 'positive',
        'growth', 'increase', 'improvement', 'advancement', 'progress'
    ]
    
    negative_words = [
        'crisis', 'failure', 'problem', 'issue', 'decline', 'decrease',
        'concern', 'worry', 'threat', 'risk', 'challenge', 'difficulty',
        'controversy', 'scandal', 'critical', 'serious', 'urgent'
    ]
    
    neutral_words = [
        'analysis', 'study', 'research', 'data', 'report', 'information',
        'statement', 'announcement', 'development', 'update', 'change'
    ]
    
    content_lower = content.lower()
    
    positive_count = sum(1 for word in positive_words if word in content_lower)
    negative_count = sum(1 for word in negative_words if word in content_lower)
    neutral_count = sum(1 for word in neutral_words if word in content_lower)
    
    total_sentiment_words = positive_count + negative_count + neutral_count
    
    if total_sentiment_words == 0:
        return {'sentiment': 'neutral', 'confidence': 0.0, 'scores': {'positive': 0, 'negative': 0, 'neutral': 0}}
    
    positive_ratio = positive_count / total_sentiment_words
    negative_ratio = negative_count / total_sentiment_words
    neutral_ratio = neutral_count / total_sentiment_words
    
    if positive_ratio > negative_ratio and positive_ratio > neutral_ratio:
        sentiment = 'positive'
        confidence = positive_ratio
    elif negative_ratio > positive_ratio and negative_ratio > neutral_ratio:
        sentiment = 'negative'
        confidence = negative_ratio
    else:
        sentiment = 'neutral'
        confidence = neutral_ratio
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'scores': {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count
        }
    }


def print_news_article_analysis(doc: Document):
    """Print analysis of a news article with extracted information."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Article Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    # Basic document info
    table.add_row("Article ID", doc.id)
    table.add_row("Source", Path(doc.source).name if doc.source else "News Article")
    table.add_row("Content Length", f"{len(doc.content):,} characters")
    table.add_row("Word Count", f"{len(doc.content.split()):,} words")
    
    # News-specific metadata
    if doc.metadata:
        if 'title' in doc.metadata:
            title = doc.metadata['title'][:60] + "..." if len(doc.metadata.get('title', '')) > 60 else doc.metadata.get('title', 'N/A')
            table.add_row("Title", title)
        
        if 'author' in doc.metadata:
            table.add_row("Author", doc.metadata['author'])
        
        if 'category' in doc.metadata:
            table.add_row("Category", doc.metadata['category'])
        
        if 'published' in doc.metadata:
            table.add_row("Published", doc.metadata['published'])
        
        if 'reading_time' in doc.metadata:
            table.add_row("Reading Time", f"{doc.metadata['reading_time']} minutes")
        
        # Sentiment analysis
        if 'sentiment' in doc.metadata:
            sentiment_data = doc.metadata['sentiment']
            sentiment_color = {
                'positive': 'green',
                'negative': 'red', 
                'neutral': 'yellow'
            }.get(sentiment_data['sentiment'], 'white')
            
            table.add_row("Sentiment", f"[{sentiment_color}]{sentiment_data['sentiment'].title()}[/{sentiment_color}] ({sentiment_data['confidence']:.2f})")
        
        # Extractor results
        if 'entities' in doc.metadata:
            entities = len(doc.metadata.get('extractors', {}).get('entities', {}))
            table.add_row("Named Entities", f"{entities} identified")
        
        if 'links' in doc.metadata:
            links = len(doc.metadata['links'])
            table.add_row("Article Links", f"{links} references")
        
        if 'summary' in doc.metadata:
            summary = doc.metadata.get('extractors', {}).get('summary', {})[:80] + "..." if len(doc.metadata.get('extractors', {}).get('summary', {})) > 80 else doc.metadata.get('extractors', {}).get('summary', {})
            table.add_row("Auto Summary", summary)
        
        if 'tags' in doc.metadata:
            tags = ", ".join(doc.metadata['tags'][:5])  # Show first 5 tags
            if len(doc.metadata['tags']) > 5:
                tags += f" (+{len(doc.metadata['tags']) - 5} more)"
            table.add_row("Article Tags", tags)
    
    console.print(table)


def print_news_insights(documents: List[Document]):
    """Print aggregated insights from news articles."""
    console.print("\n📊 [bold green]News Analysis Intelligence[/bold green]")
    
    # Aggregate news metrics
    categories = {}
    authors = {}
    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
    all_entities = {}
    all_tags = {}
    
    for doc in documents:
        # Aggregate categories
        if 'category' in doc.metadata:
            category = doc.metadata['category']
            categories[category] = categories.get(category, 0) + 1
        
        # Aggregate authors
        if 'author' in doc.metadata:
            author = doc.metadata['author']
            authors[author] = authors.get(author, 0) + 1
        
        # Aggregate sentiments
        if 'sentiment' in doc.metadata:
            sentiment = doc.metadata['sentiment']['sentiment']
            sentiments[sentiment] = sentiments.get(sentiment, 0)
        
        # Aggregate entities
        if 'entities' in doc.metadata:
            for entity in doc.metadata.get('extractors', {}).get('entities', {}):
                entity_type = entity.get('label', 'Unknown')
                entity_text = entity.get('text', '').lower()
                if entity_type not in all_entities:
                    all_entities[entity_type] = {}
                all_entities[entity_type][entity_text] = all_entities[entity_type].get(entity_text, 0) + 1
        
        # Aggregate tags
        if 'tags' in doc.metadata:
            for tag in doc.metadata['tags']:
                all_tags[tag] = all_tags.get(tag, 0) + 1
    
    # Display news distribution
    if categories or authors:
        distribution_table = Table(title="📰 News Article Distribution", show_header=True, header_style="bold yellow")
        distribution_table.add_column("Metric", style="cyan")
        distribution_table.add_column("Value", style="white")
        distribution_table.add_column("Count", style="green")
        
        # Top categories
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            distribution_table.add_row("Category", category, str(count))
        
        # Top authors
        for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True)[:3]:
            distribution_table.add_row("Author", author, str(count))
        
        console.print(distribution_table)
    
    # Display sentiment analysis
    if any(sentiments.values()):
        sentiment_table = Table(title="😊 Sentiment Analysis", show_header=True, header_style="bold green")
        sentiment_table.add_column("Sentiment", style="cyan")
        sentiment_table.add_column("Articles", style="white")
        sentiment_table.add_column("Percentage", style="yellow")
        
        total_articles = sum(sentiments.values())
        for sentiment, count in sentiments.items():
            if count > 0:
                percentage = (count / total_articles) * 100 if total_articles > 0 else 0
                color = {'positive': 'green', 'negative': 'red', 'neutral': 'yellow'}.get(sentiment, 'white')
                sentiment_table.add_row(
                    f"[{color}]{sentiment.title()}[/{color}]",
                    str(count),
                    f"{percentage:.1f}%"
                )
        
        console.print(sentiment_table)
    
    # Display top entities
    if all_entities:
        entities_table = Table(title="🏷️ Key News Entities", show_header=True, header_style="bold blue")
        entities_table.add_column("Entity Type", style="cyan")
        entities_table.add_column("Most Mentioned", style="white")
        entities_table.add_column("Frequency", style="yellow")
        
        for entity_type, entities in all_entities.items():
            if entities:
                most_common = max(entities.items(), key=lambda x: x[1])
                entities_table.add_row(entity_type, most_common[0].title(), str(most_common[1]))
        
        console.print(entities_table)
    
    # Display popular tags
    if all_tags:
        tags_table = Table(title="🏷️ Popular Article Tags", show_header=True, header_style="bold magenta")
        tags_table.add_column("Tag", style="cyan")
        tags_table.add_column("Articles", style="white")
        tags_table.add_column("Relevance", style="dim")
        
        tag_descriptions = {
            'Technology': 'Tech news and innovations',
            'Artificial Intelligence': 'AI and ML developments',
            'Climate Technology': 'Environmental and clean tech',
            'Quantum Computing': 'Quantum technology advances',
            'Investment': 'Financial and market news',
            'Research': 'Scientific research and studies'
        }
        
        for tag, count in sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:8]:
            description = tag_descriptions.get(tag, 'News category')
            tags_table.add_row(tag, str(count), description)
        
        console.print(tags_table)


def print_news_search_results(query: str, results: List[Document]):
    """Print search results optimized for news articles."""
    console.print(f"\n🔍 News Query: [bold yellow]'{query}'[/bold yellow]")
    console.print(f"📰 Found {len(results)} relevant news articles")
    
    for i, doc in enumerate(results[:3], 1):
        score = doc.metadata.get('search_score', 'N/A')
        score_text = f"Relevance: {score:.4f}" if isinstance(score, float) else f"Relevance: {score}"
        
        # Show news-specific metadata
        news_info = []
        
        if 'category' in doc.metadata:
            category = doc.metadata['category']
            news_info.append(f"📂 {category}")
        
        if 'author' in doc.metadata:
            author = doc.metadata['author']
            news_info.append(f"✍️ {author}")
        
        if 'published' in doc.metadata:
            published = doc.metadata['published']
            news_info.append(f"📅 {published}")
        
        if 'sentiment' in doc.metadata:
            sentiment_data = doc.metadata['sentiment']
            if isinstance(sentiment_data, dict):
                sentiment = sentiment_data.get('sentiment', 'neutral')
            else:
                sentiment = str(sentiment_data)
            emoji = {'positive': '😊', 'negative': '😟', 'neutral': '😐'}.get(sentiment, '😐')
            news_info.append(f"{emoji} {sentiment.title()}")
        
        if 'reading_time' in doc.metadata:
            reading_time = doc.metadata['reading_time']
            news_info.append(f"⏱️ {reading_time} min read")
        
        news_metadata = " | ".join(news_info) if news_info else "No metadata"
        
        # Content preview focusing on news content
        content_preview = doc.content[:380] + "..." if len(doc.content) > 380 else doc.content
        
        # Show title if available
        title = doc.metadata.get('title', Path(doc.source).name if doc.source else "News Article")
        if len(title) > 80:
            title = title[:77] + "..."
        
        result_text = f"""[bold]Title:[/bold] {title}
[bold]{score_text}[/bold]
[bold]Article Info:[/bold] {news_metadata}

{content_preview}"""
        
        console.print(Panel(
            result_text,
            title=f"News Article #{i}",
            title_align="left",
            border_style="green" if i == 1 else "blue",
            expand=False
        ))


def demonstrate_news_analysis_rag():
    """Demonstrate RAG system optimized for news article analysis."""
    
    print_section_header("🦙 Demo 4: News Article Analysis System", "📰")
    
    console.print("\n[bold green]This demo showcases:[/bold green]")
    console.print("• Advanced HTML parsing for web-based news articles")
    console.print("• Entity extraction for people, organizations, and events") 
    console.print("• Automated sentiment analysis and topic categorization")
    console.print("• Summary extraction for article abstracts and key points")
    console.print("• Link extraction for source verification and references")
    console.print("• News trend analysis and topic clustering")
    
    # Initialize components
    print_section_header("News Analysis System Initialization", "⚙️")
    
    console.print("🔧 Initializing HTML news parser...")
    parser = HTMLParser(config={
        "extract_links": True,          # Extract article links
        "extract_images": True,         # Extract images
        "extract_meta_tags": True,      # Extract HTML meta tags
        "preserve_structure": False,    # Clean HTML structure
        "remove_scripts": True,         # Remove scripts
        "remove_styles": True          # Remove styles
    })
    
    console.print("🎯 Setting up news-focused extractors...")
    # Path extractor to preserve source information
    path_extractor = PathExtractor("path_extractor", {"store_full_path": True, "store_filename": True, "store_directory": True, "store_extension": True})
    
    # Entity extractor for news entities
    entity_extractor = EntityExtractor("entity_extractor", {"entity_types": ["PERSON", "ORG", "GPE", "DATE", "PERCENT", "PRODUCT"], "use_fallback": True, "min_entity_length": 2})
    
    # Summary extractor for article summaries
    summary_extractor = SummaryExtractor({"summary_sentences": 3, "min_sentence_length": 10, "max_sentence_length": 500, "include_key_phrases": True, "include_statistics": True})
    
    # Link extractor for source verification
    link_extractor = LinkExtractor({"include_external": True, "include_internal": True, "include_anchors": True})
    
    console.print("🧠 Initializing news embedder...")
    embedder = OllamaEmbedder("news_embedder", {
        "model": "nomic-embed-text",
        "batch_size": 3  # Smaller batches for longer news articles
    })
    
    console.print("🗄️ Setting up news knowledge store...")
    vector_store = ChromaStore("news_analysis_store", {
        "collection_name": "news_articles",
        "persist_directory": "./demos/vectordb/news_analysis"
    })
    
    console.print("✅ News analysis system initialized!")
    
    # Process news articles
    print_section_header("News Article Processing", "📰")
    
    news_files = [
        "demos/static_samples/news_articles/ai_breakthrough.html",
        "demos/static_samples/news_articles/climate_tech_report.html",
        "demos/static_samples/news_articles/quantum_computing_milestone.html"
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
        
        parse_task = progress.add_task("Processing news articles...", total=len(news_files))
        
        for file_path in news_files:
            if not Path(file_path).exists():
                console.print(f"⚠️ News article not found: {file_path}", style="yellow")
                progress.advance(parse_task)
                continue
            
            result = parser.parse(file_path)
            # Handle both ProcessingResult and List[Document] return types
            if hasattr(result, 'documents'):
                documents = result.documents
            else:
                documents = result
            
            # Extract article metadata and perform sentiment analysis
            for doc in documents:
                # Set source file path
                doc.source = file_path
                
                # Extract HTML metadata
                article_metadata = extract_article_metadata(doc.content)
                doc.metadata.update(article_metadata)
                
                # Perform sentiment analysis
                sentiment_analysis = analyze_article_sentiment(doc.content)
                doc.metadata['sentiment'] = sentiment_analysis
            
            all_documents.extend(documents)
            
            console.print(f"📄 Processed [bold]{Path(file_path).name}[/bold]: {len(documents)} article(s)")
            progress.advance(parse_task)
    
    console.print(f"\n✅ Total news articles processed: [bold green]{len(all_documents)}[/bold green]")
    
    # Add processing timestamps and IDs to documents
    all_documents = add_processing_timestamp(all_documents)
    for doc in all_documents:
        if not doc.id:
            doc.id = generate_document_id(doc.content, doc.source)
    
    display_demo_separator()
    
    # Apply news extractors
    print_section_header("News Intelligence Extraction", "🔍")
    
    console.print("🔬 Analyzing news content with specialized extractors...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        extract_task = progress.add_task("Extracting news intelligence...", total=len(all_documents) * 4)
        
        # Apply path extractor first to preserve source information
        all_documents = path_extractor.extract(all_documents)
        progress.advance(extract_task, len(all_documents))
        
        # Apply entity extractor
        all_documents = entity_extractor.extract(all_documents)
        progress.advance(extract_task, len(all_documents))
        
        # Apply summary extractor
        all_documents = summary_extractor.extract(all_documents)
        progress.advance(extract_task, len(all_documents))
        
        # Apply link extractor
        all_documents = link_extractor.extract(all_documents)
        progress.advance(extract_task, len(all_documents))
    
    console.print("✅ News intelligence extraction complete!")
    
    # Show news insights
    print_news_insights(all_documents)
    
    # Show sample documents with full metadata
    console.print("\n📊 [bold green]Sample Documents with Full Metadata[/bold green]")
    for i, doc in enumerate(all_documents[:2]):
        display_document_with_metadata(doc, i, "News Article")
    
    display_demo_separator()
    
    # Show detailed analysis of articles
    console.print("\n📰 [bold green]Sample News Article Analysis[/bold green]")
    for i, doc in enumerate(all_documents[:3], 1):
        console.print(f"\n📄 News Article #{i}:")
        print_news_article_analysis(doc)
    
    # Generate embeddings
    print_section_header("News Content Embedding", "🧠")
    
    console.print("🔄 Generating embeddings for news articles...")
    
    # Show what will be embedded
    display_embedding_process(all_documents, "Ollama (nomic-embed-text)")
    
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
    
    console.print(f"✅ Generated embeddings for {len(all_documents)} news articles")
    
    # Store in vector database
    print_section_header("News Knowledge Base Storage", "🗄️")
    
    console.print("💾 Building searchable news knowledge base...")
    success = vector_store.add_documents(all_documents)
    if success:
        console.print(f"✅ Stored {len(all_documents)} news articles in knowledge base")
    else:
        console.print("❌ Failed to store news articles")
        return
    
    # Demonstrate news queries
    print_section_header("News Query Demonstration", "🔍")
    
    news_queries = [
        "What are the latest breakthroughs in artificial intelligence research?",
        "How is climate technology investment performing in 2024?",
        "What impact will quantum computing have on cybersecurity?",
        "Which companies are leading in AI development and innovation?",
        "What are the regulatory implications of new AI technologies?",
        "How are governments investing in clean energy technologies?",
        "What are the commercial applications of quantum computing?",
        "What challenges does the tech industry face with scaling AI?",
        "How are environmental concerns driving technology investment?"
    ]
    
    console.print("🎯 Running news queries to demonstrate article analysis and search:")
    
    for i, query in enumerate(news_queries, 1):
        console.print(f"\n[bold cyan]News Query #{i}:[/bold cyan]")
        
        # Generate query embedding
        console.print("🧠 Analyzing news query...")
        query_embeddings = embedder.embed([query])
        query_embedding = query_embeddings[0] if query_embeddings else []
        
        # Search for relevant articles
        console.print("🔍 Searching news knowledge base...")
        results = vector_store.search(query_embedding=query_embedding, top_k=3)
        
        # Show news-focused results with full metadata
        display_search_results_with_metadata(results, query)
        
        if i < len(news_queries):
            display_demo_separator()
            time.sleep(1.4)  # Pause for readability
    
    # Show news system statistics
    print_section_header("News Analysis System Analytics", "📊")
    
    info = vector_store.get_collection_info()
    if info:
        # Calculate news-specific metrics
        total_entities = sum(len(doc.metadata.get('entities', [])) for doc in all_documents)
        total_links = sum(len(doc.metadata.get('links', [])) for doc in all_documents)
        positive_articles = sum(1 for doc in all_documents if doc.metadata.get('sentiment', {}).get('sentiment') == 'positive')
        negative_articles = sum(1 for doc in all_documents if doc.metadata.get('sentiment', {}).get('sentiment') == 'negative')
        
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("News Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("News Collection", info.get("name", "N/A"))
        stats_table.add_row("Total Articles", str(info.get("document_count", "N/A")))
        stats_table.add_row("Named Entities", str(total_entities))
        stats_table.add_row("Source Links", str(total_links))
        stats_table.add_row("Positive Sentiment", str(positive_articles))
        stats_table.add_row("Negative Sentiment", str(negative_articles))
        stats_table.add_row("Embedding Model", embedder.model)
        
        console.print(stats_table)
    
    # News system summary
    print_section_header("News Analysis System Summary", "🎉")
    
    console.print("🚀 [bold green]News Article Analysis Complete![/bold green]")
    console.print("\n[bold]What this demo demonstrated:[/bold]")
    console.print("✅ Advanced HTML parsing for web-based news content")
    console.print("✅ Entity extraction for people, organizations, and events")
    console.print("✅ Automated sentiment analysis and topic categorization")
    console.print("✅ Summary extraction for article abstracts")
    console.print("✅ Link extraction for source verification")
    console.print("✅ News trend analysis and intelligent search")
    
    console.print(f"\n[bold]Why this approach is powerful for news analysis:[/bold]")
    console.print("📈 Tracks sentiment trends across topics")
    console.print("🏷️ Identifies key entities and relationships")
    console.print("🔗 Maintains source verification and credibility")
    console.print("📊 Enables topic clustering and trend analysis")
    console.print("🎯 Provides context-aware news search")
    
    console.print(f"\n📁 News knowledge base saved to: [bold]./demos/vectordb/news_analysis[/bold]")
    console.print("🔄 You can now query this news database using the CLI:")
    console.print("[dim]uv run python cli.py search 'AI quantum computing trends' --collection news_articles[/dim]")

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
        demonstrate_news_analysis_rag()
    except KeyboardInterrupt:
        console.print("\n\n👋 News demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ News demo failed: {str(e)}", style="red")
        console.print("Check that Ollama is running with the nomic-embed-text model")
        sys.exit(1)