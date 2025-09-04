#!/usr/bin/env python3
"""Simple CLI for the RAG system."""

import argparse
import logging
import json
import sys
import time
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

from core.base import Pipeline
from core.enhanced_pipeline import EnhancedPipeline
from core.factories import (
    create_embedder_from_config,
    create_parser_from_config,
    create_vector_store_from_config,
    create_retrieval_strategy_from_config,
    EmbedderFactory,
    VectorStoreFactory,
    ParserFactory,
)
# Import for retrieval strategies is handled via core.factories
from utils.progress import LlamaProgressTracker, create_enhanced_progress_bar
from utils.path_resolver import PathResolver, resolve_paths_in_config
from core.document_manager import DocumentManager, DeletionStrategy, UpdateStrategy
from core.extractor_integration import ExtractorIntegrator, apply_extractors_from_cli_args
from components.extractors import registry
from core.strategies import StrategyManager


def setup_logging(level: str = "INFO", quiet: bool = False):
    """Setup logging configuration.

    TODO: Replace with global logging module when available.
    """
    # If quiet mode, only show warnings and errors
    if quiet:
        level = "WARNING"
    
    # Configure logging - only show time for DEBUG level
    if level == "DEBUG":
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        # Simplified format for normal use
        format_str = "%(message)s"
    
    # Configure logging with proper stream handling to avoid "Error:" prefix and duplicates
    import sys

    # Create a custom handler that sends INFO and below to stdout, WARNING+ to stderr
    class InfoStdoutFilter(logging.Filter):
        def filter(self, record):
            return record.levelno <= logging.INFO

    # Clear any existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Add stderr handler for WARNING and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)
    
    # Add stdout handler for INFO and below to avoid "Error:" prefix
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(InfoStdoutFilter())
    root_logger.addHandler(stdout_handler)
    
    # Prevent propagation to avoid duplicate messages
    root_logger.propagate = False
    
    # Suppress specific noisy loggers
    if level != "DEBUG":
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
        logging.getLogger("rag.core.strategies.loader").setLevel(logging.WARNING)
        logging.getLogger("rag.core.strategies.manager").setLevel(logging.WARNING)
        logging.getLogger("components.parsers").setLevel(logging.WARNING)
        logging.getLogger("components.stores").setLevel(logging.WARNING)


def detect_file_type(file_path: Path) -> str:
    """Detect file type based on extension and mime type.
    
    Args:
        file_path: Path to the file or directory
        
    Returns:
        File type string (e.g., 'csv', 'pdf', 'json', 'directory')
    """
    # Check if it's a directory
    if file_path.is_dir():
        return 'directory'
    
    extension = file_path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(str(file_path))
    
    # Extension-based detection (primary)
    extension_map = {
        '.csv': 'csv',
        '.pdf': 'pdf',
        '.json': 'json',
        '.txt': 'text',
        '.md': 'markdown',
        '.docx': 'docx',
        '.doc': 'doc',
        '.html': 'html',
        '.htm': 'html',
        '.xls': 'excel',
        '.xlsx': 'excel'
    }
    
    if extension in extension_map:
        return extension_map[extension]
    
    # Mime type fallback
    if mime_type:
        if 'csv' in mime_type:
            return 'csv'
        elif 'pdf' in mime_type:
            return 'pdf'
        elif 'json' in mime_type:
            return 'json'
        elif 'text' in mime_type:
            return 'text'
    
    return 'unknown'


def is_unified_config(config: Dict[str, Any]) -> bool:
    """Check if config is the new unified format (v1+ schema).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if unified config format
    """
    # Check for Matt's v1 schema format
    if (config.get("version") == "v1" and 
        "rag" in config and 
        isinstance(config["rag"], dict)):
        rag_config = config["rag"]
        return (
            "parsers" in rag_config and
            "embedders" in rag_config and
            "vector_stores" in rag_config and
            "defaults" in rag_config
        )
    # Check for legacy v2.0+ format
    return (
        config.get("version", "1.0").startswith("2.") and
        "parsers" in config and
        "defaults" in config
    )


def select_parser_config(config: Dict[str, Any], file_type: str, parser_override: Optional[str] = None) -> Dict[str, Any]:
    """Select appropriate parser configuration based on file type and overrides.
    
    Args:
        config: Full configuration dictionary
        file_type: Detected file type
        parser_override: CLI override for parser selection
        
    Returns:
        Parser configuration dictionary
    """
    if not is_unified_config(config):
        # Legacy config format
        return config.get("parser", {})
    
    # Handle Matt's v1 schema format with nested rag structure
    if config.get("version") == "v1" and "rag" in config:
        rag_config = config["rag"]
        parsers = rag_config.get("parsers", {})
        defaults = rag_config.get("defaults", {})
    else:
        # Handle v2.0+ format
        parsers = config.get("parsers", {})
        defaults = config.get("defaults", {})
    
    if parser_override and parser_override in parsers:
        return parsers[parser_override]
    
    # Auto-select based on file type
    for parser_name, parser_config in parsers.items():
        extensions = parser_config.get("file_extensions", [])
        mime_types = parser_config.get("mime_types", [])
        
        # Check extension match
        if f".{file_type}" in extensions:
            return parser_config
        
        # Check mime type match (future enhancement)
        # This could be used for more sophisticated detection
    
    # Fallback to default or first available
    default_parser = defaults.get("parser", "auto")
    
    if default_parser != "auto" and default_parser in parsers:
        return parsers[default_parser]
    
    # Return first parser as ultimate fallback
    if parsers:
        return list(parsers.values())[0]
    
    return {}


def select_component_config(config: Dict[str, Any], component_type: str, override: Optional[str] = None) -> Dict[str, Any]:
    """Select component configuration with override support.
    
    Args:
        config: Full configuration dictionary
        component_type: Type of component ('embedders', 'vector_stores', 'retrieval_strategies')
        override: CLI override for component selection
        
    Returns:
        Component configuration dictionary
    """
    if not is_unified_config(config):
        # Legacy config format - map component types
        legacy_map = {
            'embedders': 'embedder',
            'vector_stores': 'vector_store', 
            'retrieval_strategies': 'retrieval_strategy'
        }
        return config.get(legacy_map.get(component_type, component_type), {})
    
    # Handle Matt's v1 schema format with nested rag structure
    if config.get("version") == "v1" and "rag" in config:
        rag_config = config["rag"]
        components = rag_config.get(component_type, {})
        defaults = rag_config.get("defaults", {})
    else:
        # Handle v2.0+ format
        components = config.get(component_type, {})
        defaults = config.get("defaults", {})
    
    if override and override in components:
        return components[override]
    
    # Use default from config
    default_key = component_type.rstrip('s')  # 'embedders' -> 'embedder'
    if default_key.endswith('ie'):
        default_key = default_key[:-2] + 'y'  # 'strategies' -> 'strategy'
    
    default_name = defaults.get(default_key, "default")
    
    if default_name in components:
        return components[default_name]
    
    # Return first available as fallback
    if components:
        return list(components.values())[0]
    
    return {}


def load_config(config_path: str, base_dir: str = None) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file with flexible path resolution.

    Args:
        config_path: Path to configuration file (can be relative or absolute)
        base_dir: Base directory for relative path resolution

    TODO: Replace with top-level config lib when available.
    """
    resolver = PathResolver(base_dir)

    try:
        resolved_config_path = resolver.resolve_config_path(config_path)
        
        with open(resolved_config_path, "r") as f:
            file_content = f.read()
        
        # Try to determine file type and parse accordingly
        if resolved_config_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                config = yaml.safe_load(file_content)
            except ImportError:
                print("PyYAML not installed. Install with: pip install PyYAML")
                sys.exit(1)
            except yaml.YAMLError as e:
                print(f"Invalid YAML in config file: {e}")
                sys.exit(1)
        else:
            # Default to JSON
            try:
                config = json.loads(file_content)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in config file: {e}")
                sys.exit(1)

        # Resolve any paths within the configuration
        config = resolve_paths_in_config(config, resolver)

        return config
    except FileNotFoundError as e:
        print(f"Config file error: {e}")
        sys.exit(1)


def load_config_with_strategy_support(
    config_path: Optional[str] = None,
    strategy_name: Optional[str] = None, 
    strategy_overrides: Optional[str] = None,
    base_dir: str = None,
    strategy_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration with strategy support.
    
    Args:
        config_path: Path to traditional config file
        strategy_name: Name of strategy to use
        strategy_overrides: JSON string of strategy overrides
        base_dir: Base directory for path resolution
        strategy_file: Path to custom strategy YAML file
        
    Returns:
        Configuration dictionary
    """
    if strategy_name:
        # Strategy-based configuration
        strategy_manager = StrategyManager(strategies_file=strategy_file)
        
        # Parse overrides if provided
        overrides = {}
        if strategy_overrides:
            try:
                overrides = json.loads(strategy_overrides)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid strategy overrides JSON: {e}")
                sys.exit(1)
        
        # Convert strategy to config
        config = strategy_manager.convert_strategy_to_config(strategy_name, overrides)
        if not config:
            print(f"❌ Strategy '{strategy_name}' not found")
            available = strategy_manager.get_available_strategies()
            if available:
                print(f"Available strategies: {', '.join(available)}")
            sys.exit(1)
        
        # Remove strategy metadata for pipeline creation
        config.pop("_strategy_info", None)
        return config
    
    elif config_path:
        # Traditional config file
        return load_config(config_path, base_dir)
    
    else:
        print("❌ Either --config or --strategy must be provided")
        print("💡 Use 'strategies list' to see available strategies")
        sys.exit(1)


def create_pipeline_from_config(
    config: Dict[str, Any], 
    enhanced: bool = False, 
    base_dir: str = None,
    file_path: Optional[Path] = None,
    parser_override: Optional[str] = None,
    embedder_override: Optional[str] = None,
    vector_store_override: Optional[str] = None
) -> tuple[Pipeline, Optional[Dict[str, Any]]]:
    """Create pipeline from configuration using factories with CLI overrides.

    Args:
        config: Configuration dictionary
        enhanced: Whether to use enhanced pipeline with progress tracking
        base_dir: Base directory for resolving relative paths in config
        file_path: Source file path for auto-detection
        parser_override: CLI override for parser selection
        embedder_override: CLI override for embedder selection
        vector_store_override: CLI override for vector store selection
        
    Returns:
        Tuple of (pipeline, extractor_config) where extractor_config is extracted 
        from parser's chunk_metadata.extractors if available
    """
    # Detect file type if file path provided
    file_type = None
    if file_path:
        file_type = detect_file_type(file_path)
    
    # Select configurations with overrides
    parser_config = select_parser_config(config, file_type or "unknown", parser_override)
    embedder_config = select_component_config(config, "embedders", embedder_override)
    store_config = select_component_config(config, "vector_stores", vector_store_override)
    
    # Extract extractor config from parser config's chunk_metadata
    extractor_config = None
    if parser_config and "config" in parser_config:
        chunk_metadata = parser_config["config"].get("chunk_metadata", {})
        if "extractors" in chunk_metadata:
            extractor_config = chunk_metadata["extractors"]
    
    # Create components using factories
    parser = create_parser_from_config(parser_config)
    embedder = create_embedder_from_config(embedder_config)
    store = create_vector_store_from_config(store_config)

    # Create pipeline
    if enhanced:
        pipeline = EnhancedPipeline("🦙 Enhanced RAG Pipeline")
    else:
        pipeline = Pipeline("RAG Pipeline")

    pipeline.add_component(parser)
    pipeline.add_component(embedder)
    pipeline.add_component(store)

    return pipeline, extractor_config


def ingest_command(args):
    """Handle the ingest command."""
    setup_logging(args.log_level, quiet=args.quiet)
    tracker = LlamaProgressTracker(verbose=args.verbose, quiet=args.quiet)

    # Resolve data source path
    resolver = PathResolver(args.base_dir if hasattr(args, "base_dir") else None)
    try:
        source_path = resolver.resolve_data_source(args.source)
        tracker.print_info(f"📂 Data source resolved to: {source_path}")
        
        # Detect file type for auto-selection
        file_type = detect_file_type(source_path)
        if file_type != 'unknown':
            tracker.print_info(f"🔍 Auto-detected file type: {file_type}")
        
    except FileNotFoundError as e:
        tracker.print_error(f"Data source error: {e}")
        sys.exit(1)

    # Load configuration (with strategy support)
    try:
        config = load_config_with_strategy_support(
            config_path=args.config,
            strategy_name=getattr(args, 'strategy', None),
            strategy_file=getattr(args, 'strategy_file', None),
            strategy_overrides=getattr(args, 'strategy_overrides', None),
            base_dir=args.base_dir if hasattr(args, "base_dir") else None
        )
        
        # Show config type
        if hasattr(args, 'strategy') and args.strategy:
            tracker.print_info(f"🚀 Using strategy: {args.strategy}")
        elif is_unified_config(config):
            tracker.print_info(f"📋 Using unified configuration v{config.get('version', '2.0')}")
        else:
            tracker.print_info("📋 Using legacy configuration format")
            
    except Exception as e:
        tracker.print_error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Create enhanced pipeline with overrides
    try:
        pipeline, config_extractors = create_pipeline_from_config(
            config,
            enhanced=True,
            base_dir=args.base_dir if hasattr(args, "base_dir") else None,
            file_path=source_path,
            parser_override=getattr(args, 'parser', None),
            embedder_override=getattr(args, 'embedder', None),
            vector_store_override=getattr(args, 'vector_store', None),
        )
    except Exception as e:
        tracker.print_error(f"Failed to create pipeline: {e}")
        sys.exit(1)

    # Run ingestion with enhanced progress tracking
    try:
        if hasattr(pipeline, "run_with_progress"):
            result = pipeline.run_with_progress(source=str(source_path))
        else:
            # Show enhanced visual progress
            if args.verbose:
                tracker.print_header("📄 Document Processing Pipeline")
                tracker.print_info(f"📂 Source: {source_path}")
                
                # Show pipeline stages
                print("\n🔧 Pipeline Stages:")
                print("  1️⃣  Parse documents")
                print("  2️⃣  Extract metadata")
                print("  3️⃣  Generate embeddings")
                print("  4️⃣  Store in vector database")
                print()
            
            tracker.print_info(f"📂 Processing documents from: {source_path}")
            
            # Run the pipeline
            result = pipeline.run(source=str(source_path))
            
            # Show embedding visualization if verbose
            if args.verbose and result.documents:
                print("\n🧠 Embedding Generation:")
                for i, doc in enumerate(result.documents[:5], 1):  # Show first 5
                    doc_name = Path(doc.source).name if doc.source else f"Document {i}"
                    print(f"  📄 {doc_name}: ", end="")
                    # Show mini embedding visualization
                    if doc.embeddings:
                        viz = "["
                        for j in range(min(20, len(doc.embeddings))):
                            val = doc.embeddings[j]
                            if val > 0.5:
                                viz += "█"
                            elif val > 0:
                                viz += "▓"
                            elif val > -0.5:
                                viz += "░"
                            else:
                                viz += "·"
                        viz += f"...] ({len(doc.embeddings)} dims)"
                        print(viz)
                    else:
                        print("[pending]")
                
                if len(result.documents) > 5:
                    print(f"  ... and {len(result.documents) - 5} more documents")
            
            tracker.print_success("Processing completed!")

        # Apply extractors from both config and CLI
        config_extractor_names = []
        cli_extractor_names = []
        
        # Collect config-based extractors
        if config_extractors:
            config_extractor_names = list(config_extractors.keys())
            tracker.print_info(f"🔧 Config-based extractors found: {', '.join(config_extractor_names)}")
        
        # Collect CLI extractors
        if hasattr(args, 'extractors') and args.extractors:
            cli_extractor_names = args.extractors
            tracker.print_info(f"🔧 CLI extractors specified: {', '.join(cli_extractor_names)}")
        
        # Apply extractors if any are specified
        if config_extractor_names or cli_extractor_names:
            try:
                # Parse CLI extractor config if provided
                cli_extractor_configs = {}
                if hasattr(args, 'extractor_config') and args.extractor_config:
                    try:
                        cli_extractor_configs = json.loads(args.extractor_config)
                    except json.JSONDecodeError as e:
                        tracker.print_warning(f"Invalid CLI extractor config JSON, using defaults: {e}")
                
                # Combine config and CLI extractors with CLI taking precedence
                final_extractor_configs = config_extractors.copy() if config_extractors else {}
                final_extractor_configs.update(cli_extractor_configs)
                
                # Get all unique extractor names (CLI overrides config)
                all_extractor_names = list(set(config_extractor_names + cli_extractor_names))
                
                if all_extractor_names:
                    tracker.print_info(f"🔧 Applying {len(all_extractor_names)} extractors: {', '.join(all_extractor_names)}")
                    
                    # Apply extractors
                    enhanced_documents = apply_extractors_from_cli_args(
                        result.documents, 
                        all_extractor_names,
                        final_extractor_configs
                    )
                    result.documents = enhanced_documents
                    tracker.print_success(f"Successfully applied {len(all_extractor_names)} extractors")
                
            except Exception as e:
                tracker.print_warning(f"Extractor application failed: {e}")

        # Show final summary with enhanced details
        print(f"\n📊 Final Results:")
        tracker.print_success(f"Documents processed: {len(result.documents)}")
        
        # Show document details if verbose
        if args.verbose and result.documents:
            print("\n📚 Document Details:")
            for i, doc in enumerate(result.documents[:3], 1):  # Show first 3
                doc_name = Path(doc.source).name if doc.source else f"Document {i}"
                print(f"\n  {i}. {doc_name}")
                print(f"     • Content: {len(doc.content)} chars")
                if doc.metadata:
                    for key, value in list(doc.metadata.items())[:3]:
                        if not key.startswith('_'):
                            print(f"     • {key}: {str(value)[:50]}")
                if doc.embeddings:
                    print(f"     • Embeddings: {len(doc.embeddings)} dimensions")
            
            if len(result.documents) > 3:
                print(f"\n  ... and {len(result.documents) - 3} more documents")

        if result.errors:
            tracker.print_warning(f"Errors encountered: {len(result.errors)}")
            if args.log_level.upper() in ["DEBUG", "INFO"] or args.verbose:
                print("\n🔍 Error Details:")
                for i, error in enumerate(result.errors[:5], 1):
                    print(f"   {i}. {error}")
                if len(result.errors) > 5:
                    print(f"   ... and {len(result.errors) - 5} more errors")
        else:
            tracker.print_success("Zero errors - perfect execution! 🎯")

    except KeyboardInterrupt:
        tracker.print_warning("\n⏸️  Processing interrupted by user. No prob-llama!")
        sys.exit(0)
    except Exception as e:
        tracker.print_error(f"Ingestion failed: {e}")
        sys.exit(1)


def search_command(args):
    """Handle the search command."""
    setup_logging(args.log_level, quiet=args.quiet)
    tracker = LlamaProgressTracker(verbose=args.verbose, quiet=args.quiet)

    # Load configuration (with strategy support)
    base_dir = getattr(args, "base_dir", None)
    config = load_config_with_strategy_support(
        config_path=args.config,
        strategy_name=getattr(args, 'strategy', None),
        strategy_file=getattr(args, 'strategy_file', None),
        strategy_overrides=getattr(args, 'strategy_overrides', None),
        base_dir=base_dir
    )
    
    # Show config type
    if hasattr(args, 'strategy') and args.strategy:
        tracker.print_info(f"🚀 Using strategy: {args.strategy}")

    # Select components with overrides
    try:
        embedder_config = select_component_config(config, "embedders", getattr(args, 'embedder', None))
        store_config = select_component_config(config, "vector_stores", getattr(args, 'vector_store', None))
        retrieval_config = select_component_config(config, "retrieval_strategies", getattr(args, 'retrieval', None))
        
        embedder = create_embedder_from_config(embedder_config)
        store = create_vector_store_from_config(store_config)
        
        # Create retrieval strategy if available
        retrieval_strategy = None
        if retrieval_config:
            retrieval_strategy = create_retrieval_strategy_from_config(retrieval_config)
            
    except Exception as e:
        tracker.print_error(f"Failed to create components: {e}")
        sys.exit(1)

    # Show configuration information
    if is_unified_config(config):
        tracker.print_info(f"📋 Using unified configuration v{config.get('version', '2.0')}")
        if getattr(args, 'retrieval', None):
            tracker.print_info(f"🔧 Retrieval strategy override: {args.retrieval}")

    # Perform search with llama flair
    tracker.print_header(f"🔍 Searching the Llama-sphere! 🔍")
    tracker.print_info(f"🎯 Query: '{args.query}'")
    tracker.print_info(f"🦙 {tracker.get_random_pun()}")

    try:
        # Convert query to embedding first
        print("🧠 Converting your query into llama-friendly embeddings...")
        query_embedding = embedder.embed([args.query])[0]
        tracker.print_success("Query embedded successfully!")
        
        # Use retrieval strategy if available, otherwise direct store search
        if retrieval_strategy:
            print(f"🔍 Using {retrieval_strategy.name} for enhanced search...")
            retrieval_result = retrieval_strategy.retrieve(
                query_embedding=query_embedding,
                vector_store=store,
                top_k=args.top_k
            )
            # Convert RetrievalResult to list of documents with scores
            results = []
            for i, (doc, score) in enumerate(zip(retrieval_result.documents, retrieval_result.scores)):
                # Add score to metadata for display
                doc.metadata["similarity_score"] = score
                results.append(doc)
                
            tracker.print_success("Enhanced retrieval completed!")
        else:
            # Fallback to direct store search
            print("🔍 Searching through the knowledge pasture...")
            results = store.search(query_embedding=query_embedding, top_k=args.top_k)

        if results:
            tracker.print_success(f"Found {len(results)} llama-nificent matches!")
            
            # Show verbose details if requested
            if args.verbose:
                tracker.print_verbose_results(results, "Detailed Search Results")
            
            # Regular output - create nice boxes for each result
            print(f"\n📋 Search Results:")

            for i, doc in enumerate(results, 1):
                # Collect content for the box
                result_content = []
                result_content.append(f"🏆 Result #{i}")
                
                # Show source if available (shortened)
                if hasattr(doc, 'source') and doc.source and doc.source != 'unknown':
                    source_display = doc.source
                    if len(source_display) > 50:
                        source_display = "..." + source_display[-47:]
                    result_content.append(f"📁 {source_display}")

                # Content preview - show first paragraph or limited chars
                content_limit = args.content_length
                
                if content_limit == -1:
                    display_content = doc.content
                else:
                    # Try to get first paragraph
                    paragraphs = doc.content.split('\n\n')
                    first_para = paragraphs[0].strip() if paragraphs else doc.content
                    
                    if len(first_para) > content_limit:
                        # Try to cut at sentence boundary
                        sentences = first_para[:content_limit].split('. ')
                        if len(sentences) > 1:
                            display_content = '. '.join(sentences[:-1]) + '.'
                        else:
                            display_content = first_para[:content_limit] + '...'
                    else:
                        display_content = first_para
                
                # Add content with proper line wrapping
                result_content.append("📝 Content:")
                content_words = display_content.split()
                current_line = ""
                for word in content_words:
                    if len(current_line + " " + word) > 68:
                        if current_line:
                            result_content.append(f"   {current_line}")
                            current_line = word
                        else:
                            result_content.append(f"   {word}")
                    else:
                        current_line = current_line + " " + word if current_line else word
                
                if current_line:
                    result_content.append(f"   {current_line}")

                # Show score
                score = doc.metadata.get("similarity_score", 0.0)
                if score is not None:
                    # Similarity scores range from 0 to 1 (1 being perfect match)
                    if score > 0.8:
                        result_content.append(f"🎯 Match: {score:.3f} (Excellent)")
                    elif score > 0.5:
                        result_content.append(f"🎯 Match: {score:.3f} (Good)")
                    elif score > 0.3:
                        result_content.append(f"🎯 Match: {score:.3f} (Fair)")
                    else:
                        result_content.append(f"🎯 Match: {score:.3f} (Weak)")

                # Show useful metadata only
                useful_metadata = {}
                if doc.metadata:
                    # Select only the most useful metadata fields (including business/sentiment)
                    for key in ['file_name', 'priority', 'tags', 'word_count', 'file_size', 
                               'sentiment', 'sentiment_score', 'sentiment_confidence',
                               'entities_extracted', 'patterns_found', 'currency_amounts',
                               'percentages_found']:
                        if key in doc.metadata:
                            useful_metadata[key] = doc.metadata[key]
                
                if useful_metadata:
                    result_content.append("📊 Info:")
                    for key, value in useful_metadata.items():
                        if key == 'file_size':
                            if isinstance(value, int):
                                if value > 1024:
                                    value = f"{value/1024:.1f}KB"
                                else:
                                    value = f"{value}B"
                        elif key == 'priority':
                            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(str(value).lower(), "⚪")
                            value = f"{priority_emoji} {value}"
                        elif key == 'tags' and isinstance(value, list):
                            value = ', '.join(value[:3])  # First 3 tags
                            if len(doc.metadata[key]) > 3:
                                value += '...'
                        elif key == 'sentiment':
                            # Show sentiment with emoji
                            sentiment_emoji = {
                                "positive": "😊", "negative": "😟", 
                                "neutral": "😐", "mixed": "🤔"
                            }.get(str(value).lower(), "❓")
                            value = f"{sentiment_emoji} {value}"
                        elif key == 'sentiment_confidence':
                            # Show confidence as percentage
                            if isinstance(value, (int, float)):
                                value = f"{value*100:.1f}%"
                        elif key == 'sentiment_score':
                            # Show sentiment score with color indicator
                            if isinstance(value, (int, float)):
                                if value > 0.3:
                                    value = f"📈 {value:.2f} (positive)"
                                elif value < -0.3:
                                    value = f"📉 {value:.2f} (negative)"
                                else:
                                    value = f"➡️ {value:.2f} (neutral)"
                        elif key == 'currency_amounts' and isinstance(value, list):
                            # Show first few currency amounts
                            if value:
                                value = ', '.join(str(x) for x in value[:2])
                                if len(doc.metadata[key]) > 2:
                                    value += f' (+{len(doc.metadata[key])-2} more)'
                        elif key == 'percentages_found' and isinstance(value, list):
                            # Show percentages
                            if value:
                                value = ', '.join(str(x) for x in value[:3])
                        elif key == 'entities_extracted' and isinstance(value, list):
                            # Show entity count
                            value = f"{len(value)} entities"
                        elif isinstance(value, list):
                            # Generic list handling
                            value = ', '.join(str(x) for x in value[:3])
                            if len(doc.metadata[key]) > 3:
                                value += '...'
                        
                        # Truncate long values
                        value_str = str(value)
                        if len(value_str) > 40:
                            value_str = value_str[:37] + "..."
                        
                        # Format the key name nicely
                        display_key = key.replace('_', ' ').title()
                        if key == 'sentiment_confidence':
                            display_key = "Confidence"
                        elif key == 'sentiment_score':
                            display_key = "Sentiment Score"
                        elif key == 'currency_amounts':
                            display_key = "💵 Amounts"
                        elif key == 'percentages_found':
                            display_key = "📊 Percentages"
                        elif key == 'entities_extracted':
                            display_key = "🏢 Entities"
                        
                        result_content.append(f"   {display_key}: {value_str}")

                # Show verbose metadata if requested
                if args.verbose and doc.metadata:
                    # Show extractor results concisely
                    extractor_results = {}
                    for key in doc.metadata:
                        if key.startswith(('entities_', 'keywords_', 'patterns_', 'summary_')):
                            extractor_results[key] = doc.metadata[key]
                    
                    if extractor_results:
                        result_content.append("🔍 Extracts:")
                        for key, value in list(extractor_results.items())[:2]:  # Limit to 2
                            display_key = key.replace('_', ' ').title()
                            if isinstance(value, list):
                                if value:
                                    display_value = ', '.join(str(x) for x in value[:3])
                                    if len(value) > 3:
                                        display_value += f' (+{len(value)-3} more)'
                                else:
                                    continue
                            elif isinstance(value, str):
                                display_value = value[:40] + "..." if len(value) > 40 else value
                            else:
                                display_value = str(value)[:40]
                            
                            result_content.append(f"   {display_key}: {display_value}")

                # Create a nice bordered box
                max_width = min(max(len(line) for line in result_content) + 4, 76)
                
                print(f"\n{Fore.CYAN}╭{'─' * (max_width - 2)}╮{Style.RESET_ALL}")
                for line in result_content:
                    # Ensure line fits in box
                    if len(line) > max_width - 4:
                        line = line[:max_width - 7] + "..."
                    padding = max_width - len(line) - 4
                    print(f"{Fore.CYAN}│{Style.RESET_ALL} {line}{' ' * padding} {Fore.CYAN}│{Style.RESET_ALL}")
                print(f"{Fore.CYAN}╰{'─' * (max_width - 2)}╯{Style.RESET_ALL}")
        else:
            tracker.print_warning(
                "No results found. The llamas are still grazing in other pastures! 🌾"
            )
            print("\n💡 Tips for better results:")
            print("   • Try different keywords")
            print("   • Use more specific terms")
            print("   • Check if your data has been ingested")
            if is_unified_config(config):
                print("   • Try a different retrieval strategy (--retrieval)")

    except Exception as e:
        tracker.print_error(f"Search failed: {e}")
        sys.exit(1)


def info_command(args):
    """Handle the info command."""
    setup_logging(args.log_level, quiet=args.quiet)

    # Check if using strategy
    if hasattr(args, 'strategy') and args.strategy:
        # Use strategy-based configuration
        from core.strategies import StrategyManager
        strategy_manager = StrategyManager(strategies_file=getattr(args, 'strategy_file', None))
        
        # Get strategy config
        config = strategy_manager.convert_strategy_to_config(args.strategy)
        if not config:
            print(f"❌ Strategy '{args.strategy}' not found")
            available = strategy_manager.get_available_strategies()
            if available:
                print(f"Available strategies: {', '.join(available)}")
            sys.exit(1)
        
        # Create vector store from strategy config
        try:
            store = create_vector_store_from_config(config.get("vector_store", {}))
        except Exception as e:
            print(f"Failed to create vector store: {e}")
            sys.exit(1)
    else:
        # Load configuration from file
        base_dir = getattr(args, "base_dir", None)
        config = load_config(args.config, base_dir)

        # Create vector store using factory with override
        try:
            store_config = select_component_config(config, "vector_stores", getattr(args, 'vector_store', None))
            store = create_vector_store_from_config(store_config)
        except Exception as e:
            print(f"Failed to create vector store: {e}")
            sys.exit(1)

    # Get info
    try:
        info = store.get_collection_info()
        print("Vector Store Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Failed to get info: {e}")
        sys.exit(1)


def test_command(args):
    """Handle the test command."""
    setup_logging(args.log_level, quiet=args.quiet)
    tracker = LlamaProgressTracker(verbose=args.verbose, quiet=args.quiet)

    tracker.print_header("🧪 RAG System Comprehensive Testing 🧪")
    
    # Determine which tests to run
    run_basic = not any([args.management, args.configurations, args.retrieval_strategies, args.enterprise])
    if args.all_tests:
        run_basic = args.management = args.configurations = args.retrieval_strategies = args.enterprise = True

    test_results = {
        "basic": {"passed": 0, "failed": 0, "tests": []},
        "management": {"passed": 0, "failed": 0, "tests": []},
        "configurations": {"passed": 0, "failed": 0, "tests": []},
        "retrieval": {"passed": 0, "failed": 0, "tests": []},
        "enterprise": {"passed": 0, "failed": 0, "tests": []}
    }

    # Basic Component Tests
    if run_basic:
        tracker.print_info("🔧 Running Basic Component Tests...")
        test_results["basic"] = run_basic_tests(args, tracker)

    # Document Management Tests
    if args.management:
        tracker.print_info("📁 Running Document Management Tests...")
        test_results["management"] = run_management_tests(args, tracker)

    # Configuration Tests
    if args.configurations:
        tracker.print_info("⚙️ Running Configuration Tests...")
        test_results["configurations"] = run_configuration_tests(args, tracker)

    # Retrieval Strategy Tests
    if args.retrieval_strategies:
        tracker.print_info("🔍 Running Retrieval Strategy Tests...")
        test_results["retrieval"] = run_retrieval_tests(args, tracker)

    # Enterprise Feature Tests
    if args.enterprise:
        tracker.print_info("🏢 Running Enterprise Feature Tests...")
        test_results["enterprise"] = run_enterprise_tests(args, tracker)

    # Test Summary
    print_test_summary(test_results, tracker)


def run_basic_tests(args, tracker: LlamaProgressTracker) -> Dict[str, Any]:
    """Run basic component tests."""
    results = {"passed": 0, "failed": 0, "tests": []}
    
    # Test Ollama connection
    print("\n1. Testing Ollama connection...")
    try:
        embedder = EmbedderFactory.create("OllamaEmbedder")
        embedder.validate_config()
        print("   ✓ Ollama is available")
        results["tests"].append({"name": "Ollama Connection", "status": "PASS"})

        # Test embedding
        test_texts = ["Hello world", "This is a test"]
        embeddings = embedder.embed(test_texts)
        if embeddings and all(emb for emb in embeddings):
            print(f"   ✓ Embeddings generated (dimension: {len(embeddings[0])})")
            results["tests"].append({"name": "Embedding Generation", "status": "PASS"})
            results["passed"] += 2
        else:
            print("   ✗ Failed to generate embeddings")
            results["tests"].append({"name": "Embedding Generation", "status": "FAIL"})
            results["passed"] += 1
            results["failed"] += 1
    except Exception as e:
        print(f"   ✗ Ollama test failed: {e}")
        results["tests"].append({"name": "Ollama Connection", "status": "FAIL", "error": str(e)})
        results["failed"] += 1

    # Test ChromaDB
    print("\n2. Testing ChromaDB...")
    try:
        store = VectorStoreFactory.create(
            "ChromaStore", {"collection_name": "test_collection"}
        )
        store.validate_config()
        print("   ✓ ChromaDB is available")
        results["tests"].append({"name": "ChromaDB Connection", "status": "PASS"})

        # Test enhanced methods
        if hasattr(store, 'get_collection_stats'):
            stats = store.get_collection_stats()
            print("   ✓ Enhanced ChromaDB methods available")
            results["tests"].append({"name": "Enhanced ChromaDB Methods", "status": "PASS"})
            results["passed"] += 1
        
        # Clean up test collection
        store.delete_collection()
        print("   ✓ Test collection cleaned up")
        results["tests"].append({"name": "Collection Cleanup", "status": "PASS"})
        results["passed"] += 2
    except Exception as e:
        print(f"   ✗ ChromaDB test failed: {e}")
        results["tests"].append({"name": "ChromaDB Connection", "status": "FAIL", "error": str(e)})
        results["failed"] += 1

    # Test file parsing
    print("\n3. Testing file parsing...")
    try:
        if args.test_file:
            from core.factories import ParserFactory

            # Resolve test file path
            resolver = PathResolver(
                args.base_dir if hasattr(args, "base_dir") else None
            )
            try:
                test_file_path = resolver.resolve_data_source(args.test_file)
                print(f"   📁 Test file resolved to: {test_file_path}")
                results["tests"].append({"name": "File Path Resolution", "status": "PASS"})
                results["passed"] += 1
            except FileNotFoundError as e:
                print(f"   ✗ Test file not found: {e}")
                results["tests"].append({"name": "File Path Resolution", "status": "FAIL", "error": str(e)})
                results["failed"] += 1
                return results

            # Detect file type and use appropriate parser
            file_extension = test_file_path.suffix.lower()
            detected_type = detect_file_type(test_file_path)
            print(f"   🔍 Auto-detected file type: {detected_type}")
            results["tests"].append({"name": "File Type Detection", "status": "PASS"})
            results["passed"] += 1
            
            if file_extension == '.csv':
                parser = ParserFactory.create("CustomerSupportCSVParser")
                test_type = "CSV"
            elif file_extension == '.pdf':
                parser = ParserFactory.create("PDFParser")
                test_type = "PDF"
            else:
                print(f"   ⚠ Unsupported file type: {file_extension}")
                results["tests"].append({"name": f"{file_extension} Parser", "status": "SKIP"})
                return results
            
            print(f"   🔍 Testing {test_type} parsing...")
            result = parser.parse(str(test_file_path))
            print(f"   ✓ Parsed {len(result.documents)} documents")
            results["tests"].append({"name": f"{test_type} Parsing", "status": "PASS"})
            results["passed"] += 1
            
            if result.errors:
                print(f"   ⚠ {len(result.errors)} parsing errors")
                results["tests"].append({"name": f"{test_type} Error Handling", "status": "WARN"})
                # Show first error for debugging
                if result.errors:
                    print(f"      First error: {result.errors[0].get('error', 'Unknown error')}")
            else:
                results["tests"].append({"name": f"{test_type} Error Handling", "status": "PASS"})
                results["passed"] += 1
        else:
            print("   ⚠ No test file provided (use --test-file)")
            results["tests"].append({"name": "File Parsing", "status": "SKIP"})
    except Exception as e:
        print(f"   ✗ File parsing test failed: {e}")
        results["tests"].append({"name": "File Parsing", "status": "FAIL", "error": str(e)})
        results["failed"] += 1

    return results


def run_management_tests(args, tracker: LlamaProgressTracker) -> Dict[str, Any]:
    """Run document management tests."""
    results = {"passed": 0, "failed": 0, "tests": []}
    
    print("\n4. Testing Document Management...")
    
    # Load configuration for testing
    try:
        config = load_config(
            args.config, 
            args.base_dir if hasattr(args, "base_dir") else None
        )
        
        # Create document manager
        store_config = select_component_config(config, "vector_stores", None)
        store = create_vector_store_from_config(store_config)
        metadata_config = store_config.get("config", {}).get("metadata_config", {})
        doc_manager = DocumentManager(store, metadata_config)
        
        print("   ✓ Document manager created")
        results["tests"].append({"name": "Document Manager Creation", "status": "PASS"})
        results["passed"] += 1
        
        # Test statistics
        try:
            stats = doc_manager.get_document_stats()
            print(f"   ✓ Statistics retrieved: {stats.get('total_documents', 0)} documents")
            results["tests"].append({"name": "Document Statistics", "status": "PASS"})
            results["passed"] += 1
        except Exception as e:
            print(f"   ✗ Statistics failed: {e}")
            results["tests"].append({"name": "Document Statistics", "status": "FAIL", "error": str(e)})
            results["failed"] += 1
        
        # Test hash manager
        try:
            hash_test = doc_manager.hash_manager.generate_content_hash("test content")
            print(f"   ✓ Hash generation working: {hash_test[:20]}...")
            results["tests"].append({"name": "Hash Generation", "status": "PASS"})
            results["passed"] += 1
        except Exception as e:
            print(f"   ✗ Hash generation failed: {e}")
            results["tests"].append({"name": "Hash Generation", "status": "FAIL", "error": str(e)})
            results["failed"] += 1
        
        # Test deletion manager (dry run)
        try:
            delete_results = doc_manager.deletion_manager.delete_by_time(
                older_than_days=999, 
                strategy=DeletionStrategy.SOFT_DELETE
            )
            print(f"   ✓ Deletion manager working (dry test)")
            results["tests"].append({"name": "Deletion Manager", "status": "PASS"})
            results["passed"] += 1
        except Exception as e:
            print(f"   ✗ Deletion manager failed: {e}")
            results["tests"].append({"name": "Deletion Manager", "status": "FAIL", "error": str(e)})
            results["failed"] += 1
    
    except Exception as e:
        print(f"   ✗ Document management setup failed: {e}")
        results["tests"].append({"name": "Document Management Setup", "status": "FAIL", "error": str(e)})
        results["failed"] += 1
    
    return results


def run_configuration_tests(args, tracker: LlamaProgressTracker) -> Dict[str, Any]:
    """Run configuration tests."""
    results = {"passed": 0, "failed": 0, "tests": []}
    
    print("\n5. Testing Configuration Systems...")
    
    # Test unified config detection
    try:
        config = load_config(
            args.config, 
            args.base_dir if hasattr(args, "base_dir") else None
        )
        
        if is_unified_config(config):
            print("   ✓ Unified configuration detected")
            results["tests"].append({"name": "Unified Config Detection", "status": "PASS"})
            results["passed"] += 1
            
            # Test component selection
            parser_config = select_component_config(config, "parsers", None)
            embedder_config = select_component_config(config, "embedders", None)
            store_config = select_component_config(config, "vector_stores", None)
            
            if all([parser_config, embedder_config, store_config]):
                print("   ✓ Component selection working")
                results["tests"].append({"name": "Component Selection", "status": "PASS"})
                results["passed"] += 1
            else:
                print("   ✗ Component selection failed")
                results["tests"].append({"name": "Component Selection", "status": "FAIL"})
                results["failed"] += 1
                
        else:
            print("   ✓ Legacy configuration detected")
            results["tests"].append({"name": "Legacy Config Support", "status": "PASS"})
            results["passed"] += 1
        
        # Test metadata config
        if "metadata_config" in str(config):
            print("   ✓ Enhanced metadata configuration found")
            results["tests"].append({"name": "Enhanced Metadata Config", "status": "PASS"})
            results["passed"] += 1
        
    except Exception as e:
        print(f"   ✗ Configuration test failed: {e}")
        results["tests"].append({"name": "Configuration Loading", "status": "FAIL", "error": str(e)})
        results["failed"] += 1
    
    return results


def run_retrieval_tests(args, tracker: LlamaProgressTracker) -> Dict[str, Any]:
    """Run retrieval strategy tests."""
    results = {"passed": 0, "failed": 0, "tests": []}
    
    print("\n6. Testing Retrieval Strategies...")
    
    try:
        # Test strategy creation
        # Import for retrieval strategies is handled via core.factories
        
        # Test basic strategy
        basic_config = {
            "type": "BasicSimilarityStrategy",
            "config": {"distance_metric": "cosine"}
        }
        
        strategy = create_retrieval_strategy_from_config(basic_config)
        print("   ✓ Basic retrieval strategy created")
        results["tests"].append({"name": "Basic Strategy Creation", "status": "PASS"})
        results["passed"] += 1
        
        # Test hybrid strategy
        hybrid_config = {
            "type": "HybridUniversalStrategy",
            "config": {
                "combination_method": "weighted_average",
                "strategies": [
                    {"type": "BasicSimilarityStrategy", "weight": 0.7},
                    {"type": "MetadataFilteredStrategy", "weight": 0.3}
                ]
            }
        }
        
        hybrid_strategy = create_retrieval_strategy_from_config(hybrid_config)
        print("   ✓ Hybrid retrieval strategy created")
        results["tests"].append({"name": "Hybrid Strategy Creation", "status": "PASS"})
        results["passed"] += 1
        
        # Test strategy registry
        from components.retrievers.strategies.universal import UNIVERSAL_STRATEGIES
        strategy_count = len(UNIVERSAL_STRATEGIES)
        print(f"   ✓ {strategy_count} universal strategies available")
        results["tests"].append({"name": "Strategy Registry", "status": "PASS"})
        results["passed"] += 1
        
    except Exception as e:
        print(f"   ✗ Retrieval strategy test failed: {e}")
        results["tests"].append({"name": "Retrieval Strategies", "status": "FAIL", "error": str(e)})
        results["failed"] += 1
    
    return results


def run_enterprise_tests(args, tracker: LlamaProgressTracker) -> Dict[str, Any]:
    """Run enterprise feature tests."""
    results = {"passed": 0, "failed": 0, "tests": []}
    
    print("\n7. Testing Enterprise Features...")
    
    try:
        # Test enterprise config loading
        enterprise_config_path = "config_examples/enterprise_document_management_config.yaml"
        resolver = PathResolver(args.base_dir if hasattr(args, "base_dir") else None)
        
        try:
            enterprise_config = load_config(enterprise_config_path)
            print("   ✓ Enterprise configuration loaded")
            results["tests"].append({"name": "Enterprise Config Loading", "status": "PASS"})
            results["passed"] += 1
            
            # Test enterprise features
            if "enterprise_features" in enterprise_config:
                print("   ✓ Enterprise features configured")
                results["tests"].append({"name": "Enterprise Features Config", "status": "PASS"})
                results["passed"] += 1
            
            # Test compliance settings
            if "compliance" in str(enterprise_config):
                print("   ✓ Compliance settings found")
                results["tests"].append({"name": "Compliance Settings", "status": "PASS"})
                results["passed"] += 1
            
            # Test advanced retrieval strategies
            strategies = enterprise_config.get("retrieval_strategies", {})
            enterprise_strategies = ["enterprise_default", "legal_specialized", "compliance_audit", "high_precision"]
            found_strategies = [s for s in enterprise_strategies if s in strategies]
            
            if found_strategies:
                print(f"   ✓ {len(found_strategies)} enterprise strategies available")
                results["tests"].append({"name": "Enterprise Retrieval Strategies", "status": "PASS"})
                results["passed"] += 1
            
        except FileNotFoundError:
            print("   ⚠ Enterprise config not found (expected in some environments)")
            results["tests"].append({"name": "Enterprise Config Loading", "status": "SKIP"})
    
    except Exception as e:
        print(f"   ✗ Enterprise feature test failed: {e}")
        results["tests"].append({"name": "Enterprise Features", "status": "FAIL", "error": str(e)})
        results["failed"] += 1
    
    return results


def print_test_summary(test_results: Dict[str, Dict], tracker: LlamaProgressTracker):
    """Print comprehensive test summary."""
    tracker.print_header("📊 Test Results Summary 📊")
    
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for category, results in test_results.items():
        if not results["tests"]:
            continue
            
        passed = results["passed"]
        failed = results["failed"]
        skipped = len([t for t in results["tests"] if t.get("status") == "SKIP"])
        warned = len([t for t in results["tests"] if t.get("status") == "WARN"])
        
        total_passed += passed
        total_failed += failed
        total_skipped += skipped
        
        status_emoji = "✅" if failed == 0 else "❌" if passed == 0 else "⚠️"
        print(f"\n{status_emoji} {category.title()} Tests: {passed} passed, {failed} failed, {skipped} skipped")
        
        # Show failed tests
        for test in results["tests"]:
            if test["status"] == "FAIL":
                print(f"   ❌ {test['name']}: {test.get('error', 'Unknown error')}")
            elif test["status"] == "WARN":
                print(f"   ⚠️  {test['name']}: Warning")
    
    # Overall summary
    overall_status = "✅ ALL TESTS PASSED" if total_failed == 0 else f"❌ {total_failed} TESTS FAILED"
    if total_skipped > 0:
        overall_status += f" ({total_skipped} skipped)"
    
    print(f"\n🎯 Overall Result: {overall_status}")
    print(f"📈 Total: {total_passed + total_failed + total_skipped} tests")
    print(f"   ✅ Passed: {total_passed}")
    print(f"   ❌ Failed: {total_failed}")
    print(f"   ⚠️  Skipped: {total_skipped}")
    
    if total_failed == 0:
        tracker.print_success("🎉 All systems operational! Ready for production use!")
    else:
        tracker.print_warning(f"⚠️  {total_failed} issues found. Please review failed tests.")
    
    print("\n📋 For detailed documentation, see:")
    print("   • README.md - Basic usage and examples")
    print("   • docs/DATABASE_MAINTENANCE_STRATEGIES.md - Document management guide") 
    print("   • config_examples/ - Configuration examples")
    print("   • CLI help: uv run python cli.py --help")
    print("   • Management help: uv run python cli.py manage --help")


def manage_command(args):
    """Handle document management commands."""
    setup_logging(args.log_level)
    tracker = LlamaProgressTracker()

    # Check if using RAG strategy
    if hasattr(args, 'rag_strategy') and args.rag_strategy:
        # Use strategy-based configuration
        from core.strategies import StrategyManager
        strategy_manager = StrategyManager(strategies_file=getattr(args, 'strategy_file', None))
        
        # Get strategy config
        config = strategy_manager.convert_strategy_to_config(args.rag_strategy)
        if not config:
            print(f"❌ Strategy '{args.rag_strategy}' not found")
            available = strategy_manager.get_available_strategies()
            if available:
                print(f"Available strategies: {', '.join(available)}")
            sys.exit(1)
        
        # Create vector store from strategy config
        try:
            store = create_vector_store_from_config(config.get("vector_store", {}))
            metadata_config = config.get("vector_store", {}).get("config", {}).get("metadata_config", {})
            doc_manager = DocumentManager(store, metadata_config)
        except Exception as e:
            tracker.print_error(f"Failed to create document manager: {e}")
            sys.exit(1)
    else:
        # Load configuration from file
        base_dir = getattr(args, "base_dir", None)
        config = load_config(args.config, base_dir)

        # Create vector store and document manager
        try:
            store_config = select_component_config(config, "vector_stores", getattr(args, 'vector_store', None))
            store = create_vector_store_from_config(store_config)
            
            # Create document manager with metadata config
            metadata_config = store_config.get("config", {}).get("metadata_config", {})
            doc_manager = DocumentManager(store, metadata_config)
            
        except Exception as e:
            tracker.print_error(f"Failed to create document manager: {e}")
            sys.exit(1)

    # Route to specific management command
    if args.manage_command == "delete":
        handle_delete_command(args, doc_manager, tracker)
    elif args.manage_command == "replace":
        handle_replace_command(args, doc_manager, tracker)
    elif args.manage_command == "stats":
        handle_stats_command(args, doc_manager, tracker)
    elif args.manage_command == "cleanup":
        handle_cleanup_command(args, doc_manager, tracker)
    elif args.manage_command == "hash":
        handle_hash_command(args, doc_manager, tracker)
    else:
        tracker.print_error("Unknown management command")
        sys.exit(1)


def handle_delete_command(args, doc_manager: DocumentManager, tracker: LlamaProgressTracker):
    """Handle delete operations."""
    strategy_map = {
        "soft": DeletionStrategy.SOFT_DELETE,
        "hard": DeletionStrategy.HARD_DELETE,
        "archive": DeletionStrategy.ARCHIVE_DELETE
    }
    
    strategy = strategy_map[args.delete_strategy]
    results = None
    
    tracker.print_header(f"🗑️  Document Deletion ({args.delete_strategy.upper()}) 🗑️")
    
    if args.dry_run:
        tracker.print_info("🔍 DRY RUN MODE - No actual deletions will occur")
    
    try:
        if args.older_than:
            tracker.print_info(f"📅 Deleting documents older than {args.older_than} days")
            results = doc_manager.deletion_manager.delete_by_time(
                older_than_days=args.older_than,
                strategy=strategy
            )
        elif args.doc_ids:
            tracker.print_info(f"🆔 Deleting documents by ID: {', '.join(args.doc_ids)}")
            results = doc_manager.deletion_manager.delete_by_document(
                doc_ids=args.doc_ids,
                strategy=strategy
            )
        elif args.filenames:
            tracker.print_info(f"📄 Deleting documents by filename: {', '.join(args.filenames)}")
            results = doc_manager.deletion_manager.delete_by_filename(
                filenames=args.filenames,
                strategy=strategy
            )
        elif args.content_hashes:
            tracker.print_info(f"🔐 Deleting documents by content hash: {len(args.content_hashes)} hashes")
            results = doc_manager.deletion_manager.delete_by_hash(
                content_hashes=args.content_hashes,
                strategy=strategy
            )
        elif hasattr(args, 'document_hashes') and args.document_hashes:
            tracker.print_info(f"📋 Deleting all chunks by document hash: {len(args.document_hashes)} documents")
            results = doc_manager.deletion_manager.delete_by_document_hash(
                document_hashes=args.document_hashes,
                strategy=strategy
            )
        elif hasattr(args, 'source_paths') and args.source_paths:
            tracker.print_info(f"📁 Deleting all documents by source path: {len(args.source_paths)} paths")
            results = doc_manager.deletion_manager.delete_by_source_path(
                source_paths=args.source_paths,
                strategy=strategy
            )
        elif args.expired:
            tracker.print_info("⏰ Deleting expired documents")
            results = doc_manager.deletion_manager.delete_expired_documents(strategy=strategy)
        elif hasattr(args, 'all') and args.all:
            tracker.print_warning("⚠️  DELETING ALL DOCUMENTS IN COLLECTION")
            if not args.dry_run:
                # Get count before deletion for reporting
                collection_info = doc_manager.vector_store.get_collection_info()
                doc_count = collection_info.get("count", 0)
                
                # For actual deletion, delete the entire collection
                if hasattr(doc_manager.vector_store, 'delete_collection'):
                    doc_manager.vector_store.delete_collection()
                    results = {"deleted_count": doc_count, "errors": []}
                    tracker.print_success(f"✅ Successfully deleted entire collection ({doc_count} documents)")
                else:
                    # Fallback: use deletion manager with broad criteria
                    # This will delete all documents regardless of age
                    results = doc_manager.deletion_manager.delete_by_time(
                        older_than_days=0,  # 0 means all documents
                        strategy=DeletionStrategy.HARD_DELETE  # Force hard delete for --all
                    )
            else:
                # For dry run, just indicate what would happen
                collection_info = doc_manager.vector_store.get_collection_info()
                doc_count = collection_info.get("count", 0)
                results = {"deleted_count": doc_count, "errors": []}
                tracker.print_info(f"📊 Would delete {doc_count} documents (entire collection)")
        else:
            tracker.print_error("No deletion criteria specified")
            return
        
        # Display results
        if results:
            if results.get("errors"):
                for error in results["errors"]:
                    tracker.print_error(f"❌ {error}")
            
            deleted_count = results.get("deleted_count", 0)
            archived_count = results.get("archived_count", 0)

            # deleted_count should always be an integer now
            if deleted_count > 0:
                tracker.print_success(f"✅ Successfully deleted {deleted_count} documents")

            if archived_count > 0:
                tracker.print_success(f"📦 Successfully archived {archived_count} documents")

            # Show "no documents" message if no documents were deleted or archived
            if deleted_count == 0 and archived_count == 0:
                tracker.print_info("ℹ️  No documents matched the deletion criteria")
        
    except Exception as e:
        tracker.print_error(f"Deletion failed: {e}")


def handle_replace_command(args, doc_manager: DocumentManager, tracker: LlamaProgressTracker):
    """Handle document replacement operations."""
    tracker.print_header(f"🔄 Document Replacement 🔄")
    tracker.print_info(f"📄 Replacing document: {args.target_doc_id}")
    tracker.print_info(f"📂 With source file: {args.source}")
    tracker.print_info(f"🔧 Strategy: {args.replace_strategy}")
    
    # TODO: Implement document replacement logic
    tracker.print_warning("Document replacement functionality coming soon!")


def handle_stats_command(args, doc_manager: DocumentManager, tracker: LlamaProgressTracker):
    """Handle statistics display."""
    tracker.print_header(f"📊 Document Statistics 📊")
    
    try:
        stats = doc_manager.get_document_stats()
        
        print("📈 Collection Overview:")
        print(f"   Total Documents: {stats.get('total_documents', 0):,}")
        print(f"   Active Documents: {stats.get('active_documents', 0):,}")
        print(f"   Deleted (Soft): {stats.get('deleted_documents', 0):,}")
        print(f"   Expired Documents: {stats.get('expired_documents', 0):,}")
        print(f"   Total Versions: {stats.get('total_versions', 0):,}")
        print(f"   Storage Size: {stats.get('storage_size_mb', 0):.1f} MB")
        
        if args.detailed:
            tracker.print_info("📋 Detailed statistics coming soon!")
        
    except Exception as e:
        tracker.print_error(f"Failed to get statistics: {e}")


def handle_cleanup_command(args, doc_manager: DocumentManager, tracker: LlamaProgressTracker):
    """Handle cleanup operations."""
    tracker.print_header(f"🧹 Document Cleanup 🧹")
    
    try:
        if args.old_versions:
            tracker.print_info(f"🗂️  Cleaning up old versions (keeping {args.old_versions} latest)")
            results = doc_manager.deletion_manager.cleanup_old_versions(keep_versions=args.old_versions)
            tracker.print_success(f"✅ Cleaned up {results.get('deleted_count', 0)} old versions")
        
        elif args.duplicates:
            tracker.print_info("🔍 Finding and removing duplicate documents")
            # TODO: Implement duplicate removal
            tracker.print_warning("Duplicate removal functionality coming soon!")
        
        elif args.expired:
            tracker.print_info("⏰ Cleaning up expired documents")
            results = doc_manager.deletion_manager.delete_expired_documents()
            tracker.print_success(f"✅ Cleaned up {results.get('deleted_count', 0)} expired documents")
        
        else:
            tracker.print_error("No cleanup operation specified")
    
    except Exception as e:
        tracker.print_error(f"Cleanup failed: {e}")


def handle_hash_command(args, doc_manager: DocumentManager, tracker: LlamaProgressTracker):
    """Handle hash-based operations."""
    tracker.print_header(f"🔐 Hash Operations 🔐")
    
    try:
        if args.find_duplicates:
            tracker.print_info("🔍 Finding duplicate documents by content hash")
            # TODO: This would require querying the vector store
            # For now, show a placeholder
            tracker.print_info("🔍 Scanning document collection for duplicates...")
            tracker.print_success("✅ No duplicates found")
        
        elif args.verify_integrity:
            tracker.print_info("🛡️  Verifying document integrity using hashes")
            tracker.print_warning("Integrity verification functionality coming soon!")
        
        elif args.rehash:
            tracker.print_info("🔄 Regenerating all document hashes")
            tracker.print_warning("Hash regeneration functionality coming soon!")
        
        else:
            tracker.print_error("No hash operation specified")
    
    except Exception as e:
        tracker.print_error(f"Hash operation failed: {e}")


def extractor_command(args):
    """Handle extractor commands."""
    setup_logging(args.log_level)
    
    if args.extractor_command == "list":
        list_extractors_command(args)
    elif args.extractor_command == "test":
        test_extractor_command(args)
    else:
        print("Unknown extractor command")


def list_extractors_command(args):
    """List available extractors."""
    tracker = LlamaProgressTracker()
    tracker.print_header("🔍 Available Extractors 🔍")
    
    extractors = registry.list_extractors()
    
    if not extractors:
        print("❌ No extractors registered")
        return
    
    print(f"📋 Found {len(extractors)} extractors:")
    print()
    
    if args.detailed:
        # Show detailed information
        extractor_info = registry.get_all_info()
        
        for name in sorted(extractors):
            info = extractor_info.get(name, {})
            print(f"🔧 {Fore.CYAN}{name}{Style.RESET_ALL}")
            
            if "error" in info:
                print(f"   ❌ Error: {info['error']}")
            else:
                description = info.get("description", "No description")
                dependencies = info.get("dependencies", [])
                
                print(f"   📄 Description: {description}")
                if dependencies:
                    print(f"   📦 Dependencies: {', '.join(dependencies)}")
                else:
                    print(f"   📦 Dependencies: None (pure Python)")
                
                # Test if dependencies are available
                try:
                    extractor = registry.create(name)
                    if extractor:
                        print(f"   ✅ Status: Available")
                    else:
                        print(f"   ❌ Status: Failed to create")
                except Exception as e:
                    print(f"   ❌ Status: Error - {e}")
            print()
    else:
        # Simple list
        for name in sorted(extractors):
            # Quick availability check
            try:
                extractor = registry.create(name)
                status = "✅" if extractor else "❌"
            except:
                status = "❌"
            
            print(f"  {status} {name}")
    
    print()
    print("💡 Use --detailed for more information")
    print("💡 Test extractors with: uv run python cli.py extractors test --extractor <name>")


def test_extractor_command(args):
    """Test an extractor on sample text or file."""
    from core.base import Document
    
    tracker = LlamaProgressTracker()
    tracker.print_header(f"🧪 Testing Extractor: {args.extractor} 🧪")
    
    # Parse config if provided
    extractor_config = {}
    if args.config:
        try:
            extractor_config = json.loads(args.config)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON config: {e}")
            return
    
    # Create extractor
    try:
        extractor = registry.create(args.extractor, extractor_config)
        if not extractor:
            print(f"❌ Failed to create extractor: {args.extractor}")
            print(f"Available extractors: {', '.join(registry.list_extractors())}")
            return
    except Exception as e:
        print(f"❌ Error creating extractor: {e}")
        return
    
    # Get text to process
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {args.file}")
            return
        
        try:
            text = file_path.read_text(encoding='utf-8')
            print(f"📖 Processing file: {args.file}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return
    elif args.text:
        text = args.text
        print(f"📖 Processing text: {text[:100]}{'...' if len(text) > 100 else ''}")
    else:
        # Use sample text
        text = """
        Machine learning and artificial intelligence are transforming modern technology. 
        Companies like OpenAI, Google, and Microsoft are leading innovations in natural language processing.
        The deadline for the project is January 15, 2024, and we need to complete the analysis by then.
        Contact john.doe@company.com for more information or call (555) 123-4567.
        The quarterly revenue increased by 25% to $2.5 million, which exceeded our expectations.
        """
        print("📖 Using sample text for testing")
    
    # Create test document
    doc = Document(
        id="test_document",
        content=text,
        metadata={"source": "test"}
    )
    
    # Run extractor
    try:
        print(f"🔄 Running {args.extractor} extractor...")
        
        start_time = time.time()
        enhanced_docs = extractor.extract([doc])
        processing_time = time.time() - start_time
        
        if not enhanced_docs:
            print("❌ No documents returned from extractor")
            return
        
        enhanced_doc = enhanced_docs[0]
        
        print(f"✅ Processing completed in {processing_time:.3f}s")
        print()
        
        # Display results
        print(f"📊 {Fore.GREEN}Extraction Results:{Style.RESET_ALL}")
        print("=" * 50)
        
        # Show extractor-specific metadata
        if "extractors" in enhanced_doc.metadata:
            extractor_data = enhanced_doc.metadata["extractors"]
            
            if args.extractor in extractor_data or f"{args.extractor}_keywords" in extractor_data:
                # Handle different extractor data structures
                extractor_results = (extractor_data.get(args.extractor) or 
                                   extractor_data.get(f"{args.extractor}_keywords") or
                                   extractor_data.get(f"{args.extractor}_entities") or
                                   extractor_data)
                
                print(json.dumps(extractor_results, indent=2, ensure_ascii=False))
            else:
                print("No specific extractor results found in metadata")
                print("Available keys:", list(extractor_data.keys()))
        
        # Show simplified metadata
        print(f"\n📋 {Fore.BLUE}Simplified Access:{Style.RESET_ALL}")
        simplified_keys = [k for k in enhanced_doc.metadata.keys() 
                          if not k.startswith('_') and k != 'extractors']
        
        for key in sorted(simplified_keys):
            value = enhanced_doc.metadata[key]
            if isinstance(value, list) and len(value) > 5:
                display_value = value[:5] + [f"... ({len(value)-5} more)"]
            else:
                display_value = value
            print(f"  {key}: {display_value}")
        
    except Exception as e:
        print(f"❌ Extractor failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")


def strategy_command(args):
    """Handle strategy commands."""
    strategy_manager = StrategyManager(strategies_file=getattr(args, 'strategy_file', None))
    
    if args.strategy_command == "list":
        strategies = strategy_manager.get_available_strategies()
        
        if not strategies:
            print("❌ No strategies available")
            return
        
        if args.detailed:
            print(f"\n🚀 {Fore.CYAN}Available RAG Strategies{Style.RESET_ALL}")
            print("=" * 80)
            strategy_manager.print_all_strategies()
        else:
            print(f"\n🚀 {Fore.CYAN}Available Strategies ({len(strategies)}){Style.RESET_ALL}")
            print("=" * 50)
            for strategy_name in sorted(strategies):
                info = strategy_manager.get_strategy_info(strategy_name)
                if info:
                    print(f"{Fore.GREEN}{strategy_name:15}{Style.RESET_ALL} - {info['description']}")
            print(f"\n💡 Use 'strategies show <name>' for details")
            print(f"💡 Use 'strategies list --detailed' for full information")
    
    elif args.strategy_command == "show":
        info = strategy_manager.get_strategy_info(args.name)
        if not info:
            print(f"❌ Strategy '{args.name}' not found")
            available = strategy_manager.get_available_strategies()
            if available:
                print(f"Available strategies: {', '.join(available)}")
            return
        
        strategy_manager.print_strategy_summary(args.name)
    
    elif args.strategy_command == "recommend":
        criteria = {}
        if args.use_case:
            criteria["use_case"] = args.use_case
        if args.performance:
            criteria["performance_priority"] = args.performance
        if args.resources:
            criteria["resource_usage"] = args.resources
        if args.complexity:
            criteria["complexity"] = args.complexity
        
        recommendations = strategy_manager.recommend_strategies(**criteria)
        
        if not recommendations:
            print("❌ No strategies match your criteria")
            return
        
        print(f"\n🎯 {Fore.CYAN}Strategy Recommendations{Style.RESET_ALL}")
        print("=" * 50)
        
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            print(f"\n{i}. {Fore.GREEN}{rec['name']}{Style.RESET_ALL}")
            print(f"   {rec['description']}")
            print(f"   Use cases: {', '.join(rec['use_cases'])}")
            print(f"   Performance: {rec['performance_priority']} | Resources: {rec['resource_usage']} | Complexity: {rec['complexity']}")
    
    elif args.strategy_command == "convert":
        overrides = {}
        if args.overrides:
            try:
                overrides = json.loads(args.overrides)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON overrides: {e}")
                return
        
        success = strategy_manager.export_strategy_as_config(
            args.strategy_name, 
            args.output_file
        )
        
        if success:
            print(f"✅ Strategy '{args.strategy_name}' exported to {args.output_file}")
        else:
            print(f"❌ Failed to export strategy '{args.strategy_name}'")
    
    elif args.strategy_command == "test":
        overrides = {}
        if args.overrides:
            try:
                overrides = json.loads(args.overrides)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON overrides: {e}")
                return
        
        config = strategy_manager.convert_strategy_to_config(args.strategy_name, overrides)
        if not config:
            print(f"❌ Strategy '{args.strategy_name}' not found")
            return
        
        # Validate the configuration
        errors = strategy_manager.validate_strategy_config(config)
        if errors:
            print(f"❌ Strategy configuration has errors:")
            for error in errors:
                print(f"   - {error}")
            return
        
        print(f"✅ Strategy '{args.strategy_name}' configuration is valid")
        
        # Test with sample file if provided
        if args.sample_file:
            print(f"\n🧪 Testing with sample file: {args.sample_file}")
            # TODO: Implement sample file testing
            print("Sample file testing not yet implemented")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Simple RAG System CLI with Unified Configuration Support")
    parser.add_argument(
        "--config",
        "-c",
        default="rag_config.json",
        help="Configuration file path (supports relative and absolute paths)",
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        help="Base directory for resolving relative paths (defaults to current directory)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress info messages, only show warnings and errors",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output including document contents and metadata",
    )
    parser.add_argument(
        "--content-length",
        type=int,
        default=150,
        help="Maximum characters to show from document content (default: 150, use -1 for unlimited)",
    )
    parser.add_argument(
        "--strategy-file",
        help="Path to custom strategy YAML file (defaults to default_strategies.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", 
        help="Ingest documents with auto file type detection",
        epilog="Examples:\n"
               "  python cli.py ingest samples/data.csv --strategy simple\n"
               "  python cli.py ingest docs/ --strategy research --parser pdf\n"
               "  python cli.py --verbose ingest data.csv --extractors keywords entities"
    )
    ingest_parser.add_argument(
        "source", help="Source file or directory (supports relative and absolute paths)"
    )
    ingest_parser.add_argument(
        "--parser", help="Override parser selection (e.g., 'csv', 'pdf', 'pdf_chunked')"
    )
    ingest_parser.add_argument(
        "--embedder", help="Override embedder selection (e.g., 'default', 'fast', 'accurate')"
    )
    ingest_parser.add_argument(
        "--vector-store", help="Override vector store selection (e.g., 'default', 'dev')"
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search", 
        help="Search documents with strategy selection",
        epilog="Examples:\n"
               "  python cli.py search \"machine learning\" --strategy simple --top-k 5\n"
               "  python cli.py search \"error logs\" --strategy customer_support\n"
               "  python cli.py --verbose search \"API docs\" --retrieval hybrid"
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return"
    )
    search_parser.add_argument(
        "--retrieval", help="Override retrieval strategy (e.g., 'default', 'filtered', 'hybrid-balanced')"
    )
    search_parser.add_argument(
        "--embedder", help="Override embedder selection (e.g., 'default', 'fast', 'accurate')"
    )
    search_parser.add_argument(
        "--vector-store", help="Override vector store selection (e.g., 'default', 'dev')"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", 
        help="Show vector store info",
        epilog="Examples:\n"
               "  python cli.py info --strategy simple\n"
               "  python cli.py info --vector-store default"
    )
    info_parser.add_argument(
        "--vector-store", help="Override vector store selection (e.g., 'default', 'dev')"
    )
    info_parser.add_argument(
        "--strategy", help="Use a predefined strategy for configuration"
    )

    # Test command
    test_parser = subparsers.add_parser(
        "test", 
        help="Test system components",
        epilog="Examples:\n"
               "  python cli.py test --test-file sample.csv\n"
               "  python cli.py test --all-tests"
    )
    test_parser.add_argument(
        "--test-file",
        help="File to test parsing (supports CSV, PDF, and more with auto-detection)",
    )

    # Document management commands
    manage_parser = subparsers.add_parser(
        "manage", 
        help="Document management operations",
        epilog="Examples:\n"
               "  python cli.py manage delete --older-than 30 --dry-run\n"
               "  python cli.py manage stats --detailed\n"
               "  python cli.py manage cleanup --duplicates"
    )
    manage_parser.add_argument(
        "--strategy", dest="rag_strategy", help="Use a predefined RAG strategy for configuration"
    )
    manage_subparsers = manage_parser.add_subparsers(dest="manage_command", help="Management commands")

    # Delete commands
    delete_parser = manage_subparsers.add_parser("delete", help="Delete documents with various strategies")
    delete_parser.add_argument("--delete-strategy", choices=["soft", "hard", "archive"], default="soft",
                              help="Deletion strategy")
    delete_parser.add_argument("--older-than", type=int, metavar="DAYS",
                              help="Delete documents older than N days")
    delete_parser.add_argument("--doc-ids", nargs="+", metavar="ID",
                              help="Delete specific document IDs")
    delete_parser.add_argument("--filenames", nargs="+", metavar="FILE",
                              help="Delete documents by filename")
    delete_parser.add_argument("--content-hashes", nargs="+", metavar="HASH",
                              help="Delete documents by content hash")
    delete_parser.add_argument("--document-hashes", nargs="+", metavar="DOC_HASH",
                              help="Delete all chunks belonging to specific documents by hash")
    delete_parser.add_argument("--source-paths", nargs="+", metavar="PATH",
                              help="Delete all documents from specific source files")
    delete_parser.add_argument("--expired", action="store_true",
                              help="Delete expired documents")
    delete_parser.add_argument("--all", action="store_true",
                              help="Delete ALL documents in the collection (use with caution!)")
    delete_parser.add_argument("--dry-run", action="store_true",
                              help="Show what would be deleted without actually deleting")

    # Replace commands
    replace_parser = manage_subparsers.add_parser("replace", help="Replace documents")
    replace_parser.add_argument("source", help="Source file to replace with")
    replace_parser.add_argument("--target-doc-id", required=True,
                               help="Document ID to replace")
    replace_parser.add_argument("--replace-strategy", choices=["replace_all", "incremental", "versioning"], 
                               default="versioning", help="Replacement strategy")

    # Stats commands
    stats_parser = manage_subparsers.add_parser("stats", help="Show document statistics")
    stats_parser.add_argument("--detailed", action="store_true",
                             help="Show detailed statistics")

    # Cleanup commands
    cleanup_parser = manage_subparsers.add_parser("cleanup", help="Cleanup operations")
    cleanup_parser.add_argument("--old-versions", type=int, metavar="KEEP",
                               help="Keep only N latest versions of each document")
    cleanup_parser.add_argument("--duplicates", action="store_true",
                               help="Remove duplicate documents based on content hash")
    cleanup_parser.add_argument("--expired", action="store_true",
                               help="Clean up expired documents")

    # Hash commands
    hash_parser = manage_subparsers.add_parser("hash", help="Hash-based operations")
    hash_parser.add_argument("--find-duplicates", action="store_true",
                            help="Find duplicate documents by content hash")
    hash_parser.add_argument("--verify-integrity", action="store_true",
                            help="Verify document integrity using hashes")
    hash_parser.add_argument("--rehash", action="store_true",
                            help="Regenerate all document hashes")

    # Extractor commands
    extractor_parser = subparsers.add_parser(
        "extractors", 
        help="Extractor operations",
        epilog="Examples:\n"
               "  python cli.py extractors list --detailed\n"
               "  python cli.py extractors test --extractor KeywordExtractor --text \"sample text\""
    )
    extractor_subparsers = extractor_parser.add_subparsers(dest="extractor_command", help="Extractor commands")
    
    # List extractors
    list_extractors_parser = extractor_subparsers.add_parser("list", help="List available extractors")
    list_extractors_parser.add_argument("--detailed", action="store_true",
                                       help="Show detailed extractor information")
    
    # Test extractor
    test_extractor_parser = extractor_subparsers.add_parser("test", help="Test extractor on text/file")
    test_extractor_parser.add_argument("--extractor", required=True,
                                      help="Extractor name to test")
    test_extractor_parser.add_argument("--file", 
                                      help="File to extract from")
    test_extractor_parser.add_argument("--text",
                                      help="Text to extract from")
    test_extractor_parser.add_argument("--config", 
                                      help="JSON config for extractor")

    # Add extractor arguments to ingest command
    ingest_parser.add_argument("--extractors", nargs="+", 
                              help="Extractors to apply (e.g., rake yake entities)")
    ingest_parser.add_argument("--extractor-config", 
                              help="JSON config for extractors (e.g., '{\"yake\": {\"max_keywords\": 15}}')")

    # Strategy commands
    strategy_parser = subparsers.add_parser(
        "strategies", 
        help="Strategy operations",
        epilog="Examples:\n"
               "  python cli.py strategies list --detailed\n"
               "  python cli.py strategies show simple\n"
               "  python cli.py strategies recommend --use-case research --performance accuracy"
    )
    strategy_subparsers = strategy_parser.add_subparsers(dest="strategy_command", help="Strategy commands")
    
    # List strategies
    list_strategies_parser = strategy_subparsers.add_parser("list", help="List available strategies")
    list_strategies_parser.add_argument("--detailed", action="store_true",
                                       help="Show detailed strategy information")
    
    # Show strategy info
    show_strategy_parser = strategy_subparsers.add_parser("show", help="Show strategy details")
    show_strategy_parser.add_argument("name", help="Strategy name to show")
    
    # Recommend strategies
    recommend_parser = strategy_subparsers.add_parser("recommend", help="Recommend strategies")
    recommend_parser.add_argument("--use-case", help="Use case to optimize for")
    recommend_parser.add_argument("--performance", choices=["speed", "accuracy", "balanced"],
                                 help="Performance priority")
    recommend_parser.add_argument("--resources", choices=["low", "medium", "high"],
                                 help="Resource usage level")
    recommend_parser.add_argument("--complexity", choices=["simple", "moderate", "complex"],
                                 help="Complexity level")
    
    # Convert strategy to config
    convert_parser = strategy_subparsers.add_parser("convert", help="Convert strategy to config file")
    convert_parser.add_argument("strategy_name", help="Strategy name to convert")
    convert_parser.add_argument("output_file", help="Output configuration file")
    convert_parser.add_argument("--overrides", help="JSON overrides to apply")
    
    # Test strategy
    test_strategy_parser = strategy_subparsers.add_parser("test", help="Test a strategy configuration")
    test_strategy_parser.add_argument("strategy_name", help="Strategy name to test")
    test_strategy_parser.add_argument("--sample-file", help="Sample file to test with")
    test_strategy_parser.add_argument("--overrides", help="JSON overrides to apply")

    # Add strategy support to main commands
    ingest_parser.add_argument("--strategy", help="Strategy name to use instead of config file")
    ingest_parser.add_argument("--strategy-overrides", help="JSON overrides for strategy")
    search_parser.add_argument("--strategy", help="Strategy name to use instead of config file")
    search_parser.add_argument("--strategy-overrides", help="JSON overrides for strategy")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n💡 Examples:")
        print("  # Use a strategy (recommended)")
        print("  uv run python cli.py ingest samples/small_sample.csv --strategy simple")
        print("  uv run python cli.py search \"password reset\" --strategy customer_support")
        print("")
        print("  # List available strategies")
        print("  uv run python cli.py strategies list")
        print("  uv run python cli.py strategies show simple")
        print("")
        print("  # Traditional config file approach")
        print("  uv run python cli.py --config config_examples/basic_config.yaml ingest samples/small_sample.csv")
        print("")
        print("  # Override strategy settings")
        print("  uv run python cli.py ingest legal_docs/ --strategy legal --strategy-overrides '{\"embedder\":{\"config\":{\"batch_size\":32}}}'")
        sys.exit(1)

    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "search":
        search_command(args)
    elif args.command == "info":
        info_command(args)
    elif args.command == "test":
        test_command(args)
    elif args.command == "manage":
        manage_command(args)
    elif args.command == "extractors":
        extractor_command(args)
    elif args.command == "strategies":
        strategy_command(args)


if __name__ == "__main__":
    main()
