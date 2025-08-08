"""
CLI subcommands for the RAG system.
"""

from .components import list_components
from .compile import compile_components
from core.factories import create_embedder_from_config, create_vector_store_from_config

# Import CLI command functions for testing
# These are imported from the main cli.py module in the parent directory
try:
    import sys
    from pathlib import Path
    
    # Add parent directory to Python path temporarily
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import the CLI commands from cli.py
    from cli import (
        ingest_command,
        search_command,
        info_command,
        test_command,
        manage_command
    )
    
    # Clean up sys.path
    if parent_dir in sys.path:
        sys.path.remove(parent_dir)
        
except ImportError:
    # If import fails, define placeholder functions
    def ingest_command(args):
        raise NotImplementedError("CLI commands not available")
    def search_command(args):
        raise NotImplementedError("CLI commands not available")
    def info_command(args):
        raise NotImplementedError("CLI commands not available")
    def test_command(args):
        raise NotImplementedError("CLI commands not available")
    def manage_command(args):
        raise NotImplementedError("CLI commands not available")

__all__ = [
    "list_components", 
    "compile_components", 
    "create_embedder_from_config", 
    "create_vector_store_from_config",
    "ingest_command",
    "search_command", 
    "info_command",
    "test_command",
    "manage_command"
]