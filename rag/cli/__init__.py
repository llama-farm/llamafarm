"""
CLI subcommands for the RAG system.
"""

from .components import list_components
from .compile import compile_components
from core.factories import create_embedder_from_config, create_vector_store_from_config

# Import CLI command functions by importing from the parent cli.py module
# We need to import the module at the parent level, not the cli package
import sys
import importlib.util
from pathlib import Path

# Get the path to the parent cli.py file
cli_py_path = Path(__file__).parent.parent / "cli.py"

if cli_py_path.exists():
    # Load the cli.py module directly
    spec = importlib.util.spec_from_file_location("_cli_module", cli_py_path)
    _cli_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_cli_module)
    
    # Import the command functions
    ingest_command = _cli_module.ingest_command
    search_command = _cli_module.search_command
    info_command = _cli_module.info_command
    test_command = _cli_module.test_command
    manage_command = _cli_module.manage_command
else:
    # Fallback functions if cli.py doesn't exist
    def ingest_command(args):
        raise NotImplementedError("CLI commands not available - cli.py not found")
    def search_command(args):
        raise NotImplementedError("CLI commands not available - cli.py not found")
    def info_command(args):
        raise NotImplementedError("CLI commands not available - cli.py not found")
    def test_command(args):
        raise NotImplementedError("CLI commands not available - cli.py not found")
    def manage_command(args):
        raise NotImplementedError("CLI commands not available - cli.py not found")

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