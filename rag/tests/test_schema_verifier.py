#!/usr/bin/env python3
"""
Schema Verifier

A comprehensive tool to validate RAG configurations against the schema.yaml.
Uses jsonschema for validation and provides detailed error reporting.
"""

import sys
import json
import yaml
import argparse
import re
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from jsonschema import validate, ValidationError, Draft7Validator, RefResolver
import logging
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()


class SchemaVerifier:
    """Verifies configurations against the RAG schema."""
    
    def __init__(self, schema_path: str = None):
        """
        Initialize the schema verifier.
        
        Args:
            schema_path: Path to the schema YAML file
        """
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "schema.yaml"
        
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self.validator = None
        self._setup_validator()
        self.actual_components = self._discover_actual_components()
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load the schema from YAML file."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        with open(self.schema_path, 'r') as f:
            schema_data = yaml.safe_load(f)
        
        # Convert the schema to JSON Schema format if needed
        if "definitions" in schema_data:
            # Already in JSON Schema format
            return schema_data
        
        # If it's a custom format, wrap it properly
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            **schema_data
        }
    
    def _setup_validator(self):
        """Setup the JSON schema validator with proper resolver."""
        # Create a resolver for handling $ref references
        resolver = RefResolver(
            base_uri=f"file://{self.schema_path.parent}/",
            referrer=self.schema
        )
        self.validator = Draft7Validator(self.schema, resolver=resolver)
    
    def _discover_actual_components(self) -> Dict[str, Set[str]]:
        """Discover actual component classes in the codebase."""
        components = {
            'parsers': set(),
            'extractors': set(),
            'embedders': set(),
            'stores': set()
        }
        
        components_dir = Path(__file__).parent.parent / "components"
        
        # Discover parsers
        parsers_dir = components_dir / "parsers"
        if parsers_dir.exists():
            for file_path in parsers_dir.rglob("*.py"):
                if not file_path.name.startswith("__") and not file_path.name.startswith("test"):
                    try:
                        with open(file_path) as f:
                            content = f.read()
                            # Find parser classes
                            classes = re.findall(r'^class\s+(\w*Parser\w*)', content, re.MULTILINE)
                            components['parsers'].update(classes)
                    except Exception:
                        pass  # Skip files that can't be read
        
        # Discover extractors
        extractors_dir = components_dir / "extractors"
        if extractors_dir.exists():
            for file_path in extractors_dir.rglob("*.py"):
                if not file_path.name.startswith("__") and not file_path.name.startswith("test"):
                    try:
                        with open(file_path) as f:
                            content = f.read()
                            classes = re.findall(r'^class\s+(\w*Extractor\w*)', content, re.MULTILINE)
                            components['extractors'].update(classes)
                    except Exception:
                        pass  # Skip files that can't be read
        
        # Discover embedders
        embedders_dir = components_dir / "embedders"
        if embedders_dir.exists():
            for file_path in embedders_dir.rglob("*.py"):
                if not file_path.name.startswith("__") and not file_path.name.startswith("test"):
                    try:
                        with open(file_path) as f:
                            content = f.read()
                            classes = re.findall(r'^class\s+(\w*Embedder\w*)', content, re.MULTILINE)
                            components['embedders'].update(classes)
                    except Exception:
                        pass  # Skip files that can't be read
        
        # Discover stores
        stores_dir = components_dir / "stores"
        if stores_dir.exists():
            for file_path in stores_dir.rglob("*.py"):
                if not file_path.name.startswith("__") and not file_path.name.startswith("test"):
                    try:
                        with open(file_path) as f:
                            content = f.read()
                            classes = re.findall(r'^class\s+(\w*Store\w*)', content, re.MULTILINE)
                            components['stores'].update(classes)
                    except Exception:
                        pass  # Skip files that can't be read
        
        # Remove base classes and helper classes
        components['parsers'].discard('BaseParser')
        components['parsers'].discard('LlamaIndexParser')
        components['parsers'].discard('ParserFactory')
        components['parsers'].discard('ParserRegistry')
        components['parsers'].discard('ToolAwareParserFactory')
        components['extractors'].discard('BaseExtractor')
        components['embedders'].discard('BaseEmbedder')
        components['stores'].discard('BaseStore')
        
        return components
    
    def verify_config(self, config_path: str) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Verify a configuration file against the schema.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Tuple of (is_valid, list_of_errors, config_data)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            return False, [f"Configuration file not found: {config_path}"], {}
        
        # Load the configuration
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    config = json.load(f)
                else:
                    return False, [f"Unsupported file format: {config_path.suffix}"], {}
        except Exception as e:
            return False, [f"Failed to parse configuration: {e}"], {}
        
        # Validate against schema
        errors = []
        
        # Check if it's a RAG configuration
        if "rag" in config:
            errors.extend(self._validate_rag_section(config["rag"]))
        
        # Reject legacy strategies format
        if "strategies" in config and isinstance(config["strategies"], list):
            errors.append("ERROR: Legacy 'strategies' format is no longer supported. Use the new RAG schema format.")
        
        return len(errors) == 0, errors, config
    
    def _validate_rag_section(self, rag_config: Dict[str, Any]) -> List[str]:
        """Validate the RAG section of a configuration."""
        errors = []
        
        # Validate databases
        if "databases" in rag_config:
            for i, db in enumerate(rag_config.get("databases", [])):
                db_errors = self._validate_database(db, i)
                errors.extend(db_errors)
        
        # Validate data processing strategies
        if "data_processing_strategies" in rag_config:
            for i, strategy in enumerate(rag_config.get("data_processing_strategies", [])):
                strategy_errors = self._validate_data_processing_strategy(strategy, i)
                errors.extend(strategy_errors)
        
        return errors
    
    def _validate_database(self, db: Dict[str, Any], index: int) -> List[str]:
        """Validate a database configuration."""
        errors = []
        prefix = f"Database[{index}]"
        
        # Required fields
        if "name" not in db:
            errors.append(f"{prefix}: Missing required field 'name'")
        
        if "type" not in db:
            errors.append(f"{prefix}: Missing required field 'type'")
        else:
            # Check if store type actually exists
            store_type = db["type"]
            if store_type not in self.actual_components['stores']:
                errors.append(f"{prefix}: Store type '{store_type}' does not exist in components/stores")
        
        # Validate default strategies reference existing strategies
        if "default_embedding_strategy" in db:
            strategy_name = db["default_embedding_strategy"]
            strategy_names = [s.get("name") for s in db.get("embedding_strategies", [])]
            if strategy_name not in strategy_names:
                errors.append(f"{prefix}: default_embedding_strategy '{strategy_name}' not found in embedding_strategies")
        
        if "default_retrieval_strategy" in db:
            strategy_name = db["default_retrieval_strategy"]
            strategy_names = [s.get("name") for s in db.get("retrieval_strategies", [])]
            if strategy_name not in strategy_names:
                errors.append(f"{prefix}: default_retrieval_strategy '{strategy_name}' not found in retrieval_strategies")
        
        # Validate embedding strategies
        for i, strategy in enumerate(db.get("embedding_strategies", [])):
            if "name" not in strategy:
                errors.append(f"{prefix}.embedding_strategies[{i}]: Missing 'name'")
            if "type" not in strategy:
                errors.append(f"{prefix}.embedding_strategies[{i}]: Missing 'type'")
            else:
                # Check if embedder type actually exists
                embedder_type = strategy["type"]
                if embedder_type not in self.actual_components['embedders']:
                    errors.append(f"{prefix}.embedding_strategies[{i}]: Embedder type '{embedder_type}' does not exist in components/embedders")
        
        # Validate retrieval strategies
        for i, strategy in enumerate(db.get("retrieval_strategies", [])):
            if "name" not in strategy:
                errors.append(f"{prefix}.retrieval_strategies[{i}]: Missing 'name'")
            if "type" not in strategy:
                errors.append(f"{prefix}.retrieval_strategies[{i}]: Missing 'type'")
        
        return errors
    
    def _validate_data_processing_strategy(self, strategy: Dict[str, Any], index: int) -> List[str]:
        """Validate a data processing strategy."""
        errors = []
        prefix = f"DataProcessingStrategy[{index}]"
        
        # Required fields
        if "name" not in strategy:
            errors.append(f"{prefix}: Missing required field 'name'")
        
        if "parsers" not in strategy or not strategy["parsers"]:
            errors.append(f"{prefix}: Missing or empty 'parsers' field")
        
        # Validate MIME type filtering
        if "allowed_mime_types" in strategy:
            if not isinstance(strategy["allowed_mime_types"], list):
                errors.append(f"{prefix}: 'allowed_mime_types' must be a list")
        
        if "allowed_extensions" in strategy:
            extensions = strategy["allowed_extensions"]
            if not isinstance(extensions, list):
                errors.append(f"{prefix}: 'allowed_extensions' must be a list")
            else:
                for ext in extensions:
                    if not ext.startswith('.'):
                        errors.append(f"{prefix}: Extension '{ext}' should start with '.'")
        
        # Validate parsers
        for i, parser in enumerate(strategy.get("parsers", [])):
            if "type" not in parser:
                errors.append(f"{prefix}.parsers[{i}]: Missing 'type'")
            else:
                # Check if parser actually exists
                parser_type = parser["type"]
                if parser_type not in self.actual_components['parsers']:
                    errors.append(f"{prefix}.parsers[{i}]: Parser type '{parser_type}' does not exist in components/parsers")
            
            # Validate parser-level MIME types and extensions
            if "mime_types" in parser:
                if not isinstance(parser["mime_types"], list):
                    errors.append(f"{prefix}.parsers[{i}]: 'mime_types' must be a list")
            
            if "file_extensions" in parser:
                extensions = parser["file_extensions"]
                if not isinstance(extensions, list):
                    errors.append(f"{prefix}.parsers[{i}]: 'file_extensions' must be a list")
                else:
                    for ext in extensions:
                        if not ext.startswith('.'):
                            errors.append(f"{prefix}.parsers[{i}]: Extension '{ext}' should start with '.'")
        
        # Validate extractors
        for i, extractor in enumerate(strategy.get("extractors", [])):
            if "type" not in extractor:
                errors.append(f"{prefix}.extractors[{i}]: Missing 'type'")
            else:
                # Check if extractor actually exists
                extractor_type = extractor["type"]
                if extractor_type not in self.actual_components['extractors']:
                    errors.append(f"{prefix}.extractors[{i}]: Extractor type '{extractor_type}' does not exist in components/extractors")
        
        return errors
    
    def verify_multiple_configs(self, config_paths: List[str]) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Verify multiple configuration files.
        
        Args:
            config_paths: List of configuration file paths
            
        Returns:
            Dictionary mapping config path to (is_valid, errors) tuple
        """
        results = {}
        for config_path in config_paths:
            is_valid, errors, _ = self.verify_config(config_path)
            results[config_path] = (is_valid, errors)
        return results
    
    def print_verification_report(self, config_path: str, is_valid: bool, errors: List[str], config: Dict[str, Any]):
        """Print a detailed verification report."""
        config_name = Path(config_path).name
        
        if is_valid:
            console.print(f"\n✅ [green]Configuration VALID:[/green] {config_name}")
            
            # Show configuration summary
            if "rag" in config:
                rag = config["rag"]
                
                # Database summary
                if "databases" in rag:
                    console.print(f"  📊 Databases: {len(rag['databases'])}")
                    for db in rag["databases"]:
                        console.print(f"     • {db.get('name', 'unnamed')} ({db.get('type', 'unknown')})")
                
                # Strategy summary
                if "data_processing_strategies" in rag:
                    console.print(f"  🔧 Processing Strategies: {len(rag['data_processing_strategies'])}")
                    for strategy in rag["data_processing_strategies"]:
                        name = strategy.get('name', 'unnamed')
                        mime_count = len(strategy.get('allowed_mime_types', []))
                        parser_count = len(strategy.get('parsers', []))
                        
                        if mime_count > 0:
                            console.print(f"     • {name} ({mime_count} MIME types, {parser_count} parsers)")
                        else:
                            console.print(f"     • {name} (accepts all types, {parser_count} parsers)")
        else:
            console.print(f"\n❌ [red]Configuration INVALID:[/red] {config_name}")
            console.print(f"\n[yellow]Found {len(errors)} error(s):[/yellow]")
            for i, error in enumerate(errors, 1):
                console.print(f"  {i}. {error}")
    
    def print_summary_table(self, results: Dict[str, Tuple[bool, List[str]]]):
        """Print a summary table of verification results."""
        table = Table(title="Schema Verification Summary", show_header=True)
        table.add_column("Configuration", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Errors", justify="center")
        table.add_column("Details")
        
        total_valid = 0
        total_errors = 0
        
        for config_path, (is_valid, errors) in results.items():
            config_name = Path(config_path).name
            status = "✅ VALID" if is_valid else "❌ INVALID"
            status_color = "green" if is_valid else "red"
            error_count = len(errors)
            
            if is_valid:
                total_valid += 1
                details = "All checks passed"
            else:
                total_errors += error_count
                # Show first error
                details = errors[0] if errors else "Unknown error"
                if len(errors) > 1:
                    details += f" (+{len(errors)-1} more)"
            
            table.add_row(
                config_name,
                f"[{status_color}]{status}[/{status_color}]",
                str(error_count) if error_count > 0 else "-",
                details
            )
        
        console.print("\n")
        console.print(table)
        
        # Summary stats
        total_configs = len(results)
        console.print(f"\n📊 Summary: {total_valid}/{total_configs} configurations valid")
        if total_errors > 0:
            console.print(f"   Total errors found: {total_errors}")


def main():
    """Main entry point for the schema verifier."""
    parser = argparse.ArgumentParser(
        description="Verify RAG configurations against the schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify a single configuration
  python schema_verifier.py config.yaml
  
  # Verify multiple configurations
  python schema_verifier.py config1.yaml config2.yaml config3.yaml
  
  # Use a custom schema file
  python schema_verifier.py --schema custom_schema.yaml config.yaml
  
  # Verify all configs in a directory
  python schema_verifier.py configs/*.yaml
        """
    )
    
    parser.add_argument(
        "configs",
        nargs="+",
        help="Path(s) to configuration file(s) to verify"
    )
    
    parser.add_argument(
        "--schema",
        default=None,
        help="Path to custom schema file (default: schema.yaml)"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary table only"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for each configuration"
    )
    
    args = parser.parse_args()
    
    # Create verifier
    try:
        verifier = SchemaVerifier(schema_path=args.schema)
    except FileNotFoundError as e:
        console.print(f"[red]❌ {e}[/red]")
        sys.exit(1)
    
    console.print(Panel.fit(
        "[bold cyan]RAG Schema Verifier[/bold cyan]\n"
        "Validating configurations against schema.yaml",
        border_style="cyan"
    ))
    
    # Verify all configurations
    results = {}
    for config_path in args.configs:
        # Handle glob patterns
        config_path = Path(config_path)
        if config_path.exists():
            is_valid, errors, config = verifier.verify_config(str(config_path))
            results[str(config_path)] = (is_valid, errors)
            
            if args.verbose and not args.summary:
                verifier.print_verification_report(str(config_path), is_valid, errors, config)
    
    # Print summary
    if results:
        verifier.print_summary_table(results)
    else:
        console.print("[yellow]No configurations found to verify[/yellow]")
        sys.exit(1)
    
    # Exit with appropriate code
    all_valid = all(is_valid for is_valid, _ in results.values())
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()