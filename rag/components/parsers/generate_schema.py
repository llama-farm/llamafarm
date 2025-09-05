#!/usr/bin/env python3
"""Generate unified parser schema from individual parser schemas."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import argparse


def load_parser_config(parser_dir: Path) -> Dict[str, Any]:
    """Load parser configuration from a parser directory.
    
    Args:
        parser_dir: Path to parser directory
        
    Returns:
        Parser configuration dictionary
    """
    config_file = parser_dir / "config.yaml"
    schema_file = parser_dir / "schema.json"
    
    config = {}
    
    # Load config.yaml
    if config_file.exists():
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
            config['metadata'] = data.get('parser', {})
            config['profiles'] = data.get('profiles', {})
    
    # Load schema.json
    if schema_file.exists():
        with open(schema_file, 'r') as f:
            config['schema'] = json.load(f)
    
    config['name'] = parser_dir.name
    return config


def generate_unified_schema(parsers_dir: Path) -> Dict[str, Any]:
    """Generate unified schema from all parser schemas.
    
    Args:
        parsers_dir: Path to parsers directory
        
    Returns:
        Unified schema dictionary
    """
    unified = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "RAG Parser Configuration",
        "description": "Unified configuration schema for all RAG parsers",
        "type": "object",
        "properties": {
            "parser_type": {
                "type": "string",
                "enum": [],
                "description": "Type of parser to use"
            },
            "parser_config": {
                "type": "object",
                "oneOf": []
            }
        },
        "required": ["parser_type"],
        "additionalProperties": False
    }
    
    parser_configs = []
    parser_metadata = {}
    
    # Scan all parser directories
    for parser_dir in parsers_dir.iterdir():
        if parser_dir.is_dir() and not parser_dir.name.startswith(('__', '.', 'base')):
            config = load_parser_config(parser_dir)
            
            if config.get('schema'):
                # Add parser type to enum
                parser_name = config.get('metadata', {}).get('name', parser_dir.name)
                unified['properties']['parser_type']['enum'].append(parser_name)
                
                # Add schema to oneOf
                schema_entry = {
                    "type": "object",
                    "properties": {
                        "parser_type": {
                            "const": parser_name
                        },
                        "parser_config": config['schema']
                    }
                }
                unified['properties']['parser_config']['oneOf'].append(schema_entry)
                
                # Store metadata
                parser_metadata[parser_name] = config.get('metadata', {})
                parser_configs.append(config)
    
    return unified, parser_metadata, parser_configs


def generate_parser_registry(parsers_dir: Path) -> str:
    """Generate Python code for parser registry.
    
    Args:
        parsers_dir: Path to parsers directory
        
    Returns:
        Python code string
    """
    code = '''"""Auto-generated parser registry."""

from typing import Dict, Type, Any
from pathlib import Path
import importlib
import logging

from .base import BaseParser

logger = logging.getLogger(__name__)


class ParserRegistry:
    """Registry for auto-discovering and loading parsers."""
    
    def __init__(self):
        self.parsers: Dict[str, Type[BaseParser]] = {}
        self._discover_parsers()
    
    def _discover_parsers(self):
        """Discover all available parsers."""
        parsers_dir = Path(__file__).parent
        
        for parser_dir in parsers_dir.iterdir():
            if parser_dir.is_dir() and not parser_dir.name.startswith(('__', '.', 'base')):
                try:
                    # Try to import the parser module
                    module_name = f"components.parsers.{parser_dir.name}.parser"
                    module = importlib.import_module(module_name)
                    
                    # Find the parser class
                    for name in dir(module):
                        obj = getattr(module, name)
                        if (isinstance(obj, type) and 
                            issubclass(obj, BaseParser) and 
                            obj != BaseParser):
                            parser_name = parser_dir.name
                            self.parsers[parser_name] = obj
                            logger.debug(f"Registered parser: {parser_name}")
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to load parser from {parser_dir.name}: {e}")
    
    def get_parser(self, name: str, config: Dict[str, Any] = None) -> BaseParser:
        """Get a parser instance by name.
        
        Args:
            name: Parser name
            config: Parser configuration
            
        Returns:
            Parser instance
        """
        if name not in self.parsers:
            raise ValueError(f"Parser '{name}' not found. Available: {list(self.parsers.keys())}")
        
        return self.parsers[name](config)
    
    def list_parsers(self) -> List[str]:
        """List all available parsers.
        
        Returns:
            List of parser names
        """
        return list(self.parsers.keys())


# Global registry instance
registry = ParserRegistry()
'''
    
    return code


def generate_types(unified_schema: Dict[str, Any], output_dir: Path):
    """Generate TypeScript types using datamodel-codegen.
    
    Args:
        unified_schema: Unified schema dictionary
        output_dir: Output directory for generated types
    """
    try:
        import subprocess
        
        # Save schema to temp file
        schema_file = output_dir / "unified_schema.json"
        with open(schema_file, 'w') as f:
            json.dump(unified_schema, f, indent=2)
        
        # Generate Python types
        subprocess.run([
            "datamodel-codegen",
            "--input", str(schema_file),
            "--output", str(output_dir / "parser_types.py"),
            "--input-file-type", "jsonschema",
            "--use-poetry-optional",
            "--use-union-operator"
        ], check=True)
        
        print(f"Generated Python types in {output_dir / 'parser_types.py'}")
        
    except ImportError:
        print("datamodel-codegen not installed. Install with: pip install datamodel-code-generator")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate types: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate unified parser schema")
    parser.add_argument(
        "--parsers-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing parser modules"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "generated",
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--generate-types",
        action="store_true",
        help="Generate Python types using datamodel-codegen"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unified schema
    print(f"Scanning parsers in {args.parsers_dir}...")
    unified_schema, metadata, configs = generate_unified_schema(args.parsers_dir)
    
    # Save unified schema
    schema_file = args.output_dir / "unified_schema.json"
    with open(schema_file, 'w') as f:
        json.dump(unified_schema, f, indent=2)
    print(f"Generated unified schema: {schema_file}")
    
    # Save parser metadata
    metadata_file = args.output_dir / "parser_metadata.yaml"
    with open(metadata_file, 'w') as f:
        yaml.dump({"parsers": metadata}, f, default_flow_style=False)
    print(f"Generated metadata: {metadata_file}")
    
    # Generate parser registry
    registry_code = generate_parser_registry(args.parsers_dir)
    registry_file = args.parsers_dir / "parser_registry.py"
    with open(registry_file, 'w') as f:
        f.write(registry_code)
    print(f"Generated parser registry: {registry_file}")
    
    # Generate types if requested
    if args.generate_types:
        generate_types(unified_schema, args.output_dir)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Parsers found: {len(unified_schema['properties']['parser_type']['enum'])}")
    print(f"  Parsers: {', '.join(unified_schema['properties']['parser_type']['enum'])}")


if __name__ == "__main__":
    main()