#!/usr/bin/env python3
"""
Configuration Schema Validator

A standalone tool to validate RAG configuration files against the schema.
Supports both the new RAG schema format and legacy formats.
"""

import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from jsonschema import validate, ValidationError, Draft7Validator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration files against the schema."""
    
    # Supported database types - single source of truth
    SUPPORTED_DATABASE_TYPES = [
        "ChromaStore", "chroma", 
        "QdrantStore", "qdrant", 
        "WeaviateStore", "weaviate", 
        "MilvusStore", "milvus", 
        "PineconeStore", "pinecone"
    ]
    
    def __init__(self, schema_path: str = None):
        """
        Initialize the validator.
        
        Args:
            schema_path: Path to the schema YAML file
        """
        if schema_path is None:
            # Default to schema.yaml in the same directory
            schema_path = Path(__file__).parent / "schema.yaml"
        
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load the schema from YAML file."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        with open(self.schema_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_config(self, config_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            return False, [f"Configuration file not found: {config_path}"]
        
        # Load the configuration
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    config = json.load(f)
                else:
                    return False, [f"Unsupported file format: {config_path.suffix}"]
        except Exception as e:
            return False, [f"Failed to parse configuration: {e}"]
        
        # Detect the format and validate accordingly
        errors = []
        
        if self._is_rag_format(config):
            errors.extend(self._validate_rag_format(config))
        elif self._is_unified_format(config):
            errors.extend(self._validate_unified_format(config))
        else:
            errors.extend(self._validate_legacy_format(config))
        
        return len(errors) == 0, errors
    
    def _is_rag_format(self, config: Dict[str, Any]) -> bool:
        """Check if config is in the new RAG format."""
        return "rag" in config and isinstance(config["rag"], dict)
    
    def _is_unified_format(self, config: Dict[str, Any]) -> bool:
        """Check if config is in the unified format."""
        return "strategies" in config and isinstance(config["strategies"], list)
    
    def _validate_rag_format(self, config: Dict[str, Any]) -> List[str]:
        """Validate RAG format configuration."""
        errors = []
        rag_config = config.get("rag", {})
        
        # Validate databases
        if "databases" not in rag_config:
            errors.append("RAG configuration missing 'databases' section")
        else:
            for i, db in enumerate(rag_config["databases"]):
                db_errors = self._validate_database(db, i)
                errors.extend(db_errors)
        
        # Validate data processing strategies
        if "data_processing_strategies" not in rag_config:
            errors.append("RAG configuration missing 'data_processing_strategies' section")
        else:
            for i, strategy in enumerate(rag_config["data_processing_strategies"]):
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
        elif db["type"] not in self.SUPPORTED_DATABASE_TYPES:
            errors.append(f"{prefix}: Invalid database type '{db['type']}'. Supported types: {', '.join(self.SUPPORTED_DATABASE_TYPES)}")
        
        # Validate default strategies if present
        if "default_embedding_strategy" in db:
            # Check if the strategy exists
            embedding_strategies = db.get("embedding_strategies", [])
            strategy_names = [s.get("name") for s in embedding_strategies if "name" in s]
            if db["default_embedding_strategy"] not in strategy_names:
                errors.append(f"{prefix}: default_embedding_strategy '{db['default_embedding_strategy']}' not found in embedding_strategies")
        
        if "default_retrieval_strategy" in db:
            # Check if the strategy exists
            retrieval_strategies = db.get("retrieval_strategies", [])
            strategy_names = [s.get("name") for s in retrieval_strategies if "name" in s]
            if db["default_retrieval_strategy"] not in strategy_names:
                errors.append(f"{prefix}: default_retrieval_strategy '{db['default_retrieval_strategy']}' not found in retrieval_strategies")
        
        # Validate embedding strategies
        for i, strategy in enumerate(db.get("embedding_strategies", [])):
            if "name" not in strategy:
                errors.append(f"{prefix}.embedding_strategies[{i}]: Missing required field 'name'")
            if "type" not in strategy:
                errors.append(f"{prefix}.embedding_strategies[{i}]: Missing required field 'type'")
        
        # Validate retrieval strategies
        for i, strategy in enumerate(db.get("retrieval_strategies", [])):
            if "name" not in strategy:
                errors.append(f"{prefix}.retrieval_strategies[{i}]: Missing required field 'name'")
            if "type" not in strategy:
                errors.append(f"{prefix}.retrieval_strategies[{i}]: Missing required field 'type'")
        
        return errors
    
    def _validate_data_processing_strategy(self, strategy: Dict[str, Any], index: int) -> List[str]:
        """Validate a data processing strategy."""
        errors = []
        prefix = f"DataProcessingStrategy[{index}]"
        
        # Required fields
        if "name" not in strategy:
            errors.append(f"{prefix}: Missing required field 'name'")
        
        # Validate parsers
        for i, parser in enumerate(strategy.get("parsers", [])):
            if "type" not in parser:
                errors.append(f"{prefix}.parsers[{i}]: Missing required field 'type'")
        
        # Validate extractors
        for i, extractor in enumerate(strategy.get("extractors", [])):
            if "type" not in extractor:
                errors.append(f"{prefix}.extractors[{i}]: Missing required field 'type'")
        
        return errors
    
    def _validate_unified_format(self, config: Dict[str, Any]) -> List[str]:
        """Validate unified format configuration."""
        errors = []
        
        for i, strategy in enumerate(config.get("strategies", [])):
            prefix = f"Strategy[{i}]"
            
            # Required fields
            if "name" not in strategy:
                errors.append(f"{prefix}: Missing required field 'name'")
            
            if "components" not in strategy:
                errors.append(f"{prefix}: Missing required field 'components'")
            else:
                # Validate components
                components = strategy["components"]
                
                if "parser" not in components:
                    errors.append(f"{prefix}.components: Missing required field 'parser'")
                
                if "embedder" not in components:
                    errors.append(f"{prefix}.components: Missing required field 'embedder'")
                
                if "vector_store" not in components:
                    errors.append(f"{prefix}.components: Missing required field 'vector_store'")
        
        return errors
    
    def _validate_legacy_format(self, config: Dict[str, Any]) -> List[str]:
        """Validate legacy format configuration."""
        errors = []
        
        # Legacy format has strategies as top-level keys
        for key, value in config.items():
            if isinstance(value, dict) and "components" in value:
                prefix = f"Strategy[{key}]"
                
                # Validate components
                components = value["components"]
                
                if "parser" not in components:
                    errors.append(f"{prefix}.components: Missing required field 'parser'")
                
                if "embedder" not in components:
                    errors.append(f"{prefix}.components: Missing required field 'embedder'")
                
                if "vector_store" not in components:
                    errors.append(f"{prefix}.components: Missing required field 'vector_store'")
        
        return errors
    
    def validate_with_jsonschema(self, config_path: str) -> Tuple[bool, List[str]]:
        """
        Validate using jsonschema library for more detailed validation.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            return False, [f"Configuration file not found: {config_path}"]
        
        # Load the configuration
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    config = json.load(f)
                else:
                    return False, [f"Unsupported file format: {config_path.suffix}"]
        except Exception as e:
            return False, [f"Failed to parse configuration: {e}"]
        
        # Validate against schema
        errors = []
        validator = Draft7Validator(self.schema)
        
        for error in validator.iter_errors(config):
            error_path = " -> ".join(str(p) for p in error.path) if error.path else "root"
            errors.append(f"{error_path}: {error.message}")
        
        return len(errors) == 0, errors


def print_validation_result(is_valid: bool, errors: List[str], config_path: str):
    """Print validation results in a nice format."""
    if is_valid:
        logger.info(f"✅ Configuration is valid: {config_path}")
    else:
        logger.error(f"❌ Configuration validation failed: {config_path}")
        logger.error(f"\nFound {len(errors)} error(s):\n")
        for i, error in enumerate(errors, 1):
            logger.error(f"  {i}. {error}")


def main():
    """Main entry point for the validator tool."""
    parser = argparse.ArgumentParser(
        description="Validate RAG configuration files against the schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a configuration file
  python validate_config.py config.yaml
  
  # Use a custom schema file
  python validate_config.py config.yaml --schema custom_schema.yaml
  
  # Use strict JSON schema validation
  python validate_config.py config.yaml --strict
  
  # Validate multiple files
  python validate_config.py config1.yaml config2.json config3.yaml
        """
    )
    
    parser.add_argument(
        "configs",
        nargs="+",
        help="Path(s) to configuration file(s) to validate"
    )
    
    parser.add_argument(
        "--schema",
        default=None,
        help="Path to custom schema file (default: schema.yaml)"
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict JSON schema validation"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show errors, no success messages"
    )
    
    args = parser.parse_args()
    
    # Create validator
    try:
        validator = ConfigValidator(schema_path=args.schema)
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    
    # Track overall success
    all_valid = True
    
    # Validate each configuration file
    for config_path in args.configs:
        if len(args.configs) > 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"Validating: {config_path}")
            logger.info(f"{'='*60}")
        
        if args.strict:
            # Use JSON schema validation
            is_valid, errors = validator.validate_with_jsonschema(config_path)
        else:
            # Use custom validation logic
            is_valid, errors = validator.validate_config(config_path)
        
        if not args.quiet or not is_valid:
            print_validation_result(is_valid, errors, config_path)
        
        if not is_valid:
            all_valid = False
    
    # Exit with appropriate code
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()