#!/usr/bin/env python3
"""
Schema Validator and Builder CI Tool

This script validates strategy schemas and component schemas,
and automatically builds consolidated schema definitions when changes occur.
"""

import yaml
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import argparse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.strategies.config import StrategyConfig
from core.strategies.loader import StrategyLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates and builds schema definitions."""
    
    def __init__(self, rag_root: Path = None):
        """Initialize validator with RAG root directory."""
        self.rag_root = rag_root or Path(__file__).parent.parent
        self.schema_file = self.rag_root / "schema.yaml"
        self.components_dir = self.rag_root / "components"
        self.config_samples_dir = self.rag_root / "config_samples"
        self.demos_dir = self.rag_root / "demos"
        self.cache_file = self.rag_root / ".schema_cache"
        
    def validate_main_schema(self) -> List[str]:
        """Validate the main schema.yaml file."""
        errors = []
        
        if not self.schema_file.exists():
            errors.append(f"Main schema file not found: {self.schema_file}")
            return errors
        
        try:
            with open(self.schema_file, 'r') as f:
                schema = yaml.safe_load(f)
            
            # Check required top-level fields
            if "$schema" not in schema:
                errors.append("Schema missing $schema field")
            
            if "properties" not in schema:
                errors.append("Schema missing properties field")
            
            if schema.get("properties", {}).get("strategies") is None:
                errors.append("Schema missing strategies property")
            
            logger.info(f"Main schema validated: {len(errors)} errors")
            
        except Exception as e:
            errors.append(f"Failed to parse schema: {e}")
        
        return errors
    
    def validate_component_schemas(self) -> Dict[str, List[str]]:
        """Validate all component schema files."""
        errors = {}
        
        # Component types to check
        component_types = ["parsers", "extractors", "embedders", "stores", "retrievers"]
        
        for comp_type in component_types:
            comp_dir = self.components_dir / comp_type
            if not comp_dir.exists():
                continue
            
            for component_dir in comp_dir.iterdir():
                if not component_dir.is_dir():
                    continue
                
                schema_file = component_dir / "schema.yaml"
                if schema_file.exists():
                    comp_errors = self._validate_component_schema(schema_file)
                    if comp_errors:
                        errors[str(schema_file.relative_to(self.rag_root))] = comp_errors
        
        logger.info(f"Validated {len(errors)} component schemas with errors")
        return errors
    
    def _validate_component_schema(self, schema_file: Path) -> List[str]:
        """Validate a single component schema file."""
        errors = []
        
        try:
            with open(schema_file, 'r') as f:
                schema = yaml.safe_load(f)
            
            # Check required fields for component schema
            required_fields = ["name", "type", "class_name", "description"]
            for field in required_fields:
                if field not in schema:
                    errors.append(f"Missing required field: {field}")
            
            # Validate config_schema if present
            if "config_schema" in schema:
                if not isinstance(schema["config_schema"], dict):
                    errors.append("config_schema must be a dictionary")
                elif "type" not in schema["config_schema"]:
                    errors.append("config_schema missing type field")
            
        except Exception as e:
            errors.append(f"Failed to parse schema: {e}")
        
        return errors
    
    def validate_strategy_configs(self) -> Dict[str, List[str]]:
        """Validate all strategy configuration files."""
        errors = {}
        
        # Check config samples
        for config_file in self.config_samples_dir.glob("*.yaml"):
            file_errors = self._validate_strategy_file(config_file)
            if file_errors:
                errors[str(config_file.relative_to(self.rag_root))] = file_errors
        
        # Check demo strategies
        demo_strategies_file = self.demos_dir / "demo_strategies.yaml"
        if demo_strategies_file.exists():
            file_errors = self._validate_strategy_file(demo_strategies_file)
            if file_errors:
                errors[str(demo_strategies_file.relative_to(self.rag_root))] = file_errors
        
        # Check default strategies
        default_strategies_file = self.rag_root / "default_strategies.yaml"
        if default_strategies_file.exists():
            file_errors = self._validate_strategy_file(default_strategies_file)
            if file_errors:
                errors[str(default_strategies_file.relative_to(self.rag_root))] = file_errors
        
        logger.info(f"Validated strategy configs: {len(errors)} files with errors")
        return errors
    
    def _validate_strategy_file(self, file_path: Path) -> List[str]:
        """Validate a strategy configuration file."""
        errors = []
        
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check if it's the new unified format
            if "strategies" in data and isinstance(data["strategies"], list):
                # Validate each strategy in the list
                for i, strategy in enumerate(data["strategies"]):
                    strategy_errors = self._validate_strategy_dict(strategy, f"Strategy {i}")
                    errors.extend(strategy_errors)
            else:
                # Legacy format - validate top-level strategies
                for name, strategy in data.items():
                    if name.startswith("usage_") or name == "strategy_templates":
                        continue
                    if isinstance(strategy, dict) and "components" in strategy:
                        strategy_errors = self._validate_strategy_dict(strategy, name)
                        errors.extend(strategy_errors)
        
        except Exception as e:
            errors.append(f"Failed to parse file: {e}")
        
        return errors
    
    def _validate_strategy_dict(self, strategy: Dict[str, Any], identifier: str) -> List[str]:
        """Validate a single strategy dictionary."""
        errors = []
        
        # Required fields
        if "name" not in strategy:
            errors.append(f"{identifier}: Missing 'name' field")
        
        if "description" not in strategy:
            errors.append(f"{identifier}: Missing 'description' field")
        
        if "components" not in strategy:
            errors.append(f"{identifier}: Missing 'components' field")
        else:
            # Validate components
            components = strategy["components"]
            
            if "parser" not in components:
                errors.append(f"{identifier}: Missing parser component")
            elif not isinstance(components["parser"], dict) or "type" not in components["parser"]:
                errors.append(f"{identifier}: Invalid parser configuration")
            
            if "embedder" not in components:
                errors.append(f"{identifier}: Missing embedder component")
            elif not isinstance(components["embedder"], dict) or "type" not in components["embedder"]:
                errors.append(f"{identifier}: Invalid embedder configuration")
            
            if "vector_store" not in components:
                errors.append(f"{identifier}: Missing vector_store component")
            elif not isinstance(components["vector_store"], dict) or "type" not in components["vector_store"]:
                errors.append(f"{identifier}: Invalid vector_store configuration")
            
            if "retrieval_strategy" not in components:
                errors.append(f"{identifier}: Missing retrieval_strategy component")
            elif not isinstance(components["retrieval_strategy"], dict) or "type" not in components["retrieval_strategy"]:
                errors.append(f"{identifier}: Invalid retrieval_strategy configuration")
            
            # Validate extractors if present
            if "extractors" in components:
                if not isinstance(components["extractors"], list):
                    errors.append(f"{identifier}: Extractors must be a list")
                else:
                    for i, extractor in enumerate(components["extractors"]):
                        if not isinstance(extractor, dict) or "type" not in extractor:
                            errors.append(f"{identifier}: Invalid extractor {i} configuration")
        
        return errors
    
    def build_consolidated_schema(self) -> Dict[str, Any]:
        """Build a consolidated schema from all component schemas."""
        consolidated = {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "components": {
                "parsers": {},
                "extractors": {},
                "embedders": {},
                "stores": {},
                "retrievers": {}
            }
        }
        
        # Collect all component schemas
        for comp_type in consolidated["components"].keys():
            comp_dir = self.components_dir / comp_type
            if not comp_dir.exists():
                continue
            
            for component_dir in comp_dir.iterdir():
                if not component_dir.is_dir():
                    continue
                
                schema_file = component_dir / "schema.yaml"
                if schema_file.exists():
                    try:
                        with open(schema_file, 'r') as f:
                            schema = yaml.safe_load(f)
                        
                        comp_name = schema.get("name", component_dir.name)
                        consolidated["components"][comp_type][comp_name] = {
                            "type": schema.get("type"),
                            "class_name": schema.get("class_name"),
                            "description": schema.get("description"),
                            "config_schema": schema.get("config_schema", {}),
                            "use_cases": schema.get("use_cases", []),
                            "dependencies": schema.get("dependencies", {})
                        }
                    except Exception as e:
                        logger.error(f"Failed to load schema from {schema_file}: {e}")
        
        return consolidated
    
    def check_for_changes(self) -> bool:
        """Check if any schema files have changed since last run."""
        current_hash = self._calculate_schema_hash()
        
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cached_hash = f.read().strip()
                
                if cached_hash == current_hash:
                    logger.info("No schema changes detected")
                    return False
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")
        
        # Save new hash
        with open(self.cache_file, 'w') as f:
            f.write(current_hash)
        
        logger.info("Schema changes detected")
        return True
    
    def _calculate_schema_hash(self) -> str:
        """Calculate hash of all schema files."""
        hasher = hashlib.md5()
        
        # Hash main schema
        if self.schema_file.exists():
            hasher.update(self.schema_file.read_bytes())
        
        # Hash component schemas
        for comp_type in ["parsers", "extractors", "embedders", "stores", "retrievers"]:
            comp_dir = self.components_dir / comp_type
            if comp_dir.exists():
                for schema_file in comp_dir.glob("*/schema.yaml"):
                    hasher.update(schema_file.read_bytes())
        
        # Hash strategy files
        for config_file in self.config_samples_dir.glob("*.yaml"):
            hasher.update(config_file.read_bytes())
        
        return hasher.hexdigest()
    
    def run_validation(self, check_changes: bool = True) -> int:
        """Run full validation and return exit code."""
        logger.info("Starting schema validation...")
        
        # Check for changes if requested
        if check_changes and not self.check_for_changes():
            logger.info("No changes detected, skipping validation")
            return 0
        
        all_errors = []
        
        # Validate main schema
        main_errors = self.validate_main_schema()
        if main_errors:
            logger.error(f"Main schema validation errors:")
            for error in main_errors:
                logger.error(f"  - {error}")
            all_errors.extend(main_errors)
        
        # Validate component schemas
        component_errors = self.validate_component_schemas()
        if component_errors:
            logger.error("Component schema validation errors:")
            for file, errors in component_errors.items():
                logger.error(f"  {file}:")
                for error in errors:
                    logger.error(f"    - {error}")
            all_errors.extend([e for errors in component_errors.values() for e in errors])
        
        # Validate strategy configs
        strategy_errors = self.validate_strategy_configs()
        if strategy_errors:
            logger.error("Strategy configuration validation errors:")
            for file, errors in strategy_errors.items():
                logger.error(f"  {file}:")
                for error in errors:
                    logger.error(f"    - {error}")
            all_errors.extend([e for errors in strategy_errors.values() for e in errors])
        
        # Build consolidated schema if no errors
        if not all_errors:
            logger.info("Building consolidated schema...")
            consolidated = self.build_consolidated_schema()
            
            # Save consolidated schema
            output_file = self.rag_root / "schemas" / "consolidated.yaml"
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w') as f:
                yaml.dump(consolidated, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Consolidated schema saved to {output_file}")
            
            # Also save as JSON
            json_file = output_file.with_suffix('.json')
            with open(json_file, 'w') as f:
                json.dump(consolidated, f, indent=2)
            
            logger.info(f"Consolidated schema also saved as JSON to {json_file}")
        
        # Print summary
        if all_errors:
            logger.error(f"\n❌ Validation failed with {len(all_errors)} errors")
            return 1
        else:
            logger.info("\n✅ All schemas validated successfully!")
            return 0


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Validate and build RAG strategy schemas"
    )
    parser.add_argument(
        "--rag-root",
        type=Path,
        help="Path to RAG root directory",
        default=Path(__file__).parent.parent
    )
    parser.add_argument(
        "--skip-cache-check",
        action="store_true",
        help="Skip checking for changes and always validate"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = SchemaValidator(args.rag_root)
    exit_code = validator.run_validation(check_changes=not args.skip_cache_check)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()