#!/usr/bin/env python3
"""
Smart Schema Validator for RAG System
Provides detailed validation, diagnostics, and suggestions for configuration files.
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import difflib
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ValidationIssue:
    """Represents a validation issue with context and suggestions"""

    def __init__(
        self,
        level: str,
        path: str,
        message: str,
        suggestion: str = None,
        example: str = None,
    ):
        self.level = level  # ERROR, WARNING, INFO
        self.path = path
        self.message = message
        self.suggestion = suggestion
        self.example = example

    def __str__(self):
        level_colors = {"ERROR": "\033[91m", "WARNING": "\033[93m", "INFO": "\033[94m"}
        reset = "\033[0m"

        output = f"{level_colors.get(self.level, '')}[{self.level}]{reset} "
        output += f"at '{self.path}': {self.message}"

        if self.suggestion:
            output += f"\n  💡 Suggestion: {self.suggestion}"

        if self.example:
            output += f"\n  📝 Example:\n{self.example}"

        return output


class SmartSchemaValidator:
    """Intelligent schema validator with helpful diagnostics"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.base_path = Path(__file__).parent.parent
        self.main_schema = self._load_main_schema()
        self.component_schemas = self._load_component_schemas()
        self.issues: List[ValidationIssue] = []
        self.suggestions: List[str] = []

    def _load_main_schema(self) -> Dict[str, Any]:
        """Load the main schema.yaml"""
        schema_path = self.base_path / "schema.yaml"
        if not schema_path.exists():
            return {}

        with open(schema_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_component_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load all component schemas"""
        components = {}

        # First try to load consolidated schemas
        consolidated_path = self.base_path / "schemas" / "consolidated.yaml"
        if consolidated_path.exists():
            with open(consolidated_path, "r") as f:
                consolidated = yaml.safe_load(f) or {}

                # Extract component schemas from consolidated file
                for comp_type in [
                    "parsers",
                    "extractors",
                    "embedders",
                    "stores",
                    "retrievers",
                ]:
                    if comp_type in consolidated:
                        for comp_name, comp_data in consolidated[comp_type].items():
                            comp_key = f"{comp_type}/{comp_name.lower()}"
                            components[comp_key] = comp_data.get("config_schema", {})

                            # Also store defaults for suggestions
                            if "defaults" in comp_data:
                                components[f"{comp_key}_defaults"] = comp_data[
                                    "defaults"
                                ]

        # Fallback to individual component schemas
        else:
            components_dir = self.base_path / "components"

            for comp_type in [
                "parsers",
                "extractors",
                "embedders",
                "stores",
                "retrievers",
            ]:
                type_dir = components_dir / comp_type
                if not type_dir.exists():
                    continue

                for comp_dir in type_dir.iterdir():
                    if comp_dir.is_dir():
                        schema_path = comp_dir / "schema.yaml"
                        if schema_path.exists():
                            with open(schema_path, "r") as f:
                                schema = yaml.safe_load(f) or {}
                                comp_key = f"{comp_type}/{comp_dir.name}"
                                components[comp_key] = schema

        return components

    def validate_file(self, file_path: str) -> Tuple[bool, List[ValidationIssue]]:
        """Validate a configuration file"""
        self.issues = []
        path = Path(file_path)

        if not path.exists():
            self.issues.append(
                ValidationIssue(
                    "ERROR",
                    file_path,
                    f"File not found",
                    f"Check the file path or create a new configuration file",
                )
            )
            return False, self.issues

        # Load the configuration
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            self.issues.append(
                ValidationIssue(
                    "ERROR",
                    file_path,
                    f"Invalid YAML syntax: {e}",
                    "Fix the YAML syntax errors. Use a YAML linter or validator.",
                )
            )
            return False, self.issues

        # Determine file type and validate accordingly
        if "strategies" in config:
            self._validate_strategy_config(config)
        elif "components" in config:
            self._validate_pipeline_config(config)
        else:
            self._validate_generic_config(config, path.name)

        return len([i for i in self.issues if i.level == "ERROR"]) == 0, self.issues

    def _validate_strategy_config(self, config: Dict[str, Any]):
        """Validate a strategy configuration file"""
        strategies = config.get("strategies", [])

        if not strategies:
            self.issues.append(
                ValidationIssue(
                    "WARNING",
                    "strategies",
                    "No strategies defined",
                    "Add at least one strategy configuration",
                    self._get_strategy_example(),
                )
            )
            return

        for idx, strategy in enumerate(strategies):
            self._validate_single_strategy(strategy, f"strategies[{idx}]")

    def _validate_single_strategy(self, strategy: Dict[str, Any], path: str):
        """Validate a single strategy configuration"""

        # Check required fields
        required_fields = ["name", "description", "components"]
        for field in required_fields:
            if field not in strategy:
                self.issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"{path}.{field}",
                        f"Required field '{field}' is missing",
                        f"Add the '{field}' field to the strategy",
                        self._get_field_example(field),
                    )
                )

        # Validate name format
        if "name" in strategy:
            name = strategy["name"]
            if not isinstance(name, str):
                self.issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"{path}.name",
                        f"Name must be a string, got {type(name).__name__}",
                        "Use a valid string identifier",
                    )
                )
            elif not name or not name[0].islower() or " " in name:
                self.issues.append(
                    ValidationIssue(
                        "WARNING",
                        f"{path}.name",
                        f"Name '{name}' doesn't follow naming convention",
                        "Use lowercase with underscores (e.g., 'my_strategy')",
                    )
                )

        # Validate components
        if "components" in strategy:
            self._validate_components(strategy["components"], f"{path}.components")

    def _validate_components(self, components: Dict[str, Any], path: str):
        """Validate strategy components configuration"""

        required_components = [
            "parser",
            "embedder",
            "vector_store",
            "retrieval_strategy",
        ]

        for comp in required_components:
            if comp not in components:
                self.issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"{path}.{comp}",
                        f"Required component '{comp}' is missing",
                        f"Add a {comp} configuration",
                        self._get_component_example(comp),
                    )
                )
            else:
                self._validate_component(components[comp], comp, f"{path}.{comp}")

        # Validate optional extractors
        if "extractors" in components:
            if not isinstance(components["extractors"], list):
                self.issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"{path}.extractors",
                        "Extractors must be a list",
                        "Change extractors to a list format",
                    )
                )
            else:
                for idx, extractor in enumerate(components["extractors"]):
                    self._validate_component(
                        extractor, "extractor", f"{path}.extractors[{idx}]"
                    )

    def _validate_component(self, component: Dict[str, Any], comp_type: str, path: str):
        """Validate a single component configuration"""

        if not isinstance(component, dict):
            self.issues.append(
                ValidationIssue(
                    "ERROR",
                    path,
                    f"Component must be a dictionary, got {type(component).__name__}",
                    "Use proper dictionary format for component configuration",
                )
            )
            return

        # Check for type field
        if "type" not in component:
            self.issues.append(
                ValidationIssue(
                    "ERROR",
                    f"{path}.type",
                    "Component type is missing",
                    f"Add a 'type' field specifying the {comp_type} type",
                    self._suggest_component_type(comp_type),
                )
            )
            return

        comp_type_value = component["type"]

        # Validate against known component types
        valid_types = self._get_valid_component_types(comp_type)
        if comp_type_value not in valid_types:
            suggestion = self._find_closest_match(comp_type_value, valid_types)
            self.issues.append(
                ValidationIssue(
                    "ERROR",
                    f"{path}.type",
                    f"Unknown {comp_type} type: '{comp_type_value}'",
                    f"Did you mean '{suggestion}'?"
                    if suggestion
                    else f"Use one of: {', '.join(valid_types[:5])}",
                )
            )

        # Validate component config if present
        if "config" in component:
            self._validate_component_config(
                component["config"], comp_type_value, comp_type, f"{path}.config"
            )

    def _validate_component_config(
        self,
        config: Dict[str, Any],
        comp_type_value: str,
        comp_category: str,
        path: str,
    ):
        """Validate component-specific configuration"""

        # Find the schema for this component
        schema_key = self._find_component_schema_key(comp_type_value, comp_category)
        if not schema_key:
            return

        component_schema = self.component_schemas.get(schema_key, {})
        if not component_schema:
            return

        schema_properties = component_schema.get("properties", {})

        # Check for unknown fields
        for field in config:
            if field not in schema_properties:
                valid_fields = list(schema_properties.keys())
                suggestion = self._find_closest_match(field, valid_fields)
                self.issues.append(
                    ValidationIssue(
                        "WARNING",
                        f"{path}.{field}",
                        f"Unknown configuration field '{field}'",
                        f"Did you mean '{suggestion}'?"
                        if suggestion
                        else f"Valid fields: {', '.join(valid_fields[:5])}",
                    )
                )

        # Validate field types and values
        for field, schema_def in schema_properties.items():
            if field in config:
                self._validate_field_value(config[field], schema_def, f"{path}.{field}")
            elif schema_def.get("required", False):
                self.issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"{path}.{field}",
                        f"Required field '{field}' is missing",
                        f"Add '{field}' with type {schema_def.get('type', 'any')}",
                    )
                )

    def _validate_field_value(self, value: Any, schema: Dict[str, Any], path: str):
        """Validate a field value against its schema"""

        expected_type = schema.get("type")
        if not expected_type:
            return

        # Handle multiple types
        if isinstance(expected_type, list):
            valid_types = expected_type
        else:
            valid_types = [expected_type]

        # Type mapping from JSON Schema to Python
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        # Check type
        valid = False
        for valid_type in valid_types:
            if valid_type in type_map:
                python_type = type_map[valid_type]
                if isinstance(python_type, tuple):
                    if isinstance(value, python_type):
                        valid = True
                        break
                else:
                    if isinstance(value, python_type):
                        valid = True
                        break

        if not valid and "null" not in valid_types:
            self.issues.append(
                ValidationIssue(
                    "ERROR",
                    path,
                    f"Invalid type: expected {'/'.join(valid_types)}, got {type(value).__name__}",
                    f"Change to {valid_types[0]} type",
                )
            )
            return

        # Additional validations
        if "enum" in schema and value not in schema["enum"]:
            self.issues.append(
                ValidationIssue(
                    "ERROR",
                    path,
                    f"Invalid value '{value}', must be one of: {', '.join(map(str, schema['enum']))}",
                    f"Use one of the allowed values",
                )
            )

        if (
            "minimum" in schema
            and isinstance(value, (int, float))
            and value < schema["minimum"]
        ):
            self.issues.append(
                ValidationIssue(
                    "ERROR",
                    path,
                    f"Value {value} is below minimum {schema['minimum']}",
                    f"Use a value >= {schema['minimum']}",
                )
            )

        if (
            "maximum" in schema
            and isinstance(value, (int, float))
            and value > schema["maximum"]
        ):
            self.issues.append(
                ValidationIssue(
                    "ERROR",
                    path,
                    f"Value {value} exceeds maximum {schema['maximum']}",
                    f"Use a value <= {schema['maximum']}",
                )
            )

        if "pattern" in schema and isinstance(value, str):
            import re

            if not re.match(schema["pattern"], value):
                self.issues.append(
                    ValidationIssue(
                        "ERROR",
                        path,
                        f"Value '{value}' doesn't match pattern {schema['pattern']}",
                        f"Format the value according to the pattern",
                    )
                )

    def _get_valid_component_types(self, category: str) -> List[str]:
        """Get valid component types for a category"""

        type_map = {
            "parser": [
                "CSVParser",
                "PDFParser",
                "MarkdownParser",
                "HTMLParser",
                "DocxParser",
                "ExcelParser",
                "PlainTextParser",
            ],
            "extractor": [
                "KeywordExtractor",
                "EntityExtractor",
                "DateTimeExtractor",
                "HeadingExtractor",
                "LinkExtractor",
                "PathExtractor",
                "PatternExtractor",
                "StatisticsExtractor",
                "SummaryExtractor",
                "TableExtractor",
            ],
            "embedder": [
                "OllamaEmbedder",
                "OpenAIEmbedder",
                "HuggingFaceEmbedder",
                "SentenceTransformerEmbedder",
            ],
            "vector_store": [
                "ChromaStore",
                "FAISSStore",
                "PineconeStore",
                "QdrantStore",
            ],
            "retrieval_strategy": [
                "BasicSimilarityStrategy",
                "MetadataFilteredStrategy",
                "MultiQueryStrategy",
                "RerankedStrategy",
                "HybridUniversalStrategy",
            ],
        }

        return type_map.get(category, [])

    def _find_closest_match(self, value: str, valid_values: List[str]) -> Optional[str]:
        """Find the closest matching value using fuzzy matching"""
        if not valid_values:
            return None

        matches = difflib.get_close_matches(value, valid_values, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def _find_component_schema_key(
        self, comp_type: str, category: str
    ) -> Optional[str]:
        """Find the schema key for a component"""

        # Map category to directory name
        category_map = {
            "parser": "parsers",
            "extractor": "extractors",
            "embedder": "embedders",
            "vector_store": "stores",
            "retrieval_strategy": "retrievers",
        }

        dir_name = category_map.get(category)
        if not dir_name:
            return None

        # Convert component type to directory name
        comp_name = comp_type.lower().replace("strategy", "").replace("store", "_store")
        comp_name = comp_name.replace("parser", "_parser").replace(
            "extractor", "_extractor"
        )
        comp_name = comp_name.replace("embedder", "_embedder")

        # Clean up double underscores
        comp_name = comp_name.replace("__", "_").strip("_")

        # Special cases
        special_cases = {
            "chromastore": "chroma_store",
            "faissstore": "faiss_store",
            "pineconestore": "pinecone_store",
            "qdrantstore": "qdrant_store",
            "basicsimilarity": "basic_similarity",
            "metadatafiltered": "metadata_filtered",
            "multiquery": "multi_query",
            "reranked": "reranked",
            "hybriduniversal": "hybrid_universal",
            "plaintext_parser": "text_parser",
        }

        comp_name = special_cases.get(comp_type.lower(), comp_name)

        return f"{dir_name}/{comp_name}"

    def _validate_pipeline_config(self, config: Dict[str, Any]):
        """Validate a pipeline configuration"""
        # Implementation for pipeline configs
        pass

    def _validate_generic_config(self, config: Dict[str, Any], filename: str):
        """Validate a generic configuration file"""
        # Implementation for other config types
        pass

    def _get_strategy_example(self) -> str:
        """Get an example strategy configuration"""
        return """
    strategies:
      - name: simple_rag
        description: "Basic RAG pipeline for general documents"
        components:
          parser:
            type: PDFParser
            config:
              combine_pages: true
          embedder:
            type: OllamaEmbedder
            config:
              model: nomic-embed-text
          vector_store:
            type: ChromaStore
            config:
              collection_name: documents
          retrieval_strategy:
            type: BasicSimilarityStrategy
            config:
              top_k: 10"""

    def _get_field_example(self, field: str) -> str:
        """Get an example for a specific field"""
        examples = {
            "name": 'name: "my_strategy"',
            "description": 'description: "A strategy for processing documents"',
            "components": "components:\n  parser:\n    type: PDFParser",
        }
        return examples.get(field, f"{field}: <value>")

    def _get_component_example(self, comp_type: str) -> str:
        """Get an example component configuration from defaults"""

        # Try to get from component defaults first
        comp_type_plural = {
            "parser": "parsers",
            "embedder": "embedders",
            "vector_store": "stores",
            "retrieval_strategy": "retrievers",
            "extractor": "extractors",
        }.get(comp_type, comp_type + "s")

        # Look for defaults in loaded schemas
        for key in self.component_schemas:
            if key.startswith(f"{comp_type_plural}/") and key.endswith("_defaults"):
                defaults = self.component_schemas[key]
                if "general_purpose" in defaults:
                    config = defaults["general_purpose"].get("config", {})
                    comp_name = key.replace(f"{comp_type_plural}/", "").replace(
                        "_defaults", ""
                    )

                    # Format as YAML
                    example = f"    {comp_type}:\n"
                    example += f"      type: {comp_name.title().replace('_', '')}\n"
                    if config:
                        example += "      config:\n"
                        for k, v in list(config.items())[
                            :3
                        ]:  # Show first 3 config items
                            if isinstance(v, str):
                                example += f'        {k}: "{v}"\n'
                            else:
                                example += f"        {k}: {v}\n"
                    return example

        # Fallback to hardcoded examples
        examples = {
            "parser": """
    parser:
      type: PDFParser
      config:
        combine_pages: true
        extract_metadata: true""",
            "embedder": """
    embedder:
      type: OllamaEmbedder
      config:
        model: nomic-embed-text
        dimension: 768""",
            "vector_store": """
    vector_store:
      type: ChromaStore
      config:
        collection_name: documents
    retrieval_strategy:
      type: BasicSimilarityStrategy
      config:
        top_k: 10
        distance_metric: cosine""",
        }
        return examples.get(comp_type, f"{comp_type}:\n  type: <ComponentType>")

    def _suggest_component_type(self, category: str) -> str:
        """Suggest a component type for a category"""
        suggestions = {
            "parser": "type: PDFParser  # or CSVParser, MarkdownParser, etc.",
            "embedder": "type: OllamaEmbedder  # or OpenAIEmbedder, HuggingFaceEmbedder",
            "vector_store": "type: ChromaStore  # or FAISSStore, PineconeStore, QdrantStore",
            "retrieval_strategy": "type: BasicSimilarityStrategy  # or MetadataFilteredStrategy, etc.",
        }
        return suggestions.get(category, f"type: <{category.title()}Type>")

    def suggest_improvements(self, config: Dict[str, Any]) -> List[str]:
        """Suggest improvements for a configuration"""
        suggestions = []

        # Check for optimization opportunities
        if "strategies" in config:
            for strategy in config.get("strategies", []):
                components = strategy.get("components", {})

                # Suggest hybrid retrieval for better results
                if (
                    components.get("retrieval_strategy", {}).get("type")
                    == "BasicSimilarityStrategy"
                ):
                    suggestions.append(
                        "💡 Consider using HybridUniversalStrategy for better retrieval results"
                    )

                # Suggest extractors for metadata enrichment
                if "extractors" not in components:
                    suggestions.append(
                        "💡 Add extractors to enrich document metadata (e.g., KeywordExtractor, EntityExtractor)"
                    )

                # Suggest optimal embedder based on use case
                embedder = components.get("embedder", {})
                if embedder.get("type") == "OllamaEmbedder":
                    suggestions.append(
                        "💡 For production, consider OpenAIEmbedder for better quality (requires API key)"
                    )

        return suggestions

    def generate_report(self, file_path: str) -> str:
        """Generate a comprehensive validation report"""
        is_valid, issues = self.validate_file(file_path)

        report = []
        report.append("=" * 60)
        report.append(f"SCHEMA VALIDATION REPORT")
        report.append(f"File: {file_path}")
        report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")

        # Summary
        error_count = len([i for i in issues if i.level == "ERROR"])
        warning_count = len([i for i in issues if i.level == "WARNING"])
        info_count = len([i for i in issues if i.level == "INFO"])

        if is_valid:
            report.append("✅ VALIDATION PASSED")
        else:
            report.append("❌ VALIDATION FAILED")

        report.append("")
        report.append(f"Issues Found:")
        report.append(f"  • Errors:   {error_count}")
        report.append(f"  • Warnings: {warning_count}")
        report.append(f"  • Info:     {info_count}")
        report.append("")

        # Detailed issues
        if issues:
            report.append("-" * 60)
            report.append("DETAILED ISSUES:")
            report.append("-" * 60)

            for issue in issues:
                report.append(str(issue))
                report.append("")

        # Suggestions
        try:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f) or {}
            suggestions = self.suggest_improvements(config)

            if suggestions:
                report.append("-" * 60)
                report.append("IMPROVEMENT SUGGESTIONS:")
                report.append("-" * 60)
                for suggestion in suggestions:
                    report.append(suggestion)
                report.append("")
        except:
            pass

        report.append("=" * 60)

        return "\n".join(report)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Smart Schema Validator for RAG configurations"
    )
    parser.add_argument("file", help="Configuration file to validate")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--fix", "-f", action="store_true", help="Attempt to auto-fix common issues"
    )
    parser.add_argument("--output", "-o", help="Output report to file")

    args = parser.parse_args()

    # Create validator
    validator = SmartSchemaValidator(verbose=args.verbose)

    # Generate report
    report = validator.generate_report(args.file)

    # Output report
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")

    # Exit with appropriate code
    is_valid, _ = validator.validate_file(args.file)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
