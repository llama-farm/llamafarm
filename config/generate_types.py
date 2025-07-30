#!/usr/bin/env python3
"""
Script to generate TypedDict classes from the JSON schema.
This ensures the type definitions stay in sync with the schema.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set


class TypeGenerator:
    """Generate TypedDict classes from JSON schema."""

    def __init__(self, schema_path: Path):
        self.schema_path = schema_path
        self.schema = self._load_schema()
        self.generated_types: List[str] = []
        self.imports: Set[str] = set()

    def _load_schema(self) -> Dict[str, Any]:
        """Load the JSON schema from file."""
        import yaml

        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_python_type(self, schema_type: str, format_type: Optional[str] = None) -> str:
        """Convert JSON schema type to Python type."""
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "List",
            "object": "Dict[str, Any]"
        }
        return type_mapping.get(schema_type, "Any")

    def _get_literal_type(self, enum_values: List[Any]) -> str:
        """Generate Literal type from enum values."""
        if not enum_values:
            return "str"

        # Convert enum values to string literals
        literal_values = []
        for value in enum_values:
            if isinstance(value, str):
                literal_values.append(f'"{value}"')
            else:
                literal_values.append(str(value))

        return f"Literal[{', '.join(literal_values)}]"

    def _generate_field_type(self, property_schema: Dict[str, Any], field_name: str, parent_context: str = "") -> str:
        """Generate type annotation for a single field."""
        schema_type = property_schema.get("type")

        # Handle enum types
        if "enum" in property_schema:
            return self._get_literal_type(property_schema["enum"])

        # Handle array types
        if schema_type == "array":
            items_schema = property_schema.get("items", {})
            if items_schema.get("type") == "string":
                return "List[str]"
            elif items_schema.get("type") == "integer":
                return "List[int]"
            elif items_schema.get("type") == "object" and "properties" in items_schema:
                # Generate specific type for array items
                item_type = self._get_array_item_type_name(field_name)
                return f"List[{item_type}]"
            else:
                return "List[Any]"

        # Handle object types
        if schema_type == "object":
            if "properties" in property_schema:
                # Generate specific type name for this object
                object_type = self._get_object_type_name(field_name, parent_context)
                return object_type
            else:
                return "Dict[str, Any]"

        # Handle basic types
        if schema_type:
            return self._get_python_type(schema_type)

        # Default to Any if type is not specified
        return "Any"

    def _get_array_item_type_name(self, field_name: str) -> str:
        """Generate type name for array items."""
        # Convert field name to PascalCase for type name
        if field_name == "prompts":
            return "PromptConfig"
        elif field_name == "models":
            return "ModelConfig"
        else:
            return f"{field_name.capitalize()}Config"

    def _get_object_type_name(self, field_name: str, parent_context: str = "") -> str:
        """Generate type name for object fields."""
        # Convert field name to PascalCase for type name
        if field_name == "rag":
            return "RAGConfig"
        elif field_name == "parser":
            return "Parser"
        elif field_name == "embedder":
            return "Embedder"
        elif field_name == "vector_store":
            return "VectorStore"
        elif field_name == "config":
            # Use specific config type based on parent context
            if parent_context == "parser":
                return "ParserConfig"
            elif parent_context == "embedder":
                return "EmbedderConfig"
            elif parent_context == "vector_store":
                return "VectorStoreConfig"
            else:
                return "Dict[str, Any]"  # Fallback to generic type
        else:
            return f"{field_name.capitalize()}Config"

    def _generate_typeddict_class(self, class_name: str, properties: Dict[str, Any], required_fields: List[str], parent_context: str = "") -> str:
        """Generate a TypedDict class definition."""
        lines = [f'class {class_name}(TypedDict):']

        # Add docstring based on class name
        if class_name == "PromptConfig":
            lines.append('    """Configuration for a single prompt."""')
        elif class_name == "ModelConfig":
            lines.append('    """Configuration for a single model."""')
        elif class_name == "RAGConfig":
            lines.append('    """RAG (Retrieval-Augmented Generation) configuration."""')
        elif class_name == "ParserConfig":
            lines.append('    """Parser configuration within RAG."""')
        elif class_name == "EmbedderConfig":
            lines.append('    """Embedder configuration within RAG."""')
        elif class_name == "VectorStoreConfig":
            lines.append('    """Vector store configuration within RAG."""')
        elif class_name == "Parser":
            lines.append('    """Parser definition in RAG configuration."""')
        elif class_name == "Embedder":
            lines.append('    """Embedder definition in RAG configuration."""')
        elif class_name == "VectorStore":
            lines.append('    """Vector store definition in RAG configuration."""')
        elif class_name == "LlamaFarmConfig":
            lines.append('    """Complete LlamaFarm configuration."""')

        # Generate field definitions
        for field_name, field_schema in properties.items():
            field_type = self._generate_field_type(field_schema, field_name, parent_context)

            # Check if field is required
            if field_name in required_fields:
                lines.append(f'    {field_name}: {field_type}')
            else:
                lines.append(f'    {field_name}: Optional[{field_type}]')

        return '\n'.join(lines)

    def generate_types(self) -> str:
        """Generate all TypedDict classes from the schema."""
        self.imports.add("from typing import TypedDict, List, Literal, Optional, Union")

        # Generate types in the correct order to avoid forward references
        # This is based on the known structure of the LlamaFarm schema
        types = []

        # 1. Generate PromptConfig (no dependencies)
        if "prompts" in self.schema.get("properties", {}):
            prompts_schema = self.schema["properties"]["prompts"]
            if "items" in prompts_schema and "properties" in prompts_schema["items"]:
                prompt_props = prompts_schema["items"]["properties"]
                prompt_required = prompts_schema["items"].get("required", [])
                prompt_class = self._generate_typeddict_class("PromptConfig", prompt_props, prompt_required)
                types.append(prompt_class)

        # 2. Generate ModelConfig (no dependencies)
        if "models" in self.schema.get("properties", {}):
            models_schema = self.schema["properties"]["models"]
            if "items" in models_schema and "properties" in models_schema["items"]:
                model_props = models_schema["items"]["properties"]
                model_required = models_schema["items"].get("required", [])
                model_class = self._generate_typeddict_class("ModelConfig", model_props, model_required)
                types.append(model_class)

        # 3. Generate RAG component config types (no dependencies)
        if "rag" in self.schema.get("properties", {}):
            rag_schema = self.schema["properties"]["rag"]
            if "properties" in rag_schema:
                # Generate config types first
                for component_name in ["parser", "embedder", "vector_store"]:
                    if component_name in rag_schema["properties"]:
                        component_schema = rag_schema["properties"][component_name]
                        if "properties" in component_schema and "config" in component_schema["properties"]:
                            config_schema = component_schema["properties"]["config"]
                            if "properties" in config_schema:
                                config_props = config_schema["properties"]
                                config_required = config_schema.get("required", [])

                                # Generate proper config class name
                                if component_name == "vector_store":
                                    config_class_name = "VectorStoreConfig"
                                else:
                                    config_class_name = f"{component_name.capitalize()}Config"

                                config_class = self._generate_typeddict_class(config_class_name, config_props, config_required, component_name)
                                types.append(config_class)

        # 4. Generate RAG component types (depend on config types)
        if "rag" in self.schema.get("properties", {}):
            rag_schema = self.schema["properties"]["rag"]
            if "properties" in rag_schema:
                for component_name in ["parser", "embedder", "vector_store"]:
                    if component_name in rag_schema["properties"]:
                        component_schema = rag_schema["properties"][component_name]
                        if "properties" in component_schema:
                            component_props = component_schema["properties"]
                            component_required = component_schema.get("required", [])

                            # Generate proper component class name
                            if component_name == "vector_store":
                                component_class_name = "VectorStore"
                            else:
                                component_class_name = component_name.capitalize()

                            component_class = self._generate_typeddict_class(component_class_name, component_props, component_required, component_name)
                            types.append(component_class)

        # 5. Generate RAGConfig (depends on component types)
        if "rag" in self.schema.get("properties", {}):
            rag_schema = self.schema["properties"]["rag"]
            if "properties" in rag_schema:
                rag_props = rag_schema["properties"]
                rag_required = rag_schema.get("required", [])
                rag_class = self._generate_typeddict_class("RAGConfig", rag_props, rag_required)
                types.append(rag_class)

        # 6. Generate main LlamaFarmConfig (depends on all other types)
        main_props = self.schema.get("properties", {})
        main_required = self.schema.get("required", [])
        main_class = self._generate_typeddict_class("LlamaFarmConfig", main_props, main_required)
        types.append(main_class)

        # Add ConfigDict alias
        types.append("# Type alias for the configuration dictionary")
        types.append("ConfigDict = Union[LlamaFarmConfig, dict]")

        # Combine all generated code
        imports_code = '\n'.join(sorted(self.imports))
        types_code = '\n\n'.join(types)

        return f'''"""
Type definitions for LlamaFarm configuration based on the JSON schema.
This file is auto-generated from schema.yaml - DO NOT EDIT MANUALLY.
"""

{imports_code}


{types_code}
'''

    def write_types_file(self, output_path: Path) -> None:
        """Write the generated types to a file."""
        content = self.generate_types()

        with open(output_path, 'w') as f:
            f.write(content)

        print(f"✅ Generated types file: {output_path}")
        print(f"📝 Generated {len(self.generated_types)} type definitions")


def main():
    """Generate types from schema and write to config_types.py."""
    script_dir = Path(__file__).parent
    schema_path = script_dir / "schema.yaml"
    output_path = script_dir / "config_types.py"

    try:
        # Generate types
        generator = TypeGenerator(schema_path)
        generator.write_types_file(output_path)

        print("\n🎉 Type generation completed successfully!")
        print(f"📁 Schema: {schema_path}")
        print(f"📁 Output: {output_path}")

        # Verify the generated file can be imported
        print("\n🔍 Verifying generated types...")
        sys.path.insert(0, str(script_dir))

        try:
            from config_types import LlamaFarmConfig, ConfigDict
            print("✅ Generated types can be imported successfully!")
        except ImportError as e:
            print(f"❌ Error importing generated types: {e}")
            return 1

        return 0

    except Exception as e:
        print(f"❌ Error generating types: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())