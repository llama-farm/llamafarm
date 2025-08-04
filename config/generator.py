import os
from typing import Optional
import yaml
from pathlib import Path

from config.config_types import ConfigDict

def generate_base_config(schema_path: Optional[str] = None) -> ConfigDict:
    """
    Generate a base config file using the JSON schema from schema.yaml
    and return it as a ConfigDict.
    """

    if schema_path is None:
        schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.yaml")

    # Load the schema
    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)
        print(schema)

    def build_default(schema_node):
        print(schema_node)
        """Recursively build a default config from a JSON schema node."""
        if "default" in schema_node:
            return schema_node["default"]
        if "type" in schema_node:
            t = schema_node["type"]
            if t == "object":
                result = {}
                properties = schema_node.get("properties", {})
                for key, prop in properties.items():
                    result[key] = build_default(prop)
                return result
            elif t == "array":
                # If items have a default, use it, else empty list
                items = schema_node.get("items", {})
                if "default" in schema_node:
                    return schema_node["default"]
                elif items:
                    # Try to build a single default item if possible
                    item_default = build_default(items)
                    if item_default is not None:
                        return [item_default]
                return []
            elif t == "string":
                return ""
            elif t == "integer":
                return 0
            elif t == "number":
                return 0.0
            elif t == "boolean":
                return False
        # Fallback: None
        return None

    # The root schema is expected to be an object
    base_config = build_default(schema)

    return base_config

# Example usage:
# generate_base_config_from_schema("schema.yaml", "llamafarm.yaml")
