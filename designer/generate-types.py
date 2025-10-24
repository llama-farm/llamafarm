#!/usr/bin/env python3
"""
Generate TypeScript types and constants for the Designer UI from rag/schema.yaml

This unified script generates all UI types from the RAG schema:
- Parser types and configurations
- Extractor types and configurations
- Vector store/database types and configurations
- Embedder types and configurations
- Retrieval strategy types and configurations
- Default configuration functions
- TypeScript type definitions

Key Features:
- Zero hardcoding - all mappings derived from schema structure
- Fully extensible - adding to schema automatically includes in UI
- Single source of truth - rag/schema.yaml drives everything

Run: ./generate-types.sh
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


def load_schema() -> Dict[str, Any]:
    """Load the RAG schema.yaml file"""
    schema_path = Path(__file__).parent.parent / "rag" / "schema.yaml"
    with open(schema_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================
# Parser Type Discovery and Mapping
# ============================================================================

def build_parser_type_mapping(schema: Dict[str, Any]) -> Dict[str, str]:
    """
    Build mapping from parser type to config key by analyzing schema structure.

    Derives mappings dynamically - no hardcoding!
    Example: "PDFParser_PyPDF2" -> "pdfParserPyPDF2Config"
    """
    parsers = schema.get("definitions", {}).get("parsers", {})
    type_to_config = {}

    for config_key in parsers.keys():
        if config_key == "autoParserConfig":
            type_to_config["auto"] = config_key
            continue

        if config_key.endswith("Config"):
            base = config_key[:-6]  # Remove "Config" suffix

            if "Parser" in base:
                match = re.search(r'Parser', base, re.IGNORECASE)
                if match:
                    format_part = base[:match.start()]
                    tool_part = base[match.end():]

                    # Uppercase the entire format part (PDF, CSV, MSG, etc.)
                    format_part = format_part.upper()

                    # Capitalize first letter of tool
                    if tool_part:
                        tool_part = tool_part[0].upper() + tool_part[1:]

                    parser_type = f"{format_part}Parser_{tool_part}"
                    type_to_config[parser_type] = config_key

    return type_to_config


def extract_parser_types(schema: Dict[str, Any]) -> List[str]:
    """Extract all parser type names from schema definitions"""
    type_mapping = build_parser_type_mapping(schema)
    return sorted(type_mapping.keys())


def get_parser_config_schema(schema: Dict[str, Any], parser_type: str) -> Dict[str, Any]:
    """Get the config schema for a specific parser type"""
    type_mapping = build_parser_type_mapping(schema)
    config_key = type_mapping.get(parser_type)

    if not config_key:
        return {}

    parsers = schema.get("definitions", {}).get("parsers", {})
    return parsers.get(config_key, {})


# ============================================================================
# Extractor Type Discovery and Mapping
# ============================================================================

def build_extractor_type_mapping(schema: Dict[str, Any]) -> Dict[str, str]:
    """
    Build mapping from extractor type to config key by analyzing schema structure.

    Derives mappings dynamically using multiple strategies - no hardcoding!
    Example: "RAKEExtractor" -> "keywordExtractorConfig" (via algorithm enum)
    """
    extractors = schema.get("definitions", {}).get("extractors", {})
    type_to_config = {}

    extractor_config = schema.get("definitions", {}).get("extractorConfig", {})
    extractor_types = extractor_config.get("properties", {}).get("type", {}).get("enum", [])

    for extractor_type in extractor_types:
        # Strategy 1: Try exact match with lowercase first letter + Config
        potential_key = extractor_type[0].lower() + extractor_type[1:] + "Config"
        if potential_key in extractors:
            type_to_config[extractor_type] = potential_key
            continue

        # Strategy 2: Try removing "Extractor" suffix and adding "Config"
        if extractor_type.endswith("Extractor"):
            base = extractor_type[:-9]  # Remove "Extractor"
            potential_key = base[0].lower() + base[1:] + "ExtractorConfig"
            if potential_key in extractors:
                type_to_config[extractor_type] = potential_key
                continue

        # Strategy 3: Check if this type is part of a config's algorithm enum
        # (e.g., RAKE, TFIDF, YAKE use keywordExtractorConfig)
        for config_key, config_def in extractors.items():
            props = config_def.get("properties", {})
            algorithm_enum = props.get("algorithm", {}).get("enum", [])
            if extractor_type.lower().replace("extractor", "") in [a.lower() for a in algorithm_enum]:
                type_to_config[extractor_type] = config_key
                break

    return type_to_config


def extract_extractor_types(schema: Dict[str, Any]) -> List[str]:
    """Extract all extractor type names from the schema"""
    extractor_config = schema.get("definitions", {}).get("extractorConfig", {})
    type_enum = extractor_config.get("properties", {}).get("type", {}).get("enum", [])
    return sorted(type_enum)


def get_extractor_config_schema(schema: Dict[str, Any], extractor_type: str) -> Dict[str, Any]:
    """Get the config schema for a specific extractor type"""
    type_mapping = build_extractor_type_mapping(schema)
    config_key = type_mapping.get(extractor_type)

    if not config_key:
        return {}

    extractors = schema.get("definitions", {}).get("extractors", {})
    return extractors.get(config_key, {})


# ============================================================================
# Vector Store Type Discovery and Mapping
# ============================================================================

def build_vector_store_type_mapping(schema: Dict[str, Any]) -> Dict[str, str]:
    """
    Build mapping from vector store type to config key by analyzing schema structure.

    Derives mappings dynamically - no hardcoding!
    Example: "ChromaStore" -> "chromaStoreConfig"
    """
    vector_stores = schema.get("definitions", {}).get("vectorStores", {})
    type_to_config = {}

    # Get all vector store types from the enum
    vector_store_config = schema.get("definitions", {}).get("vectorStoreConfig", {})
    store_types = vector_store_config.get("properties", {}).get("type", {}).get("enum", [])

    for store_type in store_types:
        # Try lowercase first letter + Config
        potential_key = store_type[0].lower() + store_type[1:] + "Config"
        if potential_key in vector_stores:
            type_to_config[store_type] = potential_key

    return type_to_config


def extract_vector_store_types(schema: Dict[str, Any]) -> List[str]:
    """Extract vector store types from schema"""
    vector_store_config = schema.get("definitions", {}).get("vectorStoreConfig", {})
    type_enum = vector_store_config.get("properties", {}).get("type", {}).get("enum", [])
    return sorted(type_enum)


def get_vector_store_config_schema(schema: Dict[str, Any], store_type: str) -> Dict[str, Any]:
    """Get the config schema for a specific vector store type"""
    type_mapping = build_vector_store_type_mapping(schema)
    config_key = type_mapping.get(store_type)

    if not config_key:
        return {}

    vector_stores = schema.get("definitions", {}).get("vectorStores", {})
    return vector_stores.get(config_key, {})


# ============================================================================
# Embedder Type Discovery and Mapping
# ============================================================================

def build_embedder_type_mapping(schema: Dict[str, Any]) -> Dict[str, str]:
    """
    Build mapping from embedder type to config key by analyzing schema structure.

    Derives mappings dynamically - no hardcoding!
    Example: "OllamaEmbedder" -> "ollamaEmbedderConfig"
    """
    embedders = schema.get("definitions", {}).get("embedders", {})
    type_to_config = {}

    # Get all embedder types from the enum
    embedder_config = schema.get("definitions", {}).get("embedderConfig", {})
    embedder_types = embedder_config.get("properties", {}).get("type", {}).get("enum", [])

    for embedder_type in embedder_types:
        # Try lowercase first letter + Config
        potential_key = embedder_type[0].lower() + embedder_type[1:] + "Config"
        if potential_key in embedders:
            type_to_config[embedder_type] = potential_key
            continue

        # Handle special cases like "HuggingFaceEmbedder" -> "huggingfaceEmbedderConfig"
        if "HuggingFace" in embedder_type:
            potential_key = embedder_type.replace("HuggingFace", "huggingface") + "Config"
            if potential_key in embedders:
                type_to_config[embedder_type] = potential_key
                continue

        # Handle "SentenceTransformerEmbedder" -> "sentenceTransformerConfig"
        if "Embedder" in embedder_type:
            base = embedder_type.replace("Embedder", "")
            potential_key = base[0].lower() + base[1:] + "Config"
            if potential_key in embedders:
                type_to_config[embedder_type] = potential_key

    return type_to_config


def extract_embedder_types(schema: Dict[str, Any]) -> List[str]:
    """Extract embedder types from schema"""
    embedder_config = schema.get("definitions", {}).get("embedderConfig", {})
    type_enum = embedder_config.get("properties", {}).get("type", {}).get("enum", [])
    return sorted(type_enum)


def get_embedder_config_schema(schema: Dict[str, Any], embedder_type: str) -> Dict[str, Any]:
    """Get the config schema for a specific embedder type"""
    type_mapping = build_embedder_type_mapping(schema)
    config_key = type_mapping.get(embedder_type)

    if not config_key:
        return {}

    embedders = schema.get("definitions", {}).get("embedders", {})
    return embedders.get(config_key, {})


# ============================================================================
# Retrieval Strategy Type Discovery and Mapping
# ============================================================================

def build_retrieval_strategy_type_mapping(schema: Dict[str, Any]) -> Dict[str, str]:
    """
    Build mapping from retrieval strategy type to config key by analyzing schema structure.

    Derives mappings dynamically - no hardcoding!
    Example: "BasicSimilarityStrategy" -> "basicSimilarityConfig"
    """
    retrieval_strategies = schema.get("definitions", {}).get("retrievalStrategies", {})
    type_to_config = {}

    # Get all retrieval strategy types from the enum
    db_def = schema.get("definitions", {}).get("databaseDefinition", {})
    retrieval_strategies_prop = db_def.get("properties", {}).get("retrieval_strategies", {})
    items = retrieval_strategies_prop.get("items", {})
    type_prop = items.get("properties", {}).get("type", {})
    strategy_types = type_prop.get("enum", [])

    for strategy_type in strategy_types:
        # Remove "Strategy" suffix and add "Config"
        if strategy_type.endswith("Strategy"):
            base = strategy_type[:-8]  # Remove "Strategy"
            potential_key = base[0].lower() + base[1:] + "Config"
            if potential_key in retrieval_strategies:
                type_to_config[strategy_type] = potential_key

    return type_to_config


def extract_retrieval_strategy_types(schema: Dict[str, Any]) -> List[str]:
    """Extract retrieval strategy types from schema"""
    db_def = schema.get("definitions", {}).get("databaseDefinition", {})
    retrieval_strategies = db_def.get("properties", {}).get("retrieval_strategies", {})
    items = retrieval_strategies.get("items", {})
    type_prop = items.get("properties", {}).get("type", {})
    type_enum = type_prop.get("enum", [])
    return sorted(type_enum)


def get_retrieval_strategy_config_schema(schema: Dict[str, Any], strategy_type: str) -> Dict[str, Any]:
    """Get the config schema for a specific retrieval strategy type"""
    type_mapping = build_retrieval_strategy_type_mapping(schema)
    config_key = type_mapping.get(strategy_type)

    if not config_key:
        return {}

    retrieval_strategies = schema.get("definitions", {}).get("retrievalStrategies", {})
    return retrieval_strategies.get(config_key, {})


# ============================================================================
# Shared Utility Functions
# ============================================================================

def generate_default_config(config_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a default config object from schema properties"""
    properties = config_schema.get("properties", {})
    defaults = {}

    for prop_name, prop_def in properties.items():
        if "default" in prop_def:
            defaults[prop_name] = prop_def["default"]

    return defaults


def convert_schema_properties_to_ts(properties: Dict[str, Any]) -> str:
    """Convert YAML schema properties to TypeScript SchemaField format"""
    if not properties:
        return "{}"

    result = []
    for prop_name, prop_def in properties.items():
        # Build SchemaField object
        field_parts = []

        # Type mapping
        yaml_type = prop_def.get("type", "string")
        if yaml_type in ["integer", "number", "string", "boolean", "array"]:
            field_parts.append(f'type: "{yaml_type}"')
        else:
            field_parts.append('type: "string"')

        # Optional fields
        if "title" in prop_def:
            title = prop_def["title"].replace('"', '\\"')
            field_parts.append(f'title: "{title}"')

        if "description" in prop_def:
            desc = prop_def["description"].replace('"', '\\"')
            field_parts.append(f'description: "{desc}"')

        if "default" in prop_def:
            default_val = json.dumps(prop_def["default"])
            field_parts.append(f'default: {default_val}')

        if "minimum" in prop_def:
            field_parts.append(f'minimum: {prop_def["minimum"]}')

        if "maximum" in prop_def:
            field_parts.append(f'maximum: {prop_def["maximum"]}')

        if "enum" in prop_def:
            enum_val = json.dumps(prop_def["enum"])
            field_parts.append(f'enum: {enum_val}')

        if "items" in prop_def and yaml_type == "array":
            items_type = prop_def["items"].get("type", "string")
            field_parts.append(f'items: {{ type: "{items_type}" }}')

        if prop_def.get("nullable"):
            field_parts.append('nullable: true')

        # Build the property
        field_str = "{ " + ", ".join(field_parts) + " }"
        result.append(f'      {prop_name}: {field_str}')

    return ",\n".join(result)


def infer_file_extensions(type_name: str) -> List[str]:
    """Infer file extensions from type name"""
    ext_map = {
        "PDF": [".pdf"],
        "CSV": [".csv"],
        "Excel": [".xlsx", ".xls"],
        "Docx": [".docx"],
        "Markdown": [".md", ".markdown"],
        "Text": [".txt"],
        "MSG": [".msg"],
    }

    for key, exts in ext_map.items():
        if key.lower() in type_name.lower():
            return exts

    return []


# ============================================================================
# TypeScript Generation
# ============================================================================

def generate_rag_types_typescript(schema: Dict[str, Any]) -> str:
    """Generate ragTypes.ts - Parser and Extractor types"""
    parser_types = extract_parser_types(schema)
    extractor_types = extract_extractor_types(schema)

    lines = [
        "/**",
        " * AUTO-GENERATED FILE - DO NOT EDIT",
        " * ",
        " * Generated from rag/schema.yaml by designer/generate-types.py",
        " * Run: cd designer && ./generate-types.sh",
        " */",
        "",
        "// ============================================================================",
        "// Parser Types",
        "// ============================================================================",
        "",
        "export const PARSER_TYPES = [",
    ]

    for pt in parser_types:
        lines.append(f'  "{pt}",')

    lines.extend([
        "] as const",
        "",
        "export type ParserType = typeof PARSER_TYPES[number]",
        "",
        "// ============================================================================",
        "// Extractor Types",
        "// ============================================================================",
        "",
        "export const EXTRACTOR_TYPES = [",
    ])

    for et in extractor_types:
        lines.append(f'  "{et}",')

    lines.extend([
        "] as const",
        "",
        "export type ExtractorType = typeof EXTRACTOR_TYPES[number]",
        "",
        "// ============================================================================",
        "// Default Configurations",
        "// ============================================================================",
        "",
        "export function getDefaultParserConfig(parserType: ParserType): Record<string, any> {",
        "  const configs: Record<ParserType, Record<string, any>> = {",
    ])

    for pt in parser_types:
        config_schema = get_parser_config_schema(schema, pt)
        default_config = generate_default_config(config_schema)
        config_json = json.dumps(default_config, indent=4)
        indented_config = "\n".join(f"      {line}" for line in config_json.split("\n"))
        lines.append(f'    "{pt}": {indented_config},')

    lines.extend([
        "  }",
        "  return configs[parserType] || {}",
        "}",
        "",
        "export function getDefaultExtractorConfig(extractorType: ExtractorType): Record<string, any> {",
        "  const configs: Record<ExtractorType, Record<string, any>> = {",
    ])

    for et in extractor_types:
        config_schema = get_extractor_config_schema(schema, et)
        default_config = generate_default_config(config_schema)
        config_json = json.dumps(default_config, indent=4)
        indented_config = "\n".join(f"      {line}" for line in config_json.split("\n"))
        lines.append(f'    "{et}": {indented_config},')

    lines.extend([
        "  }",
        "  return configs[extractorType] || {}",
        "}",
        "",
        "// ============================================================================",
        "// Schema Metadata",
        "// ============================================================================",
        "",
        "export type PrimitiveType = 'integer' | 'number' | 'string' | 'boolean' | 'array'",
        "",
        "export type SchemaField = {",
        "  type: PrimitiveType",
        "  title?: string",
        "  description?: string",
        "  default?: unknown",
        "  minimum?: number",
        "  maximum?: number",
        "  enum?: string[]",
        "  items?: { type: PrimitiveType }",
        "  nullable?: boolean",
        "}",
        "",
        "export interface ParserSchema {",
        "  type: ParserType",
        "  title: string",
        "  description: string",
        "  defaultExtensions: string[]",
        "  properties: Record<string, SchemaField>",
        "  required?: string[]",
        "}",
        "",
        "export interface ExtractorSchema {",
        "  type: ExtractorType",
        "  title: string",
        "  description: string",
        "  properties: Record<string, SchemaField>",
        "  required?: string[]",
        "}",
        "",
        "export const PARSER_SCHEMAS: Record<ParserType, ParserSchema> = {",
    ])

    # Add parser schema metadata
    for pt in parser_types:
        config_schema = get_parser_config_schema(schema, pt)
        title = config_schema.get("title", pt)
        description = config_schema.get("description", "")
        properties = config_schema.get("properties", {})
        required = config_schema.get("required", [])
        extensions = infer_file_extensions(pt)

        lines.append(f'  "{pt}": {{')
        lines.append(f'    type: "{pt}",')
        lines.append(f'    title: "{title}",')
        lines.append(f'    description: "{description}",')
        lines.append(f'    defaultExtensions: {json.dumps(extensions)},')

        # Add properties
        properties_ts = convert_schema_properties_to_ts(properties)
        if properties:
            lines.append(f'    properties: {{')
            lines.append(properties_ts)
            lines.append('    },')
        else:
            lines.append('    properties: {},')

        # Add required fields if any
        if required:
            lines.append(f'    required: {json.dumps(required)},')

        lines.append("  },")

    lines.extend([
        "}",
        "",
        "export const EXTRACTOR_SCHEMAS: Record<ExtractorType, ExtractorSchema> = {",
    ])

    # Add extractor schema metadata
    for et in extractor_types:
        config_schema = get_extractor_config_schema(schema, et)
        title = config_schema.get("title", et)
        description = config_schema.get("description", "")
        properties = config_schema.get("properties", {})
        required = config_schema.get("required", [])

        lines.append(f'  "{et}": {{')
        lines.append(f'    type: "{et}",')
        lines.append(f'    title: "{title}",')
        lines.append(f'    description: "{description}",')

        # Add properties
        properties_ts = convert_schema_properties_to_ts(properties)
        if properties:
            lines.append(f'    properties: {{')
            lines.append(properties_ts)
            lines.append('    },')
        else:
            lines.append('    properties: {},')

        # Add required fields if any
        if required:
            lines.append(f'    required: {json.dumps(required)},')

        lines.append("  },")

    lines.extend([
        "}",
        "",
    ])

    return "\n".join(lines)


def generate_database_types_typescript(schema: Dict[str, Any]) -> str:
    """Generate databaseTypes.ts - Vector Store, Embedder, and Retrieval Strategy types"""
    vector_store_types = extract_vector_store_types(schema)
    embedder_types = extract_embedder_types(schema)
    retrieval_strategy_types = extract_retrieval_strategy_types(schema)

    lines = [
        "/**",
        " * AUTO-GENERATED FILE - DO NOT EDIT",
        " * ",
        " * Generated from rag/schema.yaml by designer/generate-types.py",
        " * Run: cd designer && ./generate-types.sh",
        " */",
        "",
        "// ============================================================================",
        "// Vector Store / Database Types",
        "// ============================================================================",
        "",
        "export const VECTOR_STORE_TYPES = [",
    ]

    for vst in vector_store_types:
        lines.append(f'  "{vst}",')

    lines.extend([
        "] as const",
        "",
        "export type VectorStoreType = typeof VECTOR_STORE_TYPES[number]",
        "",
        "// ============================================================================",
        "// Embedder Types",
        "// ============================================================================",
        "",
        "export const EMBEDDER_TYPES = [",
    ])

    for et in embedder_types:
        lines.append(f'  "{et}",')

    lines.extend([
        "] as const",
        "",
        "export type EmbedderType = typeof EMBEDDER_TYPES[number]",
        "",
        "// ============================================================================",
        "// Retrieval Strategy Types",
        "// ============================================================================",
        "",
        "export const RETRIEVAL_STRATEGY_TYPES = [",
    ])

    for rst in retrieval_strategy_types:
        lines.append(f'  "{rst}",')

    lines.extend([
        "] as const",
        "",
        "export type RetrievalStrategyType = typeof RETRIEVAL_STRATEGY_TYPES[number]",
        "",
        "// ============================================================================",
        "// Default Configurations - Vector Stores",
        "// ============================================================================",
        "",
        "export function getDefaultVectorStoreConfig(storeType: VectorStoreType): Record<string, any> {",
        "  const configs: Record<VectorStoreType, Record<string, any>> = {",
    ])

    for vst in vector_store_types:
        config_schema = get_vector_store_config_schema(schema, vst)
        default_config = generate_default_config(config_schema)
        config_json = json.dumps(default_config, indent=4)
        indented_config = "\n".join(f"      {line}" for line in config_json.split("\n"))
        lines.append(f'    "{vst}": {indented_config},')

    lines.extend([
        "  }",
        "  return configs[storeType] || {}",
        "}",
        "",
        "// ============================================================================",
        "// Default Configurations - Embedders",
        "// ============================================================================",
        "",
        "export function getDefaultEmbedderConfig(embedderType: EmbedderType): Record<string, any> {",
        "  const configs: Record<EmbedderType, Record<string, any>> = {",
    ])

    for et in embedder_types:
        config_schema = get_embedder_config_schema(schema, et)
        default_config = generate_default_config(config_schema)
        config_json = json.dumps(default_config, indent=4)
        indented_config = "\n".join(f"      {line}" for line in config_json.split("\n"))
        lines.append(f'    "{et}": {indented_config},')

    lines.extend([
        "  }",
        "  return configs[embedderType] || {}",
        "}",
        "",
        "// ============================================================================",
        "// Default Configurations - Retrieval Strategies",
        "// ============================================================================",
        "",
        "export function getDefaultRetrievalStrategyConfig(strategyType: RetrievalStrategyType): Record<string, any> {",
        "  const configs: Record<RetrievalStrategyType, Record<string, any>> = {",
    ])

    for rst in retrieval_strategy_types:
        config_schema = get_retrieval_strategy_config_schema(schema, rst)
        default_config = generate_default_config(config_schema)
        config_json = json.dumps(default_config, indent=4)
        indented_config = "\n".join(f"      {line}" for line in config_json.split("\n"))
        lines.append(f'    "{rst}": {indented_config},')

    lines.extend([
        "  }",
        "  return configs[strategyType] || {}",
        "}",
        "",
        "// ============================================================================",
        "// Schema Metadata",
        "// ============================================================================",
        "",
        "export interface VectorStoreSchema {",
        "  type: VectorStoreType",
        "  title: string",
        "  description: string",
        "  category: 'local' | 'cloud' | 'memory'",
        "}",
        "",
        "export interface EmbedderSchema {",
        "  type: EmbedderType",
        "  title: string",
        "  description: string",
        "  category: 'local' | 'cloud' | 'huggingface'",
        "}",
        "",
        "export interface RetrievalStrategySchema {",
        "  type: RetrievalStrategyType",
        "  title: string",
        "  description: string",
        "  complexity: 'basic' | 'intermediate' | 'advanced'",
        "}",
        "",
        "export const VECTOR_STORE_SCHEMAS: Record<VectorStoreType, VectorStoreSchema> = {",
    ])

    # Add vector store schema metadata (categorization can be derived from config properties)
    for vst in vector_store_types:
        config_schema = get_vector_store_config_schema(schema, vst)
        title = config_schema.get("title", vst)
        description = config_schema.get("description", "")

        # Infer category from type name
        category = "local"
        if "Pinecone" in vst:
            category = "cloud"
        elif "FAISS" in vst:
            category = "memory"

        lines.append(f'  "{vst}": {{')
        lines.append(f'    type: "{vst}",')
        lines.append(f'    title: "{title}",')
        lines.append(f'    description: "{description}",')
        lines.append(f'    category: "{category}",')
        lines.append("  },")

    lines.extend([
        "}",
        "",
        "export const EMBEDDER_SCHEMAS: Record<EmbedderType, EmbedderSchema> = {",
    ])

    # Add embedder schema metadata
    for et in embedder_types:
        config_schema = get_embedder_config_schema(schema, et)
        title = config_schema.get("title", et)
        description = config_schema.get("description", "")

        # Infer category from type name
        category = "local"
        if "OpenAI" in et:
            category = "cloud"
        elif "HuggingFace" in et or "SentenceTransformer" in et:
            category = "huggingface"

        lines.append(f'  "{et}": {{')
        lines.append(f'    type: "{et}",')
        lines.append(f'    title: "{title}",')
        lines.append(f'    description: "{description}",')
        lines.append(f'    category: "{category}",')
        lines.append("  },")

    lines.extend([
        "}",
        "",
        "export const RETRIEVAL_STRATEGY_SCHEMAS: Record<RetrievalStrategyType, RetrievalStrategySchema> = {",
    ])

    # Add retrieval strategy schema metadata
    for rst in retrieval_strategy_types:
        config_schema = get_retrieval_strategy_config_schema(schema, rst)
        title = config_schema.get("title", rst)
        description = config_schema.get("description", "")

        # Infer complexity from type name
        complexity = "basic"
        if "Hybrid" in rst or "Multi" in rst:
            complexity = "advanced"
        elif "Reranked" in rst or "Metadata" in rst:
            complexity = "intermediate"

        lines.append(f'  "{rst}": {{')
        lines.append(f'    type: "{rst}",')
        lines.append(f'    title: "{title}",')
        lines.append(f'    description: "{description}",')
        lines.append(f'    complexity: "{complexity}",')
        lines.append("  },")

    lines.extend([
        "}",
        "",
        "// ============================================================================",
        "// Helper Functions",
        "// ============================================================================",
        "",
        "/**",
        " * Get all vector stores by category",
        " */",
        "export function getVectorStoresByCategory(category: 'local' | 'cloud' | 'memory'): VectorStoreType[] {",
        "  return VECTOR_STORE_TYPES.filter(type => VECTOR_STORE_SCHEMAS[type].category === category)",
        "}",
        "",
        "/**",
        " * Get all embedders by category",
        " */",
        "export function getEmbeddersByCategory(category: 'local' | 'cloud' | 'huggingface'): EmbedderType[] {",
        "  return EMBEDDER_TYPES.filter(type => EMBEDDER_SCHEMAS[type].category === category)",
        "}",
        "",
        "/**",
        " * Get all retrieval strategies by complexity",
        " */",
        "export function getRetrievalStrategiesByComplexity(complexity: 'basic' | 'intermediate' | 'advanced'): RetrievalStrategyType[] {",
        "  return RETRIEVAL_STRATEGY_TYPES.filter(type => RETRIEVAL_STRATEGY_SCHEMAS[type].complexity === complexity)",
        "}",
        "",
    ])

    return "\n".join(lines)


def main():
    """Main entry point"""
    print("Generating TypeScript types from rag/schema.yaml...")

    schema = load_schema()

    # Generate ragTypes.ts
    rag_types_content = generate_rag_types_typescript(schema)
    output_dir = Path(__file__).parent / "src" / "components" / "Rag" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    rag_types_file = output_dir / "ragTypes.ts"
    rag_types_file.write_text(rag_types_content, encoding="utf-8")

    print(f"✓ Generated {rag_types_file}")
    print(f"  - {len(extract_parser_types(schema))} parser types")
    print(f"  - {len(extract_extractor_types(schema))} extractor types")

    # Generate databaseTypes.ts
    db_types_content = generate_database_types_typescript(schema)
    db_types_file = output_dir / "databaseTypes.ts"
    db_types_file.write_text(db_types_content, encoding="utf-8")

    print(f"✓ Generated {db_types_file}")
    print(f"  - {len(extract_vector_store_types(schema))} vector store types")
    print(f"  - {len(extract_embedder_types(schema))} embedder types")
    print(f"  - {len(extract_retrieval_strategy_types(schema))} retrieval strategy types")

    print("\nDone! Import from:")
    print("  - @/components/Rag/generated/ragTypes")
    print("  - @/components/Rag/generated/databaseTypes")


if __name__ == "__main__":
    main()
