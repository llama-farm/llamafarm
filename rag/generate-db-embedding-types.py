#!/usr/bin/env python3
"""
Generate TypeScript types and constants for Database/Embedding UI from rag/schema.yaml

This script reads the RAG schema and generates:
- Vector store/database type constants
- Embedding type constants
- Retrieval strategy type constants
- Default configuration functions
- TypeScript type definitions

Run: ./generate-db-embedding-types.sh
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_schema() -> Dict[str, Any]:
    """Load the RAG schema.yaml file"""
    schema_path = Path(__file__).parent / "schema.yaml"
    with open(schema_path, "r") as f:
        return yaml.safe_load(f)


def extract_vector_store_types(schema: Dict[str, Any]) -> List[str]:
    """Extract vector store types from schema"""
    vector_store_config = schema.get("definitions", {}).get("vectorStoreConfig", {})
    type_enum = vector_store_config.get("properties", {}).get("type", {}).get("enum", [])
    return sorted(type_enum)


def extract_embedder_types(schema: Dict[str, Any]) -> List[str]:
    """Extract embedder types from schema"""
    embedder_config = schema.get("definitions", {}).get("embedderConfig", {})
    type_enum = embedder_config.get("properties", {}).get("type", {}).get("enum", [])
    return sorted(type_enum)


def extract_retrieval_strategy_types(schema: Dict[str, Any]) -> List[str]:
    """Extract retrieval strategy types from schema"""
    # From databaseDefinition.properties.retrieval_strategies.items.properties.type.enum
    db_def = schema.get("definitions", {}).get("databaseDefinition", {})
    retrieval_strategies = db_def.get("properties", {}).get("retrieval_strategies", {})
    items = retrieval_strategies.get("items", {})
    type_prop = items.get("properties", {}).get("type", {})
    type_enum = type_prop.get("enum", [])
    return sorted(type_enum)


def get_vector_store_config_schema(schema: Dict[str, Any], store_type: str) -> Dict[str, Any]:
    """Get the config schema for a specific vector store type"""
    # Map store type to config key
    type_to_config = {
        "ChromaStore": "chromaStoreConfig",
        "FAISSStore": "faissStoreConfig",
        "PineconeStore": "pineconeStoreConfig",
        "QdrantStore": "qdrantStoreConfig",
    }

    config_key = type_to_config.get(store_type)
    if not config_key:
        return {}

    vector_stores = schema.get("definitions", {}).get("vectorStores", {})
    return vector_stores.get(config_key, {})


def get_embedder_config_schema(schema: Dict[str, Any], embedder_type: str) -> Dict[str, Any]:
    """Get the config schema for a specific embedder type"""
    # Map embedder type to config key
    type_to_config = {
        "OllamaEmbedder": "ollamaEmbedderConfig",
        "HuggingFaceEmbedder": "huggingfaceEmbedderConfig",
        "OpenAIEmbedder": "openaiEmbedderConfig",
        "SentenceTransformerEmbedder": "sentenceTransformerConfig",
    }

    config_key = type_to_config.get(embedder_type)
    if not config_key:
        return {}

    embedders = schema.get("definitions", {}).get("embedders", {})
    return embedders.get(config_key, {})


def get_retrieval_strategy_config_schema(schema: Dict[str, Any], strategy_type: str) -> Dict[str, Any]:
    """Get the config schema for a specific retrieval strategy type"""
    # Map strategy type to config key
    type_to_config = {
        "BasicSimilarityStrategy": "basicSimilarityConfig",
        "MetadataFilteredStrategy": "metadataFilteredConfig",
        "MultiQueryStrategy": "multiQueryConfig",
        "RerankedStrategy": "rerankedConfig",
        "HybridUniversalStrategy": "hybridUniversalConfig",
    }

    config_key = type_to_config.get(strategy_type)
    if not config_key:
        return {}

    retrieval_strategies = schema.get("definitions", {}).get("retrievalStrategies", {})
    return retrieval_strategies.get(config_key, {})


def generate_default_config(config_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a default config object from schema properties"""
    properties = config_schema.get("properties", {})
    defaults = {}

    for prop_name, prop_def in properties.items():
        if "default" in prop_def:
            defaults[prop_name] = prop_def["default"]

    return defaults


def generate_typescript() -> str:
    """Generate the TypeScript file content"""
    schema = load_schema()

    vector_store_types = extract_vector_store_types(schema)
    embedder_types = extract_embedder_types(schema)
    retrieval_strategy_types = extract_retrieval_strategy_types(schema)

    # Build TypeScript content
    lines = [
        "/**",
        " * AUTO-GENERATED FILE - DO NOT EDIT",
        " * ",
        " * Generated from rag/schema.yaml by generate-db-embedding-types.py",
        " * Run: cd rag && ./generate-db-embedding-types.sh",
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

    # Add vector store schema metadata
    for vst in vector_store_types:
        config_schema = get_vector_store_config_schema(schema, vst)
        title = config_schema.get("title", vst)
        description = config_schema.get("description", "")

        # Categorize stores
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

        # Categorize embedders
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

        # Categorize strategies by complexity
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
    print("Generating Database/Embedding TypeScript types from rag/schema.yaml...")

    # Generate TypeScript content
    ts_content = generate_typescript()

    # Write to designer generated directory
    output_dir = Path(__file__).parent.parent / "designer" / "src" / "components" / "Rag" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "databaseTypes.ts"
    output_file.write_text(ts_content, encoding="utf-8")

    schema = load_schema()
    print(f"✓ Generated {output_file}")
    print(f"  - {len(extract_vector_store_types(schema))} vector store types")
    print(f"  - {len(extract_embedder_types(schema))} embedder types")
    print(f"  - {len(extract_retrieval_strategy_types(schema))} retrieval strategy types")
    print("\nDone! Import from: @/components/Rag/generated/databaseTypes")


if __name__ == "__main__":
    main()
