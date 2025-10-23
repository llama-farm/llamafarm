#!/usr/bin/env python3
"""
Generate TypeScript types and constants for the Designer UI from rag/schema.yaml

This script reads the RAG schema and generates:
- Parser type constants
- Extractor type constants
- Default configuration functions
- TypeScript type definitions

Run: ./generate-ui-types.sh
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


def extract_parser_types(schema: Dict[str, Any]) -> List[str]:
    """Extract all parser type names from schema definitions"""
    parsers = schema.get("definitions", {}).get("parsers", {})
    # Convert config keys to parser type names
    # e.g., "pdfParserPyPDF2Config" -> "PDFParser_PyPDF2"
    parser_types = []

    type_mapping = {
        "pdfParserPyPDF2Config": "PDFParser_PyPDF2",
        "pdfParserLlamaIndexConfig": "PDFParser_LlamaIndex",
        "csvParserPandasConfig": "CSVParser_Pandas",
        "csvParserPythonConfig": "CSVParser_Python",
        "csvParserLlamaIndexConfig": "CSVParser_LlamaIndex",
        "excelParserOpenPyXLConfig": "ExcelParser_OpenPyXL",
        "excelParserPandasConfig": "ExcelParser_Pandas",
        "excelParserLlamaIndexConfig": "ExcelParser_LlamaIndex",
        "docxParserPythonDocxConfig": "DocxParser_PythonDocx",
        "docxParserLlamaIndexConfig": "DocxParser_LlamaIndex",
        "markdownParserPythonConfig": "MarkdownParser_Python",
        "markdownParserLlamaIndexConfig": "MarkdownParser_LlamaIndex",
        "textParserPythonConfig": "TextParser_Python",
        "textParserLlamaIndexConfig": "TextParser_LlamaIndex",
        "msgParserExtractMsgConfig": "MSGParser_ExtractMsg",
        "autoParserConfig": "auto",
    }

    for config_key in parsers.keys():
        if config_key in type_mapping:
            parser_types.append(type_mapping[config_key])

    return sorted(parser_types)


def extract_extractor_types(schema: Dict[str, Any]) -> List[str]:
    """Extract all extractor type names from the schema"""
    # From extractorConfig.properties.type.enum
    extractor_config = schema.get("definitions", {}).get("extractorConfig", {})
    type_enum = extractor_config.get("properties", {}).get("type", {}).get("enum", [])
    return sorted(type_enum)


def get_parser_config_schema(schema: Dict[str, Any], parser_type: str) -> Dict[str, Any]:
    """Get the config schema for a specific parser type"""
    # Map parser type to config key
    type_to_config = {
        "PDFParser_PyPDF2": "pdfParserPyPDF2Config",
        "PDFParser_LlamaIndex": "pdfParserLlamaIndexConfig",
        "CSVParser_Pandas": "csvParserPandasConfig",
        "CSVParser_Python": "csvParserPythonConfig",
        "CSVParser_LlamaIndex": "csvParserLlamaIndexConfig",
        "ExcelParser_OpenPyXL": "excelParserOpenPyXLConfig",
        "ExcelParser_Pandas": "excelParserPandasConfig",
        "ExcelParser_LlamaIndex": "excelParserLlamaIndexConfig",
        "DocxParser_PythonDocx": "docxParserPythonDocxConfig",
        "DocxParser_LlamaIndex": "docxParserLlamaIndexConfig",
        "MarkdownParser_Python": "markdownParserPythonConfig",
        "MarkdownParser_LlamaIndex": "markdownParserLlamaIndexConfig",
        "TextParser_Python": "textParserPythonConfig",
        "TextParser_LlamaIndex": "textParserLlamaIndexConfig",
        "MSGParser_ExtractMsg": "msgParserExtractMsgConfig",
        "auto": "autoParserConfig",
    }

    config_key = type_to_config.get(parser_type)
    if not config_key:
        return {}

    parsers = schema.get("definitions", {}).get("parsers", {})
    return parsers.get(config_key, {})


def get_extractor_config_schema(schema: Dict[str, Any], extractor_type: str) -> Dict[str, Any]:
    """Get the config schema for a specific extractor type"""
    # Map extractor type to config key
    type_to_config = {
        "KeywordExtractor": "keywordExtractorConfig",
        "EntityExtractor": "entityExtractorConfig",
        "DateTimeExtractor": "dateTimeExtractorConfig",
        "HeadingExtractor": "headingExtractorConfig",
        "LinkExtractor": "linkExtractorConfig",
        "PathExtractor": "pathExtractorConfig",
        "PatternExtractor": "patternExtractorConfig",
        "ContentStatisticsExtractor": "contentStatisticsExtractorConfig",
        "SummaryExtractor": "summaryExtractorConfig",
        "TableExtractor": "tableExtractorConfig",
        "RAKEExtractor": "keywordExtractorConfig",  # Uses keyword config
        "TFIDFExtractor": "keywordExtractorConfig",  # Uses keyword config
        "YAKEExtractor": "yakeExtractorConfig",
    }

    config_key = type_to_config.get(extractor_type)
    if not config_key:
        return {}

    extractors = schema.get("definitions", {}).get("extractors", {})
    return extractors.get(config_key, {})


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


def generate_typescript() -> str:
    """Generate the TypeScript file content"""
    schema = load_schema()

    parser_types = extract_parser_types(schema)
    extractor_types = extract_extractor_types(schema)

    # Build TypeScript content
    lines = [
        "/**",
        " * AUTO-GENERATED FILE - DO NOT EDIT",
        " * ",
        " * Generated from rag/schema.yaml by generate-ui-types.py",
        " * Run: cd rag && ./generate-ui-types.sh",
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
        # Indent the JSON properly
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

        # Infer default extensions
        ext_map = {
            "PDF": [".pdf"],
            "CSV": [".csv"],
            "Excel": [".xlsx", ".xls"],
            "Docx": [".docx"],
            "Markdown": [".md", ".markdown"],
            "Text": [".txt"],
            "MSG": [".msg"],
        }
        extensions = []
        for key, exts in ext_map.items():
            if key.lower() in pt.lower():
                extensions = exts
                break

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


def main():
    """Main entry point"""
    print("Generating TypeScript types from rag/schema.yaml...")

    # Generate TypeScript content
    ts_content = generate_typescript()

    # Write to designer generated directory
    output_dir = Path(__file__).parent.parent / "designer" / "src" / "components" / "Rag" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "ragTypes.ts"
    output_file.write_text(ts_content, encoding="utf-8")

    print(f"✓ Generated {output_file}")
    print(f"  - {len(extract_parser_types(load_schema()))} parser types")
    print(f"  - {len(extract_extractor_types(load_schema()))} extractor types")
    print("\nDone! Import from: @/components/Rag/generated/ragTypes")


if __name__ == "__main__":
    main()
