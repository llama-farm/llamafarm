#!/usr/bin/env python3
"""
Consolidate all component schemas into the main schema.yaml file.
This script scans all components and updates the schemas/ directory with consolidated definitions.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

# Base paths
SCRIPT_DIR = Path(__file__).parent
RAG_DIR = SCRIPT_DIR.parent
COMPONENTS_DIR = RAG_DIR / "components"
SCHEMAS_DIR = RAG_DIR / "schemas"
MAIN_SCHEMA_PATH = RAG_DIR / "schema.yaml"

def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML file"""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return {}

def scan_component_schemas() -> Dict[str, List[Dict[str, Any]]]:
    """Scan all component directories for schema.yaml files"""
    components = defaultdict(list)
    
    # Component types to scan
    component_types = ["parsers", "extractors", "embedders", "stores", "retrievers"]
    
    for comp_type in component_types:
        type_dir = COMPONENTS_DIR / comp_type
        if not type_dir.exists():
            continue
            
        # Scan each component in the type directory
        for comp_dir in type_dir.iterdir():
            if not comp_dir.is_dir():
                continue
                
            schema_path = comp_dir / "schema.yaml"
            defaults_path = comp_dir / "defaults.yaml"
            
            if schema_path.exists():
                schema = load_yaml_file(schema_path)
                defaults = load_yaml_file(defaults_path) if defaults_path.exists() else {}
                
                # Add component metadata
                component_info = {
                    "name": comp_dir.name,
                    "type": comp_type,
                    "path": str(schema_path.relative_to(RAG_DIR)),
                    "schema": schema,
                    "defaults": defaults
                }
                
                components[comp_type].append(component_info)
                print(f"  Found {comp_type[:-1]}: {comp_dir.name}")
    
    return dict(components)

def consolidate_parsers(parsers: List[Dict]) -> Dict[str, Any]:
    """Consolidate parser schemas"""
    consolidated = {}
    
    for parser in parsers:
        name = parser["name"]
        schema = parser["schema"]
        defaults = parser["defaults"]
        
        # Convert component name to schema key format
        key = name.replace("_parser", "").replace("_", "")
        if key == "csv":
            key = "CSVParser"
        elif key == "pdf":
            key = "PDFParser"
        elif key == "docx":
            key = "DocxParser"
        elif key == "excel":
            key = "ExcelParser"
        elif key == "html":
            key = "HTMLParser"
        elif key == "markdown":
            key = "MarkdownParser"
        elif key == "text":
            key = "PlainTextParser"
        else:
            key = key.title() + "Parser"
        
        consolidated[key] = {
            "description": schema.get("description", f"Parser for {name}"),
            "config_schema": schema.get("properties", {}),
            "defaults": defaults
        }
    
    return consolidated

def consolidate_extractors(extractors: List[Dict]) -> Dict[str, Any]:
    """Consolidate extractor schemas"""
    consolidated = {}
    
    for extractor in extractors:
        name = extractor["name"]
        schema = extractor["schema"]
        defaults = extractor["defaults"]
        
        # Convert component name to schema key format
        key = name.replace("_extractor", "").replace("_", "")
        key = ''.join(word.capitalize() for word in key.split('_')) + "Extractor"
        
        consolidated[key] = {
            "description": schema.get("description", f"Extractor for {name}"),
            "config_schema": schema.get("properties", {}),
            "defaults": defaults
        }
    
    return consolidated

def consolidate_embedders(embedders: List[Dict]) -> Dict[str, Any]:
    """Consolidate embedder schemas"""
    consolidated = {}
    
    for embedder in embedders:
        name = embedder["name"]
        schema = embedder["schema"]
        defaults = embedder["defaults"]
        
        # Convert component name to schema key format
        key = name.replace("_embedder", "").replace("_", "")
        if key == "ollama":
            key = "OllamaEmbedder"
        elif key == "openai":
            key = "OpenAIEmbedder"
        elif key == "huggingface":
            key = "HuggingFaceEmbedder"
        elif key == "sentence_transformer":
            key = "SentenceTransformerEmbedder"
        else:
            key = key.title() + "Embedder"
        
        consolidated[key] = {
            "description": schema.get("description", f"Embedder using {name}"),
            "config_schema": schema.get("properties", {}),
            "required": schema.get("required", []),
            "defaults": defaults
        }
    
    return consolidated

def consolidate_stores(stores: List[Dict]) -> Dict[str, Any]:
    """Consolidate store schemas"""
    consolidated = {}
    
    for store in stores:
        name = store["name"]
        schema = store["schema"]
        defaults = store["defaults"]
        
        # Convert component name to schema key format
        key = name.replace("_store", "").replace("_", "")
        if key == "chroma":
            key = "ChromaStore"
        elif key == "faiss":
            key = "FAISSStore"
        elif key == "pinecone":
            key = "PineconeStore"
        elif key == "qdrant":
            key = "QdrantStore"
        else:
            key = key.title() + "Store"
        
        consolidated[key] = {
            "description": schema.get("description", f"Vector store using {name}"),
            "config_schema": schema.get("properties", {}),
            "required": schema.get("required", []),
            "defaults": defaults
        }
    
    return consolidated

def consolidate_retrievers(retrievers: List[Dict]) -> Dict[str, Any]:
    """Consolidate retriever schemas"""
    consolidated = {}
    
    for retriever in retrievers:
        name = retriever["name"]
        schema = retriever["schema"]
        defaults = retriever["defaults"]
        
        # Convert component name to schema key format
        if name == "basic_similarity":
            key = "BasicSimilarityStrategy"
        elif name == "metadata_filtered":
            key = "MetadataFilteredStrategy"
        elif name == "multi_query":
            key = "MultiQueryStrategy"
        elif name == "reranked":
            key = "RerankedStrategy"
        elif name == "hybrid_universal":
            key = "HybridUniversalStrategy"
        else:
            key = name.replace("_", "").title() + "Strategy"
        
        consolidated[key] = {
            "description": schema.get("description", f"Retrieval strategy: {name}"),
            "config_schema": schema.get("properties", {}),
            "defaults": defaults
        }
    
    return consolidated

def write_consolidated_schemas(components: Dict[str, List[Dict]]):
    """Write consolidated schemas to the schemas/ directory"""
    
    # Ensure schemas directory exists
    SCHEMAS_DIR.mkdir(exist_ok=True)
    
    # Consolidate each component type
    consolidated_data = {}
    
    if "parsers" in components:
        parsers_schema = consolidate_parsers(components["parsers"])
        with open(SCHEMAS_DIR / "parsers.yaml", 'w') as f:
            yaml.dump({"parsers": parsers_schema}, f, default_flow_style=False, sort_keys=False)
        consolidated_data["parsers"] = parsers_schema
        print(f"✓ Wrote {len(parsers_schema)} parser schemas to schemas/parsers.yaml")
    
    if "extractors" in components:
        extractors_schema = consolidate_extractors(components["extractors"])
        with open(SCHEMAS_DIR / "extractors.yaml", 'w') as f:
            yaml.dump({"extractors": extractors_schema}, f, default_flow_style=False, sort_keys=False)
        consolidated_data["extractors"] = extractors_schema
        print(f"✓ Wrote {len(extractors_schema)} extractor schemas to schemas/extractors.yaml")
    
    if "embedders" in components:
        embedders_schema = consolidate_embedders(components["embedders"])
        with open(SCHEMAS_DIR / "embedders.yaml", 'w') as f:
            yaml.dump({"embedders": embedders_schema}, f, default_flow_style=False, sort_keys=False)
        consolidated_data["embedders"] = embedders_schema
        print(f"✓ Wrote {len(embedders_schema)} embedder schemas to schemas/embedders.yaml")
    
    if "stores" in components:
        stores_schema = consolidate_stores(components["stores"])
        with open(SCHEMAS_DIR / "stores.yaml", 'w') as f:
            yaml.dump({"stores": stores_schema}, f, default_flow_style=False, sort_keys=False)
        consolidated_data["stores"] = stores_schema
        print(f"✓ Wrote {len(stores_schema)} store schemas to schemas/stores.yaml")
    
    if "retrievers" in components:
        retrievers_schema = consolidate_retrievers(components["retrievers"])
        with open(SCHEMAS_DIR / "retrievers.yaml", 'w') as f:
            yaml.dump({"retrievers": retrievers_schema}, f, default_flow_style=False, sort_keys=False)
        consolidated_data["retrievers"] = retrievers_schema
        print(f"✓ Wrote {len(retrievers_schema)} retriever schemas to schemas/retrievers.yaml")
    
    # Write consolidated JSON version
    with open(SCHEMAS_DIR / "consolidated.json", 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    print(f"✓ Wrote consolidated JSON to schemas/consolidated.json")
    
    # Write consolidated YAML version
    with open(SCHEMAS_DIR / "consolidated.yaml", 'w') as f:
        yaml.dump(consolidated_data, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Wrote consolidated YAML to schemas/consolidated.yaml")
    
    return consolidated_data

def update_main_schema(consolidated_data: Dict[str, Any]):
    """Update the main schema.yaml with new component definitions"""
    
    # Load current main schema
    main_schema = load_yaml_file(MAIN_SCHEMA_PATH)
    
    # Check for new components
    changes_made = False
    
    # Update definitions section if it exists
    if "definitions" not in main_schema:
        main_schema["definitions"] = {}
    
    # Track changes
    added_components = []
    updated_components = []
    
    # Process each component type
    for comp_type, components in consolidated_data.items():
        if comp_type not in main_schema["definitions"]:
            main_schema["definitions"][comp_type] = {}
        
        for comp_name, comp_schema in components.items():
            if comp_name not in main_schema["definitions"][comp_type]:
                # New component found
                added_components.append(f"{comp_type}.{comp_name}")
                main_schema["definitions"][comp_type][comp_name] = comp_schema
                changes_made = True
            else:
                # Check if schema changed
                existing = main_schema["definitions"][comp_type][comp_name]
                if existing != comp_schema:
                    updated_components.append(f"{comp_type}.{comp_name}")
                    # Only update config_schema to preserve other fields
                    if "config_schema" in comp_schema:
                        existing["config_schema"] = comp_schema["config_schema"]
                    changes_made = True
    
    if changes_made:
        # Create backup
        backup_path = MAIN_SCHEMA_PATH.with_suffix('.yaml.bak')
        with open(backup_path, 'w') as f:
            original = load_yaml_file(MAIN_SCHEMA_PATH)
            yaml.dump(original, f, default_flow_style=False)
        print(f"✓ Created backup at {backup_path}")
        
        # Write updated schema
        with open(MAIN_SCHEMA_PATH, 'w') as f:
            yaml.dump(main_schema, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✅ Updated main schema.yaml")
        if added_components:
            print(f"  Added {len(added_components)} new components:")
            for comp in added_components[:5]:
                print(f"    - {comp}")
            if len(added_components) > 5:
                print(f"    ... and {len(added_components) - 5} more")
        
        if updated_components:
            print(f"  Updated {len(updated_components)} existing components")
    else:
        print("\n✅ No changes needed to main schema.yaml")
    
    return changes_made

def validate_schemas():
    """Validate that all schemas are properly formatted"""
    errors = []
    
    # Check each component schema
    for comp_type in ["parsers", "extractors", "embedders", "stores", "retrievers"]:
        type_dir = COMPONENTS_DIR / comp_type
        if not type_dir.exists():
            continue
        
        for comp_dir in type_dir.iterdir():
            if not comp_dir.is_dir():
                continue
            
            schema_path = comp_dir / "schema.yaml"
            if schema_path.exists():
                schema = load_yaml_file(schema_path)
                
                # Check for required fields
                if "$schema" not in schema:
                    errors.append(f"{schema_path}: Missing $schema field")
                if "type" not in schema:
                    errors.append(f"{schema_path}: Missing type field")
                if "properties" not in schema:
                    errors.append(f"{schema_path}: Missing properties field")
    
    if errors:
        print("\n⚠️  Validation errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ All schemas validated successfully")
        return True

def main():
    """Main consolidation process"""
    print("=" * 60)
    print("Schema Consolidation Tool")
    print("=" * 60)
    
    # Step 1: Scan components
    print("\n1. Scanning component schemas...")
    components = scan_component_schemas()
    
    total_components = sum(len(comps) for comps in components.values())
    print(f"  Found {total_components} total components")
    
    # Step 2: Consolidate schemas
    print("\n2. Consolidating schemas...")
    consolidated_data = write_consolidated_schemas(components)
    
    # Step 3: Update main schema (optional)
    print("\n3. Checking main schema for updates...")
    update_main_schema(consolidated_data)
    
    # Step 4: Validate
    print("\n4. Validating schemas...")
    is_valid = validate_schemas()
    
    print("\n" + "=" * 60)
    if is_valid:
        print("✅ Schema consolidation completed successfully!")
    else:
        print("⚠️  Schema consolidation completed with warnings")
    print("=" * 60)
    
    return 0 if is_valid else 1

if __name__ == "__main__":
    exit(main())