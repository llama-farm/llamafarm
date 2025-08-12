#!/usr/bin/env python3
"""
Update all component schema.yaml and defaults.yaml files to match JSON Schema format
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

# Base path for components
COMPONENTS_PATH = Path(__file__).parent.parent / "components"

def create_json_schema_header(component_type: str, component_name: str, title: str, description: str) -> Dict[str, Any]:
    """Create standard JSON Schema header"""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": f"components/{component_type}/{component_name}/schema.yaml",
        "title": title,
        "description": description,
        "type": "object",
        "additionalProperties": False
    }

def update_markdown_parser():
    """Update Markdown Parser schema and defaults"""
    schema_path = COMPONENTS_PATH / "parsers" / "markdown_parser" / "schema.yaml"
    defaults_path = COMPONENTS_PATH / "parsers" / "markdown_parser" / "defaults.yaml"
    
    # Schema
    schema = create_json_schema_header(
        "parsers", "markdown_parser",
        "Markdown Parser Configuration",
        "Parses Markdown files with structure extraction"
    )
    schema["properties"] = {
        "extract_metadata": {
            "type": "boolean",
            "default": True,
            "description": "Extract YAML frontmatter"
        },
        "extract_headings": {
            "type": "boolean",
            "default": True,
            "description": "Extract heading structure"
        },
        "extract_links": {
            "type": "boolean",
            "default": True,
            "description": "Extract all links"
        },
        "extract_code_blocks": {
            "type": "boolean",
            "default": True,
            "description": "Extract code blocks"
        },
        "chunk_by_headings": {
            "type": "boolean",
            "default": False,
            "description": "Split by headings"
        },
        "preserve_formatting": {
            "type": "boolean",
            "default": False,
            "description": "Preserve Markdown formatting"
        },
        "heading_level_split": {
            "type": "integer",
            "default": 2,
            "minimum": 1,
            "maximum": 6,
            "description": "Heading level for splitting"
        }
    }
    
    with open(schema_path, 'w') as f:
        f.write("# Markdown Parser Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)
    
    # Defaults
    defaults = {
        "general_purpose": {
            "name": "General Purpose Markdown",
            "description": "Standard Markdown parsing",
            "config": {
                "extract_metadata": True,
                "extract_headings": True,
                "extract_links": True,
                "extract_code_blocks": True,
                "chunk_by_headings": False,
                "preserve_formatting": False,
                "heading_level_split": 2
            },
            "recommended_for": [
                "Documentation",
                "README files",
                "Blog posts"
            ]
        },
        "technical_docs": {
            "name": "Technical Documentation",
            "description": "Technical docs with code blocks",
            "config": {
                "extract_metadata": True,
                "extract_headings": True,
                "extract_links": True,
                "extract_code_blocks": True,
                "chunk_by_headings": True,
                "preserve_formatting": True,
                "heading_level_split": 2
            },
            "recommended_for": [
                "API documentation",
                "Technical guides",
                "Developer docs"
            ]
        }
    }
    
    with open(defaults_path, 'w') as f:
        f.write("# Markdown Parser Default Configurations\n")
        f.write("# These provide ready-to-use configurations for common use cases\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Updated Markdown Parser")

def update_docx_parser():
    """Update DOCX Parser schema and defaults"""
    schema_path = COMPONENTS_PATH / "parsers" / "docx_parser" / "schema.yaml"
    defaults_path = COMPONENTS_PATH / "parsers" / "docx_parser" / "defaults.yaml"
    
    # Schema
    schema = create_json_schema_header(
        "parsers", "docx_parser",
        "DOCX Parser Configuration",
        "Parses Microsoft Word documents"
    )
    schema["properties"] = {
        "extract_metadata": {
            "type": "boolean",
            "default": True,
            "description": "Extract document properties"
        },
        "extract_headers_footers": {
            "type": "boolean",
            "default": True,
            "description": "Include headers and footers"
        },
        "extract_comments": {
            "type": "boolean",
            "default": True,
            "description": "Extract comments"
        },
        "extract_tables": {
            "type": "boolean",
            "default": True,
            "description": "Extract tables"
        },
        "extract_images": {
            "type": "boolean",
            "default": False,
            "description": "Extract images"
        },
        "preserve_formatting": {
            "type": "boolean",
            "default": False,
            "description": "Preserve text formatting"
        }
    }
    
    with open(schema_path, 'w') as f:
        f.write("# DOCX Parser Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)
    
    # Defaults
    defaults = {
        "general_purpose": {
            "name": "General Purpose DOCX",
            "description": "Standard Word document parsing",
            "config": {
                "extract_metadata": True,
                "extract_headers_footers": True,
                "extract_comments": True,
                "extract_tables": True,
                "extract_images": False,
                "preserve_formatting": False
            },
            "recommended_for": [
                "Reports",
                "Letters",
                "General documents"
            ]
        },
        "with_comments": {
            "name": "Documents with Comments",
            "description": "Extract documents with review comments",
            "config": {
                "extract_metadata": True,
                "extract_headers_footers": True,
                "extract_comments": True,
                "extract_tables": True,
                "extract_images": False,
                "preserve_formatting": True
            },
            "recommended_for": [
                "Reviewed documents",
                "Collaborative docs",
                "Draft documents"
            ]
        }
    }
    
    with open(defaults_path, 'w') as f:
        f.write("# DOCX Parser Default Configurations\n")
        f.write("# These provide ready-to-use configurations for common use cases\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Updated DOCX Parser")

def update_excel_parser():
    """Update Excel Parser schema and defaults"""
    schema_path = COMPONENTS_PATH / "parsers" / "excel_parser" / "schema.yaml"
    defaults_path = COMPONENTS_PATH / "parsers" / "excel_parser" / "defaults.yaml"
    
    # Schema
    schema = create_json_schema_header(
        "parsers", "excel_parser",
        "Excel Parser Configuration",
        "Parses Excel spreadsheets"
    )
    schema["properties"] = {
        "sheet_names": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "default": None,
            "description": "Specific sheets to parse"
        },
        "combine_sheets": {
            "type": "boolean",
            "default": False,
            "description": "Combine all sheets"
        },
        "extract_formulas": {
            "type": "boolean",
            "default": False,
            "description": "Extract cell formulas"
        },
        "extract_charts": {
            "type": "boolean",
            "default": False,
            "description": "Extract chart metadata"
        },
        "table_format": {
            "type": "string",
            "enum": ["csv", "markdown", "json", "text"],
            "default": "markdown",
            "description": "Table output format"
        },
        "header_row": {
            "type": "integer",
            "default": 0,
            "minimum": 0,
            "description": "Header row index"
        }
    }
    
    with open(schema_path, 'w') as f:
        f.write("# Excel Parser Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)
    
    # Defaults
    defaults = {
        "general_purpose": {
            "name": "General Purpose Excel",
            "description": "Standard Excel parsing",
            "config": {
                "sheet_names": None,
                "combine_sheets": False,
                "extract_formulas": False,
                "extract_charts": False,
                "table_format": "markdown",
                "header_row": 0
            },
            "recommended_for": [
                "Spreadsheets",
                "Data tables",
                "Reports"
            ]
        },
        "financial_data": {
            "name": "Financial Data",
            "description": "Financial spreadsheets with formulas",
            "config": {
                "sheet_names": None,
                "combine_sheets": True,
                "extract_formulas": True,
                "extract_charts": True,
                "table_format": "markdown",
                "header_row": 0
            },
            "recommended_for": [
                "Financial reports",
                "Budget sheets",
                "Accounting data"
            ]
        }
    }
    
    with open(defaults_path, 'w') as f:
        f.write("# Excel Parser Default Configurations\n")
        f.write("# These provide ready-to-use configurations for common use cases\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Updated Excel Parser")

def update_html_parser():
    """Update HTML Parser schema and defaults"""
    schema_path = COMPONENTS_PATH / "parsers" / "html_parser" / "schema.yaml"
    defaults_path = COMPONENTS_PATH / "parsers" / "html_parser" / "defaults.yaml"
    
    # Schema
    schema = create_json_schema_header(
        "parsers", "html_parser",
        "HTML Parser Configuration",
        "Parses HTML documents and web pages"
    )
    schema["properties"] = {
        "extract_metadata": {
            "type": "boolean",
            "default": True,
            "description": "Extract meta tags"
        },
        "extract_links": {
            "type": "boolean",
            "default": True,
            "description": "Extract hyperlinks"
        },
        "extract_images": {
            "type": "boolean",
            "default": True,
            "description": "Extract image sources"
        },
        "preserve_structure": {
            "type": "boolean",
            "default": False,
            "description": "Preserve HTML structure"
        },
        "remove_scripts": {
            "type": "boolean",
            "default": True,
            "description": "Remove JavaScript"
        },
        "remove_styles": {
            "type": "boolean",
            "default": True,
            "description": "Remove CSS"
        },
        "text_only": {
            "type": "boolean",
            "default": False,
            "description": "Extract only text"
        }
    }
    
    with open(schema_path, 'w') as f:
        f.write("# HTML Parser Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)
    
    # Defaults
    defaults = {
        "general_purpose": {
            "name": "General Purpose HTML",
            "description": "Standard HTML parsing",
            "config": {
                "extract_metadata": True,
                "extract_links": True,
                "extract_images": True,
                "preserve_structure": False,
                "remove_scripts": True,
                "remove_styles": True,
                "text_only": False
            },
            "recommended_for": [
                "Web pages",
                "HTML documents",
                "Email templates"
            ]
        },
        "text_extraction": {
            "name": "Text Only Extraction",
            "description": "Extract only text content",
            "config": {
                "extract_metadata": False,
                "extract_links": False,
                "extract_images": False,
                "preserve_structure": False,
                "remove_scripts": True,
                "remove_styles": True,
                "text_only": True
            },
            "recommended_for": [
                "Content extraction",
                "Text analysis",
                "Simplified parsing"
            ]
        }
    }
    
    with open(defaults_path, 'w') as f:
        f.write("# HTML Parser Default Configurations\n")
        f.write("# These provide ready-to-use configurations for common use cases\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Updated HTML Parser")

def update_text_parser():
    """Update Text Parser schema and defaults"""
    schema_path = COMPONENTS_PATH / "parsers" / "text_parser" / "schema.yaml"
    defaults_path = COMPONENTS_PATH / "parsers" / "text_parser" / "defaults.yaml"
    
    # Schema
    schema = create_json_schema_header(
        "parsers", "text_parser",
        "Text Parser Configuration",
        "Parses plain text files"
    )
    schema["properties"] = {
        "encoding": {
            "type": "string",
            "default": "utf-8",
            "description": "Text encoding"
        },
        "chunk_size": {
            "type": "integer",
            "default": 1000,
            "minimum": 100,
            "maximum": 10000,
            "description": "Chunk size in characters"
        },
        "chunk_overlap": {
            "type": "integer",
            "default": 200,
            "minimum": 0,
            "maximum": 1000,
            "description": "Overlap between chunks"
        },
        "split_by": {
            "type": "string",
            "enum": ["characters", "words", "sentences", "paragraphs", "lines"],
            "default": "characters",
            "description": "Splitting method"
        },
        "preserve_whitespace": {
            "type": "boolean",
            "default": False,
            "description": "Preserve whitespace"
        }
    }
    
    with open(schema_path, 'w') as f:
        f.write("# Text Parser Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)
    
    # Defaults
    defaults = {
        "general_purpose": {
            "name": "General Purpose Text",
            "description": "Standard text parsing",
            "config": {
                "encoding": "utf-8",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "split_by": "characters",
                "preserve_whitespace": False
            },
            "recommended_for": [
                "Text files",
                "Log files",
                "Plain documents"
            ]
        },
        "large_documents": {
            "name": "Large Documents",
            "description": "Large text files with overlap",
            "config": {
                "encoding": "utf-8",
                "chunk_size": 2000,
                "chunk_overlap": 400,
                "split_by": "paragraphs",
                "preserve_whitespace": False
            },
            "recommended_for": [
                "Books",
                "Large reports",
                "Long articles"
            ]
        }
    }
    
    with open(defaults_path, 'w') as f:
        f.write("# Text Parser Default Configurations\n")
        f.write("# These provide ready-to-use configurations for common use cases\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Updated Text Parser")

def main():
    """Update all component schemas"""
    print("Updating component schemas to JSON Schema format...")
    
    # Update parsers
    update_markdown_parser()
    update_docx_parser()
    update_excel_parser()
    update_html_parser()
    update_text_parser()
    
    print("\n✅ All parser components updated!")
    print("\nNote: This script updated the parsers. Similar updates needed for:")
    print("- Extractors")
    print("- Embedders")
    print("- Retrievers")
    print("- Stores")

if __name__ == "__main__":
    main()