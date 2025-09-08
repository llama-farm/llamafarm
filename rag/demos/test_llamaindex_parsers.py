#!/usr/bin/env python3
"""Test that LlamaIndex parsers are properly exposed."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from components.parsers.parser_factory import ToolAwareParserFactory
from components.parsers.parser_registry import ParserRegistry

def test_llamaindex_exposure():
    """Test that all LlamaIndex parsers are available."""
    
    print("=" * 60)
    print("Testing LlamaIndex Parser Exposure")
    print("=" * 60)
    
    # Test parser discovery
    parsers = ToolAwareParserFactory.discover_parsers()
    print(f"\n✓ Discovered {len(parsers)} parser types:")
    for parser_type in parsers:
        print(f"  - {parser_type}: {len(parsers[parser_type])} implementations")
    
    # Test registry
    registry = ParserRegistry()
    
    # Check for LlamaIndex parsers
    llamaindex_parsers = [
        "MarkdownParser_LlamaIndex",
        "PDFParser_LlamaIndex", 
        "TextParser_LlamaIndex",
        "DocxParser_LlamaIndex",
        "CSVParser_LlamaIndex",
        "ExcelParser_LlamaIndex"
    ]
    
    print("\n" + "=" * 60)
    print("LlamaIndex Parser Availability:")
    print("=" * 60)
    
    for parser_name in llamaindex_parsers:
        # Check if parser exists in registry
        parser_info = registry.data["parsers"].get(parser_name)
        if parser_info:
            print(f"\n✓ {parser_name}:")
            print(f"  Tool: {parser_info['tool']}")
            print(f"  Description: {parser_info['description']}")
            print(f"  Extensions: {', '.join(parser_info['supported_extensions'])}")
            print(f"  Capabilities: {len(parser_info['capabilities'])} features")
            
            # Try to create the parser
            try:
                parser = ToolAwareParserFactory.create_parser(parser_name=parser_name)
                if parser:
                    print(f"  ✓ Can be instantiated")
                else:
                    print(f"  ⚠ Failed to instantiate")
            except Exception as e:
                print(f"  ⚠ Error creating parser: {e}")
        else:
            print(f"\n✗ {parser_name} NOT FOUND in registry")
    
    # Check file type mappings
    print("\n" + "=" * 60)
    print("File Type Support (LlamaIndex):")
    print("=" * 60)
    
    file_types = {
        ".md": "Markdown",
        ".pdf": "PDF",
        ".txt": "Text",
        ".docx": "DOCX",
        ".csv": "CSV",
        ".xlsx": "Excel"
    }
    
    for ext, name in file_types.items():
        parsers_for_ext = registry.get_parsers_for_extension(ext)
        llamaindex_parsers_for_ext = [
            p for p in parsers_for_ext 
            if "LlamaIndex" in p.get("tool", "")
        ]
        
        if llamaindex_parsers_for_ext:
            print(f"\n{name} ({ext}):")
            for p in llamaindex_parsers_for_ext:
                print(f"  ✓ {p['parser']} ({p['tool']})")
        else:
            print(f"\n{name} ({ext}): No LlamaIndex parser")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    total_parsers = len(registry.data["parsers"])
    llamaindex_count = sum(
        1 for name in registry.data["parsers"] 
        if "LlamaIndex" in registry.data["parsers"][name].get("tool", "")
    )
    
    print(f"Total parsers in registry: {total_parsers}")
    print(f"LlamaIndex parsers: {llamaindex_count}")
    print(f"Coverage: {llamaindex_count}/{len(llamaindex_parsers)} expected parsers")
    
    # Test creating a parser with config
    print("\n" + "=" * 60)
    print("Testing Parser Creation with Config:")
    print("=" * 60)
    
    test_config = {
        "chunk_size": 500,
        "chunk_strategy": "semantic",
        "extract_metadata": True
    }
    
    try:
        parser = ToolAwareParserFactory.create_parser(
            parser_name="TextParser_LlamaIndex",
            config=test_config
        )
        if parser:
            print(f"✓ Successfully created TextParser_LlamaIndex with custom config")
            print(f"  Config: {parser.config}")
    except Exception as e:
        print(f"✗ Failed to create parser with config: {e}")

if __name__ == "__main__":
    test_llamaindex_exposure()