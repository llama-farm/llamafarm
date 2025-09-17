#!/usr/bin/env python3
"""
Demo script showing how to use LlamaIndex parsers in the RAG system.

This script demonstrates:
1. Using different LlamaIndex parsers for various file types
2. Advanced chunking strategies
3. Fallback mechanisms for robust parsing
4. Integration with the RAG pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from components.parsers.parser_factory import ToolAwareParserFactory
from core import ProcessingPipeline
from config import load_config
import json


def demo_pdf_with_llamaindex():
    """Demonstrate advanced PDF parsing with LlamaIndex."""
    print("\n" + "="*60)
    print("DEMO: Advanced PDF Parsing with LlamaIndex")
    print("="*60)
    
    # Create PDF parser with LlamaIndex
    parser = ToolAwareParserFactory.create_parser(
        parser_name="PDFParser_LlamaIndex",
        config={
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "chunk_strategy": "semantic",  # Advanced semantic chunking
            "extract_metadata": True,
            "extract_images": True,
            "extract_tables": True,
            "fallback_strategies": [
                "llama_pdf_reader",
                "llama_pymupdf_reader",
                "direct_pymupdf",
                "pypdf2_fallback"
            ]
        }
    )
    
    # Parse a sample PDF
    pdf_path = "samples/pdfs/test_document.pdf"
    if Path(pdf_path).exists():
        result = parser.parse(pdf_path)
        
        print(f"\n✓ Parsed {len(result.documents)} chunks from PDF")
        if result.documents:
            doc = result.documents[0]
            print(f"  First chunk preview: {doc.content[:200]}...")
            print(f"  Metadata: {json.dumps(doc.metadata, indent=2)}")
        
        if result.errors:
            print(f"\n⚠ Encountered {len(result.errors)} errors:")
            for error in result.errors[:3]:
                print(f"  - {error}")
    else:
        print(f"⚠ Sample PDF not found at {pdf_path}")


def demo_markdown_with_llamaindex():
    """Demonstrate heading-aware markdown parsing with LlamaIndex."""
    print("\n" + "="*60)
    print("DEMO: Structured Markdown Parsing with LlamaIndex")
    print("="*60)
    
    # Create Markdown parser with LlamaIndex
    parser = ToolAwareParserFactory.create_parser(
        parser_name="MarkdownParser_LlamaIndex",
        config={
            "chunk_size": 1200,
            "chunk_overlap": 150,
            "chunk_strategy": "headings",  # Split by markdown headings
            "extract_metadata": True,
            "extract_code_blocks": True,
            "extract_tables": True,
            "extract_links": True,
            "preserve_structure": True
        }
    )
    
    # Parse a sample markdown file
    md_path = "samples/documents/machine_learning_guide.md"
    if Path(md_path).exists():
        result = parser.parse(md_path)
        
        print(f"\n✓ Parsed {len(result.documents)} chunks from Markdown")
        if result.documents:
            doc = result.documents[0]
            print(f"  First chunk preview: {doc.content[:200]}...")
            
            # Show extracted code blocks if any
            if "code_blocks" in doc.metadata:
                print(f"  Extracted {len(doc.metadata['code_blocks'])} code blocks")
            
            # Show extracted links if any
            if "links" in doc.metadata:
                print(f"  Extracted {len(doc.metadata['links'])} links")
    else:
        print(f"⚠ Sample markdown not found at {md_path}")


def demo_excel_with_llamaindex():
    """Demonstrate multi-sheet Excel parsing with LlamaIndex."""
    print("\n" + "="*60)
    print("DEMO: Multi-Sheet Excel Processing with LlamaIndex")
    print("="*60)
    
    # Create Excel parser with LlamaIndex
    parser = ToolAwareParserFactory.create_parser(
        parser_name="ExcelParser_LlamaIndex",
        config={
            "chunk_size": 500,  # Rows per chunk
            "chunk_strategy": "rows",
            "sheets": None,  # Process all sheets
            "combine_sheets": False,
            "extract_metadata": True,
            "extract_formulas": True,
            "header_row": 0,
            "na_values": ["", "NA", "N/A", "null", "None"]
        }
    )
    
    # Parse a sample Excel file
    excel_path = "samples/data/financial_report.xlsx"
    if Path(excel_path).exists():
        result = parser.parse(excel_path)
        
        print(f"\n✓ Parsed {len(result.documents)} chunks from Excel")
        
        # Show sheet information
        sheets = set()
        for doc in result.documents:
            if "sheet_name" in doc.metadata:
                sheets.add(doc.metadata["sheet_name"])
        
        if sheets:
            print(f"  Processed sheets: {', '.join(sheets)}")
        
        # Show formula extraction
        for doc in result.documents:
            if "formulas" in doc.metadata and doc.metadata["formulas"]:
                print(f"  Found formulas in sheet '{doc.metadata.get('sheet_name', 'unknown')}'")
                break
    else:
        print(f"⚠ Sample Excel file not found at {excel_path}")


def demo_csv_with_llamaindex():
    """Demonstrate intelligent CSV parsing with LlamaIndex."""
    print("\n" + "="*60)
    print("DEMO: Smart CSV Processing with LlamaIndex")
    print("="*60)
    
    # Create CSV parser with LlamaIndex
    parser = ToolAwareParserFactory.create_parser(
        parser_name="CSVParser_LlamaIndex",
        config={
            "chunk_size": 1000,
            "chunk_strategy": "semantic",  # Semantic understanding of rows
            "field_mapping": {
                "ticket_id": "id",
                "description": "content",
                "priority": "priority_level"
            },
            "extract_metadata": True,
            "combine_fields": True,
            "na_values": ["", "NA", "N/A", "null", "None"]
        }
    )
    
    # Parse a sample CSV file
    csv_path = "samples/csv/small_sample.csv"
    if Path(csv_path).exists():
        result = parser.parse(csv_path)
        
        print(f"\n✓ Parsed {len(result.documents)} chunks from CSV")
        if result.documents:
            doc = result.documents[0]
            print(f"  First chunk preview: {doc.content[:200]}...")
            
            # Show field mapping results
            if "mapped_fields" in doc.metadata:
                print(f"  Applied field mappings: {doc.metadata['mapped_fields']}")
    else:
        print(f"⚠ Sample CSV not found at {csv_path}")


def demo_text_with_llamaindex():
    """Demonstrate multi-format text parsing with LlamaIndex."""
    print("\n" + "="*60)
    print("DEMO: Multi-Format Text Processing with LlamaIndex")
    print("="*60)
    
    # Create Text parser with LlamaIndex
    parser = ToolAwareParserFactory.create_parser(
        parser_name="TextParser_LlamaIndex",
        config={
            "chunk_size": 1200,
            "chunk_overlap": 100,
            "chunk_strategy": "semantic",  # Smart semantic chunking
            "extract_metadata": True,
            "preserve_code_structure": True,
            "detect_language": True,
            "include_prev_next_rel": True  # Include chunk relationships
        }
    )
    
    # Parse different text formats
    text_files = [
        "samples/documents/ai_overview.txt",
        "samples/code/example.py",
        "samples/configs/settings.yaml"
    ]
    
    for file_path in text_files:
        if Path(file_path).exists():
            result = parser.parse(file_path)
            
            file_type = Path(file_path).suffix
            print(f"\n✓ Parsed {file_type} file: {len(result.documents)} chunks")
            
            if result.documents:
                doc = result.documents[0]
                
                # Show language detection
                if "detected_language" in doc.metadata:
                    print(f"  Detected language: {doc.metadata['detected_language']}")
                
                # Show chunk relationships
                if "prev_chunk_id" in doc.metadata or "next_chunk_id" in doc.metadata:
                    print(f"  Chunk relationships preserved")
        else:
            print(f"⚠ Sample file not found at {file_path}")


def demo_docx_with_llamaindex():
    """Demonstrate DOCX parsing with formatting preservation."""
    print("\n" + "="*60)
    print("DEMO: DOCX Document Processing with LlamaIndex")
    print("="*60)
    
    # Create DOCX parser with LlamaIndex
    parser = ToolAwareParserFactory.create_parser(
        parser_name="DocxParser_LlamaIndex",
        config={
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "chunk_strategy": "paragraphs",
            "extract_metadata": True,
            "extract_tables": True,
            "extract_images": True,
            "preserve_formatting": True,
            "include_header_footer": True
        }
    )
    
    # Parse a sample DOCX file
    docx_path = "samples/documents/report.docx"
    if Path(docx_path).exists():
        result = parser.parse(docx_path)
        
        print(f"\n✓ Parsed {len(result.documents)} chunks from DOCX")
        if result.documents:
            doc = result.documents[0]
            print(f"  First chunk preview: {doc.content[:200]}...")
            
            # Show preserved formatting
            if "formatting" in doc.metadata:
                print(f"  Formatting preserved: {doc.metadata['formatting']}")
            
            # Show extracted tables
            if "tables" in doc.metadata and doc.metadata["tables"]:
                print(f"  Extracted {len(doc.metadata['tables'])} tables")
    else:
        print(f"⚠ Sample DOCX not found at {docx_path}")


def compare_parsers():
    """Compare traditional parser with LlamaIndex parser."""
    print("\n" + "="*60)
    print("COMPARISON: Traditional vs LlamaIndex Parsers")
    print("="*60)
    
    test_file = "samples/documents/machine_learning_guide.md"
    
    if not Path(test_file).exists():
        print(f"⚠ Test file not found: {test_file}")
        return
    
    # Parse with traditional parser
    traditional_parser = ToolAwareParserFactory.create_parser(
        parser_name="MarkdownParser_Python",
        config={
            "chunk_size": 1200,
            "chunk_overlap": 150,
            "extract_metadata": True
        }
    )
    
    traditional_result = traditional_parser.parse(test_file)
    
    # Parse with LlamaIndex parser
    llamaindex_parser = ToolAwareParserFactory.create_parser(
        parser_name="MarkdownParser_LlamaIndex",
        config={
            "chunk_size": 1200,
            "chunk_overlap": 150,
            "chunk_strategy": "headings",
            "extract_metadata": True,
            "preserve_structure": True
        }
    )
    
    llamaindex_result = llamaindex_parser.parse(test_file)
    
    # Compare results
    print("\nComparison Results:")
    print(f"  Traditional Parser: {len(traditional_result.documents)} chunks")
    print(f"  LlamaIndex Parser: {len(llamaindex_result.documents)} chunks")
    
    # Compare metadata richness
    traditional_metadata = traditional_result.documents[0].metadata if traditional_result.documents else {}
    llamaindex_metadata = llamaindex_result.documents[0].metadata if llamaindex_result.documents else {}
    
    print(f"\n  Traditional metadata fields: {len(traditional_metadata)}")
    print(f"  LlamaIndex metadata fields: {len(llamaindex_metadata)}")
    
    # Show unique LlamaIndex features
    unique_features = set(llamaindex_metadata.keys()) - set(traditional_metadata.keys())
    if unique_features:
        print(f"\n  Unique LlamaIndex features: {', '.join(unique_features)}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("LlamaIndex Parser Demonstration Suite")
    print("="*60)
    print("\nThis demo showcases the advanced capabilities of LlamaIndex parsers")
    print("compared to traditional parsing approaches.")
    
    # Run individual demos
    try:
        demo_pdf_with_llamaindex()
    except Exception as e:
        print(f"\n✗ PDF demo failed: {e}")
    
    try:
        demo_markdown_with_llamaindex()
    except Exception as e:
        print(f"\n✗ Markdown demo failed: {e}")
    
    try:
        demo_excel_with_llamaindex()
    except Exception as e:
        print(f"\n✗ Excel demo failed: {e}")
    
    try:
        demo_csv_with_llamaindex()
    except Exception as e:
        print(f"\n✗ CSV demo failed: {e}")
    
    try:
        demo_text_with_llamaindex()
    except Exception as e:
        print(f"\n✗ Text demo failed: {e}")
    
    try:
        demo_docx_with_llamaindex()
    except Exception as e:
        print(f"\n✗ DOCX demo failed: {e}")
    
    # Run comparison
    try:
        compare_parsers()
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nKey Advantages of LlamaIndex Parsers:")
    print("1. Semantic chunking strategies")
    print("2. Multiple fallback mechanisms")
    print("3. Rich metadata extraction")
    print("4. Format-specific optimizations")
    print("5. Chunk relationship tracking")
    print("6. Advanced structure preservation")


if __name__ == "__main__":
    main()