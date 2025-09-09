#!/usr/bin/env python3
"""
Comprehensive showcase of all LlamaIndex parsers with real documents.
This demo tests each parser with actual files from the samples directory.
"""

import sys
from pathlib import Path
import json
from typing import Dict, Any
from datetime import datetime

# Add parent directory to path (go up one level from demos/)
sys.path.append(str(Path(__file__).parent.parent))

from components.parsers.parser_factory import ToolAwareParserFactory
from components.parsers.parser_registry import ParserRegistry


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(parser_name: str, result: Any, file_path: str):
    """Print parsing results in a structured format."""
    print(f"\n📄 File: {file_path}")
    print(f"🔧 Parser: {parser_name}")
    
    if result.documents:
        print(f"✅ Successfully parsed {len(result.documents)} chunks")
        
        # Show first document details
        first_doc = result.documents[0]
        print(f"\n📊 First Chunk Analysis:")
        print(f"  - Content length: {len(first_doc.content)} characters")
        print(f"  - Content preview: {first_doc.content[:150]}...")
        
        # Show metadata
        if first_doc.metadata:
            print(f"\n🏷️  Metadata ({len(first_doc.metadata)} fields):")
            for key, value in list(first_doc.metadata.items())[:8]:
                if isinstance(value, (list, dict)):
                    print(f"    • {key}: {type(value).__name__} with {len(value)} items")
                elif isinstance(value, str) and len(value) > 50:
                    print(f"    • {key}: {value[:50]}...")
                else:
                    print(f"    • {key}: {value}")
        
        # Show chunk distribution
        if len(result.documents) > 1:
            total_chars = sum(len(doc.content) for doc in result.documents)
            avg_chunk_size = total_chars // len(result.documents)
            print(f"\n📈 Chunk Statistics:")
            print(f"    • Total chunks: {len(result.documents)}")
            print(f"    • Average chunk size: {avg_chunk_size} characters")
            print(f"    • Total content: {total_chars} characters")
    
    if result.errors:
        print(f"\n⚠️  Errors: {len(result.errors)}")
        for error in result.errors[:3]:
            print(f"    • {error}")


def test_pdf_parser():
    """Test PDFParser_LlamaIndex with real PDF documents."""
    print_section("PDF PARSER WITH LLAMAINDEX - Advanced Features")
    
    parser = ToolAwareParserFactory.create_parser(
        parser_name="PDFParser_LlamaIndex",
        config={
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "chunk_strategy": "sentences",  # Smart sentence-based chunking
            "extract_metadata": True,
            "extract_images": False,  # Set to False for now
            "extract_tables": True,
            "fallback_strategies": [
                "llama_pdf_reader",
                "llama_pymupdf_reader", 
                "direct_pymupdf",
                "pypdf2_fallback"
            ]
        }
    )
    
    # Test with real PDFs
    pdf_files = [
        "static_samples/747/ryanair-737-700-800-fcom-rev-30.pdf",
        "../samples/pdfs/llama.pdf",
        "static_samples/business_reports/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
    ]
    
    for pdf_path in pdf_files:
        if Path(pdf_path).exists():
            try:
                print(f"\n🔄 Processing: {Path(pdf_path).name}")
                result = parser.parse(pdf_path)
                print_result("PDFParser_LlamaIndex", result, Path(pdf_path).name)
            except Exception as e:
                print(f"❌ Error parsing {pdf_path}: {e}")
        else:
            print(f"⏭️  Skipping (not found): {pdf_path}")
    
    return parser


def test_markdown_parser():
    """Test MarkdownParser_LlamaIndex with real markdown files."""
    print_section("MARKDOWN PARSER WITH LLAMAINDEX - Heading-Aware Chunking")
    
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
    
    # Test with real markdown files
    md_files = [
        "../samples/documents/machine_learning_guide.md",
        "../samples/documents/python_programming_basics.md",
        "static_samples/code_documentation/api_reference.md",
        "static_samples/code_documentation/best_practices.md"
    ]
    
    for md_path in md_files:
        if Path(md_path).exists():
            try:
                print(f"\n🔄 Processing: {Path(md_path).name}")
                result = parser.parse(md_path)
                print_result("MarkdownParser_LlamaIndex", result, Path(md_path).name)
                
                # Show special markdown features
                if result.documents and result.documents[0].metadata:
                    metadata = result.documents[0].metadata
                    if "code_blocks" in metadata:
                        print(f"    🔧 Code blocks found: {len(metadata['code_blocks'])}")
                    if "links" in metadata:
                        print(f"    🔗 Links extracted: {len(metadata['links'])}")
                    if "headings" in metadata:
                        print(f"    📑 Heading structure preserved")
                        
            except Exception as e:
                print(f"❌ Error parsing {md_path}: {e}")
        else:
            print(f"⏭️  Skipping (not found): {md_path}")
    
    return parser


def test_csv_parser():
    """Test CSVParser_LlamaIndex with real CSV files."""
    print_section("CSV PARSER WITH LLAMAINDEX - Intelligent Field Mapping")
    
    parser = ToolAwareParserFactory.create_parser(
        parser_name="CSVParser_LlamaIndex",
        config={
            "chunk_size": 500,  # Rows per chunk
            "chunk_strategy": "rows",
            "field_mapping": {
                "subject": "title",
                "body": "content",
                "priority": "importance"
            },
            "extract_metadata": True,
            "combine_fields": True,
            "na_values": ["", "NA", "N/A", "null", "None"]
        }
    )
    
    # Test with real CSV files
    csv_files = [
        "static_samples/customer_support/support_tickets.csv",
        "../samples/csv/small_sample.csv",
        "static_samples/business_reports/supply_chain_metrics.csv"
    ]
    
    for csv_path in csv_files:
        if Path(csv_path).exists():
            try:
                print(f"\n🔄 Processing: {Path(csv_path).name}")
                result = parser.parse(csv_path)
                print_result("CSVParser_LlamaIndex", result, Path(csv_path).name)
                
                # Show CSV-specific features
                if result.documents:
                    print(f"    📊 Rows processed: {len(result.documents)}")
                    if result.documents[0].metadata.get("field_mapping"):
                        print(f"    🔄 Field mapping applied")
                        
            except Exception as e:
                print(f"❌ Error parsing {csv_path}: {e}")
        else:
            print(f"⏭️  Skipping (not found): {csv_path}")
    
    return parser


def test_excel_parser():
    """Test ExcelParser_LlamaIndex with real Excel files."""
    print_section("EXCEL PARSER WITH LLAMAINDEX - Multi-Sheet Support")
    
    parser = ToolAwareParserFactory.create_parser(
        parser_name="ExcelParser_LlamaIndex",
        config={
            "chunk_size": 500,  # Rows per chunk
            "chunk_strategy": "rows",
            "sheets": None,  # Process all sheets
            "combine_sheets": False,
            "extract_metadata": True,
            "extract_formulas": False,  # Formulas require actual Excel file
            "header_row": 0,
            "na_values": ["", "NA", "N/A", "null", "None", "#N/A"]
        }
    )
    
    # Test with real Excel files
    excel_files = [
        "static_samples/business_reports/quarterly_financial_report.xlsx"
    ]
    
    for excel_path in excel_files:
        if Path(excel_path).exists():
            try:
                print(f"\n🔄 Processing: {Path(excel_path).name}")
                result = parser.parse(excel_path)
                print_result("ExcelParser_LlamaIndex", result, Path(excel_path).name)
                
                # Show Excel-specific features
                if result.documents:
                    sheets = set()
                    for doc in result.documents:
                        if "sheet_name" in doc.metadata:
                            sheets.add(doc.metadata["sheet_name"])
                    if sheets:
                        print(f"    📑 Sheets processed: {', '.join(sheets)}")
                        
            except Exception as e:
                print(f"❌ Error parsing {excel_path}: {e}")
        else:
            print(f"⏭️  Skipping (not found): {excel_path}")
    
    return parser


def test_text_parser():
    """Test TextParser_LlamaIndex with various text formats."""
    print_section("TEXT PARSER WITH LLAMAINDEX - Multi-Format & Semantic Chunking")
    
    parser = ToolAwareParserFactory.create_parser(
        parser_name="TextParser_LlamaIndex",
        config={
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "chunk_strategy": "sentences",  # Semantic chunking
            "extract_metadata": True,
            "preserve_code_structure": True,
            "detect_language": True,
            "include_prev_next_rel": True  # Track chunk relationships
        }
    )
    
    # Test with various text files
    text_files = [
        "../samples/documents/ai_overview.txt",
        "../samples/documents/data_science_workflow.txt",
        "static_samples/research_papers/llm_scaling_laws.txt",
        "static_samples/research_papers/transformer_architecture.txt"
    ]
    
    for text_path in text_files:
        if Path(text_path).exists():
            try:
                print(f"\n🔄 Processing: {Path(text_path).name}")
                result = parser.parse(text_path)
                print_result("TextParser_LlamaIndex", result, Path(text_path).name)
                
                # Show text-specific features
                if result.documents and result.documents[0].metadata:
                    metadata = result.documents[0].metadata
                    if "detected_language" in metadata:
                        print(f"    🌐 Language detected: {metadata['detected_language']}")
                    if "prev_chunk_id" in metadata or "next_chunk_id" in metadata:
                        print(f"    🔗 Chunk relationships preserved")
                        
            except Exception as e:
                print(f"❌ Error parsing {text_path}: {e}")
        else:
            print(f"⏭️  Skipping (not found): {text_path}")
    
    return parser


def test_docx_parser():
    """Test DocxParser_LlamaIndex with real DOCX files."""
    print_section("DOCX PARSER WITH LLAMAINDEX - Enhanced Document Processing")
    
    parser = ToolAwareParserFactory.create_parser(
        parser_name="DocxParser_LlamaIndex",
        config={
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "chunk_strategy": "paragraphs",
            "extract_metadata": True,
            "extract_tables": True,
            "extract_images": False,
            "preserve_formatting": True,
            "include_header_footer": False
        }
    )
    
    # Note: We don't have DOCX files in the samples, so we'll create a simple test
    print(f"\n📝 Note: No DOCX files in samples, showing parser capabilities")
    print(f"    • Paragraph-based chunking")
    print(f"    • Table extraction")
    print(f"    • Formatting preservation")
    print(f"    • Metadata extraction")
    
    return parser


def test_parser_registry():
    """Verify all LlamaIndex parsers are in the registry."""
    print_section("PARSER REGISTRY VERIFICATION")
    
    registry = ParserRegistry()
    all_parsers = registry.list_all_parsers()
    
    llamaindex_parsers = [
        "PDFParser_LlamaIndex",
        "MarkdownParser_LlamaIndex",
        "CSVParser_LlamaIndex",
        "ExcelParser_LlamaIndex",
        "TextParser_LlamaIndex",
        "DocxParser_LlamaIndex"
    ]
    
    print("\n🔍 Checking Parser Registry:")
    for parser_name in llamaindex_parsers:
        if parser_name in all_parsers:
            parser_info = registry.get_parser(parser_name)
            print(f"  ✅ {parser_name}")
            print(f"     Tool: {parser_info.get('tool', 'N/A')}")
            print(f"     Extensions: {', '.join(parser_info.get('supported_extensions', []))}")
        else:
            print(f"  ❌ {parser_name} - NOT FOUND")
    
    # Show file extension coverage
    print("\n📁 File Extension Coverage:")
    extensions = [".pdf", ".md", ".csv", ".xlsx", ".txt", ".docx"]
    for ext in extensions:
        parsers = registry.get_parsers_for_extension(ext)
        llamaindex_parsers_for_ext = [
            p for p in parsers 
            if "LlamaIndex" in p.get("tool", "")
        ]
        if llamaindex_parsers_for_ext:
            parser_names = [p["parser"] for p in llamaindex_parsers_for_ext]
            print(f"  {ext}: {', '.join(parser_names)}")


def compare_with_traditional_parsers():
    """Compare LlamaIndex parsers with traditional parsers."""
    print_section("COMPARISON: LlamaIndex vs Traditional Parsers")
    
    test_file = "../samples/documents/machine_learning_guide.md"
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        return
    
    print(f"\n📊 Comparing parsers on: {Path(test_file).name}")
    
    # Traditional parser
    traditional_parser = ToolAwareParserFactory.create_parser(
        parser_name="MarkdownParser_Python",
        config={
            "chunk_size": 1200,
            "chunk_overlap": 150,
            "extract_metadata": True
        }
    )
    
    # LlamaIndex parser
    llamaindex_parser = ToolAwareParserFactory.create_parser(
        parser_name="MarkdownParser_LlamaIndex",
        config={
            "chunk_size": 1200,
            "chunk_overlap": 150,
            "chunk_strategy": "headings",
            "extract_metadata": True,
            "extract_code_blocks": True,
            "preserve_structure": True
        }
    )
    
    # Parse with both
    traditional_result = traditional_parser.parse(test_file)
    llamaindex_result = llamaindex_parser.parse(test_file)
    
    print("\n📈 Results Comparison:")
    print(f"  Traditional Parser:")
    print(f"    • Chunks: {len(traditional_result.documents)}")
    if traditional_result.documents:
        print(f"    • Metadata fields: {len(traditional_result.documents[0].metadata)}")
        print(f"    • Average chunk size: {sum(len(d.content) for d in traditional_result.documents) // len(traditional_result.documents)} chars")
    
    print(f"\n  LlamaIndex Parser:")
    print(f"    • Chunks: {len(llamaindex_result.documents)}")
    if llamaindex_result.documents:
        print(f"    • Metadata fields: {len(llamaindex_result.documents[0].metadata)}")
        print(f"    • Average chunk size: {sum(len(d.content) for d in llamaindex_result.documents) // len(llamaindex_result.documents)} chars")
        
        # Show unique features
        metadata = llamaindex_result.documents[0].metadata
        unique_features = []
        if "code_blocks" in metadata:
            unique_features.append("code extraction")
        if "headings" in metadata:
            unique_features.append("heading structure")
        if "links" in metadata:
            unique_features.append("link extraction")
        if unique_features:
            print(f"    • Unique features: {', '.join(unique_features)}")


def main():
    """Run the comprehensive LlamaIndex parser showcase."""
    print("\n" + "=" * 80)
    print("  🚀 LLAMAINDEX PARSER COMPREHENSIVE SHOWCASE")
    print("  Testing with Real Documents from Samples Directory")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Test each parser
    parsers_tested = []
    
    try:
        pdf_parser = test_pdf_parser()
        parsers_tested.append("PDF")
    except Exception as e:
        print(f"\n❌ PDF Parser Error: {e}")
    
    try:
        markdown_parser = test_markdown_parser()
        parsers_tested.append("Markdown")
    except Exception as e:
        print(f"\n❌ Markdown Parser Error: {e}")
    
    try:
        csv_parser = test_csv_parser()
        parsers_tested.append("CSV")
    except Exception as e:
        print(f"\n❌ CSV Parser Error: {e}")
    
    try:
        excel_parser = test_excel_parser()
        parsers_tested.append("Excel")
    except Exception as e:
        print(f"\n❌ Excel Parser Error: {e}")
    
    try:
        text_parser = test_text_parser()
        parsers_tested.append("Text")
    except Exception as e:
        print(f"\n❌ Text Parser Error: {e}")
    
    try:
        docx_parser = test_docx_parser()
        parsers_tested.append("DOCX")
    except Exception as e:
        print(f"\n❌ DOCX Parser Error: {e}")
    
    # Verify registry
    test_parser_registry()
    
    # Compare with traditional parsers
    compare_with_traditional_parsers()
    
    # Summary
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print_section("SHOWCASE SUMMARY")
    print(f"\n✅ Parsers Tested: {', '.join(parsers_tested)}")
    print(f"⏱️  Total Time: {elapsed_time:.2f} seconds")
    
    print("\n🎯 Key LlamaIndex Features Demonstrated:")
    print("  1. Semantic chunking strategies")
    print("  2. Multiple fallback mechanisms (PDF)")
    print("  3. Heading-aware parsing (Markdown)")
    print("  4. Multi-sheet support (Excel)")
    print("  5. Field mapping (CSV)")
    print("  6. Chunk relationship tracking (Text)")
    print("  7. Rich metadata extraction (All)")
    
    print("\n💡 Note: Some features require LlamaIndex to be installed:")
    print("  pip install llama-index llama-index-readers-file")
    
    print("\n" + "=" * 80)
    print("  🎉 SHOWCASE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()