#!/usr/bin/env python3
"""
Demo: MIME Type Filtering in Action

This script demonstrates the two-tier MIME type filtering system:
1. Strategy-level filtering (what files the strategy accepts)
2. Parser-level routing (which parser handles each file type)
"""

from pathlib import Path
from core.mime_type_filter import MimeTypeFilter
from core.strategies.loader import StrategyLoader
import yaml

def demonstrate_mime_filtering():
    """Demonstrate MIME type filtering with real strategies."""
    
    print("\n" + "="*80)
    print("🎯 MIME Type Filtering Demonstration")
    print("="*80)
    
    # Initialize components
    mime_filter = MimeTypeFilter()
    loader = StrategyLoader(strategies_file="demos/demo_strategies.yaml")
    
    # Test files
    test_files = [
        Path("report.pdf"),
        Path("data.csv"),
        Path("document.docx"),
        Path("notes.md"),
        Path("spreadsheet.xlsx"),
        Path("script.py"),
        Path("webpage.html"),
        Path("config.json")
    ]
    
    print("\n📁 Test Files:")
    for f in test_files:
        mime_type, ext = mime_filter.get_mime_type(f)
        print(f"  • {f.name:20} MIME: {mime_type:50} Extension: {ext}")
    
    # Test CSV-only strategy
    print("\n" + "-"*80)
    print("📊 Testing CSV-Only Strategy (csv_processing)")
    print("-"*80)
    
    csv_strategy = {
        "name": "csv_processing",
        "allowed_mime_types": ["text/csv", "text/tab-separated-values"],
        "allowed_extensions": [".csv", ".tsv"]
    }
    
    result = mime_filter.validate_strategy_files(test_files, csv_strategy)
    print(f"✅ Accepted files: {[f.name for f in result['accepted']]}")
    print(f"❌ Rejected files: {[f.name for f in result['rejected']]}")
    
    # Test multi-format business strategy
    print("\n" + "-"*80)
    print("💼 Testing Multi-Format Business Strategy")
    print("-"*80)
    
    business_strategy = {
        "name": "business_processing",
        "allowed_mime_types": [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv"
        ],
        "allowed_extensions": [".pdf", ".docx", ".xlsx", ".csv"]
    }
    
    result = mime_filter.validate_strategy_files(test_files, business_strategy)
    print(f"✅ Accepted files: {[f.name for f in result['accepted']]}")
    print(f"❌ Rejected files: {[f.name for f in result['rejected']]}")
    
    # Demonstrate parser routing
    print("\n" + "-"*80)
    print("🔄 Parser Routing for Business Documents")
    print("-"*80)
    
    business_parsers = [
        {
            "type": "PDFParser_LlamaIndex",
            "mime_types": ["application/pdf"],
            "file_extensions": [".pdf", ".PDF"],
            "priority": 10
        },
        {
            "type": "DocxParser_LlamaIndex",
            "mime_types": ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            "file_extensions": [".docx", ".DOCX"],
            "priority": 10
        },
        {
            "type": "ExcelParser_LlamaIndex",
            "mime_types": ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
            "file_extensions": [".xlsx", ".XLSX"],
            "priority": 10
        },
        {
            "type": "CSVParser_Pandas",
            "mime_types": ["text/csv"],
            "file_extensions": [".csv", ".CSV"],
            "priority": 5
        }
    ]
    
    # Route accepted files to parsers
    accepted_files = result['accepted']
    for file_path in accepted_files:
        best_parser = mime_filter.find_matching_parser(file_path, business_parsers)
        if best_parser:
            print(f"  📄 {file_path.name:20} → {best_parser['type']}")
    
    # Test generic strategy (accepts all)
    print("\n" + "-"*80)
    print("🌐 Testing Generic Strategy (accepts all files)")
    print("-"*80)
    
    generic_strategy = {
        "name": "text_processing",
        "allowed_mime_types": [],  # Empty = accept all
        "allowed_extensions": []   # Empty = accept all
    }
    
    result = mime_filter.validate_strategy_files(test_files, generic_strategy)
    print(f"✅ Accepted files: {len(result['accepted'])} (all files)")
    print(f"❌ Rejected files: {len(result['rejected'])}")
    
    # Demonstrate priority-based parser selection
    print("\n" + "-"*80)
    print("⚡ Priority-Based Parser Selection")
    print("-"*80)
    
    pdf_parsers = [
        {
            "type": "BasicPDFParser",
            "mime_types": ["application/pdf"],
            "priority": 1
        },
        {
            "type": "AdvancedPDFParser",
            "mime_types": ["application/pdf"],
            "priority": 10  # Higher priority wins
        }
    ]
    
    pdf_file = Path("document.pdf")
    best_parser = mime_filter.find_matching_parser(pdf_file, pdf_parsers)
    print(f"  For {pdf_file.name}, selected parser: {best_parser['type']} (higher priority)")
    
    print("\n" + "="*80)
    print("✅ Demonstration Complete!")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("  1️⃣  Strategy-level filtering (allowed_mime_types, allowed_extensions)")
    print("  2️⃣  Parser-level routing based on MIME types and extensions")
    print("  3️⃣  Priority-based parser selection when multiple parsers match")
    print("  4️⃣  Generic strategies that accept all file types")
    print("  5️⃣  Specialized strategies for specific file types only")
    print()

if __name__ == "__main__":
    demonstrate_mime_filtering()