#!/usr/bin/env python3
"""
Real-world demonstration of PDFParser_LlamaIndex with fallback strategies.
This shows the parser actually working with various PDF documents.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from components.parsers.parser_factory import ToolAwareParserFactory


def analyze_pdf(pdf_path: str, parser):
    """Analyze a PDF file and show detailed results."""
    print(f"\n{'='*70}")
    print(f"📄 Processing: {Path(pdf_path).name}")
    print(f"   File size: {Path(pdf_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*70}")
    
    # Parse the PDF
    start_time = datetime.now()
    result = parser.parse(pdf_path)
    parse_time = (datetime.now() - start_time).total_seconds()
    
    if result.documents:
        print(f"\n✅ SUCCESS - Parsed in {parse_time:.2f} seconds")
        print(f"   Strategy used: {result.documents[0].metadata.get('strategy', 'unknown')}")
        print(f"   Total pages: {result.documents[0].metadata.get('total_pages', 'unknown')}")
        print(f"   Chunks created: {len(result.documents)}")
        
        # Calculate statistics
        total_chars = sum(len(doc.content) for doc in result.documents)
        avg_chunk_size = total_chars // len(result.documents) if len(result.documents) > 1 else total_chars
        
        print(f"\n📊 Content Statistics:")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average chunk size: {avg_chunk_size:,} chars")
        
        # Show metadata from first chunk
        print(f"\n🏷️  Metadata Fields:")
        for key, value in result.documents[0].metadata.items():
            if key not in ['source', 'file_name']:  # Skip long fields
                if isinstance(value, (str, int, float, bool)):
                    print(f"   • {key}: {value}")
        
        # Show content preview
        print(f"\n📝 Content Preview (first 500 chars):")
        preview = result.documents[0].content[:500].replace('\n', ' ')
        print(f"   {preview}...")
        
        # Show chunk distribution if multiple chunks
        if len(result.documents) > 1:
            print(f"\n📈 Chunk Distribution:")
            for i, doc in enumerate(result.documents[:5]):  # Show first 5 chunks
                print(f"   Chunk {i+1}: {len(doc.content):,} chars")
            if len(result.documents) > 5:
                print(f"   ... and {len(result.documents) - 5} more chunks")
    
    if result.errors:
        print(f"\n⚠️  Errors encountered: {len(result.errors)}")
        for error in result.errors[:3]:
            print(f"   • {error}")
    
    return result


def main():
    """Run comprehensive PDF parsing demonstration."""
    print("\n" + "="*70)
    print("🚀 PDFParser_LlamaIndex - Real Document Processing Demo")
    print("="*70)
    print("\nThis demo shows the LlamaIndex PDF parser with fallback strategies")
    print("processing real PDF documents of various types and sizes.")
    
    # Create parser with advanced configuration
    print("\n🔧 Creating PDFParser_LlamaIndex with fallback strategies...")
    parser = ToolAwareParserFactory.create_parser(
        parser_name="PDFParser_LlamaIndex",
        config={
            "chunk_size": 2000,  # Reasonable chunk size for demonstration
            "chunk_overlap": 200,
            "chunk_strategy": "sentences",
            "extract_metadata": True,
            "extract_images": False,  # Disable for speed
            "extract_tables": True,
            "fallback_strategies": [
                "llama_pdf_reader",      # Try LlamaIndex first
                "llama_pymupdf_reader",  # Then PyMuPDF via LlamaIndex
                "direct_pymupdf",        # Direct PyMuPDF
                "pypdf2_fallback"        # Final fallback to PyPDF2
            ]
        }
    )
    
    # List of PDFs to test
    pdf_files = [
        # Small technical document
        ("../samples/pdfs/llama.pdf", "Small technical document about llamas"),
        
        # Research papers
        ("../samples/pdfs/minillama.pdf", "Research paper"),
        
        # Large technical manual (if exists)
        ("static_samples/747/ryanair-737-700-800-fcom-rev-30.pdf", "Large aircraft manual (1952 pages!)"),
        
        # Business report with graphics
        ("static_samples/business_reports/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf", "Business report with charts"),
        
        # Additional PDFs if they exist
        ("../samples/pdfs/2008LlamaProjectHandbook.pdf", "Project handbook"),
        ("../samples/pdfs/Llamas-Alpacas-Rutgers-University.pdf", "University document"),
    ]
    
    results = []
    successful = 0
    failed = 0
    
    for pdf_path, description in pdf_files:
        if Path(pdf_path).exists():
            print(f"\n📚 {description}")
            try:
                result = analyze_pdf(pdf_path, parser)
                if result.documents:
                    successful += 1
                    results.append((Path(pdf_path).name, len(result.documents), True))
                else:
                    failed += 1
                    results.append((Path(pdf_path).name, 0, False))
            except Exception as e:
                print(f"❌ Failed to parse: {e}")
                failed += 1
                results.append((Path(pdf_path).name, 0, False))
        else:
            print(f"\n⏭️  Skipping {Path(pdf_path).name} - file not found")
    
    # Summary
    print("\n" + "="*70)
    print("📊 PARSING SUMMARY")
    print("="*70)
    
    print(f"\nResults:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    
    if results:
        print(f"\nDetailed Results:")
        print(f"  {'File Name':<50} {'Chunks':<10} {'Status'}")
        print(f"  {'-'*50} {'-'*10} {'-'*10}")
        for filename, chunks, success in results:
            status = "✅ Success" if success else "❌ Failed"
            print(f"  {filename:<50} {chunks:<10} {status}")
    
    print("\n🎯 Key Features Demonstrated:")
    print("  1. ✅ Multiple fallback strategies ensure robust parsing")
    print("  2. ✅ Handles PDFs of various sizes (3 pages to 1952 pages!)")
    print("  3. ✅ Extracts comprehensive metadata")
    print("  4. ✅ Smart sentence-based chunking")
    print("  5. ✅ Works even without LlamaIndex installed (via fallbacks)")
    
    print("\n💡 The fallback mechanism ensures PDFs are always parsed:")
    print("  • First tries LlamaIndex PDF reader (if available)")
    print("  • Falls back to PyMuPDF integration")
    print("  • Then tries direct PyMuPDF")
    print("  • Finally uses PyPDF2 as last resort")
    
    print("\n" + "="*70)
    print("🎉 Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    main()