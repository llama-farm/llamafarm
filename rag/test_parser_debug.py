#!/usr/bin/env python3
"""Debug parsers to see what content they're extracting"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Test each parser directly
def test_csv_parser():
    """Test CSV parser directly"""
    from components.parsers.csv_parser_pandas import CSVParser_Pandas
    
    parser = CSVParser_Pandas({
        'chunk_size': 800,
        'chunk_overlap': 100,
        'chunk_strategy': 'rows',
        'extract_metadata': True
    })
    
    csv_file = Path(__file__).parent / "demos/static_samples/customer_support/support_tickets.csv"
    
    print("Testing CSV Parser:")
    print(f"File: {csv_file}")
    
    try:
        documents = parser.parse(str(csv_file))
        print(f"Parsed {len(documents)} documents")
        
        for i, doc in enumerate(documents[:2], 1):
            print(f"\nDocument {i}:")
            print(f"  Content length: {len(doc.content)}")
            print(f"  Content preview: {doc.content[:200]}...")
            print(f"  Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_markdown_parser():
    """Test Markdown parser directly"""
    from components.parsers.markdown_parser_python import MarkdownParser_Python
    
    parser = MarkdownParser_Python({
        'chunk_size': 1000,
        'chunk_strategy': 'sections',
        'extract_metadata': True,
        'extract_code_blocks': True,
        'extract_links': True
    })
    
    md_file = Path(__file__).parent / "demos/static_samples/code_documentation/api_reference.md"
    
    print("\n\nTesting Markdown Parser:")
    print(f"File: {md_file}")
    
    try:
        documents = parser.parse(str(md_file))
        print(f"Parsed {len(documents)} documents")
        
        for i, doc in enumerate(documents[:2], 1):
            print(f"\nDocument {i}:")
            print(f"  Content length: {len(doc.content)}")
            print(f"  Content preview: {doc.content[:200]}...")
            print(f"  Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_pdf_parser():
    """Test PDF parser directly"""
    from components.parsers.pdf_parser_pypdf2 import PDFParser_PyPDF2
    
    parser = PDFParser_PyPDF2({
        'chunk_size': 1000,
        'chunk_overlap': 150,
        'chunk_strategy': 'paragraphs',
        'extract_metadata': True
    })
    
    pdf_file = Path(__file__).parent / "demos/static_samples/fda_letters/761315_2025_Orig1s000OtherActionLtrs.pdf"
    
    print("\n\nTesting PDF Parser:")
    print(f"File: {pdf_file}")
    
    try:
        documents = parser.parse(str(pdf_file))
        print(f"Parsed {len(documents)} documents")
        
        for i, doc in enumerate(documents[:2], 1):
            print(f"\nDocument {i}:")
            print(f"  Content length: {len(doc.content)}")
            print(f"  Content preview: {doc.content[:200]}...")
            if "BLA" in doc.content:
                print("  *** Contains 'BLA' ***")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_csv_parser()
    test_markdown_parser() 
    test_pdf_parser()