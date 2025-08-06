#!/usr/bin/env python3
"""
Test script for new extractors: TableExtractor, LinkExtractor, HeadingExtractor.
"""

import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_table_extractor():
    """Test TableExtractor functionality."""
    print("🧪 Testing TableExtractor...")
    
    try:
        from components.extractors.table_extractor.table_extractor import TableExtractor
        
        # Test text with various table formats
        test_text = """Document with multiple table formats:

1. Pipe table:
| Name | Age | City |
|------|-----|------|
| John | 25  | NYC  |  
| Jane | 30  | LA   |

2. Grid table:
+------+-----+------+
| Name | Age | City |
+======+=====+======+
| Bob  | 35  | SF   |
| Alice| 28  | DC   |
+------+-----+------+

3. CSV-like data:
Product,Price,Stock
Laptop,999,10
Phone,599,25
Tablet,299,15

4. HTML table marker:
<table>
<tr><th>Item</th><th>Count</th></tr>
<tr><td>Books</td><td>50</td></tr>
<tr><td>Pens</td><td>100</td></tr>
</table>

Some regular text here without tables.
"""
        
        # Test extractor
        extractor = TableExtractor({
            "table_formats": ["pipe", "grid", "csv"],
            "min_columns": 2,
            "min_rows": 2,
            "detect_headers": True
        })
        
        # Create Document object for the new interface
        from core.base import Document
        test_doc = Document(content=test_text, metadata={"doc_id": "test_doc"})
        enhanced_docs = extractor.extract([test_doc])
        
        # Get the result from the enhanced document
        result = enhanced_docs[0].metadata
        
        assert "tables" in result, "Should extract tables"
        tables = result["tables"]
        
        # Should find multiple tables
        assert len(tables) >= 2, f"Expected at least 2 tables, found {len(tables)}"
        
        # Check that each table has expected structure
        for table in tables:
            assert "type" in table, "Table should have type"
            assert "data" in table, "Table should have data"
            data = table["data"]
            assert len(data) >= 2, "Table should have at least 2 rows"
            if data:
                assert len(data[0]) >= 2, "Table should have at least 2 columns"
        
        print(f"✅ TableExtractor - Found {len(tables)} tables")
        
        # Print details about found tables
        for i, table in enumerate(tables):
            data = table.get("data", [])
            rows = len(data)
            cols = len(data[0]) if data else 0
            print(f"   Table {i+1}: {table['type']} format, {cols}x{rows}")
            if table.get('headers'):
                print(f"     Headers: {table['headers'][:3]}...")  # Show first 3 headers
        
        return True
        
    except ImportError as e:
        print(f"⚠️  TableExtractor - Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ TableExtractor - Test failed: {e}")
        return False


def test_link_extractor():
    """Test LinkExtractor functionality."""
    print("\n🧪 Testing LinkExtractor...")
    
    try:
        from components.extractors.link_extractor.link_extractor import LinkExtractor
        
        # Test text with various link types
        test_text = """Contact information and links:

Website: https://example.com
Email: contact@example.com
Phone: +1-555-123-4567 or (555) 987-6543
GitHub: https://github.com/user/repo
Twitter: @username
LinkedIn: linkedin.com/in/profile

File paths:
- /path/to/file.txt
- C:\\Users\\Documents\\report.pdf
- ./relative/path/data.csv

More URLs:
- http://old-site.org
- https://secure-site.net/page?param=value
- ftp://ftp.example.com/files/

Social media:
Follow us on Twitter @company or Facebook facebook.com/company
"""
        
        # Test extractor
        extractor = LinkExtractor({
            "extract_urls": True,
            "extract_emails": True,
            "extract_phone_numbers": True,
            "extract_social_media": True,
            "extract_file_paths": True,
            "categorize_domains": True,
            "validate_urls": False  # Skip validation for testing
        })
        
        # Create Document object for the new interface
        from core.base import Document
        test_doc = Document(content=test_text, metadata={"doc_id": "test_doc"})
        enhanced_docs = extractor.extract([test_doc])
        
        # Get the result from the enhanced document
        result = enhanced_docs[0].metadata
        
        # Check basic structure
        assert "urls" in result, "Should extract URLs"
        assert "emails" in result, "Should extract emails"
        
        # Should find URLs
        urls = result.get("urls", [])
        assert len(urls) >= 3, f"Expected at least 3 URLs, found {len(urls)}"
        
        # Should find emails
        emails = result.get("emails", [])
        assert len(emails) >= 1, f"Expected at least 1 email, found {len(emails)}"
        
        # Should find phone numbers
        phones = result.get("phone_numbers", [])
        assert len(phones) >= 1, f"Expected at least 1 phone number, found {len(phones)}"
        
        print(f"✅ LinkExtractor - Extracted links successfully")
        print(f"   URLs: {len(urls)}")
        print(f"   Emails: {len(emails)}")
        print(f"   Phone numbers: {len(phones)}")
        
        if "mentions" in result:
            mentions = result["mentions"]
            print(f"   Mentions: {len(mentions)}")
        
        if "file_paths" in result:
            paths = result["file_paths"]
            print(f"   File paths: {len(paths)}")
        
        # Show some examples
        if urls:
            print(f"   Example URLs: {urls[:2]}")
        if emails:
            print(f"   Example emails: {emails[:2]}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  LinkExtractor - Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ LinkExtractor - Test failed: {e}")
        return False


def test_heading_extractor():
    """Test HeadingExtractor functionality."""
    print("\n🧪 Testing HeadingExtractor...")
    
    try:
        from components.extractors.heading_extractor.heading_extractor import HeadingExtractor
        
        # Test text with various heading formats
        test_text = """DOCUMENT TITLE

# Main Heading (Markdown)

This is some content under the main heading.

## Section 1 (Markdown Level 2)

Content for section 1.

Section 2 (Underlined)
=====================

More content here.

Subsection 2.1 (Underlined)
---------------------------

### Section 3 (Markdown Level 3)

1. NUMBERED SECTION ONE
2. NUMBERED SECTION TWO

Some regular text here.

CAPS HEADING EXAMPLE

More content after caps heading.

#### Section 4 (Markdown Level 4)

Final section content.
"""
        
        # Test extractor
        extractor = HeadingExtractor({
            "extract_markdown_headings": True,
            "extract_underlined_headings": True,
            "extract_numbered_headings": True,
            "extract_caps_headings": True,
            "min_heading_length": 3,
            "max_heading_length": 100,
            "generate_toc": True,
            "analyze_structure": True
        })
        
        # Create Document object for the new interface
        from core.base import Document
        test_doc = Document(content=test_text, metadata={"doc_id": "test_doc"})
        enhanced_docs = extractor.extract([test_doc])
        
        # Get the result from the enhanced document
        result = enhanced_docs[0].metadata
        
        # Check basic structure
        assert "headings" in result, "Should extract headings"
        headings = result["headings"]
        
        # Should find multiple headings
        assert len(headings) >= 5, f"Expected at least 5 headings, found {len(headings)}"
        
        # Check heading structure
        for heading in headings:
            assert "text" in heading, "Heading should have text"
            assert "level" in heading, "Heading should have level"
            assert "type" in heading, "Heading should have type"
            assert "line_number" in heading, "Heading should have line number"
        
        # Check for different heading types
        heading_types = {h["type"] for h in headings}
        assert len(heading_types) >= 2, f"Expected multiple heading types, found {heading_types}"
        
        print(f"✅ HeadingExtractor - Found {len(headings)} headings")
        print(f"   Heading types: {sorted(heading_types)}")
        
        # Show heading structure
        for heading in headings[:5]:  # Show first 5 headings
            print(f"   {heading['level']}: {heading['text'][:50]}... ({heading['type']})")
        
        # Check for table of contents
        if "table_of_contents" in result:
            toc = result["table_of_contents"]
            print(f"   Generated TOC with {len(toc)} entries")
        
        # Check for hierarchy analysis
        if "hierarchy" in result:
            hierarchy = result["hierarchy"]
            print(f"   Document structure: max depth {hierarchy.get('max_depth', 'unknown')}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  HeadingExtractor - Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ HeadingExtractor - Test failed: {e}")
        return False


def run_extractor_tests():
    """Run all extractor tests."""
    print("🚀 Running New Extractor Tests")
    print("=" * 50)
    
    tests = [
        ("TableExtractor", test_table_extractor),
        ("LinkExtractor", test_link_extractor),
        ("HeadingExtractor", test_heading_extractor),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} - Unexpected error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Extractor Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All extractor tests passed!")
    else:
        print("⚠️  Some tests failed or were skipped due to missing dependencies")
    
    return passed, total


if __name__ == "__main__":
    run_extractor_tests()