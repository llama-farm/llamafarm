#!/usr/bin/env python3
"""
Test script for new parsers: DocxParser, PlainTextParser, HTMLParser, ExcelParser.
"""

import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_plain_text_parser():
    """Test PlainTextParser functionality."""
    print("🧪 Testing PlainTextParser...")
    
    try:
        from components.parsers.text_parser import PlainTextParser
        
        # Create test text file
        test_content = """# Document Title
        
This is a test document with some structure.

## Section 1
- Item 1
- Item 2
- Item 3

## Section 2
This section has some content with numbers: 123, 456.

```python
def hello():
    print("Hello World")
```

Final paragraph with some text.
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            # Test parser
            parser = PlainTextParser(name="PlainTextParser", config={
                "preserve_line_breaks": True,
                "detect_structure": True,
                "strip_empty_lines": True
            })
            
            docs = parser.parse(temp_file)
            
            assert len(docs) == 1, f"Expected 1 document, got {len(docs)}"
            doc = docs[0]
            
            assert doc.content, "Document content should not be empty"
            assert doc.metadata["parser_type"] == "PlainTextParser"
            assert "line_count" in doc.metadata
            assert "word_count" in doc.metadata
            assert doc.metadata["word_count"] > 0
            
            print(f"✅ PlainTextParser - Document parsed successfully")
            print(f"   Content length: {len(doc.content)} chars")
            print(f"   Word count: {doc.metadata.get('word_count', 0)}")
            print(f"   Line count: {doc.metadata.get('line_count', 0)}")
            
            if "has_headers" in doc.metadata:
                print(f"   Has headers: {doc.metadata['has_headers']}")
            if "has_lists" in doc.metadata:
                print(f"   Has lists: {doc.metadata['has_lists']}")
            
            return True
            
        finally:
            os.unlink(temp_file)
            
    except ImportError as e:
        print(f"⚠️  PlainTextParser - Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ PlainTextParser - Test failed: {e}")
        return False


def test_html_parser():
    """Test HTMLParser functionality."""
    print("\\n🧪 Testing HTMLParser...")
    
    try:
        from components.parsers.html_parser import HTMLParser
        
        # Create test HTML file
        test_html = """<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
    <meta name="description" content="A test HTML document">
    <meta name="author" content="Test Author">
</head>
<body>
    <h1>Main Title</h1>
    <p>This is a test paragraph with <a href="https://example.com">a link</a>.</p>
    
    <h2>Section with List</h2>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
    
    <h2>Section with Table</h2>
    <table>
        <tr>
            <th>Name</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Test</td>
            <td>123</td>
        </tr>
    </table>
    
    <img src="test.jpg" alt="Test image">
    
    <script>console.log("This should be removed");</script>
</body>
</html>"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(test_html)
            temp_file = f.name
        
        try:
            # Test parser
            parser = HTMLParser(name="HTMLParser", config={
                "extract_links": True,
                "extract_images": True,
                "extract_tables": True,
                "extract_meta": True,
                "remove_scripts": True
            })
            
            docs = parser.parse(temp_file)
            
            assert len(docs) == 1, f"Expected 1 document, got {len(docs)}"
            doc = docs[0]
            
            assert doc.content, "Document content should not be empty"
            assert doc.metadata["parser_type"] == "HTMLParser"
            assert "title" in doc.metadata
            assert doc.metadata["title"] == "Test Document"
            
            print(f"✅ HTMLParser - Document parsed successfully")
            print(f"   Title: {doc.metadata.get('title', 'N/A')}")
            print(f"   Content length: {len(doc.content)} chars")
            
            if "links" in doc.metadata:
                print(f"   Links found: {len(doc.metadata['links'])}")
            if "images" in doc.metadata:
                print(f"   Images found: {len(doc.metadata['images'])}")
            if "tables" in doc.metadata:
                print(f"   Tables found: {len(doc.metadata['tables'])}")
            
            # Verify script was removed
            assert "console.log" not in doc.content, "Scripts should be removed"
            
            return True
            
        finally:
            os.unlink(temp_file)
            
    except ImportError as e:
        print(f"⚠️  HTMLParser - Import failed (beautifulsoup4 required): {e}")
        return False
    except Exception as e:
        print(f"❌ HTMLParser - Test failed: {e}")
        return False


def test_docx_parser():
    """Test DocxParser functionality."""
    print("\\n🧪 Testing DocxParser...")
    
    try:
        from components.parsers.docx_parser import DocxParser
        
        print("⚠️  DocxParser - Skipping test (requires actual .docx file and python-docx)")
        print("   To test manually: create a test.docx file and run:")
        print("   parser = DocxParser({})")
        print("   docs = parser.parse('test.docx')")
        
        # Test basic instantiation
        parser = DocxParser(name="DocxParser", config={
            "extract_tables": True,
            "include_document_properties": True
        })
        
        print("✅ DocxParser - Basic instantiation successful")
        return True
        
    except ImportError as e:
        print(f"⚠️  DocxParser - Import failed (python-docx required): {e}")
        print("   Install with: pip install python-docx")
        return False
    except Exception as e:
        print(f"❌ DocxParser - Test failed: {e}")
        return False


def test_excel_parser():
    """Test ExcelParser functionality."""
    print("\\n🧪 Testing ExcelParser...")
    
    try:
        from components.parsers.excel_parser import ExcelParser
        
        print("⚠️  ExcelParser - Skipping test (requires actual .xlsx file and pandas/openpyxl)")
        print("   To test manually: create a test.xlsx file and run:")
        print("   parser = ExcelParser({})")
        print("   docs = parser.parse('test.xlsx')")
        
        # Test basic instantiation
        parser = ExcelParser(name="ExcelParser", config={
            "parse_all_sheets": True,
            "include_sheet_stats": True
        })
        
        print("✅ ExcelParser - Basic instantiation successful")
        return True
        
    except ImportError as e:
        print(f"⚠️  ExcelParser - Import failed (pandas/openpyxl required): {e}")
        print("   Install with: pip install pandas openpyxl xlrd")
        return False
    except Exception as e:
        print(f"❌ ExcelParser - Test failed: {e}")
        return False


def test_markdown_parser():
    """Test MarkdownParser functionality."""
    print("\\n🧪 Testing MarkdownParser...")
    
    try:
        from components.parsers.markdown_parser import MarkdownParser
        
        # Create test markdown file
        test_markdown = """---
title: Test Document  
author: Test Author
tags: [test, markdown]
---

# Main Title

This is a test markdown document with various elements.

## Section 1

Here's some content with **bold** and *italic* text.

### Subsection

- Item 1
- Item 2
- Item 3

## Section 2

Here's a code block:

```python
def hello():
    print("Hello World")
```

And a [link to example](https://example.com).

| Name | Value |
|------|-------|
| Test | 123   |
| Demo | 456   |
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(test_markdown)
            temp_file = f.name
        
        try:
            # Test parser
            parser = MarkdownParser(name="MarkdownParser", config={
                "extract_frontmatter": True,
                "chunk_by_headings": True,
                "preserve_code_blocks": True,
                "extract_links": True
            })
            
            docs = parser.parse(temp_file)
            
            assert len(docs) >= 1, f"Expected at least 1 document, got {len(docs)}"
            doc = docs[0]
            
            assert doc.content, "Document content should not be empty"
            assert doc.metadata["parser_type"] == "MarkdownParser"
            
            print(f"✅ MarkdownParser - Document parsed successfully")
            print(f"   Documents created: {len(docs)}")
            print(f"   Content length: {len(doc.content)} chars")
            
            if "frontmatter" in doc.metadata:
                print(f"   Frontmatter extracted: {bool(doc.metadata['frontmatter'])}")
            if "headings" in doc.metadata:
                print(f"   Headings found: {len(doc.metadata['headings'])}")
            if "links" in doc.metadata:
                print(f"   Links found: {len(doc.metadata['links'])}")
            
            return True
            
        finally:
            os.unlink(temp_file)
            
    except ImportError as e:
        print(f"⚠️  MarkdownParser - Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ MarkdownParser - Test failed: {e}")
        return False


def run_parser_tests():
    """Run all parser tests."""
    print("🚀 Running New Parser Tests")
    print("=" * 50)
    
    tests = [
        ("PlainTextParser", test_plain_text_parser),
        ("HTMLParser", test_html_parser),
        ("MarkdownParser", test_markdown_parser),
        ("DocxParser", test_docx_parser),
        ("ExcelParser", test_excel_parser),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} - Unexpected error: {e}")
    
    print("\\n" + "=" * 50)
    print(f"📊 Parser Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All parser tests passed!")
    else:
        print("⚠️  Some tests failed or were skipped due to missing dependencies")
    
    return passed, total


if __name__ == "__main__":
    run_parser_tests()