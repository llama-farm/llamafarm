"""Tests for Text Parser component."""

import pytest
from pathlib import Path
import sys
import tempfile
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.base import Document
from components.parsers.text_parser.text_parser import PlainTextParser


class TestPlainTextParser:
    """Test PlainTextParser functionality."""
    
    @pytest.fixture
    def sample_text_content(self):
        """Sample text content for testing."""
        return """# Sample Document Title

This is a sample document for testing the text parser functionality.
It contains multiple paragraphs and different text structures.

## Section 1: Introduction

This section introduces the content. It has several sentences that should
be parsed correctly by the text parser component.

## Section 2: Data

Here are some key points:
- Point one with important information
- Point two with additional details
- Point three with concluding remarks

### Subsection 2.1

More detailed information in a subsection format.

## Conclusion

The document concludes with this final section that summarizes the content."""
    
    @pytest.fixture
    def temp_text_file(self, sample_text_content):
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def default_parser(self):
        """Create default text parser."""
        return PlainTextParser(name="PlainTextParser", config={
            "encoding": "utf-8",
            "chunk_size": None,
            "preserve_line_breaks": True,
            "strip_empty_lines": True,
            "detect_structure": True
        })
    
    def test_parser_initialization(self):
        """Test parser initialization with different configs."""
        # Default config
        parser = PlainTextParser(name="PlainTextParser")
        assert parser is not None
        assert parser.encoding == "auto"
        assert parser.preserve_line_breaks == True
        
        # Custom config
        custom_config = {
            "encoding": "utf-8",
            "chunk_size": 1000,
            "preserve_line_breaks": False,
            "strip_empty_lines": False,
            "detect_structure": False
        }
        parser = PlainTextParser(name="PlainTextParser", config=custom_config)
        assert parser.encoding == "utf-8"
        assert parser.chunk_size == 1000
        assert parser.preserve_line_breaks == False
        assert parser.detect_structure == False
    
    def test_basic_text_parsing(self, default_parser, temp_text_file):
        """Test basic text file parsing."""
        result = default_parser.parse(temp_text_file)
        
        # Should return ProcessingResult
        from core.base import ProcessingResult
        assert isinstance(result, ProcessingResult)
        assert isinstance(result.documents, list)
        assert len(result.documents) > 0
        assert all(isinstance(doc, Document) for doc in result.documents)
        
        # Check document content
        doc = result.documents[0]
        assert len(doc.content) > 0
        assert "Sample Document Title" in doc.content
        assert doc.id is not None
        assert doc.source == temp_text_file
    
    def test_metadata_extraction(self, default_parser, temp_text_file):
        """Test metadata extraction from parsed file."""
        result = default_parser.parse(temp_text_file)
        doc = result.documents[0]
        
        # Should have file metadata
        assert "file_path" in doc.metadata
        assert "file_name" in doc.metadata
        assert "file_size" in doc.metadata
        assert "parser_type" in doc.metadata
        assert doc.metadata["parser_type"] == "PlainTextParser"
        
        # Should have content statistics
        assert "line_count" in doc.metadata
        assert "character_count" in doc.metadata
        assert "word_count" in doc.metadata
        assert doc.metadata["word_count"] > 0
    
    def test_structure_detection(self, default_parser, temp_text_file):
        """Test structure detection in text content."""
        result = default_parser.parse(temp_text_file)
        doc = result.documents[0]
        
        # Should detect structure elements
        if "has_headers" in doc.metadata:
            assert isinstance(doc.metadata["has_headers"], bool)
        if "has_lists" in doc.metadata:
            assert isinstance(doc.metadata["has_lists"], bool)
        if "headers" in doc.metadata:
            assert isinstance(doc.metadata["headers"], list)
    
    def test_chunking_functionality(self, sample_text_content):  
        """Test document chunking with size limits."""
        # Create parser with small chunk size
        chunking_parser = PlainTextParser(name="PlainTextParser", config={
            "chunk_size": 200,  # Small chunk size
            "preserve_line_breaks": True
        })
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text_content)
            temp_path = f.name
        
        try:
            result = chunking_parser.parse(temp_path)
            
            # Should create multiple chunks for large content
            if len(sample_text_content) > 200:
                assert len(result.documents) > 1
                
                # Check chunk metadata - updated for new hash-based system
                for i, doc in enumerate(result.documents):
                    # New system uses chunk_index (0-based) and chunk_id
                    assert "chunk_index" in doc.metadata
                    assert "chunk_id" in doc.metadata
                    assert "chunk_hash" in doc.metadata
                    assert "total_chunks" in doc.metadata
                    
                    # Verify chunk_index is 0-based
                    assert doc.metadata["chunk_index"] == i
                    
                    # Verify chunk_id format: doc_{doc_hash}_chunk_{index}_{content_hash}
                    assert doc.metadata["chunk_id"].startswith("doc_")
                    assert f"_chunk_{i}_" in doc.metadata["chunk_id"]
                    
                    # Check content size
                    assert len(doc.content) <= 250  # Allow some flexibility
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_encoding_detection(self, default_parser):
        """Test encoding detection and handling."""
        # Create file with UTF-8 content
        test_content = "Test content with special characters: åäö ñüé"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', encoding='utf-8', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = default_parser.parse(temp_path)
            
            # Should parse successfully
            assert len(result.documents) == 1
            assert "special characters" in result.documents[0].content
            assert "encoding" in result.documents[0].metadata
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_empty_file_handling(self, default_parser):
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            result = default_parser.parse(temp_path)
            
            # Should handle gracefully
            assert isinstance(result.documents, list)
            assert len(result.documents) == 1
            assert result.documents[0].content == ""
            assert result.documents[0].metadata["word_count"] == 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_nonexistent_file_handling(self, default_parser):
        """Test handling of nonexistent files."""
        with pytest.raises(FileNotFoundError):
            default_parser.parse("/nonexistent/path/file.txt")
    
    def test_can_parse_method(self, default_parser):
        """Test the can_parse method for file type detection."""
        # Should handle text file extensions
        assert default_parser.can_parse("document.txt") == True
        assert default_parser.can_parse("logfile.log") == True
        assert default_parser.can_parse("readme.README") == True
        
        # Should reject other file types
        assert default_parser.can_parse("document.pdf") == False
        assert default_parser.can_parse("image.jpg") == False
        assert default_parser.can_parse("data.csv") == False
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        extensions = PlainTextParser.get_supported_extensions()
        
        assert isinstance(extensions, list)
        assert ".txt" in extensions
        assert ".log" in extensions
        assert len(extensions) > 0
    
    def test_strip_empty_lines_option(self, sample_text_content):
        """Test empty line stripping functionality."""
        # Add extra empty lines to content
        content_with_empty_lines = sample_text_content + "\n\n\n\nExtra content\n\n\n"
        
        # Parser with strip_empty_lines=True
        strip_parser = PlainTextParser(name="PlainTextParser", config={
            "strip_empty_lines": True
        })
        
        # Parser with strip_empty_lines=False  
        no_strip_parser = PlainTextParser(name="PlainTextParser", config={
            "strip_empty_lines": False
        })
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content_with_empty_lines)
            temp_path = f.name
        
        try:
            result_stripped = strip_parser.parse(temp_path)
            result_not_stripped = no_strip_parser.parse(temp_path)
            
            # Stripped version should have fewer lines
            stripped_lines = result_stripped.documents[0].metadata.get("line_count", 0)
            not_stripped_lines = result_not_stripped.documents[0].metadata.get("line_count", 0)
            
            assert stripped_lines <= not_stripped_lines
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_preserve_line_breaks_option(self, sample_text_content):
        """Test line break preservation option."""
        # Parser that preserves line breaks
        preserve_parser = PlainTextParser(name="PlainTextParser", config={
            "preserve_line_breaks": True
        })
        
        # Parser that doesn't preserve line breaks
        no_preserve_parser = PlainTextParser(name="PlainTextParser", config={
            "preserve_line_breaks": False
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text_content)
            temp_path = f.name
        
        try:
            result_preserved = preserve_parser.parse(temp_path)
            result_not_preserved = no_preserve_parser.parse(temp_path)
            
            # Should both parse successfully
            assert len(result_preserved.documents) > 0
            assert len(result_not_preserved.documents) > 0
            
            # Content should be different
            preserved_content = result_preserved.documents[0].content
            not_preserved_content = result_not_preserved.documents[0].content
            
            # Both should contain the main text
            assert "Sample Document Title" in preserved_content
            assert "Sample Document Title" in not_preserved_content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_description(self):
        """Test parser description method."""
        description = PlainTextParser.get_description()
        assert isinstance(description, str)
        assert len(description) > 0
        assert "plain text" in description.lower()


if __name__ == "__main__":
    pytest.main([__file__])