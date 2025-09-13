"""
Test MIME Type Filtering

Tests the two-tier MIME type filtering system for strategies and parsers.
"""

import unittest
from pathlib import Path
from rag.core.mime_type_filter import MimeTypeFilter, filter_files_for_strategy, get_parser_for_file


class TestMimeTypeFiltering(unittest.TestCase):
    """Test MIME type filtering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = MimeTypeFilter()
        
        # Create test file paths (don't need to exist for type checking)
        self.test_files = [
            Path("test.pdf"),
            Path("document.docx"),
            Path("data.csv"),
            Path("notes.md"),
            Path("config.json"),
            Path("image.png"),
            Path("script.py"),
            Path("report.xlsx")
        ]
    
    def test_mime_type_detection(self):
        """Test MIME type detection for various file types."""
        test_cases = [
            (Path("test.pdf"), "application/pdf", ".pdf"),
            (Path("doc.docx"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
            (Path("data.csv"), "text/csv", ".csv"),
            (Path("notes.md"), "text/markdown", ".md"),
            (Path("config.json"), "application/json", ".json"),
        ]
        
        for file_path, expected_mime, expected_ext in test_cases:
            mime_type, extension = self.filter.get_mime_type(file_path)
            self.assertEqual(extension, expected_ext, f"Extension mismatch for {file_path}")
            self.assertEqual(mime_type, expected_mime, f"MIME type mismatch for {file_path}")
    
    def test_strategy_level_filtering_pdf_only(self):
        """Test strategy that only accepts PDF files."""
        strategy_config = {
            "name": "pdf_only_strategy",
            "allowed_mime_types": ["application/pdf"],
            "allowed_extensions": [".pdf"]
        }
        
        # Test individual files
        pdf_allowed, _ = self.filter.is_file_allowed_by_strategy(Path("test.pdf"), strategy_config)
        self.assertTrue(pdf_allowed, "PDF should be allowed")
        
        csv_allowed, reason = self.filter.is_file_allowed_by_strategy(Path("data.csv"), strategy_config)
        self.assertFalse(csv_allowed, "CSV should not be allowed")
        self.assertIn("not in allowed types", reason)
        
        # Test batch validation
        result = self.filter.validate_strategy_files(self.test_files, strategy_config)
        self.assertEqual(len(result['accepted']), 1)
        self.assertEqual(result['accepted'][0].name, "test.pdf")
        self.assertEqual(len(result['rejected']), 7)
    
    def test_strategy_level_filtering_spreadsheets(self):
        """Test strategy that accepts spreadsheet files."""
        strategy_config = {
            "name": "spreadsheet_strategy",
            "allowed_mime_types": [
                "text/csv",
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ],
            "allowed_extensions": [".csv", ".xls", ".xlsx"]
        }
        
        result = self.filter.validate_strategy_files(self.test_files, strategy_config)
        accepted_names = [f.name for f in result['accepted']]
        self.assertIn("data.csv", accepted_names)
        self.assertIn("report.xlsx", accepted_names)
        self.assertNotIn("test.pdf", accepted_names)
    
    def test_strategy_with_no_restrictions(self):
        """Test strategy that accepts all file types."""
        strategy_config = {
            "name": "generic_strategy",
            "allowed_mime_types": [],  # Empty means accept all
            "allowed_extensions": []   # Empty means accept all
        }
        
        result = self.filter.validate_strategy_files(self.test_files, strategy_config)
        self.assertEqual(len(result['accepted']), len(self.test_files))
        self.assertEqual(len(result['rejected']), 0)
    
    def test_parser_matching(self):
        """Test finding the best parser for a file."""
        parsers = [
            {
                "type": "PDFParser_LlamaIndex",
                "mime_types": ["application/pdf"],
                "file_extensions": [".pdf", ".PDF"],
                "priority": 10
            },
            {
                "type": "CSVParser_Pandas",
                "mime_types": ["text/csv"],
                "file_extensions": [".csv", ".tsv"]
            },
            {
                "type": "TextParser_Python",
                "file_extensions": [".txt", ".md", ".log"]
            },
            {
                "type": "GenericParser",
                # No restrictions - accepts all
            }
        ]
        
        # Test PDF matching
        pdf_parser = self.filter.find_matching_parser(Path("test.pdf"), parsers)
        self.assertIsNotNone(pdf_parser)
        self.assertEqual(pdf_parser['type'], "PDFParser_LlamaIndex")
        
        # Test CSV matching
        csv_parser = self.filter.find_matching_parser(Path("data.csv"), parsers)
        self.assertIsNotNone(csv_parser)
        self.assertEqual(csv_parser['type'], "CSVParser_Pandas")
        
        # Test Markdown matching (no MIME type, just extension)
        md_parser = self.filter.find_matching_parser(Path("notes.md"), parsers)
        self.assertIsNotNone(md_parser)
        self.assertEqual(md_parser['type'], "TextParser_Python")
        
        # Test fallback to generic parser
        json_parser = self.filter.find_matching_parser(Path("config.json"), parsers)
        self.assertIsNotNone(json_parser)
        self.assertEqual(json_parser['type'], "GenericParser")
    
    def test_file_assignment_to_parsers(self):
        """Test assigning multiple files to appropriate parsers."""
        parsers = [
            {
                "type": "PDFParser",
                "mime_types": ["application/pdf"]
            },
            {
                "type": "SpreadsheetParser",
                "file_extensions": [".csv", ".xlsx", ".xls"]
            },
            {
                "type": "TextParser",
                "mime_types": ["text/plain", "text/markdown"],
                "file_extensions": [".txt", ".md"]
            }
        ]
        
        files = [
            Path("doc1.pdf"),
            Path("doc2.pdf"),
            Path("data.csv"),
            Path("report.xlsx"),
            Path("notes.md"),
            Path("readme.txt"),
            Path("unknown.xyz")  # No matching parser
        ]
        
        assignments = self.filter.assign_files_to_parsers(files, parsers)
        
        # Check PDF assignments
        self.assertIn(0, assignments)
        self.assertEqual(len(assignments[0]), 2)
        
        # Check spreadsheet assignments
        self.assertIn(1, assignments)
        self.assertEqual(len(assignments[1]), 2)
        
        # Check text assignments
        self.assertIn(2, assignments)
        self.assertEqual(len(assignments[2]), 2)
        
        # Check unassigned
        self.assertIn('unassigned', assignments)
        self.assertEqual(len(assignments['unassigned']), 1)
    
    def test_parser_priority(self):
        """Test that parser priority is considered in matching."""
        parsers = [
            {
                "type": "LowPriorityPDFParser",
                "mime_types": ["application/pdf"],
                "priority": 1
            },
            {
                "type": "HighPriorityPDFParser",
                "mime_types": ["application/pdf"],
                "priority": 10
            }
        ]
        
        # Higher priority parser should be selected
        best_parser = self.filter.find_matching_parser(Path("test.pdf"), parsers)
        self.assertEqual(best_parser['type'], "HighPriorityPDFParser")
    
    def test_extension_only_matching(self):
        """Test matching based only on file extensions (no MIME type)."""
        strategy_config = {
            "name": "extension_only",
            "allowed_extensions": [".py", ".js", ".java"]
        }
        
        test_files = [
            Path("script.py"),
            Path("app.js"),
            Path("Main.java"),
            Path("doc.pdf")
        ]
        
        result = self.filter.validate_strategy_files(test_files, strategy_config)
        self.assertEqual(len(result['accepted']), 3)
        self.assertEqual(len(result['rejected']), 1)
        self.assertEqual(result['rejected'][0].name, "doc.pdf")


if __name__ == "__main__":
    unittest.main()