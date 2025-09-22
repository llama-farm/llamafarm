"""Essential parser tests - real tests with minimal mocking."""

import pytest
import tempfile
from pathlib import Path

from components.parsers.text.python_parser import TextParser_Python
from components.parsers.csv.python_parser import CSVParser_Python
from components.parsers.markdown.python_parser import MarkdownParser_Python


class TestParsers:
    """Core parser functionality tests."""

    def test_text_parser(self, temp_dir):
        """Test basic text parsing."""
        parser = TextParser_Python()
        from pathlib import Path

        file_path = Path(temp_dir) / "test.txt"
        file_path.write_text("This is a test document.\nWith multiple lines.")

        result = parser.parse(str(file_path))
        assert result is not None
        assert len(result.documents) >= 1

    def test_csv_parser(self):
        """Test CSV parsing with real data."""
        parser = CSVParser_Python()
        csv_content = """name,age,city
John,30,New York
Jane,25,Boston"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            result = parser.parse(f.name)
            assert result is not None
            assert len(result.documents) > 0

            Path(f.name).unlink()

    def test_markdown_parser(self, temp_dir):
        """Test markdown parsing."""
        parser = MarkdownParser_Python()
        from pathlib import Path

        file_path = Path(temp_dir) / "test.md"
        file_path.write_text("# Header\n\nContent with **bold**.")

        result = parser.parse(str(file_path))
        assert result is not None
        assert len(result.documents) >= 1

    def test_parser_error_handling(self):
        """Test parser handles errors gracefully."""
        parser = CSVParser_Python()
        result = parser.parse("nonexistent.csv")

        assert len(result.documents) == 0
        assert len(result.errors) > 0
