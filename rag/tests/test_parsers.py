"""Tests for CSV parsers (aligned with current parser API)."""

from components.parsers.csv.python_parser import CSVParser_Python


class TestCSVParser:
    """Test the generic CSV parser."""

    def test_basic_parsing(self, sample_csv_file: str):
        """Test basic CSV parsing functionality."""
        parser = CSVParser_Python(config={"chunk_size": 1000})

        result = parser.parse(sample_csv_file)

        assert len(result.documents) >= 1
        assert len(result.errors) == 0

        doc = result.documents[0]
        assert "Login Issue" in doc.content
        assert "Cannot login to the system" in doc.content
        # Type/priority appear in table content for Python parser
        assert "Incident" in doc.content
        assert "medium" in doc.content

    def test_custom_configuration(self, sample_csv_file: str):
        """Test parser with custom configuration."""
        parser = CSVParser_Python(config={"chunk_size": 1000})
        result = parser.parse(sample_csv_file)

        doc = result.documents[0]
        assert "Login Issue" in doc.content
        assert "medium" in doc.content

    def test_invalid_file(self):
        """Test parsing of non-existent file."""
        parser = CSVParser_Python()
        result = parser.parse("nonexistent_file.csv")

        assert len(result.documents) == 0
        assert len(result.errors) > 0


class TestCustomerSupportCSVParser:
    """Customer support CSV parsing behavior checks (generic parser)."""

    def test_customer_support_parsing(self, sample_csv_file: str):
        parser = CSVParser_Python(config={"chunk_size": 1000})
        result = parser.parse(sample_csv_file)

        assert len(result.documents) >= 1
        assert len(result.errors) == 0

        doc = result.documents[0]
        assert "Login Issue" in doc.content
        assert "Cannot login to the system" in doc.content
        assert "Reset your password" in doc.content
        assert "Incident" in doc.content

    def test_priority_values_present(self, sample_csv_file: str):
        parser = CSVParser_Python(config={"chunk_size": 1000})
        result = parser.parse(sample_csv_file)

        doc_text = "\n".join(d.content for d in result.documents)
        assert any(p in doc_text for p in ["medium", "high", "critical"])
