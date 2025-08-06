"""Tests for CSV parsers."""

import pytest
from pathlib import Path

from components.parsers.csv_parser import CSVParser, CustomerSupportCSVParser


class TestCSVParser:
    """Test the generic CSV parser."""

    def test_basic_parsing(self, sample_csv_file: str):
        """Test basic CSV parsing functionality."""
        parser = CSVParser(
            config={
                "content_fields": ["subject", "body"],
                "metadata_fields": ["type", "priority"],
            }
        )

        result = parser.parse(sample_csv_file)

        assert len(result.documents) == 3
        assert len(result.errors) == 0

        doc = result.documents[0]
        assert "Login Issue" in doc.content
        assert "Cannot login to the system" in doc.content
        assert doc.metadata["type"] == "Incident"
        assert doc.metadata["priority"] == "medium"

    def test_custom_configuration(self, sample_csv_file: str):
        """Test parser with custom configuration."""
        config = {
            "content_fields": ["subject"],
            "metadata_fields": ["priority", "language"],
            "combine_content": False,
        }

        parser = CSVParser(config=config)
        result = parser.parse(sample_csv_file)

        doc = result.documents[0]
        assert doc.content == "Login Issue"
        assert "body" not in doc.content
        assert doc.metadata["priority"] == "medium"
        assert doc.metadata["language"] == "en"

    def test_invalid_file(self):
        """Test parsing of non-existent file."""
        parser = CSVParser()
        result = parser.parse("nonexistent_file.csv")

        assert len(result.documents) == 0
        assert len(result.errors) > 0


class TestCustomerSupportCSVParser:
    """Test the customer support specialized parser."""

    def test_customer_support_parsing(self, sample_csv_file: str):
        """Test customer support CSV parsing with tags."""
        parser = CustomerSupportCSVParser()
        result = parser.parse(sample_csv_file)

        assert len(result.documents) == 3
        assert len(result.errors) == 0

        doc = result.documents[0]
        assert "Login Issue" in doc.content
        assert "Cannot login to the system" in doc.content
        assert "Reset your password" in doc.content
        assert doc.metadata["type"] == "Incident"
        assert doc.metadata["tags"] == ["Login", "Authentication"]

    def test_priority_mapping(self, sample_csv_file: str):
        """Test priority numeric mapping."""
        parser = CustomerSupportCSVParser()
        result = parser.parse(sample_csv_file)

        # Find documents by priority
        medium_doc = next(
            d for d in result.documents if d.metadata["priority"] == "medium"
        )
        high_doc = next(d for d in result.documents if d.metadata["priority"] == "high")
        critical_doc = next(
            d for d in result.documents if d.metadata["priority"] == "critical"
        )

        assert medium_doc.metadata["priority_numeric"] == 2
        assert high_doc.metadata["priority_numeric"] == 3
        assert critical_doc.metadata["priority_numeric"] == 4

    def test_tag_extraction(self, sample_csv_file: str):
        """Test tag extraction from multiple tag columns."""
        parser = CustomerSupportCSVParser()
        result = parser.parse(sample_csv_file)

        # Check tags are properly extracted
        for doc in result.documents:
            assert "tags" in doc.metadata
            assert isinstance(doc.metadata["tags"], list)
            assert len(doc.metadata["tags"]) >= 1
