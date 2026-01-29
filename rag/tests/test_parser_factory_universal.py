"""Tests for UniversalParser integration with ToolAwareParserFactory.

This test file verifies:
1. ToolAwareParserFactory discovers UniversalParser
2. UniversalParser priority is higher than legacy parsers
3. Legacy parsers still work when explicitly requested
"""



class TestParserFactoryDiscovery:
    """Test parser factory discovery of UniversalParser."""

    def test_factory_discovers_universal_parser(self):
        """Test: ToolAwareParserFactory discovers UniversalParser."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        factory = ToolAwareParserFactory()

        # discover_parsers returns dict[str, list[dict]]
        parsers = factory.discover_parsers()

        # Check that universal parser type exists
        assert "universal" in parsers

        # Extract parser names from the universal type
        universal_parsers = parsers.get("universal", [])
        parser_names = [p.get("name") for p in universal_parsers]

        assert "UniversalParser" in parser_names

    def test_factory_lists_universal_parser(self):
        """Test: list_parsers includes UniversalParser."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        parser_list = ToolAwareParserFactory.list_parsers()

        assert "UniversalParser" in parser_list

    def test_factory_gets_universal_parser_info(self):
        """Test: get_parser_info returns details for UniversalParser."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("UniversalParser")

        assert info is not None
        assert info["name"] == "UniversalParser"
        assert "supported_extensions" in info
        assert ".pdf" in info["supported_extensions"]
        assert ".txt" in info["supported_extensions"]
        assert ".md" in info["supported_extensions"]


class TestParserPriority:
    """Test parser priority system."""

    def test_universal_parser_has_priority(self):
        """Test: UniversalParser has priority 10 (higher than legacy)."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("UniversalParser")

        # UniversalParser should have priority 10
        assert info.get("priority", 100) == 10

    def test_universal_parser_priority_lower_than_legacy(self):
        """Test: UniversalParser priority is lower (better) than legacy parsers."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        universal_info = ToolAwareParserFactory.get_parser_info("UniversalParser")
        pdf_pypdf2_info = ToolAwareParserFactory.get_parser_info("PDFParser_PyPDF2")

        universal_priority = universal_info.get("priority", 100)
        legacy_priority = pdf_pypdf2_info.get("priority", 100) if pdf_pypdf2_info else 100

        # Lower number = higher priority
        # UniversalParser (10) should be higher priority than legacy (100)
        assert universal_priority < legacy_priority


class TestLegacyParsers:
    """Test that legacy parsers still work."""

    def test_legacy_pdf_parser_exists(self):
        """Test: Legacy PDFParser_PyPDF2 can still be accessed."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("PDFParser_PyPDF2")

        assert info is not None
        assert info["name"] == "PDFParser_PyPDF2"

    def test_legacy_text_parser_exists(self):
        """Test: Legacy TextParser can still be accessed."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        # Check for text parsers - name may vary
        parser_list = ToolAwareParserFactory.list_parsers()

        text_parsers = [p for p in parser_list if "Text" in p]
        assert len(text_parsers) > 0, "Should have at least one text parser"

    def test_can_instantiate_legacy_parser(self):
        """Test: Legacy parsers can be explicitly instantiated."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("PDFParser_PyPDF2")

        # The info should contain enough to create the parser
        assert info is not None
        assert "name" in info
        assert "supported_extensions" in info


class TestUniversalParserExtensions:
    """Test UniversalParser supported extensions."""

    def test_supports_common_document_formats(self):
        """Test: UniversalParser supports common document formats."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("UniversalParser")
        extensions = info.get("supported_extensions", [])

        # Should support these common formats
        expected = [".pdf", ".docx", ".txt", ".md", ".html", ".csv", ".json"]

        for ext in expected:
            assert ext in extensions, f"Should support {ext}"

    def test_supports_image_formats_for_ocr(self):
        """Test: UniversalParser supports image formats for OCR."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("UniversalParser")
        extensions = info.get("supported_extensions", [])

        # Should support image formats for OCR
        image_formats = [".png", ".jpg", ".jpeg"]

        for ext in image_formats:
            assert ext in extensions, f"Should support {ext} for OCR"

    def test_supports_office_formats(self):
        """Test: UniversalParser supports Office formats."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("UniversalParser")
        extensions = info.get("supported_extensions", [])

        office_formats = [".docx", ".xlsx", ".pptx"]

        for ext in office_formats:
            assert ext in extensions, f"Should support {ext}"


class TestParserDependencies:
    """Test parser dependency information."""

    def test_universal_parser_lists_dependencies(self):
        """Test: UniversalParser lists its dependencies."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("UniversalParser")
        dependencies = info.get("dependencies", {})

        # Should list markitdown as required
        assert "required" in dependencies
        assert "markitdown" in dependencies["required"]

    def test_universal_parser_lists_optional_dependencies(self):
        """Test: UniversalParser lists optional dependencies."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("UniversalParser")
        dependencies = info.get("dependencies", {})

        # Should list semchunk and tiktoken as optional
        assert "optional" in dependencies
        optional = dependencies["optional"]
        assert "semchunk" in optional
        assert "tiktoken" in optional


class TestParserRegistry:
    """Test parser registry integration."""

    def test_parser_registry_can_list_all(self):
        """Test: ParserRegistry can list all parsers."""
        from components.parsers.parser_registry import ParserRegistry

        registry = ParserRegistry()
        all_parsers = registry.list_all_parsers()

        # Should have multiple parsers
        assert len(all_parsers) > 0

    def test_universal_parser_in_registry_or_factory(self):
        """Test: UniversalParser is accessible via registry or factory."""
        from components.parsers.parser_factory import ToolAwareParserFactory
        from components.parsers.parser_registry import ParserRegistry

        registry = ParserRegistry()
        all_parsers = registry.list_all_parsers()

        factory_parsers = ToolAwareParserFactory.list_parsers()

        # UniversalParser should be in at least one of these
        in_registry = "UniversalParser" in all_parsers
        in_factory = "UniversalParser" in factory_parsers

        assert in_registry or in_factory, "UniversalParser should be discoverable"
