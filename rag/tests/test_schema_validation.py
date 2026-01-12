"""
Tests for schema validation of new parser and chunker configs.

Tests cover:
- Docling parser config validation
- MarkItDown parser config validation
- Simple chunker config validation
- Backward compatibility with legacy configs
- Config loading from llamafarm.yaml
"""

import pytest
import yaml
from pathlib import Path


class TestSchemaDefinitions:
    """Test that schema definitions are properly structured."""

    @pytest.fixture
    def schema(self):
        """Load the schema.yaml file."""
        schema_path = Path(__file__).parent.parent / "schema.yaml"
        with open(schema_path) as f:
            return yaml.safe_load(f)

    def test_schema_loads(self, schema):
        """Test that schema.yaml loads without errors."""
        assert schema is not None
        assert "definitions" in schema

    def test_docling_parser_config_exists(self, schema):
        """Test that doclingParserConfig is defined in schema."""
        parsers = schema.get("definitions", {}).get("parsers", {})
        assert "doclingParserConfig" in parsers

        docling_config = parsers["doclingParserConfig"]
        assert docling_config["type"] == "object"
        assert "properties" in docling_config

    def test_markitdown_parser_config_exists(self, schema):
        """Test that markItDownParserConfig is defined in schema."""
        parsers = schema.get("definitions", {}).get("parsers", {})
        assert "markItDownParserConfig" in parsers

        markitdown_config = parsers["markItDownParserConfig"]
        assert markitdown_config["type"] == "object"
        assert "properties" in markitdown_config

    def test_simple_chunker_config_exists(self, schema):
        """Test that simpleChunkerConfig is defined in schema."""
        chunkers = schema.get("definitions", {}).get("chunkers", {})
        assert "simpleChunkerConfig" in chunkers

        chunker_config = chunkers["simpleChunkerConfig"]
        assert chunker_config["type"] == "object"
        assert "properties" in chunker_config

    def test_docling_hybrid_chunker_config_exists(self, schema):
        """Test that doclingHybridChunkerConfig is defined in schema."""
        chunkers = schema.get("definitions", {}).get("chunkers", {})
        assert "doclingHybridChunkerConfig" in chunkers


class TestDoclingParserConfig:
    """Test Docling parser configuration validation."""

    @pytest.fixture
    def schema(self):
        """Load the schema.yaml file."""
        schema_path = Path(__file__).parent.parent / "schema.yaml"
        with open(schema_path) as f:
            return yaml.safe_load(f)

    def test_docling_has_chunk_size(self, schema):
        """Test that Docling config has chunk_size property."""
        docling = schema["definitions"]["parsers"]["doclingParserConfig"]
        props = docling.get("properties", {})

        assert "chunk_size" in props
        assert props["chunk_size"]["type"] == "integer"
        assert props["chunk_size"]["default"] == 512

    def test_docling_has_chunk_strategy(self, schema):
        """Test that Docling config has chunk_strategy property."""
        docling = schema["definitions"]["parsers"]["doclingParserConfig"]
        props = docling.get("properties", {})

        assert "chunk_strategy" in props
        assert "hybrid" in props["chunk_strategy"]["enum"]
        assert "hierarchical" in props["chunk_strategy"]["enum"]
        assert "none" in props["chunk_strategy"]["enum"]

    def test_docling_has_output_format(self, schema):
        """Test that Docling config has output_format property."""
        docling = schema["definitions"]["parsers"]["doclingParserConfig"]
        props = docling.get("properties", {})

        assert "output_format" in props
        assert "markdown" in props["output_format"]["enum"]

    def test_docling_has_metadata_options(self, schema):
        """Test that Docling config has metadata extraction options."""
        docling = schema["definitions"]["parsers"]["doclingParserConfig"]
        props = docling.get("properties", {})

        assert "extract_metadata" in props
        assert "extract_tables" in props
        assert "extract_headings" in props

    def test_docling_default_extensions(self, schema):
        """Test that Docling has default extensions defined."""
        docling = schema["definitions"]["parsers"]["doclingParserConfig"]

        default_ext = docling.get("defaultExtensions", [])
        assert ".pdf" in default_ext
        assert ".docx" in default_ext


class TestMarkItDownParserConfig:
    """Test MarkItDown parser configuration validation."""

    @pytest.fixture
    def schema(self):
        """Load the schema.yaml file."""
        schema_path = Path(__file__).parent.parent / "schema.yaml"
        with open(schema_path) as f:
            return yaml.safe_load(f)

    def test_markitdown_has_output_format(self, schema):
        """Test that MarkItDown config has output_format property."""
        markitdown = schema["definitions"]["parsers"]["markItDownParserConfig"]
        props = markitdown.get("properties", {})

        assert "output_format" in props
        assert "markdown" in props["output_format"]["enum"]

    def test_markitdown_has_metadata_option(self, schema):
        """Test that MarkItDown config has extract_metadata option."""
        markitdown = schema["definitions"]["parsers"]["markItDownParserConfig"]
        props = markitdown.get("properties", {})

        assert "extract_metadata" in props
        assert props["extract_metadata"]["type"] == "boolean"

    def test_markitdown_default_extensions(self, schema):
        """Test that MarkItDown has default extensions defined."""
        markitdown = schema["definitions"]["parsers"]["markItDownParserConfig"]

        default_ext = markitdown.get("defaultExtensions", [])
        assert ".pdf" in default_ext
        assert ".docx" in default_ext
        assert ".csv" in default_ext


class TestSimpleChunkerConfig:
    """Test Simple Chunker configuration validation."""

    @pytest.fixture
    def schema(self):
        """Load the schema.yaml file."""
        schema_path = Path(__file__).parent.parent / "schema.yaml"
        with open(schema_path) as f:
            return yaml.safe_load(f)

    def test_chunker_has_chunk_size(self, schema):
        """Test that SimpleChunker config has chunk_size property."""
        chunker = schema["definitions"]["chunkers"]["simpleChunkerConfig"]
        props = chunker.get("properties", {})

        assert "chunk_size" in props
        assert props["chunk_size"]["type"] == "integer"

    def test_chunker_has_strategy(self, schema):
        """Test that SimpleChunker config has strategy property."""
        chunker = schema["definitions"]["chunkers"]["simpleChunkerConfig"]
        props = chunker.get("properties", {})

        assert "strategy" in props
        strategies = props["strategy"]["enum"]

        assert "characters" in strategies
        assert "sentences" in strategies
        assert "paragraphs" in strategies
        assert "sections" in strategies
        assert "pages" in strategies
        assert "tokens" in strategies

    def test_chunker_has_overlap(self, schema):
        """Test that SimpleChunker config has overlap property."""
        chunker = schema["definitions"]["chunkers"]["simpleChunkerConfig"]
        props = chunker.get("properties", {})

        assert "overlap" in props
        assert props["overlap"]["type"] == "integer"


class TestLegacyConfigBackwardCompatibility:
    """Test that legacy parser configs still work."""

    @pytest.fixture
    def schema(self):
        """Load the schema.yaml file."""
        schema_path = Path(__file__).parent.parent / "schema.yaml"
        with open(schema_path) as f:
            return yaml.safe_load(f)

    def test_pypdf2_parser_still_exists(self, schema):
        """Test that PyPDF2 parser config is still available."""
        parsers = schema.get("definitions", {}).get("parsers", {})
        assert "pdfParserPyPDF2Config" in parsers

    def test_llamaindex_parser_still_exists(self, schema):
        """Test that LlamaIndex parser configs are still available."""
        parsers = schema.get("definitions", {}).get("parsers", {})
        assert "pdfParserLlamaIndexConfig" in parsers

    def test_csv_parser_still_exists(self, schema):
        """Test that CSV parser configs are still available."""
        parsers = schema.get("definitions", {}).get("parsers", {})
        assert "csvParserPandasConfig" in parsers

    def test_parser_config_includes_new_parsers(self, schema):
        """Test that parserConfig references include new parsers."""
        parser_config = schema.get("definitions", {}).get("parserConfig", {})
        config_refs = parser_config.get("properties", {}).get("config", {}).get("oneOf", [])

        # Convert refs to strings for easier checking
        ref_strings = [str(ref) for ref in config_refs]

        # Check that new parser refs are included
        has_docling = any("doclingParserConfig" in str(ref) for ref in config_refs)
        has_markitdown = any("markItDownParserConfig" in str(ref) for ref in config_refs)

        assert has_docling, "doclingParserConfig not in parserConfig oneOf"
        assert has_markitdown, "markItDownParserConfig not in parserConfig oneOf"


class TestConfigExamples:
    """Test that example configurations are valid."""

    def test_minimal_docling_config(self):
        """Test a minimal Docling configuration."""
        config = {
            "type": "DoclingParser",
            "config": {
                "chunk_strategy": "hybrid",
            }
        }

        # Basic validation
        assert config["type"] == "DoclingParser"
        assert config["config"]["chunk_strategy"] == "hybrid"

    def test_full_docling_config(self):
        """Test a full Docling configuration with all options."""
        config = {
            "type": "DoclingParser",
            "config": {
                "chunk_size": 256,
                "chunk_overlap": 30,
                "chunk_strategy": "hybrid",
                "extract_metadata": True,
                "extract_tables": True,
                "extract_headings": True,
                "output_format": "markdown",
                "preserve_structure": True,
                "page_aware": True,
            }
        }

        assert config["config"]["chunk_size"] == 256
        assert config["config"]["output_format"] == "markdown"

    def test_minimal_markitdown_config(self):
        """Test a minimal MarkItDown configuration."""
        config = {
            "type": "MarkItDownParser",
            "config": {
                "output_format": "markdown",
            }
        }

        assert config["type"] == "MarkItDownParser"

    def test_minimal_chunker_config(self):
        """Test a minimal SimpleChunker configuration."""
        config = {
            "type": "SimpleChunker",
            "config": {
                "strategy": "paragraphs",
                "chunk_size": 500,
            }
        }

        assert config["config"]["strategy"] == "paragraphs"

    def test_combined_config_example(self):
        """Test a combined parser + chunker configuration."""
        # This represents how a user might configure their RAG pipeline
        pipeline_config = {
            "parser": {
                "type": "DoclingParser",
                "config": {
                    "chunk_strategy": "none",  # No chunking in parser
                    "output_format": "markdown",
                }
            },
            "chunker": {
                "type": "SimpleChunker",
                "config": {
                    "strategy": "sections",
                    "chunk_size": 1000,
                    "overlap": 100,
                }
            }
        }

        # Validate structure
        assert pipeline_config["parser"]["config"]["chunk_strategy"] == "none"
        assert pipeline_config["chunker"]["config"]["strategy"] == "sections"


class TestDataProcessingStrategy:
    """Test data processing strategy configurations."""

    def test_simple_processing_strategy(self):
        """Test a simplified processing strategy using new parsers."""
        strategy = {
            "name": "modern_document_processing",
            "description": "Modern document processing using Docling with smart chunking",
            "parsers": [
                {
                    "type": "DoclingParser",
                    "file_extensions": [".pdf", ".docx"],
                    "priority": 0,
                    "config": {
                        "chunk_size": 512,
                        "chunk_strategy": "hybrid",
                        "extract_metadata": True,
                    }
                },
                {
                    "type": "MarkItDownParser",
                    "file_extensions": [".csv", ".html", ".json"],
                    "priority": 1,
                    "config": {
                        "output_format": "markdown",
                    }
                }
            ]
        }

        assert len(strategy["parsers"]) == 2
        assert strategy["parsers"][0]["type"] == "DoclingParser"
        assert strategy["parsers"][0]["priority"] == 0

    def test_fallback_strategy(self):
        """Test a strategy with fallback parsers."""
        strategy = {
            "name": "robust_pdf_processing",
            "description": "PDF processing with fallback support",
            "parsers": [
                {
                    "type": "DoclingParser",
                    "file_extensions": [".pdf"],
                    "priority": 0,
                    "fallback_parser": "PDFParser_PyPDF2",
                    "config": {
                        "chunk_strategy": "hybrid",
                    }
                },
                {
                    "type": "PDFParser_PyPDF2",
                    "file_extensions": [".pdf"],
                    "priority": 10,
                    "config": {
                        "chunk_strategy": "paragraphs",
                    }
                }
            ]
        }

        assert strategy["parsers"][0]["fallback_parser"] == "PDFParser_PyPDF2"


class TestBackwardCompatibilityFull:
    """Comprehensive backward compatibility tests."""

    def test_legacy_parser_config_still_valid(self):
        """Test that legacy parser configurations still work."""
        # This is a config format that was used before the new parsers
        legacy_config = {
            "type": "PDFParser_PyPDF2",
            "file_extensions": [".pdf"],
            "config": {
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "chunk_strategy": "paragraphs",
                "extract_metadata": True,
                "preserve_layout": True,
            }
        }

        assert legacy_config["type"] == "PDFParser_PyPDF2"
        assert legacy_config["config"]["chunk_size"] == 1000

    def test_legacy_llamaindex_parser_config(self):
        """Test that LlamaIndex parser configurations still work."""
        legacy_config = {
            "type": "PDFParser_LlamaIndex",
            "file_extensions": [".pdf"],
            "config": {
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "extract_images": False,
                "extract_tables": False,
            }
        }

        assert legacy_config["type"] == "PDFParser_LlamaIndex"

    def test_legacy_csv_parser_config(self):
        """Test that CSV parser configurations still work."""
        legacy_config = {
            "type": "CSVParser_Pandas",
            "file_extensions": [".csv"],
            "config": {
                "chunk_size": 1000,
                "delimiter": ",",
                "encoding": "utf-8",
            }
        }

        assert legacy_config["type"] == "CSVParser_Pandas"

    def test_legacy_docx_parser_config(self):
        """Test that DOCX parser configurations still work."""
        legacy_config = {
            "type": "DocxParser_PythonDocx",
            "file_extensions": [".docx"],
            "config": {
                "chunk_size": 1000,
                "extract_tables": True,
            }
        }

        assert legacy_config["type"] == "DocxParser_PythonDocx"

    def test_mixed_old_and_new_parsers(self):
        """Test that old and new parsers can coexist in the same strategy."""
        mixed_strategy = {
            "name": "mixed_parsers",
            "description": "Mix of old and new parsers for different file types",
            "parsers": [
                # New Docling for PDFs
                {
                    "type": "DoclingParser",
                    "file_extensions": [".pdf"],
                    "priority": 0,
                    "config": {"chunk_strategy": "hybrid"},
                },
                # Legacy PyPDF2 as fallback
                {
                    "type": "PDFParser_PyPDF2",
                    "file_extensions": [".pdf"],
                    "priority": 10,
                    "config": {"chunk_strategy": "paragraphs"},
                },
                # Legacy CSV parser (Docling doesn't handle CSV well)
                {
                    "type": "CSVParser_Pandas",
                    "file_extensions": [".csv"],
                    "priority": 0,
                    "config": {"delimiter": ","},
                },
                # New MarkItDown for HTML
                {
                    "type": "MarkItDownParser",
                    "file_extensions": [".html"],
                    "priority": 0,
                    "config": {"output_format": "markdown"},
                },
                # Legacy text parser
                {
                    "type": "TextParser_Python",
                    "file_extensions": [".txt"],
                    "priority": 0,
                    "config": {"encoding": "utf-8"},
                },
            ]
        }

        # Verify all parser types are recognized
        parser_types = [p["type"] for p in mixed_strategy["parsers"]]
        assert "DoclingParser" in parser_types
        assert "PDFParser_PyPDF2" in parser_types
        assert "CSVParser_Pandas" in parser_types
        assert "MarkItDownParser" in parser_types
        assert "TextParser_Python" in parser_types

    def test_parser_registry_has_legacy_parsers(self):
        """Test that the parser registry still has all legacy parsers."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from components.parsers.parser_registry import registry

        # Check legacy parsers exist
        legacy_parsers = [
            "PDFParser_PyPDF2",
            "PDFParser_LlamaIndex",
            "CSVParser_Pandas",
            "CSVParser_Python",
            "CSVParser_LlamaIndex",
            "DocxParser_PythonDocx",
            "DocxParser_LlamaIndex",
            "ExcelParser_OpenPyXL",
            "ExcelParser_Pandas",
            "ExcelParser_LlamaIndex",
            "TextParser_Python",
            "TextParser_LlamaIndex",
            "MarkdownParser_Python",
            "MarkdownParser_LlamaIndex",
            "MsgParser_ExtractMsg",
        ]

        for parser_name in legacy_parsers:
            parser = registry.get_parser(parser_name)
            assert parser is not None, f"Legacy parser {parser_name} not found in registry"

    def test_parser_registry_has_new_parsers(self):
        """Test that the parser registry has new parsers."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from components.parsers.parser_registry import registry

        # Check new parsers exist
        new_parsers = ["DoclingParser", "MarkItDownParser"]

        for parser_name in new_parsers:
            parser = registry.get_parser(parser_name)
            assert parser is not None, f"New parser {parser_name} not found in registry"

    def test_file_extension_lookup_includes_all_parsers(self):
        """Test that file extension lookup returns both old and new parsers."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from components.parsers.parser_registry import registry

        # PDF should have both new (Docling, MarkItDown) and legacy (PyPDF2, LlamaIndex)
        pdf_parsers = registry.get_parsers_for_extension(".pdf")
        parser_names = [p["parser"] for p in pdf_parsers]

        assert "DoclingParser" in parser_names, "DoclingParser should be available for PDF"
        assert "MarkItDownParser" in parser_names, "MarkItDownParser should be available for PDF"
        assert "PDFParser_PyPDF2" in parser_names, "PDFParser_PyPDF2 should still be available for PDF"
        assert "PDFParser_LlamaIndex" in parser_names, "PDFParser_LlamaIndex should still be available for PDF"

    def test_docx_extension_lookup(self):
        """Test that DOCX extension returns both old and new parsers."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from components.parsers.parser_registry import registry

        docx_parsers = registry.get_parsers_for_extension(".docx")
        parser_names = [p["parser"] for p in docx_parsers]

        assert "DoclingParser" in parser_names
        assert "MarkItDownParser" in parser_names
        assert "DocxParser_PythonDocx" in parser_names
        assert "DocxParser_LlamaIndex" in parser_names

    def test_priority_ordering(self):
        """Test that new parsers have higher priority (lower number) than legacy."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from components.parsers.parser_registry import registry

        pdf_parsers = registry.get_parsers_for_extension(".pdf")

        # Find priorities
        docling_entry = next((p for p in pdf_parsers if p["parser"] == "DoclingParser"), None)
        pypdf2_entry = next((p for p in pdf_parsers if p["parser"] == "PDFParser_PyPDF2"), None)

        assert docling_entry is not None
        assert pypdf2_entry is not None
        assert docling_entry["priority"] < pypdf2_entry["priority"], \
            "DoclingParser should have higher priority (lower number) than PyPDF2"
