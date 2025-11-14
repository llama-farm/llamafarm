"""Integration tests for preprocessing pipeline."""

import tempfile
from pathlib import Path

import pytest
from config.datamodel import DataProcessingStrategy, Parser

from components.preprocessors.factory import PreprocessorFactory
from core.blob_processor import BlobProcessor


class TestPreprocessingIntegration:
    """Test end-to-end preprocessing integration."""

    def test_blob_processor_with_markitdown_preprocessor(self):
        """Test BlobProcessor with MarkItDown preprocessor."""
        try:
            # Create a strategy with MarkItDown preprocessor
            from config.datamodel import Preprocessor

            strategy = DataProcessingStrategy(
                name="test_with_preprocessing",
                description="Test strategy with MarkItDown preprocessing",
                preprocessors=[
                    Preprocessor(
                        type="MarkItDownPreprocessor",
                        priority=10,
                        config={},
                        file_include_patterns=["*.html", "*.docx", "*.pptx"],
                    )
                ],
                parsers=[
                    Parser(
                        type="TextParser_Python",
                        priority=100,
                        config={"chunk_size": 500, "chunk_overlap": 50},
                        file_include_patterns=["*"],
                    )
                ],
            )

            processor = BlobProcessor(strategy)

            # Verify preprocessors were initialized
            assert len(processor.preprocessors) == 1
            assert processor.preprocessors[0][0].type == "MarkItDownPreprocessor"

        except ImportError as e:
            pytest.skip(f"Required dependencies not installed: {e}")

    def test_markitdown_html_to_markdown(self):
        """Test MarkItDown preprocessor with HTML input."""
        try:
            from config.datamodel import Preprocessor

            strategy = DataProcessingStrategy(
                name="test_html_preprocessing",
                description="Test HTML preprocessing",
                preprocessors=[
                    Preprocessor(
                        type="MarkItDownPreprocessor",
                        priority=10,
                        config={},
                        file_include_patterns=["*.html"],
                    )
                ],
                parsers=[
                    Parser(
                        type="TextParser_Python",
                        priority=100,
                        config={"chunk_size": 500, "chunk_overlap": 50},
                        file_include_patterns=["*"],
                    )
                ],
            )

            processor = BlobProcessor(strategy)

            # Create a test HTML file
            html_content = """
            <!DOCTYPE html>
            <html>
            <head><title>Test Document</title></head>
            <body>
                <h1>Main Heading</h1>
                <p>This is a test paragraph with <strong>bold text</strong>.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                    <li>Item 3</li>
                </ul>
            </body>
            </html>
            """

            blob_data = html_content.encode("utf-8")
            metadata = {"filename": "test.html"}

            # Process the blob
            documents = processor.process_blob(blob_data, metadata)

            # Verify preprocessing occurred
            assert len(documents) > 0

            # Check that markdown conversion happened
            first_doc = documents[0]
            assert "preprocessed" in first_doc.metadata
            assert first_doc.metadata["preprocessed"] is True
            assert first_doc.metadata["preprocessor"] == "MarkItDownPreprocessor"

            # Verify content was converted
            content = first_doc.content
            assert "Main Heading" in content
            assert "test paragraph" in content

        except ImportError as e:
            pytest.skip(f"MarkItDown not installed: {e}")

    def test_preprocessor_chaining_concept(self):
        """Test that preprocessors can be configured to chain."""
        try:
            from config.datamodel import Preprocessor

            # This tests the configuration structure for chaining
            # (e.g., OCR -> MarkItDown -> Parser)
            strategy = DataProcessingStrategy(
                name="test_chaining",
                description="Test preprocessor chaining",
                preprocessors=[
                    # OCR would run first (priority 10)
                    Preprocessor(
                        type="PaddleOCRPreprocessor",
                        priority=10,
                        config={},
                        file_include_patterns=["*.pdf"],
                    ),
                    # MarkItDown would run second (priority 20)
                    Preprocessor(
                        type="MarkItDownPreprocessor",
                        priority=20,
                        config={},
                        file_include_patterns=["*.docx", "*.pptx"],
                    ),
                ],
                parsers=[
                    Parser(
                        type="TextParser_Python",
                        priority=100,
                        config={},
                        file_include_patterns=["*"],
                    )
                ],
            )

            processor = BlobProcessor(strategy)

            # Verify preprocessors are sorted by priority
            # Note: PaddleOCR may not be installed, so only MarkItDown might load
            assert len(processor.preprocessors) >= 1

            # If both loaded, verify priority sorting
            if len(processor.preprocessors) == 2:
                assert processor.preprocessors[0][0].priority == 10
                assert processor.preprocessors[1][0].priority == 20
            else:
                # Only MarkItDown loaded
                assert processor.preprocessors[0][0].type == "MarkItDownPreprocessor"

        except ImportError as e:
            pytest.skip(f"Required dependencies not installed: {e}")

    def test_sample_file_preprocessing(self):
        """Test preprocessing with actual sample files if available."""
        # Use relative path from repository root
        repo_root = Path(__file__).parent.parent.parent
        samples_dir = repo_root / ".plans" / "samples"

        if not samples_dir.exists():
            pytest.skip("Sample files directory not found")

        # Look for HTML files
        html_files = list(samples_dir.glob("*.html"))
        if not html_files:
            pytest.skip("No HTML sample files found")

        try:
            from config.datamodel import Preprocessor

            strategy = DataProcessingStrategy(
                name="test_real_samples",
                description="Test with real sample files",
                preprocessors=[
                    Preprocessor(
                        type="MarkItDownPreprocessor",
                        priority=10,
                        config={},
                        file_include_patterns=["*.html"],
                    )
                ],
                parsers=[
                    Parser(
                        type="TextParser_Python",
                        priority=100,
                        config={"chunk_size": 1000, "chunk_overlap": 100},
                        file_include_patterns=["*"],
                    )
                ],
            )

            processor = BlobProcessor(strategy)

            # Test with first HTML file
            test_file = html_files[0]
            with open(test_file, "rb") as f:
                blob_data = f.read()

            metadata = {"filename": test_file.name}
            documents = processor.process_blob(blob_data, metadata)

            # Verify processing succeeded
            assert len(documents) > 0
            print(f"\n✅ Successfully preprocessed {test_file.name}")
            print(f"   Generated {len(documents)} document chunks")
            print(f"   First chunk length: {len(documents[0].content)} chars")

        except ImportError as e:
            pytest.skip(f"MarkItDown not installed: {e}")


class TestPreprocessorFactory:
    """Test preprocessor factory functionality."""

    def test_factory_discovers_preprocessors(self):
        """Test that factory discovers all preprocessors."""
        preprocessors = PreprocessorFactory.discover_preprocessors()

        # Should find at least markitdown and ocr types
        assert "markitdown" in preprocessors
        assert "ocr" in preprocessors

        # MarkItDown should have config
        markitdown_configs = preprocessors["markitdown"]
        assert len(markitdown_configs) > 0
        assert markitdown_configs[0]["name"] == "MarkItDownPreprocessor"

    def test_factory_creates_markitdown(self):
        """Test factory can create MarkItDown preprocessor."""
        try:
            preprocessor = PreprocessorFactory.create(
                "MarkItDownPreprocessor",
                config={"preserve_structure": True}
            )

            assert preprocessor is not None
            assert preprocessor.can_process("test.html", {})
            assert preprocessor.can_process("test.docx", {})

        except ImportError:
            pytest.skip("MarkItDown not installed")

    def test_factory_error_handling(self):
        """Test factory handles missing preprocessors."""
        with pytest.raises(ValueError, match="not found"):
            PreprocessorFactory.create("NonExistentPreprocessor", config={})


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
