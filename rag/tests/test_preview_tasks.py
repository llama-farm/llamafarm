"""Tests for the preview Celery task - TDD Red Phase.

All tests written FIRST and will fail until implementation is complete.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestPreviewTask:
    """Tests for the preview Celery task."""

    @pytest.fixture
    def mock_project_dir(self, tmp_path: Path) -> Path:
        """Create a mock project directory with config."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir(parents=True)

        # Create minimal llamafarm.yaml
        config_content = """
namespace: test
name: test_project
rag:
  databases:
    - name: default
      type: ChromaStore
      config: {}
      embedding_strategy:
        type: OllamaEmbedder
        config:
          model: nomic-embed-text
  data_processing_strategies:
    - name: default
      parsers:
        - type: TextParser_Python
          file_include_patterns:
            - "*.txt"
          config:
            chunk_size: 200
            chunk_overlap: 20
          priority: 1
      extractors: []
"""
        (project_dir / "llamafarm.yaml").write_text(config_content)

        return project_dir

    @pytest.fixture
    def sample_file(self, tmp_path: Path) -> Path:
        """Create a sample file for testing."""
        file_path = tmp_path / "sample.txt"
        file_path.write_text("This is sample content for testing the preview task.")
        return file_path

    def test_preview_task_exists(self):
        """The preview_document task can be imported."""
        from tasks.preview_tasks import preview_document_task

        assert preview_document_task is not None
        assert callable(preview_document_task)

    def test_preview_task_has_correct_name(self):
        """The task has the correct Celery task name."""
        from tasks.preview_tasks import preview_document_task

        assert preview_document_task.name == "rag.preview_document"

    @patch("tasks.preview_tasks.PreviewHandler")
    @patch("tasks.preview_tasks._get_blob_processor")
    def test_preview_document_task_returns_valid_result(
        self,
        mock_get_processor,
        mock_handler_class,
        mock_project_dir: Path,
        sample_file: Path,
    ):
        """Task returns PreviewResult as dict."""
        from core.base import PreviewChunk, PreviewResult

        # Setup mocks
        mock_blob_processor = Mock()
        mock_get_processor.return_value = mock_blob_processor

        mock_result = PreviewResult(
            original_text="This is sample content",
            chunks=[
                PreviewChunk(
                    chunk_index=0,
                    content="This is sample",
                    start_position=0,
                    end_position=14,
                )
            ],
            file_info={"filename": "sample.txt", "size": 50},
            parser_used="TextParser_Python",
            chunk_strategy="characters",
            chunk_size=200,
            chunk_overlap=20,
            total_chunks=1,
        )

        mock_handler = Mock()
        mock_handler.generate_preview.return_value = mock_result
        mock_handler_class.return_value = mock_handler

        from tasks.preview_tasks import preview_document_task

        # Call the task
        result = preview_document_task(
            project_dir=str(mock_project_dir),
            file_path=str(sample_file),
            database="default",
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "original_text" in result
        assert "chunks" in result
        assert "total_chunks" in result
        assert result["total_chunks"] == 1

    @patch("tasks.preview_tasks.PreviewHandler")
    @patch("tasks.preview_tasks._get_blob_processor")
    def test_preview_document_task_with_overrides(
        self,
        mock_get_processor,
        mock_handler_class,
        mock_project_dir: Path,
        sample_file: Path,
    ):
        """Task accepts and applies chunk setting overrides."""
        from core.base import PreviewChunk, PreviewResult

        mock_blob_processor = Mock()
        mock_get_processor.return_value = mock_blob_processor

        mock_result = PreviewResult(
            original_text="Test",
            chunks=[PreviewChunk(0, "Test", 0, 4)],
            file_info={},
            parser_used="Test",
            chunk_strategy="characters",
            chunk_size=500,
            chunk_overlap=100,
            total_chunks=1,
        )

        mock_handler = Mock()
        mock_handler.generate_preview.return_value = mock_result
        mock_handler_class.return_value = mock_handler

        from tasks.preview_tasks import preview_document_task

        # Call with overrides
        _result = preview_document_task(
            project_dir=str(mock_project_dir),
            file_path=str(sample_file),
            database="default",
            chunk_size=500,
            chunk_overlap=100,
            chunk_strategy="sentences",
        )

        # Verify overrides were passed to handler
        mock_handler.generate_preview.assert_called_once()
        call_kwargs = mock_handler.generate_preview.call_args[1]
        assert call_kwargs.get("chunk_size_override") == 500
        assert call_kwargs.get("chunk_overlap_override") == 100
        assert call_kwargs.get("chunk_strategy_override") == "sentences"

    def test_preview_document_task_invalid_file_raises(
        self,
        mock_project_dir: Path,
    ):
        """Task raises appropriate error for invalid file."""
        from tasks.preview_tasks import preview_document_task

        with pytest.raises((FileNotFoundError, ValueError)):
            preview_document_task(
                project_dir=str(mock_project_dir),
                file_path="/nonexistent/file.txt",
                database="default",
            )

    def test_preview_document_task_invalid_project_raises(
        self,
        sample_file: Path,
    ):
        """Task raises error for invalid project directory."""
        from tasks.preview_tasks import preview_document_task

        with pytest.raises((FileNotFoundError, ValueError)):
            preview_document_task(
                project_dir="/nonexistent/project",
                file_path=str(sample_file),
                database="default",
            )

    @patch("tasks.preview_tasks.PreviewHandler")
    @patch("tasks.preview_tasks._get_blob_processor")
    def test_preview_task_uses_correct_database_config(
        self,
        mock_get_processor,
        mock_handler_class,
        mock_project_dir: Path,
        sample_file: Path,
    ):
        """Task uses the specified database configuration."""
        from core.base import PreviewResult

        mock_blob_processor = Mock()
        mock_get_processor.return_value = mock_blob_processor

        mock_result = PreviewResult(
            original_text="Test",
            chunks=[],
            file_info={},
            parser_used="Test",
            chunk_strategy="characters",
            chunk_size=200,
            chunk_overlap=20,
            total_chunks=0,
        )

        mock_handler = Mock()
        mock_handler.generate_preview.return_value = mock_result
        mock_handler_class.return_value = mock_handler

        from tasks.preview_tasks import preview_document_task

        # Call with specific database
        preview_document_task(
            project_dir=str(mock_project_dir),
            file_path=str(sample_file),
            database="custom_db",
        )

        # Verify _get_blob_processor was called with correct database
        mock_get_processor.assert_called_once()
        call_args = mock_get_processor.call_args
        assert (
            "custom_db" in str(call_args) or call_args[1].get("database") == "custom_db"
        )

    @patch("tasks.preview_tasks.PreviewHandler")
    @patch("tasks.preview_tasks._get_blob_processor")
    def test_preview_task_handles_binary_files(
        self,
        mock_get_processor,
        mock_handler_class,
        mock_project_dir: Path,
        tmp_path: Path,
    ):
        """Task handles binary files (like PDF)."""
        from core.base import PreviewChunk, PreviewResult

        # Create a mock binary file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 mock content")

        mock_blob_processor = Mock()
        mock_get_processor.return_value = mock_blob_processor

        mock_result = PreviewResult(
            original_text="Extracted PDF content",
            chunks=[PreviewChunk(0, "Extracted PDF content", 0, 21)],
            file_info={"filename": "test.pdf"},
            parser_used="PDFParser",
            chunk_strategy="characters",
            chunk_size=200,
            chunk_overlap=20,
            total_chunks=1,
        )

        mock_handler = Mock()
        mock_handler.generate_preview.return_value = mock_result
        mock_handler_class.return_value = mock_handler

        from tasks.preview_tasks import preview_document_task

        result = preview_document_task(
            project_dir=str(mock_project_dir),
            file_path=str(pdf_file),
            database="default",
        )

        assert result["parser_used"] == "PDFParser"
        assert result["total_chunks"] == 1


class TestPreviewTaskHelpers:
    """Tests for helper functions in preview_tasks module."""

    @pytest.fixture
    def mock_project_dir(self, tmp_path: Path) -> Path:
        """Create a mock project directory with config."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir(parents=True)

        config_content = """\
version: v1
namespace: test
name: test_project
runtime:
  type: local
  config: {}
rag:
  databases:
    - name: default
      type: ChromaStore
      config: {}
  data_processing_strategies:
    - name: default
      parsers:
        - type: TextParser_Python
          file_include_patterns:
            - "*.txt"
          config:
            chunk_size: 200
            chunk_overlap: 20
          priority: 1
      extractors: []
"""
        (project_dir / "llamafarm.yaml").write_text(config_content)

        return project_dir

    def test_get_blob_processor_returns_configured_processor(
        self,
        mock_project_dir: Path,
    ):
        """_get_blob_processor returns a properly configured BlobProcessor."""
        from tasks.preview_tasks import _get_blob_processor

        processor = _get_blob_processor(
            str(mock_project_dir),
            database="default",
        )

        assert processor is not None
        # Verify it has parsers configured
        assert hasattr(processor, "parsers")

    def test_get_blob_processor_uses_database_strategy(
        self,
        mock_project_dir: Path,
    ):
        """_get_blob_processor uses the strategy from the database config."""
        from tasks.preview_tasks import _get_blob_processor

        # This should load the processing strategy associated with the database
        processor = _get_blob_processor(
            str(mock_project_dir),
            database="default",
        )

        # Verify the processor was configured
        assert processor is not None
