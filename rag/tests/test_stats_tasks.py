"""Tests for RAG stats Celery tasks."""

import tempfile
from pathlib import Path
from unittest.mock import Mock


class TestEstimateDocumentCount:
    """Tests for document count estimation helper."""

    def test_estimate_from_unique_source_files(self):
        """Test document count estimation from unique source files."""
        from tasks.stats_tasks import _estimate_document_count

        mock_api = Mock()
        mock_collection = Mock()

        # Simulate 6 chunks from 3 unique documents
        mock_collection.get.return_value = {
            "metadatas": [
                {"source_file": "doc1.pdf"},
                {"source_file": "doc1.pdf"},
                {"source_file": "doc1.pdf"},
                {"source_file": "doc2.pdf"},
                {"source_file": "doc2.pdf"},
                {"source_file": "doc3.pdf"},
            ]
        }
        mock_api.vector_store.collection = mock_collection

        count = _estimate_document_count(mock_api, chunk_count=6)
        assert count == 3

    def test_estimate_using_source_key(self):
        """Test estimation using 'source' metadata key."""
        from tasks.stats_tasks import _estimate_document_count

        mock_api = Mock()
        mock_collection = Mock()

        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "/path/to/doc1.pdf"},
                {"source": "/path/to/doc1.pdf"},
                {"source": "/path/to/doc2.pdf"},
            ]
        }
        mock_api.vector_store.collection = mock_collection

        count = _estimate_document_count(mock_api, chunk_count=3)
        assert count == 2

    def test_estimate_using_filename_key(self):
        """Test estimation using 'filename' metadata key."""
        from tasks.stats_tasks import _estimate_document_count

        mock_api = Mock()
        mock_collection = Mock()

        mock_collection.get.return_value = {
            "metadatas": [
                {"filename": "report.pdf"},
                {"filename": "report.pdf"},
            ]
        }
        mock_api.vector_store.collection = mock_collection

        count = _estimate_document_count(mock_api, chunk_count=2)
        assert count == 1

    def test_estimate_fallback_when_query_fails(self):
        """Test fallback estimation when metadata query fails."""
        from tasks.stats_tasks import _estimate_document_count

        mock_api = Mock()
        mock_api.vector_store.collection.get.side_effect = Exception("Query failed")

        # Should fall back to chunk_count / 10
        count = _estimate_document_count(mock_api, chunk_count=100)
        assert count == 10

    def test_estimate_fallback_no_source_metadata(self):
        """Test fallback when metadata has no source info."""
        from tasks.stats_tasks import _estimate_document_count

        mock_api = Mock()
        mock_collection = Mock()

        mock_collection.get.return_value = {
            "metadatas": [
                {"chunk_index": 0},
                {"chunk_index": 1},
            ]
        }
        mock_api.vector_store.collection = mock_collection

        # No source info found, should fall back
        count = _estimate_document_count(mock_api, chunk_count=20)
        assert count == 2  # 20 / 10

    def test_estimate_with_zero_chunks(self):
        """Test estimation with zero chunks."""
        from tasks.stats_tasks import _estimate_document_count

        mock_api = Mock()
        mock_api.vector_store.collection.get.side_effect = Exception("Query failed")

        count = _estimate_document_count(mock_api, chunk_count=0)
        assert count == 0

    def test_estimate_minimum_one_document(self):
        """Test that estimate returns at least 1 for non-zero chunks."""
        from tasks.stats_tasks import _estimate_document_count

        mock_api = Mock()
        mock_api.vector_store.collection.get.side_effect = Exception("Query failed")

        # Even with just 5 chunks, should return at least 1
        count = _estimate_document_count(mock_api, chunk_count=5)
        assert count >= 1


class TestGetStorageSizes:
    """Tests for storage size calculation helper."""

    def test_storage_sizes_with_data_and_index(self):
        """Test storage size calculation with data and index files."""
        from tasks.stats_tasks import _get_storage_sizes

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data file
            data_file = Path(tmpdir) / "data.db"
            data_file.write_bytes(b"x" * 1000)

            # Create index file
            index_file = Path(tmpdir) / "index.idx"
            index_file.write_bytes(b"y" * 500)

            collection_size, index_size = _get_storage_sizes(tmpdir)

            assert collection_size == 1000
            assert index_size == 500

    def test_storage_sizes_multiple_index_types(self):
        """Test that various index file extensions are recognized."""
        from tasks.stats_tasks import _get_storage_sizes

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create various index files
            (Path(tmpdir) / "hnsw.idx").write_bytes(b"a" * 100)
            (Path(tmpdir) / "vectors.index").write_bytes(b"b" * 200)
            (Path(tmpdir) / "embeddings.bin").write_bytes(b"c" * 300)

            # Create data file
            (Path(tmpdir) / "metadata.json").write_bytes(b"d" * 50)

            collection_size, index_size = _get_storage_sizes(tmpdir)

            assert index_size == 600  # 100 + 200 + 300
            assert collection_size == 50

    def test_storage_sizes_nested_directories(self):
        """Test storage size calculation with nested directories."""
        from tasks.stats_tasks import _get_storage_sizes

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()

            (Path(tmpdir) / "root.db").write_bytes(b"x" * 100)
            (subdir / "nested.db").write_bytes(b"y" * 200)
            (subdir / "nested.idx").write_bytes(b"z" * 50)

            collection_size, index_size = _get_storage_sizes(tmpdir)

            assert collection_size == 300  # 100 + 200
            assert index_size == 50

    def test_storage_sizes_nonexistent_dir(self):
        """Test storage sizes with nonexistent directory."""
        from tasks.stats_tasks import _get_storage_sizes

        collection_size, index_size = _get_storage_sizes("/nonexistent/path/xyz")

        assert collection_size == 0
        assert index_size == 0

    def test_storage_sizes_empty_dir(self):
        """Test storage sizes with empty directory."""
        from tasks.stats_tasks import _get_storage_sizes

        with tempfile.TemporaryDirectory() as tmpdir:
            collection_size, index_size = _get_storage_sizes(tmpdir)

            assert collection_size == 0
            assert index_size == 0


class TestGetEmbeddingDimension:
    """Tests for the _get_embedding_dimension helper."""

    def test_extracts_from_default_strategy(self):
        """Test extraction from explicitly configured default strategy."""
        from tasks.stats_tasks import _get_embedding_dimension

        mock_strategy1 = Mock()
        mock_strategy1.name = "small"
        mock_strategy1.config = {"dimension": 384}

        mock_strategy2 = Mock()
        mock_strategy2.name = "large"
        mock_strategy2.config = {"dimension": 1536}

        mock_db = Mock()
        mock_db.embedding_strategies = [mock_strategy1, mock_strategy2]
        mock_db.default_embedding_strategy = "large"

        assert _get_embedding_dimension(mock_db) == 1536

    def test_uses_first_strategy_when_no_default(self):
        """Test that first strategy is used when no default specified."""
        from tasks.stats_tasks import _get_embedding_dimension

        mock_strategy1 = Mock()
        mock_strategy1.name = "first"
        mock_strategy1.config = {"dimension": 512}

        mock_strategy2 = Mock()
        mock_strategy2.name = "second"
        mock_strategy2.config = {"dimension": 768}

        mock_db = Mock()
        mock_db.embedding_strategies = [mock_strategy1, mock_strategy2]
        mock_db.default_embedding_strategy = None

        assert _get_embedding_dimension(mock_db) == 512

    def test_returns_default_when_no_strategies(self):
        """Test default value when no embedding strategies configured."""
        from tasks.stats_tasks import _get_embedding_dimension

        mock_db = Mock()
        mock_db.embedding_strategies = []

        assert _get_embedding_dimension(mock_db) == 768

    def test_returns_default_when_strategies_is_none(self):
        """Test default value when embedding_strategies is None."""
        from tasks.stats_tasks import _get_embedding_dimension

        mock_db = Mock()
        mock_db.embedding_strategies = None

        assert _get_embedding_dimension(mock_db) == 768

    def test_returns_default_when_config_missing_dimension(self):
        """Test default when strategy config lacks dimension key."""
        from tasks.stats_tasks import _get_embedding_dimension

        mock_strategy = Mock()
        mock_strategy.name = "no_dimension"
        mock_strategy.config = {"model": "nomic-embed-text"}

        mock_db = Mock()
        mock_db.embedding_strategies = [mock_strategy]
        mock_db.default_embedding_strategy = None

        assert _get_embedding_dimension(mock_db) == 768

    def test_handles_non_dict_config(self):
        """Test graceful handling when config is not a dict."""
        from tasks.stats_tasks import _get_embedding_dimension

        mock_strategy = Mock()
        mock_strategy.name = "weird"
        mock_strategy.config = "not a dict"

        mock_db = Mock()
        mock_db.embedding_strategies = [mock_strategy]
        mock_db.default_embedding_strategy = None

        assert _get_embedding_dimension(mock_db) == 768

    def test_finds_default_not_at_first_position(self):
        """Test finding default strategy when it's not the first in list."""
        from tasks.stats_tasks import _get_embedding_dimension

        strategies = []
        for _, (name, dim) in enumerate([("a", 256), ("b", 512), ("c", 1024)]):
            s = Mock()
            s.name = name
            s.config = {"dimension": dim}
            strategies.append(s)

        mock_db = Mock()
        mock_db.embedding_strategies = strategies
        mock_db.default_embedding_strategy = "c"

        assert _get_embedding_dimension(mock_db) == 1024


class TestExtractFilename:
    """Tests for the _extract_filename helper."""

    def test_extract_simple_filename(self):
        """Test extracting filename from simple path."""
        from tasks.stats_tasks import _extract_filename

        assert _extract_filename("document.pdf") == "document.pdf"

    def test_extract_from_unix_path(self):
        """Test extracting filename from Unix path."""
        from tasks.stats_tasks import _extract_filename

        assert _extract_filename("/path/to/document.pdf") == "document.pdf"

    def test_extract_from_windows_path(self):
        """Test extracting filename from Windows path."""
        from tasks.stats_tasks import _extract_filename

        assert _extract_filename("C:\\Users\\docs\\document.pdf") == "document.pdf"

    def test_extract_from_mixed_path(self):
        """Test extracting filename from mixed path separators."""
        from tasks.stats_tasks import _extract_filename

        assert _extract_filename("/path/to\\document.pdf") == "document.pdf"

    def test_extract_empty_source(self):
        """Test extracting filename from empty string."""
        from tasks.stats_tasks import _extract_filename

        assert _extract_filename("") == "unknown"

    def test_extract_none_source(self):
        """Test extracting filename from None."""
        from tasks.stats_tasks import _extract_filename

        assert _extract_filename(None) == "unknown"


class TestListDocumentsTaskImport:
    """Test that list documents task can be imported and has correct metadata."""

    def test_list_documents_task_is_registered(self):
        """Test that the list documents task is properly registered."""
        from tasks.stats_tasks import rag_list_database_documents_task

        assert rag_list_database_documents_task.name == "rag.list_database_documents"

    def test_list_documents_task_can_be_imported(self):
        """Test that the list documents task can be imported."""
        from tasks.stats_tasks import rag_list_database_documents_task

        assert rag_list_database_documents_task is not None
        assert callable(rag_list_database_documents_task)


class TestListDocumentsTaskAggregation:
    """Tests for document aggregation logic in list_database_documents task."""

    def test_aggregates_chunks_by_source(self):
        """Test that chunks from the same source are aggregated by the real task."""
        from unittest.mock import patch

        from core.base import Document

        # Create test chunks from two documents
        chunks = [
            Document(
                id="chunk1",
                content="",
                source="doc1.pdf",
                metadata={"size": 1000, "parser_type": "PDFParser"},
            ),
            Document(
                id="chunk2",
                content="",
                source="doc1.pdf",
                metadata={"size": 1000, "parser_type": "PDFParser"},
            ),
            Document(
                id="chunk3",
                content="",
                source="doc2.pdf",
                metadata={"size": 2000, "parser_type": "TextParser"},
            ),
        ]

        # Mock the config and search API to test the actual task logic
        mock_db = Mock()
        mock_db.name = "test_db"  # Set .name attribute explicitly (Mock(name=...) is for repr)

        mock_config = Mock()
        mock_config.rag.databases = [mock_db]

        mock_search_api = Mock()
        mock_search_api.vector_store.list_documents.return_value = (chunks, len(chunks))

        with (
            patch("tasks.stats_tasks.load_config", return_value=mock_config),
            patch("tasks.stats_tasks.DatabaseSearchAPI", return_value=mock_search_api),
        ):
            from tasks.stats_tasks import rag_list_database_documents_task

            # Call the actual task (use .run() to bypass Celery machinery)
            result = rag_list_database_documents_task.run(
                project_dir="/fake/path",
                database="test_db",
            )

        # Verify the task correctly aggregated the chunks
        assert result["total_count"] == 2
        docs_by_name = {d["filename"]: d for d in result["documents"]}
        assert docs_by_name["doc1.pdf"]["chunk_count"] == 2
        assert docs_by_name["doc2.pdf"]["chunk_count"] == 1
        assert docs_by_name["doc1.pdf"]["size_bytes"] == 1000
        assert docs_by_name["doc2.pdf"]["size_bytes"] == 2000


class TestStatsTaskTaskImport:
    """Test that stats task can be imported and has correct metadata."""

    def test_task_is_registered(self):
        """Test that the stats task is properly registered."""
        from tasks.stats_tasks import rag_get_database_stats_task

        assert rag_get_database_stats_task.name == "rag.get_database_stats"

    def test_task_module_imports(self):
        """Test that all stats task components can be imported."""
        from tasks.stats_tasks import (
            StatsTask,
            _estimate_document_count,
            _extract_filename,
            _get_storage_sizes,
            rag_get_database_stats_task,
            rag_list_database_documents_task,
        )

        assert StatsTask is not None
        assert rag_get_database_stats_task is not None
        assert rag_list_database_documents_task is not None
        assert callable(_estimate_document_count)
        assert callable(_get_storage_sizes)
        assert callable(_extract_filename)

    def test_task_exported_from_package(self):
        """Test that task is exported from tasks package."""
        from tasks import rag_get_database_stats_task

        assert rag_get_database_stats_task.name == "rag.get_database_stats"
