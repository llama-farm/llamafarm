"""
Tests for the DocsContextService.
"""

import pytest
from pathlib import Path
from services.docs_context_service import DocsContextService, KEYWORD_TO_DOCS


class TestDocsContextService:
    """Test suite for DocsContextService."""

    def test_initialization_with_custom_root(self, tmp_path):
        """Test service initialization with custom docs root."""
        # Create some fake docs
        docs_root = tmp_path / "docs"
        docs_root.mkdir()
        (docs_root / "test.md").write_text("# Test Doc\n\nTest content")

        # Should initialize without error
        service = DocsContextService(docs_root=docs_root)
        assert service.docs_root == docs_root
        assert isinstance(service._doc_cache, dict)

    def test_initialization_with_missing_root(self, tmp_path):
        """Test service handles missing docs root gracefully."""
        missing_root = tmp_path / "nonexistent"

        service = DocsContextService(docs_root=missing_root)
        # Should not raise, but cache should be empty
        assert len(service._doc_cache) == 0

    def test_load_doc_cache(self, tmp_path):
        """Test loading documentation files into cache."""
        docs_root = tmp_path / "docs"
        docs_root.mkdir()

        # Create some docs matching our keyword mappings
        rag_doc = docs_root / "rag"
        rag_doc.mkdir()
        (rag_doc / "index.md").write_text("# RAG Guide\n\nRAG content here")

        cli_doc = docs_root / "cli"
        cli_doc.mkdir()
        (cli_doc / "index.md").write_text("# CLI Guide\n\nCLI commands here")

        service = DocsContextService(docs_root=docs_root)

        # Check that docs were loaded
        assert "rag/index.md" in service._doc_cache
        assert "cli/index.md" in service._doc_cache
        assert "RAG content here" in service._doc_cache["rag/index.md"]
        assert "CLI commands here" in service._doc_cache["cli/index.md"]

    def test_match_docs_for_query_single_keyword(self, tmp_path):
        """Test matching docs based on single keyword."""
        docs_root = tmp_path / "docs"
        docs_root.mkdir()

        rag_dir = docs_root / "rag"
        rag_dir.mkdir()
        (rag_dir / "index.md").write_text("# RAG Guide\n\nRAG documentation")

        service = DocsContextService(docs_root=docs_root)

        # Query with RAG keyword
        results = service.match_docs_for_query("How do I use RAG?")
        assert len(results) >= 1
        assert any("rag/index.md" in doc_path for doc_path, _ in results)

    def test_match_docs_for_query_multiple_keywords(self, tmp_path):
        """Test matching docs with multiple keywords."""
        docs_root = tmp_path / "docs"
        docs_root.mkdir()

        # Create RAG docs
        rag_dir = docs_root / "rag"
        rag_dir.mkdir()
        (rag_dir / "index.md").write_text("# RAG Guide")

        # Create CLI docs
        cli_dir = docs_root / "cli"
        cli_dir.mkdir()
        (cli_dir / "index.md").write_text("# CLI Guide")
        (cli_dir / "lf-datasets.md").write_text("# Dataset Commands")

        service = DocsContextService(docs_root=docs_root)

        # Query with multiple keywords
        results = service.match_docs_for_query("How do I use CLI commands for datasets?")

        # Should match both CLI and datasets docs (limited by max_docs default of 2)
        assert len(results) <= 2
        doc_paths = [doc_path for doc_path, _ in results]
        assert any("cli" in path or "dataset" in path.lower() for path in doc_paths)

    def test_match_docs_for_query_no_matches(self, tmp_path):
        """Test query with no matching keywords."""
        docs_root = tmp_path / "docs"
        docs_root.mkdir()

        service = DocsContextService(docs_root=docs_root)

        # Query with no matching keywords
        results = service.match_docs_for_query("xyzzy nonexistent topic")
        assert len(results) == 0

    def test_match_docs_respects_max_docs(self, tmp_path):
        """Test that max_docs parameter limits results."""
        docs_root = tmp_path / "docs"
        docs_root.mkdir()

        # Create multiple docs
        for i in range(5):
            doc_dir = docs_root / f"doc{i}"
            doc_dir.mkdir()
            (doc_dir / "index.md").write_text(f"Doc {i}")

        service = DocsContextService(docs_root=docs_root)

        # Manually add docs to cache for testing
        for i in range(5):
            service._doc_cache[f"doc{i}/index.md"] = f"Doc {i} content"

        # Mock a query that would match all docs
        # We need to modify the keyword mapping temporarily
        original_mapping = KEYWORD_TO_DOCS.copy()
        KEYWORD_TO_DOCS["test"] = [f"doc{i}/index.md" for i in range(5)]

        try:
            results = service.match_docs_for_query("test", max_docs=2)
            assert len(results) == 2
        finally:
            # Restore original mapping
            KEYWORD_TO_DOCS.clear()
            KEYWORD_TO_DOCS.update(original_mapping)

    def test_match_docs_truncates_long_docs(self, tmp_path):
        """Test that long docs are truncated."""
        docs_root = tmp_path / "docs"
        docs_root.mkdir()

        rag_dir = docs_root / "rag"
        rag_dir.mkdir()

        # Create a very long doc
        long_content = "\n".join([f"Line {i}" for i in range(200)])
        (rag_dir / "index.md").write_text(long_content)

        service = DocsContextService(docs_root=docs_root)

        results = service.match_docs_for_query("rag", max_lines_per_doc=50)

        assert len(results) == 1
        doc_path, content = results[0]
        assert "truncated" in content.lower()
        assert content.count("\n") < 200  # Should be truncated

    def test_format_doc_for_context(self):
        """Test formatting a single doc for context injection."""
        service = DocsContextService(docs_root=Path("/fake"))

        formatted = service.format_doc_for_context(
            "rag/index.md",
            "# RAG Guide\n\nContent here"
        )

        assert "## Documentation: rag/index.md" in formatted
        assert "# RAG Guide" in formatted
        assert "Content here" in formatted
        assert "---" in formatted

    def test_format_multiple_docs(self):
        """Test formatting multiple docs together."""
        service = DocsContextService(docs_root=Path("/fake"))

        docs = [
            ("rag/index.md", "RAG content"),
            ("cli/index.md", "CLI content"),
        ]

        formatted = service.format_multiple_docs(docs)

        assert "rag/index.md" in formatted
        assert "cli/index.md" in formatted
        assert "RAG content" in formatted
        assert "CLI content" in formatted

    def test_format_multiple_docs_empty(self):
        """Test formatting empty docs list."""
        service = DocsContextService(docs_root=Path("/fake"))

        formatted = service.format_multiple_docs([])
        assert formatted == ""

    def test_keyword_mappings_are_lowercase(self):
        """Test that all keyword mappings are lowercase for case-insensitive matching."""
        for keyword in KEYWORD_TO_DOCS.keys():
            assert keyword == keyword.lower(), f"Keyword '{keyword}' is not lowercase"

    def test_case_insensitive_matching(self, tmp_path):
        """Test that keyword matching is case-insensitive."""
        docs_root = tmp_path / "docs"
        docs_root.mkdir()

        rag_dir = docs_root / "rag"
        rag_dir.mkdir()
        (rag_dir / "index.md").write_text("# RAG Guide")

        service = DocsContextService(docs_root=docs_root)

        # Test various capitalizations
        for query in ["How do I use RAG?", "how do i use rag?", "How Do I Use Rag?"]:
            results = service.match_docs_for_query(query)
            assert len(results) >= 1, f"Failed for query: {query}"
            assert any("rag/index.md" in doc_path for doc_path, _ in results)
