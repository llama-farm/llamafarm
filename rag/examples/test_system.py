#!/usr/bin/env python3
"""Test script for the RAG system."""

import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
import unittest
import tempfile
import json
import csv
from pathlib import Path

from core.base import Document
from parsers.csv_parser import CustomerSupportCSVParser
from embedders.ollama_embedder import OllamaEmbedder
from stores.chroma_store import ChromaStore


class TestCSVParser(unittest.TestCase):
    """Test CSV parser functionality."""

    def setUp(self):
        """Create test CSV file."""
        self.test_data = [
            {
                "subject": "Login Issue",
                "body": "Cannot login to the system",
                "answer": "Reset your password",
                "type": "Incident",
                "queue": "IT Support",
                "priority": "medium",
                "language": "en",
                "tag_1": "Login",
                "tag_2": "Authentication",
            },
            {
                "subject": "Data Loss",
                "body": "Lost important files",
                "answer": "Files recovered from backup",
                "type": "Incident",
                "queue": "Technical Support",
                "priority": "high",
                "language": "en",
                "tag_1": "DataLoss",
                "tag_2": "Recovery",
            },
        ]

        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )

        fieldnames = [
            "subject",
            "body",
            "answer",
            "type",
            "queue",
            "priority",
            "language",
            "tag_1",
            "tag_2",
        ]
        writer = csv.DictWriter(self.temp_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in self.test_data:
            writer.writerow(row)

        self.temp_file.close()

    def tearDown(self):
        """Clean up test file."""
        Path(self.temp_file.name).unlink(missing_ok=True)

    def test_csv_parsing(self):
        """Test basic CSV parsing."""
        parser = CustomerSupportCSVParser()
        result = parser.parse(self.temp_file.name)

        self.assertEqual(len(result.documents), 2)
        self.assertEqual(len(result.errors), 0)

        # Check first document
        doc1 = result.documents[0]
        self.assertIn("Login Issue", doc1.content)
        self.assertIn("Cannot login to the system", doc1.content)
        self.assertIn("Reset your password", doc1.content)
        self.assertEqual(doc1.metadata["priority"], "medium")
        self.assertEqual(doc1.metadata["tags"], ["Login", "Authentication"])

    def test_custom_config(self):
        """Test parser with custom configuration."""
        config = {
            "content_fields": ["subject", "body"],
            "metadata_fields": ["type", "priority"],
        }

        parser = CustomerSupportCSVParser(config=config)
        result = parser.parse(self.temp_file.name)

        doc = result.documents[0]
        self.assertNotIn(
            "Reset your password", doc.content
        )  # Answer should not be in content
        self.assertIn("type", doc.metadata)
        self.assertIn("priority", doc.metadata)


class TestOllamaEmbedder(unittest.TestCase):
    """Test Ollama embedder (requires Ollama running)."""

    def setUp(self):
        """Skip if Ollama not available."""
        try:
            self.embedder = OllamaEmbedder(config={"model": "nomic-embed-text"})
            self.embedder.validate_config()
            self.ollama_available = True
        except Exception:
            self.ollama_available = False

    def test_embedding_generation(self):
        """Test embedding generation."""
        if not self.ollama_available:
            self.skipTest("Ollama not available")

        texts = ["Hello world", "This is a test document"]
        embeddings = self.embedder.embed(texts)

        self.assertEqual(len(embeddings), 2)
        self.assertTrue(
            all(isinstance(emb, list) and len(emb) > 0 for emb in embeddings)
        )


class TestChromaStore(unittest.TestCase):
    """Test ChromaDB store functionality."""

    def setUp(self):
        """Create test store."""
        import tempfile

        self.temp_dir = tempfile.mkdtemp()

        self.store = ChromaStore(
            config={
                "collection_name": "test_collection",
                "persist_directory": self.temp_dir,
            }
        )

    def tearDown(self):
        """Clean up test store."""
        try:
            self.store.delete_collection()
        except:
            pass

        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_search_documents(self):
        """Test adding and searching documents."""
        # Create test documents with embeddings
        docs = [
            Document(
                id="doc1",
                content="This is about login issues",
                metadata={"type": "login", "priority": "high"},
                embeddings=[0.1, 0.2, 0.3, 0.4],  # Dummy embedding
            ),
            Document(
                id="doc2",
                content="This discusses data backup procedures",
                metadata={"type": "backup", "priority": "medium"},
                embeddings=[0.5, 0.6, 0.7, 0.8],  # Dummy embedding
            ),
        ]

        # Add documents
        success = self.store.add_documents(docs)
        self.assertTrue(success)

        # Search by embedding
        results = self.store.search(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=2)
        self.assertEqual(len(results), 2)

        # First result should be more similar
        self.assertEqual(results[0].id, "doc1")


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def setUp(self):
        """Create test data and temporary directories."""
        import tempfile

        # Create test CSV
        self.test_csv = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )

        test_data = [
            {
                "subject": "Password Reset",
                "body": "User cannot reset password",
                "answer": "Follow password reset procedure",
                "type": "Incident",
                "priority": "low",
                "tag_1": "Password",
            }
        ]

        fieldnames = ["subject", "body", "answer", "type", "priority", "tag_1"]
        writer = csv.DictWriter(self.test_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in test_data:
            writer.writerow(row)

        self.test_csv.close()

        # Create temporary directory for ChromaDB
        import tempfile

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files."""
        Path(self.test_csv.name).unlink(missing_ok=True)

        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_pipeline(self):
        """Test the complete pipeline without Ollama (mock embeddings)."""
        from core.base import Pipeline

        # Create components
        parser = CustomerSupportCSVParser()

        # Mock embedder that just returns dummy embeddings
        class MockEmbedder:
            def __init__(self):
                self.name = "MockEmbedder"

            def validate_config(self):
                return True

            def process(self, documents):
                for doc in documents:
                    doc.embeddings = [0.1] * 384  # Dummy embedding
                return type("Result", (), {"documents": documents, "errors": []})()

        store = ChromaStore(
            config={
                "collection_name": "test_pipeline",
                "persist_directory": self.temp_dir,
            }
        )

        # Create and run pipeline
        pipeline = Pipeline("Test Pipeline")
        pipeline.add_component(parser)
        pipeline.add_component(MockEmbedder())
        pipeline.add_component(store)

        result = pipeline.run(source=self.test_csv.name)

        # Verify results
        self.assertEqual(len(result.documents), 1)
        self.assertEqual(len(result.errors), 0)

        # Verify document was stored
        info = store.get_collection_info()
        self.assertEqual(info["count"], 1)


def run_system_tests():
    """Run comprehensive system tests."""
    print("Running RAG System Tests")
    print("=" * 50)

    # Run unit tests
    unittest.main(verbosity=2, exit=False)

    print("\n" + "=" * 50)
    print("Manual Tests (requires manual verification)")
    print("=" * 50)

    # Test CLI functionality
    print("\n1. Testing CLI init command...")
    try:
        from cli import create_sample_config

        create_sample_config()
        print("   ✓ Sample config created successfully")

        # Verify config file was created
        if Path("rag_config.json").exists():
            with open("rag_config.json") as f:
                config = json.load(f)
                print(f"   ✓ Config loaded with {len(config)} sections")

    except Exception as e:
        print(f"   ✗ CLI test failed: {e}")

    print("\n2. Testing sample data parsing...")
    try:
        sample_file = "filtered-english-incident-tickets.csv"
        if Path(sample_file).exists():
            parser = CustomerSupportCSVParser()
            result = parser.parse(sample_file)
            print(f"   ✓ Parsed {len(result.documents)} documents from sample file")
            print(f"   ✓ Found {len(result.errors)} parsing errors")

            if result.documents:
                doc = result.documents[0]
                print(f"   ✓ First document has {len(doc.metadata)} metadata fields")
                print(f"   ✓ Content length: {len(doc.content)} characters")
        else:
            print(f"   ⚠ Sample file not found: {sample_file}")

    except Exception as e:
        print(f"   ✗ Sample parsing test failed: {e}")

    print("\nTest Summary:")
    print("- Core functionality tested with unit tests")
    print("- CLI interface verified")
    print("- Sample data parsing tested")
    print("- To test with Ollama, ensure it's running with nomic-embed-text model")
    print("- Run 'python cli.py test' for interactive testing")


if __name__ == "__main__":
    run_system_tests()
