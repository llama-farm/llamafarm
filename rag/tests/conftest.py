"""Pytest configuration and fixtures for the RAG system tests."""

import tempfile
import shutil
from pathlib import Path
from typing import Generator
import pytest

from rag.core.base import Document


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            content="This is about login issues and password resets",
            metadata={
                "type": "login",
                "priority": "high",
                "tags": ["Login", "Password"],
            },
            embeddings=[0.1, 0.2, 0.3, 0.4],
        ),
        Document(
            id="doc2",
            content="Data backup procedures and system maintenance",
            metadata={
                "type": "backup",
                "priority": "medium",
                "tags": ["Backup", "System"],
            },
            embeddings=[0.5, 0.6, 0.7, 0.8],
        ),
        Document(
            id="doc3",
            content="Security breach incident requiring immediate attention",
            metadata={
                "type": "security",
                "priority": "critical",
                "tags": ["Security", "Breach"],
            },
            embeddings=[0.9, 0.1, 0.5, 0.3],
        ),
    ]


@pytest.fixture
def sample_csv_data() -> str:
    """Create sample CSV data for testing."""
    return """subject,body,answer,type,queue,priority,language,tag_1,tag_2
Login Issue,Cannot login to the system,Reset your password,Incident,IT Support,medium,en,Login,Authentication
Data Loss,Lost important files,Files recovered from backup,Incident,Technical Support,high,en,DataLoss,Recovery
Security Alert,Potential security breach detected,Investigation started,Security,Security Team,critical,en,Security,Breach"""


@pytest.fixture
def sample_csv_file(temp_dir: str, sample_csv_data: str) -> str:
    """Create a temporary CSV file with sample data."""
    csv_path = Path(temp_dir) / "test_data.csv"
    csv_path.write_text(sample_csv_data)
    return str(csv_path)


@pytest.fixture
def mock_ollama_available():
    """Mock Ollama availability for tests that don't require actual Ollama."""
    import requests_mock

    with requests_mock.Mocker() as m:
        m.get(
            "http://localhost:11434/api/tags",
            json={"models": [{"name": "nomic-embed-text:latest"}]},
        )
        m.post(
            "http://localhost:11434/api/embeddings",
            json={"embedding": [0.1, 0.2, 0.3, 0.4] * 192},  # 768 dimensions
        )
        yield m


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "ollama: marks tests that require Ollama to be running"
    )
    config.addinivalue_line("markers", "chromadb: marks tests that require ChromaDB")
