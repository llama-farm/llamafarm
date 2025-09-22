"""Essential pytest configuration and fixtures."""

import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator
import pytest

# Add parent directories to path so 'rag' module can be imported
rag_dir = Path(__file__).parent.parent  # /path/to/rag
project_root = rag_dir.parent  # /path/to/llamafarm-1
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(rag_dir))

from core.base import Document


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
            content="This is a test document about machine learning",
            metadata={"type": "technical"},
            embeddings=[0.1, 0.2, 0.3],
        ),
        Document(
            id="doc2",
            content="Another document about data processing",
            metadata={"type": "technical"},
            embeddings=[0.4, 0.5, 0.6],
        ),
    ]


@pytest.fixture
def sample_csv_file(temp_dir: str) -> str:
    """Create a temporary CSV file with sample data."""
    csv_path = Path(temp_dir) / "test.csv"
    csv_path.write_text("name,value\ntest,123\nsample,456")
    return str(csv_path)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "slow: slow tests")
