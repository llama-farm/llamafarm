"""
Shared pytest fixtures for Universal Runtime tests.
"""

import os

import pytest
import torch


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_llm: marks tests as requiring an LLM model (may be slow or skipped in CI)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (may be skipped with -m 'not slow')",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require LLM if the environment variable is set."""
    skip_llm = pytest.mark.skip(
        reason="Skipping LLM tests (set RUN_LLM_TESTS=1 to enable)"
    )
    for item in items:
        # Skip unless RUN_LLM_TESTS=1 is set
        if "requires_llm" in item.keywords and os.environ.get("RUN_LLM_TESTS", "").strip() != "1":
            item.add_marker(skip_llm)


@pytest.fixture(scope="session")
def device():
    """Get optimal device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "Hello, this is a test sentence."


@pytest.fixture
def sample_texts():
    """Multiple sample texts for batch testing."""
    return [
        "Hello, this is a test sentence.",
        "Machine learning is fascinating.",
        "The quick brown fox jumps over the lazy dog.",
    ]


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "user", "content": "What is 2+2?"},
    ]


# Model IDs for testing (using smallest/fastest models)
TEST_MODELS = {
    "language": "hf-internal-testing/tiny-random-gpt2",
    "encoder": "sentence-transformers/all-MiniLM-L6-v2",
}


@pytest.fixture
def test_model_ids():
    """Return test model IDs."""
    return TEST_MODELS
