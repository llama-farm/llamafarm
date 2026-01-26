"""
Shared pytest fixtures for Universal Runtime tests.
"""

import gc

import pytest
import torch

# Run GC before starting test session to prevent GC during model imports
# This helps avoid segfaults on Python 3.12+ with transformers/torch
gc.collect()


@pytest.fixture(autouse=True, scope="function")
def gc_between_tests():
    """Manage garbage collection to prevent segfaults with torch/numpy.

    On Python 3.11+ in CI environments, garbage collection during or after
    tests involving torch/numpy/scipy can cause segfaults due to C extension
    module interactions. We disable automatic GC during tests and only run
    it at controlled points.
    """
    # Disable automatic GC during the test
    gc.disable()
    yield
    # Re-enable GC but don't force a collection (let Python handle it naturally)
    gc.enable()


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
