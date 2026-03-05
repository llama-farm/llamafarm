"""
Shared pytest fixtures for Universal Runtime tests.
"""

import gc

import pytest
import torch

# Disable automatic garbage collection for the entire test session.
# On Python 3.11+ in CI environments, GC during or after tests involving
# torch/numpy/scipy can cause segfaults due to C extension module interactions.
# The GC is triggered during pytest's fixture teardown, which crashes the process.
gc.disable()

# Run one final GC while it's still safe (before any torch models are loaded)
gc.collect()


@pytest.fixture(autouse=True, scope="session")
def gc_disabled_for_session():
    """Keep GC disabled for the entire test session to prevent segfaults.

    This is necessary because pytest's fixture teardown triggers GC,
    and torch/numpy/scipy objects can cause segfaults when collected
    in CI environments with Python 3.11+.
    """
    # GC is already disabled at module load time
    yield
    # Don't re-enable GC - let the process exit cleanly


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
