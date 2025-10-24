"""
Shared pytest fixtures for Universal Runtime tests.
"""

import pytest
import torch
from PIL import Image
import io
import base64
import numpy as np


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


@pytest.fixture
def sample_image():
    """Create a simple test image (RGB)."""
    # Create a 224x224 RGB image with random colors
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    return img


@pytest.fixture
def sample_image_base64(sample_image):
    """Sample image as base64 string."""
    buffered = io.BytesIO()
    sample_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@pytest.fixture
def sample_audio():
    """Create a simple test audio signal."""
    # Create 1 second of silence at 16kHz (Whisper's sample rate)
    sample_rate = 16000
    duration = 1.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    return audio


@pytest.fixture
def sample_audio_base64(sample_audio):
    """Sample audio as base64 (simulated)."""
    # For testing, just return a placeholder
    # Real tests would need actual audio encoding
    return base64.b64encode(sample_audio.tobytes()).decode("utf-8")


# Model IDs for testing (using smallest/fastest models)
TEST_MODELS = {
    "language": "hf-internal-testing/tiny-random-gpt2",
    "encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "vision_classification": "hf-internal-testing/tiny-random-vit",
    "vision_clip": "openai/clip-vit-base-patch32",  # CLIP needs real model
    "audio": "openai/whisper-tiny",  # Smallest Whisper
    "multimodal": "Salesforce/blip-image-captioning-base",  # Smallest BLIP
}


@pytest.fixture
def test_model_ids():
    """Return test model IDs."""
    return TEST_MODELS
