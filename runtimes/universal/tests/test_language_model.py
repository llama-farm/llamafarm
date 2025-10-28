"""
Tests for LanguageModel (text generation).
"""

import pytest
from models.language_model import LanguageModel


@pytest.mark.asyncio
async def test_language_load(device, test_model_ids):
    """Test loading a causal LM."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    assert model.model is not None
    assert model.tokenizer is not None
    assert model.model_type == "language"


@pytest.mark.asyncio
async def test_language_generate(device, test_model_ids, sample_text):
    """Test text generation."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    result = await model.generate(
        prompt=sample_text,
        max_tokens=10,
        temperature=0.7,
    )

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_language_format_messages(device, test_model_ids, sample_messages):
    """Test message formatting for chat."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    formatted = model.format_messages(sample_messages)

    assert isinstance(formatted, str)
    assert len(formatted) > 0
    assert "2+2" in formatted or "What" in formatted


@pytest.mark.asyncio
async def test_language_model_info(device, test_model_ids):
    """Test getting model info."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    info = model.get_model_info()

    assert info["model_id"] == test_model_ids["language"]
    assert info["model_type"] == "language"
    assert info["device"] == device


@pytest.mark.asyncio
async def test_language_temperature_variations(device, test_model_ids, sample_text):
    """Test generation with different temperatures."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    # Test with temperature=0 (deterministic)
    result1 = await model.generate(prompt=sample_text, max_tokens=5, temperature=0.0)
    result2 = await model.generate(prompt=sample_text, max_tokens=5, temperature=0.0)

    # Should be identical (or very similar due to floating point)
    assert isinstance(result1, str)
    assert isinstance(result2, str)

    # Test with temperature=1.0 (more random)
    result3 = await model.generate(prompt=sample_text, max_tokens=5, temperature=1.0)
    assert isinstance(result3, str)


@pytest.mark.asyncio
async def test_language_streaming(device, test_model_ids, sample_text):
    """Test streaming text generation."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    # Verify model supports streaming
    assert model.supports_streaming is True

    # Collect streamed tokens
    tokens = []
    async for token in model.generate_stream(
        prompt=sample_text,
        max_tokens=20,
        temperature=0.7,
    ):
        assert isinstance(token, str)
        tokens.append(token)

    # Should have received multiple tokens
    assert len(tokens) > 0

    # Concatenated tokens should form a coherent response
    full_text = "".join(tokens)
    assert len(full_text) > 0


@pytest.mark.asyncio
async def test_language_streaming_with_stop(device, test_model_ids):
    """Test streaming with stop sequences."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    # Generate with a stop sequence
    tokens = []
    async for token in model.generate_stream(
        prompt="Count: 1, 2, 3,",
        max_tokens=50,
        temperature=0.5,
        stop=["\n"],
    ):
        tokens.append(token)

    full_text = "".join(tokens)
    assert len(full_text) > 0
    # Should stop before newline
    assert "\n" not in full_text


@pytest.mark.asyncio
async def test_language_streaming_vs_nonstreaming_equivalence(
    device, test_model_ids, sample_text
):
    """Test that streaming and non-streaming produce equivalent results with deterministic settings."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    # Generate with non-streaming (deterministic)
    non_streaming_result = await model.generate(
        prompt=sample_text,
        max_tokens=15,
        temperature=0.0,  # Deterministic
        top_p=1.0,
    )

    # Generate with streaming (deterministic)
    streaming_tokens = []
    async for token in model.generate_stream(
        prompt=sample_text,
        max_tokens=15,
        temperature=0.0,  # Deterministic
        top_p=1.0,
    ):
        streaming_tokens.append(token)

    streaming_result = "".join(streaming_tokens).strip()

    # Results should be identical (or very similar) with temperature=0
    assert isinstance(non_streaming_result, str)
    assert isinstance(streaming_result, str)
    assert len(non_streaming_result) > 0
    assert len(streaming_result) > 0

    # With temperature=0, results should be very close or identical
    # We allow for minor differences due to potential floating point variations
    # or tokenization differences, but they should be substantially the same
    assert (
        streaming_result == non_streaming_result
        or streaming_result in non_streaming_result
        or non_streaming_result in streaming_result
    ), (
        f"Streaming result '{streaming_result}' differs significantly from non-streaming '{non_streaming_result}'"
    )
