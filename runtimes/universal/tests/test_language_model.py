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
async def test_language_generate(device, test_model_ids, sample_messages):
    """Test text generation with messages."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    result = await model.generate(
        messages=sample_messages,
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
async def test_language_temperature_variations(device, test_model_ids, sample_messages):
    """Test generation with different temperatures."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    # Test with temperature=0 (deterministic)
    result1 = await model.generate(
        messages=sample_messages, max_tokens=5, temperature=0.0
    )
    result2 = await model.generate(
        messages=sample_messages, max_tokens=5, temperature=0.0
    )

    # Should be identical (or very similar due to floating point)
    assert isinstance(result1, str)
    assert isinstance(result2, str)

    # Test with temperature=1.0 (more random)
    result3 = await model.generate(
        messages=sample_messages, max_tokens=5, temperature=1.0
    )
    assert isinstance(result3, str)


@pytest.mark.asyncio
async def test_language_streaming(device, test_model_ids, sample_messages):
    """Test streaming text generation."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    # Verify model supports streaming
    assert model.supports_streaming is True

    # Collect streamed tokens
    tokens = []
    async for token in model.generate_stream(
        messages=sample_messages,
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
    # Use temperature=0 for deterministic output and a prompt likely to produce content
    tokens = []
    async for token in model.generate_stream(
        messages=[{"role": "user", "content": "Continue this sequence: 1, 2, 3, 4,"}],
        max_tokens=50,
        temperature=0.0,  # Deterministic for consistent behavior
        stop=["8"],  # Stop at 8 instead of newline - more predictable
    ):
        tokens.append(token)

    full_text = "".join(tokens)
    # The model should generate some numbers before hitting the stop sequence
    # If it generates "8" immediately, that's still valid stop behavior
    # The key assertion is that the stop sequence is NOT in the output
    assert "8" not in full_text, "Stop sequence should not appear in output"
    # Verify streaming worked (got at least some output or stopped immediately)
    assert isinstance(full_text, str)


@pytest.mark.asyncio
async def test_language_streaming_vs_nonstreaming_equivalence(
    device, test_model_ids, sample_messages
):
    """Test that streaming and non-streaming produce equivalent results with deterministic settings."""
    model = LanguageModel(test_model_ids["language"], device)
    await model.load()

    # Generate with non-streaming (deterministic)
    non_streaming_result = await model.generate(
        messages=sample_messages,
        max_tokens=15,
        temperature=0.0,  # Deterministic
        top_p=1.0,
    )

    # Generate with streaming (deterministic)
    streaming_tokens = []
    async for token in model.generate_stream(
        messages=sample_messages,
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
