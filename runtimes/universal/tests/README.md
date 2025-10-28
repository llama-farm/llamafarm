# Universal Runtime Tests

Comprehensive test suite for all model types in the Universal Runtime.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_language_model.py       # Text generation tests
├── test_encoder_model.py         # Embeddings & classification tests
├── test_diffusion_model.py       # Image generation tests
├── test_vision_model.py          # Image understanding tests
├── test_audio_model.py           # Speech-to-text tests
└── test_multimodal_model.py      # Vision-language tests
```

## Running Tests

### Run all tests
```bash
cd runtimes/universal
uv run python -m pytest tests/
```

### Run specific test file
```bash
uv run python -m pytest tests/test_language_model.py
```

### Run specific test
```bash
uv run python -m pytest tests/test_encoder_model.py::test_encoder_embed_single
```

### Run fast tests only (skip slow model downloads)
```bash
uv run python -m pytest tests/ -m "not slow"
```

### Run with verbose output
```bash
uv run python -m pytest tests/ -v
```

### Run with coverage
```bash
uv run python -m pytest tests/ --cov=models --cov-report=html
```

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.slow` - Tests that download large models (CLIP, Whisper, BLIP)
- `@pytest.mark.asyncio` - Async tests (all model tests)

## Test Models

The test suite uses small/fast models to keep tests quick:

| Model Type | Test Model | Size | Notes |
|------------|------------|------|-------|
| CausalLM | `hf-internal-testing/tiny-random-gpt2` | ~1MB | Tiny test model |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` | ~90MB | Fast embeddings |
| Diffusion | `hf-internal-testing/tiny-stable-diffusion-torch` | ~500MB | Test diffusion |
| Vision (classification) | `hf-internal-testing/tiny-random-vit` | ~1MB | Tiny ViT |
| Vision (CLIP) | `openai/clip-vit-base-patch32` | ~600MB | Full CLIP (marked slow) |
| Audio | `openai/whisper-tiny` | ~150MB | Smallest Whisper (marked slow) |
| Multimodal | `Salesforce/blip-image-captioning-base` | ~1GB | Smallest BLIP (marked slow) |

## Test Coverage

### CausalLMModel Tests
- ✅ Model loading
- ✅ Text generation
- ✅ Message formatting for chat
- ✅ Temperature variations
- ✅ Model info retrieval

### EncoderModel Tests
- ✅ Model loading
- ✅ Single text embedding
- ✅ Batch text embedding
- ✅ Embedding normalization
- ✅ Semantic similarity
- ✅ Generate not supported (raises NotImplementedError)

### DiffusionModel Tests
- ✅ Model loading
- ✅ Image generation from text
- ✅ Multiple image generation
- ✅ Negative prompts
- ✅ Seed-based reproducibility
- ✅ Different schedulers

### VisionModel Tests
- ✅ Classification model loading
- ✅ Image classification
- ✅ Batch classification
- ✅ Base64 image input
- ✅ CLIP zero-shot classification
- ✅ CLIP image embeddings
- ✅ CLIP text embeddings

### AudioModel Tests
- ✅ Model loading
- ✅ Audio transcription
- ✅ Base64 audio input
- ✅ Timestamp generation
- ✅ Temperature control
- ✅ Audio translation
- ✅ Generate alias

### MultimodalModel Tests
- ✅ Model loading
- ✅ Image captioning
- ✅ Base64 image input
- ✅ Beam search variations
- ✅ Visual question answering
- ✅ Generic generate() method
- ✅ Visual chat (LLaVA-style)

## Common Test Patterns

### Testing with fixtures
```python
@pytest.mark.asyncio
async def test_something(device, test_model_ids, sample_image):
    """Test with pre-configured fixtures."""
    model = VisionModel(test_model_ids["vision_classification"], device)
    await model.load()
    result = await model.classify([sample_image])
    assert result is not None
```

### Testing exceptions
```python
@pytest.mark.asyncio
async def test_raises_error(device, test_model_ids):
    """Test that invalid operations raise errors."""
    model = EncoderModel(test_model_ids["encoder"], device)
    await model.load()

    with pytest.raises(NotImplementedError):
        await model.generate("test")
```

### Skipping slow tests
```python
@pytest.mark.asyncio
@pytest.mark.slow  # Will be skipped with -m "not slow"
async def test_large_model(device, test_model_ids):
    """Test that requires large model download."""
    model = AudioModel(test_model_ids["audio"], device)
    await model.load()
    # ... test code
```

## Continuous Integration

For CI environments, run fast tests only:
```bash
# Skip slow model downloads in CI
uv run python -m pytest tests/ -m "not slow"
```

For full test suite (nightly builds):
```bash
# Run all tests including slow ones
uv run python -m pytest tests/
```

## Troubleshooting

### Tests fail to download models
- Check internet connection
- Verify HuggingFace access (some models require authentication)
- Try running once locally to cache models

### Out of memory errors
- Use smaller test models
- Run tests sequentially: `pytest tests/ -n 1`
- Skip slow tests: `pytest tests/ -m "not slow"`

### Import errors
- Ensure all dependencies installed: `uv sync`
- Check Python path includes runtime directory

## Adding New Tests

When adding a new model type:

1. Create `test_<model_type>_model.py`
2. Add test model ID to `conftest.py`
3. Write tests for:
   - Model loading
   - Core functionality
   - Edge cases
   - Error handling
4. Add appropriate markers (`@pytest.mark.slow` if needed)
5. Update this README

## Dependencies

Required packages (in `pyproject.toml`):
- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `torch` - PyTorch
- `transformers` - HuggingFace transformers
- `diffusers` - HuggingFace diffusers
- `pillow` - Image processing
- `numpy` - Numerical operations

Optional packages:
- `pytest-cov` - Coverage reporting
- `pytest-timeout` - Test timeouts
- `pytest-xdist` - Parallel test execution
