"""Live KV cache save/load test — requires a GGUF model on disk.

Skip gracefully if no model is available. Not run in CI.
Usage: KV_TEST_MODEL_PATH=/path/to/model.gguf pytest tests/test_kv_live.py -v
"""

import os

import pytest

GGUF_MODEL_PATH = os.environ.get("KV_TEST_MODEL_PATH", "")


@pytest.fixture
def llama_model():
    """Load a Llama model for KV testing, skip if unavailable."""
    if not GGUF_MODEL_PATH or not os.path.isfile(GGUF_MODEL_PATH):
        pytest.skip(
            "Set KV_TEST_MODEL_PATH to a .gguf file to run live KV tests"
        )

    try:
        from llamafarm_llama import Llama
    except ImportError:
        pytest.skip("llamafarm_llama not installed")

    llm = Llama(model_path=GGUF_MODEL_PATH, n_ctx=2048, n_gpu_layers=-1)
    yield llm
    # Clean up GPU/memory resources
    del llm


def test_kv_live_save_restore(llama_model):
    """Verify KV state round-trips through save/load."""
    llm = llama_model

    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    tokens = llm.tokenize(prompt, add_special=False, parse_special=True)
    assert len(tokens) > 0

    # Process tokens to populate KV cache
    assert llm._decode_batch(tokens), "decode_batch failed"

    # Save KV state
    kv_data = llm.state_seq_save(0)
    assert len(kv_data) > 0, "state_seq_save returned empty"

    # Reset KV cache before restoring to verify true round-trip
    llm.kv_cache_clear()

    # Restore KV state
    consumed = llm.state_seq_load(kv_data)
    assert consumed > 0, "state_seq_load consumed 0 bytes"
    assert consumed == len(kv_data), (
        f"state_seq_load consumed {consumed} != {len(kv_data)}"
    )

    # Validate restored state is functional by sampling a token
    token_id = llm.sample()
    assert token_id is not None, "sample() returned None after KV restore"
