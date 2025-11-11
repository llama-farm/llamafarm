"""Model format detection utilities for Universal Runtime.

Detects whether a HuggingFace model repository contains GGUF or transformers format files.
"""

import os
from typing import Optional
from huggingface_hub import snapshot_download
import logging

logger = logging.getLogger(__name__)

# Cache detection results to avoid repeated filesystem checks
_format_cache: dict[str, str] = {}


def detect_model_format(model_id: str, token: Optional[str] = None) -> str:
    """
    Detect if a HuggingFace model is GGUF or transformers format.

    This function downloads (or locates in cache) the model repository and checks
    for .gguf files to determine the format. Results are cached to avoid repeated
    filesystem operations.

    Args:
        model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-0.6B-GGUF")
        token: Optional HuggingFace authentication token for gated models

    Returns:
        "gguf" if model contains .gguf files, "transformers" otherwise

    Raises:
        Exception: If model cannot be downloaded or accessed

    Examples:
        >>> detect_model_format("unsloth/Qwen3-0.6B-GGUF")
        "gguf"
        >>> detect_model_format("google/gemma-3-1b-it")
        "transformers"
    """
    # Check cache first
    if model_id in _format_cache:
        logger.debug(f"Using cached format for {model_id}: {_format_cache[model_id]}")
        return _format_cache[model_id]

    logger.info(f"Detecting format for model: {model_id}")

    try:
        # Download/locate model in HuggingFace cache
        # Use allow_patterns to only fetch small files for detection (faster)
        local_path = snapshot_download(
            repo_id=model_id,
            token=token,
            allow_patterns=["*.gguf", "config.json", "*.safetensors", "*.bin"],
            ignore_patterns=["*.msgpack", "*.h5", "*.onnx"],
        )

        # Check for GGUF files in the snapshot directory
        for filename in os.listdir(local_path):
            if filename.endswith('.gguf'):
                logger.info(f"Detected GGUF format: found {filename}")
                _format_cache[model_id] = "gguf"
                return "gguf"

        # No GGUF files found - assume transformers format
        logger.info(f"Detected transformers format (no .gguf files found)")
        _format_cache[model_id] = "transformers"
        return "transformers"

    except Exception as e:
        logger.error(f"Error detecting model format for {model_id}: {e}")
        raise


def get_gguf_file_path(model_id: str, token: Optional[str] = None) -> str:
    """
    Get the full path to a GGUF file in the HuggingFace cache.

    This function ensures the model is downloaded and returns the path to the
    first .gguf file found in the model repository.

    Args:
        model_id: HuggingFace model identifier
        token: Optional HuggingFace authentication token for gated models

    Returns:
        Full absolute path to .gguf file

    Raises:
        FileNotFoundError: If no GGUF file found in the model repository

    Examples:
        >>> path = get_gguf_file_path("unsloth/Qwen3-0.6B-GGUF")
        >>> path.endswith('.gguf')
        True
    """
    logger.info(f"Locating GGUF file for model: {model_id}")

    # Ensure model is in cache
    local_path = snapshot_download(repo_id=model_id, token=token)

    # Find first .gguf file
    gguf_files = []
    for filename in os.listdir(local_path):
        if filename.endswith('.gguf'):
            gguf_files.append(filename)

    if not gguf_files:
        raise FileNotFoundError(
            f"No GGUF file found in model repository: {model_id}\n"
            f"Repository path: {local_path}\n"
            f"Available files: {os.listdir(local_path)}"
        )

    # If multiple GGUF files, log warning and use first
    if len(gguf_files) > 1:
        logger.warning(
            f"Multiple GGUF files found in {model_id}: {gguf_files}. "
            f"Using first file: {gguf_files[0]}"
        )

    gguf_path = os.path.join(local_path, gguf_files[0])
    logger.info(f"Found GGUF file: {gguf_path}")
    return gguf_path


def clear_format_cache():
    """Clear the format detection cache.

    Useful for testing or when model repositories are updated.
    """
    global _format_cache
    _format_cache = {}
    logger.debug("Format detection cache cleared")
