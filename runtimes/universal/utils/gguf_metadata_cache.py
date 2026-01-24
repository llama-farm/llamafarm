"""Shared GGUF metadata cache for efficient metadata extraction.

This module provides a centralized cache for GGUF file metadata to avoid
redundant file reads. GGUF metadata reading is expensive (~4-5 seconds for
large models), so caching significantly improves performance.

The cache stores:
- File size and context length (for context_calculator)
- Chat template (for jinja_tools)
- Special tokens (for jinja_tools)
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GGUFMetadata:
    """Cached metadata from a GGUF file."""

    file_path: str
    file_size_bytes: int
    file_size_mb: float
    n_ctx_train: int | None = None
    chat_template: str | None = None
    bos_token: str = ""
    eos_token: str = ""
    # Raw fields for any additional lookups
    _raw_fields: dict[str, Any] = field(default_factory=dict, repr=False)


# Global cache: path -> GGUFMetadata
_metadata_cache: dict[str, GGUFMetadata] = {}
_cache_lock = threading.Lock()


def get_gguf_metadata_cached(gguf_path: str) -> GGUFMetadata:
    """Get GGUF metadata, using cache if available.

    This function reads the GGUF file once and caches all commonly needed
    metadata (file size, context length, chat template, special tokens).
    Subsequent calls for the same path return the cached data instantly.

    Args:
        gguf_path: Absolute path to the GGUF file

    Returns:
        GGUFMetadata with all extracted information

    Raises:
        FileNotFoundError: If GGUF file doesn't exist
    """
    # Normalize path for consistent cache keys
    normalized_path = os.path.normpath(os.path.abspath(gguf_path))

    with _cache_lock:
        if normalized_path in _metadata_cache:
            logger.debug(f"Using cached GGUF metadata for: {normalized_path}")
            return _metadata_cache[normalized_path]

    # Not in cache - read from file (outside lock to avoid blocking)
    logger.info(f"Reading GGUF metadata (will be cached): {normalized_path}")
    metadata = _read_gguf_metadata(normalized_path)

    with _cache_lock:
        _metadata_cache[normalized_path] = metadata

    return metadata


def _read_gguf_metadata(gguf_path: str) -> GGUFMetadata:
    """Read all metadata from a GGUF file in a single pass.

    This is an internal function that performs the actual file reading.
    Use get_gguf_metadata_cached() for cached access.
    """
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    file_size = os.path.getsize(gguf_path)

    metadata = GGUFMetadata(
        file_path=gguf_path,
        file_size_bytes=file_size,
        file_size_mb=file_size / (1024 * 1024),
    )

    try:
        from gguf import GGUFReader

        reader = GGUFReader(gguf_path)

        # Extract all needed metadata in a single pass through fields
        bos_id = None
        eos_id = None
        tokens_data = None

        for key, field in reader.fields.items():
            # Store raw fields for debugging
            metadata._raw_fields[key] = field

            # Context length fields
            context_field_names = ["context_length", "n_ctx_train", "n_ctx"]
            if any(target in key for target in context_field_names) and field.data:
                try:
                    n_ctx_train = field.parts[field.data[0]]
                    if n_ctx_train:
                        metadata.n_ctx_train = int(n_ctx_train)
                        logger.debug(
                            f"Found context size in field '{key}': {n_ctx_train}"
                        )
                except (IndexError, ValueError, TypeError):
                    pass

            # Chat template
            if key == "tokenizer.chat_template":
                if hasattr(field, "parts") and field.parts:
                    # Use only the last part which contains the actual string data
                    # GGUF field.parts structure for strings:
                    #   parts[0]: field name length (8 bytes)
                    #   parts[1]: field name (bytes)
                    #   parts[2]: type indicator (4 bytes)
                    #   parts[3]: string length (8 bytes)
                    #   parts[-1]: the actual string data
                    try:
                        template_bytes = bytes(field.parts[-1])
                        metadata.chat_template = template_bytes.decode("utf-8")
                        logger.debug(
                            f"Found chat template ({len(metadata.chat_template)} chars)"
                        )
                    except (IndexError, UnicodeDecodeError) as e:
                        logger.warning(f"Failed to decode chat template: {e}")
                elif hasattr(field, "data"):
                    # Older format fallback
                    try:
                        metadata.chat_template = bytes(field.data).decode("utf-8")
                    except UnicodeDecodeError as e:
                        logger.warning(
                            f"Failed to decode chat template (fallback): {e}"
                        )

            # BOS token ID
            if key == "tokenizer.ggml.bos_token_id":
                if hasattr(field, "parts") and field.parts:
                    with contextlib.suppress(IndexError, ValueError, TypeError):
                        bos_id = int(field.parts[0][0])
                elif hasattr(field, "data"):
                    with contextlib.suppress(IndexError, ValueError, TypeError):
                        bos_id = int(field.data[0])

            # EOS token ID
            if key == "tokenizer.ggml.eos_token_id":
                if hasattr(field, "parts") and field.parts:
                    with contextlib.suppress(IndexError, ValueError, TypeError):
                        eos_id = int(field.parts[0][0])
                elif hasattr(field, "data"):
                    with contextlib.suppress(IndexError, ValueError, TypeError):
                        eos_id = int(field.data[0])

            # Tokens array (for resolving BOS/EOS IDs to strings)
            if key == "tokenizer.ggml.tokens":
                if hasattr(field, "parts"):
                    tokens_data = field.parts
                elif hasattr(field, "data"):
                    tokens_data = field.data

        # Resolve token IDs to strings
        if tokens_data is not None:
            if bos_id is not None and bos_id < len(tokens_data):
                try:
                    token_bytes = tokens_data[bos_id]
                    if isinstance(token_bytes, (bytes, bytearray)):
                        metadata.bos_token = token_bytes.decode(
                            "utf-8", errors="replace"
                        )
                    elif isinstance(token_bytes, str):
                        metadata.bos_token = token_bytes
                except (IndexError, UnicodeDecodeError):
                    pass

            if eos_id is not None and eos_id < len(tokens_data):
                try:
                    token_bytes = tokens_data[eos_id]
                    if isinstance(token_bytes, (bytes, bytearray)):
                        metadata.eos_token = token_bytes.decode(
                            "utf-8", errors="replace"
                        )
                    elif isinstance(token_bytes, str):
                        metadata.eos_token = token_bytes
                except (IndexError, UnicodeDecodeError):
                    pass

        logger.debug(
            f"GGUF metadata extracted: n_ctx={metadata.n_ctx_train}, "
            f"template={len(metadata.chat_template or '')} chars, "
            f"bos='{metadata.bos_token}', eos='{metadata.eos_token}'"
        )

    except ImportError:
        logger.warning("gguf package not installed, limited metadata available")
    except Exception as e:
        logger.warning(f"Error reading GGUF metadata: {e}")

    return metadata


def clear_metadata_cache(gguf_path: str | None = None) -> None:
    """Clear the GGUF metadata cache.

    Args:
        gguf_path: If provided, only clear cache for this specific path.
                   If None, clear the entire cache.
    """
    global _metadata_cache

    with _cache_lock:
        if gguf_path:
            normalized_path = os.path.normpath(os.path.abspath(gguf_path))
            if normalized_path in _metadata_cache:
                del _metadata_cache[normalized_path]
                logger.debug(f"Cleared GGUF metadata cache for: {normalized_path}")
        else:
            _metadata_cache = {}
            logger.debug("Cleared all GGUF metadata cache")


def get_cache_stats() -> dict:
    """Get statistics about the metadata cache.

    Returns:
        Dict with cache statistics (entry count, paths cached)
    """
    with _cache_lock:
        return {
            "entry_count": len(_metadata_cache),
            "cached_paths": list(_metadata_cache.keys()),
        }
