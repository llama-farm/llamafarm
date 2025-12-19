"""
LlamaFarm's custom llama.cpp Python bindings.

On first import, automatically downloads the appropriate binary if not bundled.

Example:
    >>> from llamafarm_llama import Llama
    >>> llm = Llama(model_path="model.gguf", n_ctx=2048, n_gpu_layers=-1)
    >>> response = llm.create_chat_completion(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=100,
    ... )
    >>> print(response["choices"][0]["message"]["content"])
"""

from __future__ import annotations

import logging

# Set up logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import binary management
from ._binary import (
    LLAMA_CPP_VERSION,
    clear_cache,
    get_binary_info,
    get_lib_path,
    get_platform_key,
)

# Ensure binary is available (downloads if needed)
_lib_path = get_lib_path()

# Import main classes
from .llama import Llama
from .types import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    EmbeddingResponse,
)

# Import logging control
from ._bindings import set_llama_log_level

__version__ = "0.1.0"
__llama_cpp_version__ = LLAMA_CPP_VERSION

__all__ = [
    # Main class
    "Llama",
    # Types
    "ChatMessage",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "EmbeddingResponse",
    # Utilities
    "get_binary_info",
    "get_platform_key",
    "clear_cache",
    # Logging control
    "set_llama_log_level",
    # Version info
    "__version__",
    "__llama_cpp_version__",
]
