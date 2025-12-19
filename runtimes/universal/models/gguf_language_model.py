"""
GGUF language model wrapper using llama-cpp-python.

Provides the same interface as LanguageModel but uses llama-cpp-python for
GGUF quantized models, enabling faster inference and lower memory usage.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from utils.context_calculator import get_default_context_size
from utils.model_format import get_gguf_file_path

from .base import BaseModel

if TYPE_CHECKING:
    from llamafarm_llama import Llama

logger = logging.getLogger(__name__)


class GGUFLanguageModel(BaseModel):
    """Wrapper for GGUF models using llama-cpp-python.

    This class provides an interface compatible with LanguageModel but uses
    llama-cpp-python for inference with GGUF quantized models. GGUF models
    offer:
    - 50-75% smaller file sizes (4-bit/8-bit quantization)
    - 2-3x faster inference on Apple Silicon (Metal)
    - Significantly lower memory requirements
    - Optimized CPU inference

    The model is automatically configured for the target device (Metal/CUDA/CPU)
    and supports both streaming and non-streaming generation.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        token: str | None = None,
        n_ctx: int | None = None,
        preferred_quantization: str | None = None,
    ):
        """Initialize GGUF language model.

        Args:
            model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-0.6B-GGUF")
            device: Target device ("cuda", "mps", or "cpu")
            token: Optional HuggingFace authentication token for gated models
            n_ctx: Optional context window size. If None, will be computed automatically
                   based on available memory and model defaults.
            preferred_quantization: Optional quantization preference (e.g., "Q4_K_M", "Q8_0").
                                    If None, defaults to Q4_K_M. Only downloads the specified
                                    quantization to save disk space.
        """
        super().__init__(model_id, device, token=token)
        self.model_type = "language"
        self.supports_streaming = True
        self.llama: Llama | None = None
        self.requested_n_ctx = self.n_ctx = n_ctx  # Store requested value
        self.actual_n_ctx: int | None = None  # Will be computed during load()
        self.preferred_quantization = preferred_quantization
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def load(self) -> None:
        """Load the GGUF model using llama-cpp-python.

        This method:
        1. Locates the .gguf file in the HuggingFace cache
        2. Computes optimal context size based on memory and configuration
        3. Configures GPU layers based on the target device
        4. Initializes the llama-cpp-python Llama instance
        5. Runs initialization in a thread pool (blocking operation)

        Raises:
            FileNotFoundError: If no .gguf file found in model repository
            Exception: If model loading fails
        """

        logger.info(f"Loading GGUF model: {self.model_id}")

        # Get path to .gguf file in HF cache
        # This will intelligently select and download only the preferred quantization
        gguf_path = get_gguf_file_path(
            self.model_id,
            self.token,
            preferred_quantization=self.preferred_quantization,
        )

        # On Windows, convert backslashes to forward slashes for llama.cpp compatibility
        # The underlying C library can have issues with Windows-style paths
        if sys.platform == "win32":
            gguf_path = gguf_path.replace("\\", "/")

        logger.info(f"GGUF file located at: {gguf_path}")

        # Compute optimal context size
        self.actual_n_ctx, warnings = get_default_context_size(
            model_id=self.model_id,
            gguf_path=gguf_path,
            device=self.device,
            config_n_ctx=self.requested_n_ctx,
        )

        # Log warnings to stderr
        for warning in warnings:
            logger.warning(warning)

        logger.info(f"Using context size: {self.actual_n_ctx}")

        # Configure GPU layers based on device
        # Note: llama-cpp-python automatically uses whatever backend it was compiled with
        # (CUDA, ROCm, Metal, Vulkan, etc.). We just tell it whether to use GPU or CPU.
        if self.device != "cpu":
            n_gpu_layers = -1  # Use all layers on GPU (any backend)
            logger.info(
                f"Configuring for GPU acceleration on {self.device} (all layers on GPU)"
            )
        else:
            n_gpu_layers = 0  # CPU only
            logger.info("Configuring for CPU-only inference")

        # Load model using llama-cpp-python
        # Run in thread pool since Llama() initialization is blocking
        loop = asyncio.get_running_loop()

        def _load_model():
            import os

            try:
                from llamafarm_llama import Llama
            except ImportError as e:
                raise ImportError(
                    "llamafarm-llama is required for GGUF models but is not installed. "
                    "Install it with: pip install llamafarm-llama"
                ) from e

            # Verify file exists and is readable before attempting to load
            if not os.path.exists(gguf_path):
                raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
            if not os.access(gguf_path, os.R_OK):
                raise PermissionError(f"GGUF file not readable: {gguf_path}")

            file_size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
            logger.info(f"Loading GGUF file ({file_size_mb:.1f} MB): {gguf_path}")

            try:
                return Llama(
                    model_path=gguf_path,
                    n_ctx=self.actual_n_ctx,  # Use computed context size
                    n_gpu_layers=n_gpu_layers,
                    n_threads=None,  # Auto-detect optimal threads
                    verbose=False,  # Disable verbose logging (managed by ggml_logging)
                    seed=-1,  # Random seed (-1 = random)
                )
            except ValueError as e:
                # Provide more helpful error message for common issues
                error_msg = str(e)
                if "Failed to load model from file" in error_msg:
                    logger.error(
                        f"llama.cpp failed to load model. This can be caused by:\n"
                        f"  1. Corrupted GGUF file - try deleting and re-downloading\n"
                        f"  2. Incompatible llama-cpp-python wheel - try reinstalling\n"
                        f"  3. Unsupported GGUF format version\n"
                        f"  File: {gguf_path}\n"
                        f"  Size: {file_size_mb:.1f} MB\n"
                        f"  Context: {self.actual_n_ctx}"
                    )
                raise

        self.llama = await loop.run_in_executor(self._executor, _load_model)

        logger.info(
            f"GGUF model loaded successfully on {self.device} "
            f"with {n_gpu_layers} GPU layers and context size {self.actual_n_ctx}"
        )

    def format_messages(self, messages: list[dict]) -> str:
        """Format chat messages into a prompt string.

        Converts OpenAI-style chat messages into a single prompt string
        suitable for the model. Uses a simple template format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Formatted prompt string

        Examples:
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful"},
            ...     {"role": "user", "content": "Hello"}
            ... ]
            >>> model.format_messages(messages)
            'System: You are helpful\\nUser: Hello\\nAssistant:'
        """
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Add final prompt for assistant response
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        thinking_budget: int | None = None,
    ) -> str:
        """Generate chat completion (non-streaming).

        Uses llama-cpp's create_chat_completion() which applies the chat template
        embedded in the GGUF metadata. This is essential for models like Qwen
        that use special tokens and thinking tags.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_p: Nucleus sampling threshold
            stop: List of stop sequences to end generation
            thinking_budget: Maximum tokens for thinking before forcing </think>

        Returns:
            Generated text as a string

        Raises:
            AssertionError: If model not loaded
        """
        assert self.llama is not None, "Model not loaded. Call load() first."

        max_tokens = max_tokens or 512
        loop = asyncio.get_running_loop()

        def _generate():
            try:
                # Set up logits processor for thinking budget if specified
                logits_processor = None
                if thinking_budget is not None:
                    from utils.thinking import ThinkingBudgetProcessor

                    logits_processor = [
                        ThinkingBudgetProcessor(
                            self.llama, max_thinking_tokens=thinking_budget
                        )
                    ]

                # Use create_chat_completion which applies the model's chat template
                return self.llama.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop or [],
                    logits_processor=logits_processor,
                )
            except Exception as e:
                logger.error(
                    f"Error during llama-cpp-python chat completion: {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"Chat completion failed: {e}") from e

        try:
            result = await loop.run_in_executor(self._executor, _generate)
            content = result["choices"][0]["message"]["content"]
            return content.strip() if content else ""
        except Exception as e:
            logger.error(f"Error extracting chat completion result: {e}", exc_info=True)
            raise ValueError(f"Unexpected result from chat completion: {e}") from e

    async def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        thinking_budget: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion with streaming (async generator).

        Uses llama-cpp's create_chat_completion() with streaming, which applies
        the chat template embedded in the GGUF metadata.

        Thinking budget is enforced via logits processor, which forces the model
        to generate </think> when the budget is reached.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_p: Nucleus sampling threshold
            stop: List of stop sequences to end generation
            thinking_budget: Maximum tokens for thinking before forcing </think>

        Yields:
            Generated text tokens as strings

        Raises:
            AssertionError: If model not loaded
        """
        assert self.llama is not None, "Model not loaded. Call load() first."

        max_tokens = max_tokens or 512
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _generate_stream():
            """Run chat completion in separate thread."""
            try:
                thinking_tokens = 0
                in_thinking = False
                thinking_ended = False
                accumulated_text = ""

                # Set up logits processor for thinking budget enforcement
                logits_processor = None
                if thinking_budget is not None:
                    from utils.thinking import ThinkingBudgetProcessor

                    logits_processor = [
                        ThinkingBudgetProcessor(
                            self.llama, max_thinking_tokens=thinking_budget
                        )
                    ]

                for chunk in self.llama.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop or [],
                    stream=True,
                    logits_processor=logits_processor,
                ):
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        accumulated_text += content

                        # Track thinking state
                        if "<think>" in accumulated_text.lower() and not in_thinking:
                            in_thinking = True
                        if "</think>" in accumulated_text.lower():
                            thinking_ended = True
                            in_thinking = False

                        # Count thinking tokens
                        if in_thinking and not thinking_ended:
                            thinking_tokens += 1

                        future = asyncio.run_coroutine_threadsafe(
                            queue.put(content), loop
                        )
                        future.result()
            except Exception as e:
                logger.error(f"Error in GGUF chat stream: {e}", exc_info=True)
                future = asyncio.run_coroutine_threadsafe(queue.put(e), loop)
                future.result()
            finally:
                future = asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                future.result()

        loop.run_in_executor(self._executor, _generate_stream)

        # Yield tokens as they arrive, propagate exceptions
        while True:
            item = await queue.get()
            if item is None:
                break
            elif isinstance(item, Exception):
                raise item
            else:
                yield item

    async def unload(self) -> None:
        """Unload GGUF model and free resources."""
        logger.info(f"Unloading GGUF language model: {self.model_id}")

        # Clear llama-cpp-python instance
        self.llama = None

        # Shutdown thread pool executor
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True, cancel_futures=True)
            # Create new executor for potential future use
            self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info(f"GGUF language model unloaded: {self.model_id}")

    def __del__(self):
        """Cleanup thread pool executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
