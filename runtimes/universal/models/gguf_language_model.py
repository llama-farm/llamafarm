"""
GGUF language model wrapper using llama-cpp-python.

Provides the same interface as LanguageModel but uses llama-cpp-python for
GGUF quantized models, enabling faster inference and lower memory usage.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from llama_cpp import Llama

from utils.context_calculator import get_default_context_size
from utils.model_format import get_gguf_file_path

from .base import BaseModel

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
            return Llama(
                model_path=gguf_path,
                n_ctx=self.actual_n_ctx,  # Use computed context size
                n_gpu_layers=n_gpu_layers,
                n_threads=None,  # Auto-detect optimal threads
                verbose=False,  # Disable verbose logging
                seed=-1,  # Random seed (-1 = random)
            )

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
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text completion (non-streaming).

        Args:
            prompt: Input prompt string
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_p: Nucleus sampling threshold
            stop: List of stop sequences to end generation

        Returns:
            Generated text as a string

        Raises:
            AssertionError: If model not loaded
        """
        assert self.llama is not None, "Model not loaded. Call load() first."

        max_tokens = max_tokens or 512

        # Run generation in thread pool (blocking call)
        loop = asyncio.get_running_loop()

        def _generate():
            try:
                return self.llama(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop or [],
                    echo=False,  # Don't echo the prompt in output
                )
            except Exception as e:
                logger.error(
                    f"Error during llama-cpp-python generation: {e}", exc_info=True
                )
                raise RuntimeError(f"Text generation failed: {e}") from e

        # Validate llama-cpp result structure before extracting text
        try:
            result = await loop.run_in_executor(self._executor, _generate)
            # Extract text from llama-cpp result
            generated_text = result["choices"][0]["text"]
            return generated_text.strip()
        except Exception as e:
            logger.error(
                f"Error validating llama-cpp result structure: {e}", exc_info=True
            )
            raise ValueError(f"Unexpected result structure: {result}") from e

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text completion with streaming (async generator).

        Yields tokens as they are generated, enabling real-time streaming
        responses. The generation runs in a separate thread and tokens are
        passed via an async queue.

        Args:
            prompt: Input prompt string
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_p: Nucleus sampling threshold
            stop: List of stop sequences to end generation

        Yields:
            Generated text tokens as strings

        Raises:
            AssertionError: If model not loaded

        Examples:
            >>> async for token in model.generate_stream("Hello"):
            ...     print(token, end='')
        """
        assert self.llama is not None, "Model not loaded. Call load() first."

        max_tokens = max_tokens or 512

        # Create a queue for passing tokens and exceptions between threads
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _generate():
            """Run generation in separate thread."""
            try:
                for chunk in self.llama(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop or [],
                    stream=True,
                ):
                    text = chunk["choices"][0]["text"]
                    # Put token in queue (thread-safe via run_coroutine_threadsafe)
                    future = asyncio.run_coroutine_threadsafe(queue.put(text), loop)
                    future.result()  # Wait for put to complete
            except Exception as e:
                logger.error(f"Error in GGUF generation: {e}", exc_info=True)
                # Put exception in queue so it can be propagated to caller
                future = asyncio.run_coroutine_threadsafe(queue.put(e), loop)
                future.result()
            finally:
                # Signal completion (None sentinel)
                future = asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                future.result()

        # Start generation in thread pool
        loop.run_in_executor(self._executor, _generate)

        # Yield tokens as they arrive, propagate exceptions
        while True:
            item = await queue.get()
            if item is None:
                # Completion sentinel
                break
            elif isinstance(item, Exception):
                # Propagate exception from streaming thread
                raise item
            else:
                # Regular token
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
