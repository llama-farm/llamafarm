"""
MLX language model wrapper using mlx-lm.

Provides the same interface as LanguageModel but uses mlx-lm for
MLX-optimized models on Apple Silicon, enabling faster inference
and lower memory usage.
"""

import asyncio
import logging
import platform
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from .base import BaseModel

logger = logging.getLogger(__name__)


def _check_mlx_platform() -> tuple[bool, str]:
    """Check if the current platform supports MLX.

    Returns:
        Tuple of (is_supported, error_message)
    """
    system = platform.system()
    if system != "Darwin":
        return False, (
            f"MLX models are only supported on macOS with Apple Silicon. "
            f"Current platform: {system}. "
            f"Please use a GGUF or transformers model instead."
        )

    # Check if it's Apple Silicon (M1/M2/M3/M4)
    machine = platform.machine()
    if machine != "arm64":
        return False, (
            f"MLX models require Apple Silicon (M1/M2/M3/M4). "
            f"Current architecture: {machine}. "
            f"Please use a GGUF or transformers model instead."
        )

    return True, ""


class MLXLanguageModel(BaseModel):
    """Wrapper for MLX models using mlx-lm.

    This class provides an interface compatible with LanguageModel but uses
    mlx-lm for inference with MLX-optimized models. MLX models offer:
    - Optimized performance on Apple Silicon (M1/M2/M3/M4)
    - Lower memory usage through quantization
    - Native Metal acceleration
    - Fast inference with low latency

    The model is automatically configured for Apple Silicon Metal backend
    and supports both streaming and non-streaming generation.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        token: str | None = None,
    ):
        """Initialize MLX language model.

        Args:
            model_id: HuggingFace model identifier (e.g., "Qwen/Qwen3-4B-MLX-4bit")
            device: Target device (should be "mps" for MLX models, but accepts any)
            token: Optional HuggingFace authentication token for gated models
        """
        super().__init__(model_id, device, token=token)
        self.model_type = "language"
        self.supports_streaming = True
        self.mlx_model = None
        self.mlx_tokenizer = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def load(self) -> None:
        """Load the MLX model using mlx-lm.

        This method:
        1. Checks platform compatibility (macOS with Apple Silicon)
        2. Imports mlx_lm library (late import to allow optional dependency)
        3. Downloads and loads the model from HuggingFace
        4. Initializes tokenizer
        5. Runs initialization in a thread pool (blocking operation)

        Raises:
            RuntimeError: If platform doesn't support MLX
            ImportError: If mlx_lm is not installed
            Exception: If model loading fails
        """
        # Check platform compatibility first
        is_supported, error_msg = _check_mlx_platform()
        if not is_supported:
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Try to import mlx_lm
        try:
            import mlx_lm
        except ImportError as e:
            error_msg = (
                "mlx_lm is required for MLX models. "
                "Install it with: pip install mlx-lm (macOS only)"
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from e

        logger.info(f"Loading MLX model: {self.model_id}")

        # Load model using mlx_lm
        # Run in thread pool since model loading can be blocking
        loop = asyncio.get_running_loop()

        def _load_model():
            model, tokenizer = mlx_lm.load(
                self.model_id,
                tokenizer_config={
                    "trust_remote_code": True,
                    "token": self.token,
                },
            )
            return model, tokenizer

        try:
            self.mlx_model, self.mlx_tokenizer = await loop.run_in_executor(
                self._executor, _load_model
            )
            logger.info(f"MLX model loaded successfully: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}", exc_info=True)
            raise

    def format_messages(self, messages: list[dict]) -> str:
        """Format chat messages into a prompt string.

        Converts OpenAI-style chat messages into a single prompt string
        suitable for the model. Attempts to use the tokenizer's chat
        template if available.

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
        # Try to use tokenizer's apply_chat_template if available
        if self.mlx_tokenizer and hasattr(self.mlx_tokenizer, "apply_chat_template"):
            try:
                result = self.mlx_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if isinstance(result, str):
                    return result
            except Exception as e:
                logger.debug(f"Could not use chat template: {e}")

        # Fallback to simple concatenation
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

        Uses mlx_lm's generate method to produce text from the model.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_p: Nucleus sampling threshold
            stop: List of stop sequences to end generation (not fully supported)
            thinking_budget: Not used for MLX models (included for API compatibility)

        Returns:
            Generated text as a string

        Raises:
            AssertionError: If model not loaded
        """
        assert self.mlx_model is not None, "Model not loaded. Call load() first."
        assert self.mlx_tokenizer is not None, (
            "Tokenizer not loaded. Call load() first."
        )

        try:
            import mlx_lm
        except ImportError as e:
            raise ImportError("mlx_lm is required for MLX models") from e

        max_tokens = max_tokens or 512
        loop = asyncio.get_running_loop()

        # Format messages into prompt
        prompt = self.format_messages(messages)

        def _generate():
            try:
                # Create a sampler with temperature and top_p
                from mlx_lm.sample_utils import make_sampler

                sampler = make_sampler(temp=temperature, top_p=top_p)

                # Generate using mlx_lm
                response = mlx_lm.generate(
                    self.mlx_model,
                    self.mlx_tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
                return response
            except Exception as e:
                logger.error(f"Error during MLX generation: {e}", exc_info=True)
                raise RuntimeError(f"MLX generation failed: {e}") from e

        try:
            result = await loop.run_in_executor(self._executor, _generate)
            return result.strip() if result else ""
        except Exception as e:
            logger.error(f"Error extracting MLX generation result: {e}", exc_info=True)
            raise

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

        Uses mlx_lm's stream generation to yield tokens as they're generated.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_p: Nucleus sampling threshold
            stop: List of stop sequences to end generation
            thinking_budget: Not used for MLX models (included for API compatibility)

        Yields:
            Generated text tokens as strings

        Raises:
            AssertionError: If model not loaded
        """
        assert self.mlx_model is not None, "Model not loaded. Call load() first."
        assert self.mlx_tokenizer is not None, (
            "Tokenizer not loaded. Call load() first."
        )

        try:
            import mlx_lm
        except ImportError as e:
            raise ImportError("mlx_lm is required for MLX models") from e

        max_tokens = max_tokens or 512
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Format messages into prompt
        prompt = self.format_messages(messages)

        def _generate_stream():
            """Run streaming generation in separate thread."""
            try:
                # Create a sampler with temperature and top_p
                from mlx_lm.sample_utils import make_sampler

                sampler = make_sampler(temp=temperature, top_p=top_p)

                for response in mlx_lm.stream_generate(
                    self.mlx_model,
                    self.mlx_tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    # mlx_lm.stream_generate yields GenerationResponse objects
                    # Extract the text attribute
                    text = response.text if hasattr(response, "text") else str(response)
                    future = asyncio.run_coroutine_threadsafe(queue.put(text), loop)
                    future.result()

                    # Check for stop sequences
                    if stop:
                        for stop_seq in stop:
                            if stop_seq in text:
                                break
            except Exception as e:
                logger.error(f"Error in MLX stream: {e}", exc_info=True)
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
        """Unload MLX model and free resources."""
        logger.info(f"Unloading MLX language model: {self.model_id}")

        # Clear MLX model and tokenizer instances
        self.mlx_model = None
        self.mlx_tokenizer = None

        # Shutdown thread pool executor
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True, cancel_futures=True)
            # Create new executor for potential future use
            self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info(f"MLX language model unloaded: {self.model_id}")

    def __del__(self):
        """Cleanup thread pool executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
