"""
GGUF language model wrapper using llama-cpp.

Provides the same interface as LanguageModel but uses llama-cpp for
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
from utils.context_manager import ContextBudget, ContextManager, ContextUsage
from utils.model_format import get_gguf_file_path
from utils.token_counter import TokenCounter

from .base import BaseModel

if TYPE_CHECKING:
    from llamafarm_llama import Llama

logger = logging.getLogger(__name__)


class GGUFLanguageModel(BaseModel):
    """Wrapper for GGUF models using llama-cpp.

    This class provides an interface compatible with LanguageModel but uses
    llama-cpp for inference with GGUF quantized models. GGUF models
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
        n_batch: int | None = None,
        n_gpu_layers: int | None = None,
        n_threads: int | None = None,
        flash_attn: bool | None = None,
        use_mmap: bool | None = None,
        use_mlock: bool | None = None,
        cache_type_k: str | None = None,
        cache_type_v: str | None = None,
        preferred_quantization: str | None = None,
    ):
        """Initialize GGUF language model.

        Args:
            model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-0.6B-GGUF")
            device: Target device ("cuda", "mps", or "cpu")
            token: Optional HuggingFace authentication token for gated models
            n_ctx: Optional context window size. If None, will be computed automatically
                   based on available memory and model defaults.
            n_batch: Optional batch size for prompt processing. If None, defaults to 2048.
                     Critical for memory: lower values (e.g., 512) reduce compute buffer size.
            n_gpu_layers: Optional number of layers to offload to GPU. If None, will be
                          auto-detected based on device. Use -1 for all layers.
            n_threads: Optional number of CPU threads. If None, auto-detected.
                       Set to match CPU core count (e.g., 6 for Jetson Orin Nano).
            flash_attn: Optional flag to enable/disable flash attention. If None,
                        defaults to True for faster inference on supported hardware.
            use_mmap: Optional flag for memory-mapped file loading. If None, defaults to True.
                      Recommended True on memory-constrained devices for efficient swapping.
            use_mlock: Optional flag to lock model in RAM. If None, defaults to False.
                       Set False on 8GB devices to allow OS memory management.
            cache_type_k: Optional KV cache key quantization type (e.g., "q4_0", "q8_0", "f16").
                          Using "q4_0" can reduce KV cache memory by ~4x. Critical for
                          memory-constrained devices like Jetson Orin Nano (8GB shared).
            cache_type_v: Optional KV cache value quantization type. Same options as cache_type_k.
                          Setting both to "q4_0" provides maximum memory savings.
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
        self.requested_n_batch = n_batch  # Store requested value (None = default 2048)
        self.requested_n_gpu_layers = (
            n_gpu_layers  # Store requested value (None = auto)
        )
        self.requested_n_threads = n_threads  # Store requested value (None = auto)
        self.requested_flash_attn = (
            flash_attn  # Store requested value (None = default True)
        )
        self.requested_use_mmap = (
            use_mmap  # Store requested value (None = default True)
        )
        self.requested_use_mlock = (
            use_mlock  # Store requested value (None = default False)
        )
        self.requested_cache_type_k = (
            cache_type_k  # Store requested value (None = default f16)
        )
        self.requested_cache_type_v = (
            cache_type_v  # Store requested value (None = default f16)
        )
        self.preferred_quantization = preferred_quantization
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Context management (initialized during load())
        self._token_counter: TokenCounter | None = None
        self._context_manager: ContextManager | None = None

        # Cached GGUF metadata (extracted once during load())
        self._chat_template: str | None = None
        self._special_tokens: dict[str, str] | None = None

    def _get_available_memory_mb(self) -> int | None:
        """Get available system memory in MB for Memory Guard check.

        This helps prevent OOM errors on memory-constrained devices like Jetson
        by detecting low memory conditions before attempting to allocate large buffers.

        Returns:
            Available memory in MB, or None if unable to determine.
        """
        try:
            # Try Linux /proc/meminfo first (works on Jetson and most Linux)
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemAvailable" in line:
                        # Format: "MemAvailable:   1234567 kB"
                        return int(line.split()[1]) // 1024
        except (FileNotFoundError, PermissionError, OSError):
            pass

        # Fallback: try psutil if available
        try:
            import psutil

            return int(psutil.virtual_memory().available / (1024 * 1024))
        except ImportError:
            pass

        # Unable to determine available memory
        return None

    async def load(self) -> None:
        """Load the GGUF model using llama-cpp.

        This method:
        1. Locates the .gguf file in the HuggingFace cache
        2. Computes optimal context size based on memory and configuration
        3. Configures GPU layers based on the target device
        4. Initializes the llama-cpp Llama instance
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

        # Store path for later use (e.g., Jinja2 template extraction)
        self._gguf_path = gguf_path

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

        # Configure GPU layers for llama.cpp
        # Use explicitly requested value if provided, otherwise auto-detect
        from utils.device import get_gguf_gpu_layers

        if self.requested_n_gpu_layers is not None:
            n_gpu_layers = self.requested_n_gpu_layers
            logger.info(f"Using configured n_gpu_layers: {n_gpu_layers}")
        else:
            n_gpu_layers = get_gguf_gpu_layers()
            logger.info(f"Auto-detected n_gpu_layers: {n_gpu_layers}")

        # Configure batch size (critical for memory on constrained devices)
        # Default 2048 for fast prompt processing, but lower values reduce memory
        n_batch = self.requested_n_batch if self.requested_n_batch is not None else 2048

        # Memory Guard: Check available memory and reduce n_batch if needed
        # This prevents "Error 12" (CUDA OOM) on memory-constrained devices like Jetson
        available_mb = self._get_available_memory_mb()
        if available_mb is not None and available_mb < 3000 and n_batch > 512:
            logger.warning(
                f"Low memory detected ({available_mb}MB available). "
                f"Reducing n_batch from {n_batch} to 512 to prevent OOM."
            )
            n_batch = 512

        logger.info(f"Using n_batch: {n_batch}")

        # Configure thread count (None = auto-detect in Llama class)
        n_threads = self.requested_n_threads
        if n_threads is not None:
            logger.info(f"Using configured n_threads: {n_threads}")

        # Configure flash attention (default True for faster inference)
        flash_attn = (
            self.requested_flash_attn if self.requested_flash_attn is not None else True
        )
        logger.info(f"Using flash_attn: {flash_attn}")

        # Configure memory mapping (default True for efficient memory management)
        use_mmap = (
            self.requested_use_mmap if self.requested_use_mmap is not None else True
        )
        logger.info(f"Using use_mmap: {use_mmap}")

        # Configure memory locking (default False to allow OS memory management)
        use_mlock = (
            self.requested_use_mlock if self.requested_use_mlock is not None else False
        )
        logger.info(f"Using use_mlock: {use_mlock}")

        # Configure KV cache quantization (None = default f16, use q4_0 for memory savings)
        cache_type_k = self.requested_cache_type_k
        cache_type_v = self.requested_cache_type_v
        if cache_type_k is not None:
            logger.info(f"Using cache_type_k: {cache_type_k}")
        if cache_type_v is not None:
            logger.info(f"Using cache_type_v: {cache_type_v}")

        # Load model using llama-cpp
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
                    n_batch=n_batch,  # Batch size for prompt processing
                    n_gpu_layers=n_gpu_layers,  # GPU layer offloading
                    n_threads=n_threads,  # CPU threads (None = auto)
                    flash_attn=flash_attn,  # Flash attention optimization
                    use_mmap=use_mmap,  # Memory-mapped file loading
                    use_mlock=use_mlock,  # Lock model in RAM
                    cache_type_k=cache_type_k,  # KV cache key quantization
                    cache_type_v=cache_type_v,  # KV cache value quantization
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
                        f"  2. Incompatible llama-cpp binary - try reinstalling\n"
                        f"  3. Unsupported GGUF format version\n"
                        f"  File: {gguf_path}\n"
                        f"  Size: {file_size_mb:.1f} MB\n"
                        f"  Context: {self.actual_n_ctx}"
                    )
                raise

        try:
            self.llama = await loop.run_in_executor(self._executor, _load_model)

            # Initialize context management
            self._token_counter = TokenCounter(self.llama)
            budget = ContextBudget.from_context_size(self.actual_n_ctx)
            self._context_manager = ContextManager(self._token_counter, budget)

            # Pre-extract and cache GGUF metadata for chat template rendering
            # This avoids re-reading the large GGUF file on every request
            try:
                from utils.jinja_tools import (
                    get_chat_template_from_gguf,
                    get_special_tokens_from_gguf,
                    supports_native_tools,
                )

                self._chat_template = get_chat_template_from_gguf(gguf_path)
                if self._chat_template:
                    has_tools = supports_native_tools(self._chat_template)
                    logger.info(
                        f"Chat template cached ({len(self._chat_template)} chars), "
                        f"supports_native_tools={has_tools}"
                    )
                else:
                    logger.debug("No chat template found in GGUF metadata")

                self._special_tokens = get_special_tokens_from_gguf(gguf_path)
                logger.debug(
                    f"Special tokens cached: bos='{self._special_tokens.get('bos_token', '')}', "
                    f"eos='{self._special_tokens.get('eos_token', '')}'"
                )
            except Exception as e:
                logger.warning(f"Failed to cache GGUF metadata: {e}")
                self._chat_template = None
                self._special_tokens = None

            logger.info(
                f"GGUF model loaded successfully on {self.device} "
                f"with {n_gpu_layers} GPU layers and context size {self.actual_n_ctx}"
            )
        except Exception:
            # Clean up executor if load fails to prevent resource leak
            if hasattr(self, "_executor"):
                self._executor.shutdown(wait=False)
            raise

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

    @property
    def token_counter(self) -> TokenCounter | None:
        """Get the token counter for this model."""
        return self._token_counter

    @property
    def context_manager(self) -> ContextManager | None:
        """Get the context manager for this model."""
        return self._context_manager

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.

        Raises:
            RuntimeError: If model not loaded.
        """
        if self._token_counter is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._token_counter.count_tokens(text)

    def validate_context(self, messages: list[dict]) -> ContextUsage:
        """Validate messages fit within context and return usage info.

        Args:
            messages: List of chat messages to validate.

        Returns:
            ContextUsage with token counts and overflow status.

        Raises:
            RuntimeError: If model not loaded.
        """
        if self._context_manager is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._context_manager.validate_messages(messages)

    def _render_with_jinja2(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> str | None:
        """Try to render messages with tools using Jinja2 template.

        This uses the model's native chat template (cached from GGUF metadata) to render
        the prompt with tool definitions, which produces better results for models
        that were trained with native tool calling support.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tools: List of tool definitions in OpenAI format

        Returns:
            Rendered prompt string if the model supports native tools, None otherwise.
        """
        # Use cached template (extracted once during load())
        template = self._chat_template
        if not template:
            logger.debug("Jinja2 rendering skipped: no chat template cached")
            return None

        try:
            from utils.jinja_tools import (
                render_chat_with_tools,
                supports_native_tools,
            )

            has_tools = supports_native_tools(template)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Using cached chat template ({len(template)} chars), "
                    f"supports_native_tools={has_tools}"
                )
                # Log first 500 chars of template for debugging
                logger.debug(f"Template preview: {template[:500]}...")

            if not has_tools:
                logger.debug(
                    "Jinja2 rendering skipped: template does not support native tools "
                    "('tools' variable not found in template)"
                )
                return None

            # Use cached special tokens
            special_tokens = self._special_tokens or {}

            # Debug log tools being used in Jinja2 path
            if logger.isEnabledFor(logging.DEBUG):
                import json

                tool_names = [
                    t.get("function", {}).get("name", "unknown") for t in tools
                ]
                logger.debug(f"Tools provided (Jinja2 path): {tool_names}")
                logger.debug(f"Full tool definitions:\n{json.dumps(tools, indent=2)}")

            # Render the template with tools
            prompt = render_chat_with_tools(
                template=template,
                messages=messages,
                tools=tools,
                add_generation_prompt=True,
                bos_token=special_tokens.get("bos_token", ""),
                eos_token=special_tokens.get("eos_token", ""),
            )

            logger.debug(
                f"Rendered prompt with Jinja2 native tool support "
                f"({len(prompt)} chars, {len(tools)} tools)"
            )
            return prompt

        except Exception as e:
            logger.debug(f"Jinja2 tool rendering failed, will use fallback: {e}")
            return None

    def _prepare_messages_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> list[dict]:
        """Prepare messages with tool definitions using prompt injection.

        This is the fallback approach when Jinja2 rendering is not available.
        Tools are injected into the system message using XML format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tools: Optional list of tool definitions in OpenAI format
            tool_choice: Tool choice strategy:
                - None or "auto": Model may call tools (default)
                - "none": Model should not call tools
                - "required": Model must call at least one tool
                - {"type": "function", "function": {"name": "X"}}: Must call specific function

        Returns:
            Messages with tools injected (if tools provided)
        """
        if not tools:
            return messages

        # Debug log tools and tool_choice
        if logger.isEnabledFor(logging.DEBUG):
            import json

            tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
            logger.debug(f"Tools provided: {tool_names}")
            logger.debug(f"Tool choice: {tool_choice}")
            logger.debug(f"Full tool definitions:\n{json.dumps(tools, indent=2)}")

        # Inject tools into messages using prompt-based approach
        from utils.tool_calling import inject_tools_into_messages

        logger.debug(
            f"Using prompt-based tool injection with tool_choice={tool_choice}"
        )
        return inject_tools_into_messages(messages, tools, tool_choice=tool_choice)

    async def _generate_from_prompt(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        thinking_budget: int | None,
    ) -> str:
        """Generate completion from a pre-formatted prompt string.

        This is used when Jinja2 rendering produces a prompt with native tool support.

        Args:
            prompt: Pre-formatted prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop: List of stop sequences
            thinking_budget: Maximum tokens for thinking

        Returns:
            Generated text as a string
        """
        assert self.llama is not None, "Model not loaded"

        loop = asyncio.get_running_loop()

        # Capture llama reference for nested function (type checker can't see through closures)
        llama = self.llama

        def _generate():
            try:
                # Set up logits processor for thinking budget if specified
                logits_processor = None
                if thinking_budget is not None:
                    from utils.thinking import ThinkingBudgetProcessor

                    logits_processor = ThinkingBudgetProcessor(
                        llama, max_thinking_tokens=thinking_budget
                    )

                # Use create_completion for raw prompts (no chat template applied)
                return llama.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop or [],
                    logits_processor=logits_processor,
                )
            except Exception as e:
                logger.error(
                    f"Error during llama-cpp completion: {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"Completion failed: {e}") from e

        try:
            result = await loop.run_in_executor(self._executor, _generate)
            content = result["choices"][0]["message"]["content"]
            return content.strip() if content else ""
        except Exception as e:
            logger.error(f"Error extracting completion result: {e}", exc_info=True)
            raise ValueError(f"Unexpected result from completion: {e}") from e

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        thinking_budget: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> str:
        """Generate chat completion (non-streaming).

        For tool calling, this method first tries to use the model's native Jinja2
        template with tool support. If the model doesn't support native tools,
        falls back to prompt-based tool injection.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_p: Nucleus sampling threshold
            stop: List of stop sequences to end generation
            thinking_budget: Maximum tokens for thinking before forcing </think>
            tools: Optional list of tool definitions in OpenAI format
            tool_choice: Optional tool choice strategy ("auto", "none", "required")

        Returns:
            Generated text as a string

        Raises:
            AssertionError: If model not loaded
        """
        assert self.llama is not None, "Model not loaded. Call load() first."

        max_tokens = max_tokens or 512

        # Try Jinja2 native tool rendering first (if tools provided)
        if tools:
            jinja2_prompt = self._render_with_jinja2(messages, tools)
            if jinja2_prompt is not None:
                # Debug log the full prompt being sent to the LLM
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"[generate] Final prompt (Jinja2 rendered, {len(jinja2_prompt)} chars):\n"
                        f"{'=' * 60}\n{jinja2_prompt}\n{'=' * 60}"
                    )
                # Use the pre-formatted prompt directly
                return await self._generate_from_prompt(
                    prompt=jinja2_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    thinking_budget=thinking_budget,
                )

        # Fallback: use prompt injection + chat completion
        prepared_messages = self._prepare_messages_with_tools(
            messages, tools, tool_choice
        )

        # Debug log the prepared messages (prompt injection path)
        if logger.isEnabledFor(logging.DEBUG):
            import json

            logger.debug(
                f"[generate] Prepared messages ({len(prepared_messages)} messages):\n"
                f"{'=' * 60}\n{json.dumps(prepared_messages, indent=2)}\n{'=' * 60}"
            )
        loop = asyncio.get_running_loop()

        def _generate():
            try:
                # Set up logits processor for thinking budget if specified
                logits_processor = None
                if thinking_budget is not None:
                    from utils.thinking import ThinkingBudgetProcessor

                    logits_processor = ThinkingBudgetProcessor(
                        self.llama, max_thinking_tokens=thinking_budget
                    )

                # Use create_chat_completion which applies the model's chat template
                return self.llama.create_chat_completion(
                    messages=prepared_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop or [],
                    logits_processor=logits_processor,
                )
            except Exception as e:
                logger.error(
                    f"Error during llama-cpp chat completion: {e}",
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

    async def _stream_from_prompt(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        thinking_budget: int | None,
    ) -> AsyncGenerator[str, None]:
        """Stream completion from a pre-formatted prompt string.

        This is used when Jinja2 rendering produces a prompt with native tool support.

        Args:
            prompt: Pre-formatted prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop: List of stop sequences
            thinking_budget: Maximum tokens for thinking

        Yields:
            Generated text tokens as strings
        """
        assert self.llama is not None, "Model not loaded"

        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Capture llama reference for nested function (type checker can't see through closures)
        llama = self.llama

        def _generate_stream():
            """Run completion in separate thread."""
            try:
                thinking_tokens = 0
                in_thinking = False
                thinking_ended = False
                accumulated_text = ""

                # Set up logits processor for thinking budget enforcement
                logits_processor = None
                if thinking_budget is not None:
                    from utils.thinking import ThinkingBudgetProcessor

                    logits_processor = ThinkingBudgetProcessor(
                        llama, max_thinking_tokens=thinking_budget
                    )

                for chunk in llama.create_completion(
                    prompt=prompt,
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
                logger.error(f"Error in GGUF completion stream: {e}", exc_info=True)
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

    async def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        thinking_budget: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion with streaming (async generator).

        For tool calling, this method first tries to use the model's native Jinja2
        template with tool support. If the model doesn't support native tools,
        falls back to prompt-based tool injection.

        Thinking budget is enforced via logits processor, which forces the model
        to generate </think> when the budget is reached.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_p: Nucleus sampling threshold
            stop: List of stop sequences to end generation
            thinking_budget: Maximum tokens for thinking before forcing </think>
            tools: Optional list of tool definitions in OpenAI format
            tool_choice: Optional tool choice strategy ("auto", "none", "required")

        Yields:
            Generated text tokens as strings

        Raises:
            AssertionError: If model not loaded
        """
        assert self.llama is not None, "Model not loaded. Call load() first."

        max_tokens = max_tokens or 512

        # Try Jinja2 native tool rendering first (if tools provided)
        if tools:
            jinja2_prompt = self._render_with_jinja2(messages, tools)
            if jinja2_prompt is not None:
                # Debug log the full prompt being sent to the LLM
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"[generate_stream] Final prompt (Jinja2 rendered, {len(jinja2_prompt)} chars):\n"
                        f"{'=' * 60}\n{jinja2_prompt}\n{'=' * 60}"
                    )
                # Use the pre-formatted prompt directly
                async for token in self._stream_from_prompt(
                    prompt=jinja2_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    thinking_budget=thinking_budget,
                ):
                    yield token
                return

        # Fallback: use prompt injection + chat completion
        prepared_messages = self._prepare_messages_with_tools(
            messages, tools, tool_choice
        )

        # Debug log the prepared messages (prompt injection path)
        if logger.isEnabledFor(logging.DEBUG):
            import json

            logger.debug(
                f"[generate_stream] Prepared messages ({len(prepared_messages)} messages):\n"
                f"{'=' * 60}\n{json.dumps(prepared_messages, indent=2)}\n{'=' * 60}"
            )

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

                    logits_processor = ThinkingBudgetProcessor(
                        self.llama, max_thinking_tokens=thinking_budget
                    )

                for chunk in self.llama.create_chat_completion(
                    messages=prepared_messages,
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

        # Clear llama-cpp instance
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
