"""
High-level Llama interface, API-compatible with llama-cpp-python.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from ._bindings import ensure_backend, ffi, get_lib, get_mtmd_lib
from .types import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    EmbeddingResponse,
)

logger = logging.getLogger(__name__)


class Llama:
    """
    High-level interface to llama.cpp, API-compatible with llama-cpp-python.

    Example:
        >>> from llamafarm_llama import Llama
        >>> llm = Llama(model_path="model.gguf", n_ctx=2048, n_gpu_layers=-1)
        >>> response = llm.create_chat_completion(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100,
        ... )
        >>> print(response["choices"][0]["message"]["content"])
    """

    # GGML type enum values for KV cache quantization
    # These reduce memory usage at the cost of some precision
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1    # Default - full precision
    GGML_TYPE_Q4_0 = 2   # 4-bit quantization - significant memory savings
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8   # 8-bit quantization - balanced

    # Map string names to enum values
    GGML_TYPE_MAP = {
        "f32": GGML_TYPE_F32,
        "f16": GGML_TYPE_F16,
        "q4_0": GGML_TYPE_Q4_0,
        "q4_1": GGML_TYPE_Q4_1,
        "q5_0": GGML_TYPE_Q5_0,
        "q5_1": GGML_TYPE_Q5_1,
        "q8_0": GGML_TYPE_Q8_0,
    }

    def __init__(
        self,
        model_path: str,
        *,
        mmproj_path: Optional[str] = None,  # Multimodal projector path for audio/vision
        n_ctx: int = 2048,
        n_batch: int = 2048,  # Larger batch = faster prompt processing
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        n_gpu_layers: int = 0,
        main_gpu: int = 0,
        tensor_split: Optional[List[float]] = None,
        vocab_only: bool = False,
        use_mmap: bool = True,
        use_mlock: bool = False,
        seed: int = -1,
        rope_freq_base: float = 0.0,
        rope_freq_scale: float = 0.0,
        embedding: bool = False,
        flash_attn: bool = True,  # Enable by default for faster inference
        cache_type_k: Optional[str] = None,  # KV cache key quantization (e.g., "q4_0", "q8_0", "f16")
        cache_type_v: Optional[str] = None,  # KV cache value quantization (e.g., "q4_0", "q8_0", "f16")
        verbose: bool = True,
        warmup: bool = True,  # Run warmup inference to compile shaders
        **kwargs,
    ):
        """
        Initialize a Llama model.

        Args:
            model_path: Path to the GGUF model file.
            mmproj_path: Path to multimodal projector file (.gguf) for audio/vision models.
                         If provided, enables audio/vision input via create_chat_completion_with_audio().
            n_ctx: Context window size.
            n_batch: Batch size for prompt processing. Larger = faster TTFT.
            n_threads: Number of threads for generation. None = auto.
            n_threads_batch: Number of threads for batch processing. None = auto.
            n_gpu_layers: Number of layers to offload to GPU. -1 = all.
            main_gpu: Main GPU to use.
            tensor_split: How to split tensors across GPUs.
            vocab_only: Only load vocabulary.
            use_mmap: Use memory mapping.
            use_mlock: Lock model in memory.
            seed: Random seed. -1 = random.
            rope_freq_base: RoPE frequency base.
            rope_freq_scale: RoPE frequency scale.
            embedding: Enable embedding mode.
            flash_attn: Use flash attention for faster inference.
            cache_type_k: KV cache key quantization type. Options: "f32", "f16" (default),
                          "q8_0", "q4_0". Lower precision = less memory but slightly
                          reduced quality. "q4_0" can reduce KV cache memory by ~4x.
            cache_type_v: KV cache value quantization type. Same options as cache_type_k.
                          Setting both to "q4_0" significantly reduces memory on devices
                          like Jetson Orin Nano (8GB shared memory).
            verbose: Print verbose output.
            warmup: Run warmup inference to pre-compile GPU shaders.
                    This moves shader compilation cost to load time,
                    significantly reducing TTFT on first request.
        """
        # Ensure backend is initialized
        ensure_backend()

        self._lib = get_lib()
        self._verbose = verbose

        # Windows path fix (match existing behavior)
        if sys.platform == "win32":
            model_path = model_path.replace("\\", "/")

        self._model_path = model_path
        self._embedding_mode = embedding

        # Get default thread count
        import os

        cpu_count = os.cpu_count() or 4
        if n_threads is None:
            n_threads = min(cpu_count, 8)
        if n_threads_batch is None:
            n_threads_batch = n_threads

        # Model parameters
        model_params = self._lib.llama_model_default_params()
        # In llama.cpp b7376+, n_gpu_layers=-1 is interpreted literally (no layers)
        # Use a large number (999) to mean "all layers on GPU"
        if n_gpu_layers < 0:
            n_gpu_layers = 999  # Offload all layers to GPU
        model_params.n_gpu_layers = n_gpu_layers
        model_params.main_gpu = main_gpu
        model_params.vocab_only = vocab_only
        model_params.use_mmap = use_mmap
        model_params.use_mlock = use_mlock

        if tensor_split:
            # Create tensor split array
            ts_array = ffi.new("float[]", tensor_split)
            model_params.tensor_split = ts_array
            self._tensor_split = ts_array  # Keep reference
        else:
            self._tensor_split = None

        # Load model
        if verbose:
            logger.info(f"Loading model: {model_path}")

        model_path_bytes = model_path.encode("utf-8")
        self._model = self._lib.llama_load_model_from_file(
            model_path_bytes, model_params
        )

        if self._model == ffi.NULL:
            raise RuntimeError(f"Failed to load model: {model_path}")

        # Context parameters
        ctx_params = self._lib.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_batch
        ctx_params.n_ubatch = n_batch  # Match ubatch to batch for max GPU parallelism
        ctx_params.n_threads = n_threads
        ctx_params.n_threads_batch = n_threads_batch
        ctx_params.embeddings = embedding

        # Performance optimizations for GPU inference
        # flash_attn_type: -1 = auto, 0 = disabled, 1 = enabled
        # Use auto (-1) by default to let llama.cpp decide based on hardware
        ctx_params.flash_attn_type = 1 if flash_attn else -1
        # Offload KV cache to GPU (critical for performance)
        ctx_params.offload_kqv = True
        # Offload operations to GPU
        ctx_params.op_offload = True

        # KV cache quantization for memory savings
        # Setting type_k and type_v to q4_0 can reduce KV cache memory by ~4x
        # This is critical for memory-constrained devices like Jetson Orin Nano
        if cache_type_k is not None:
            type_k_lower = cache_type_k.lower()
            if type_k_lower in self.GGML_TYPE_MAP:
                ctx_params.type_k = self.GGML_TYPE_MAP[type_k_lower]
                if verbose:
                    logger.info(f"Using cache_type_k: {cache_type_k}")
            else:
                logger.warning(
                    f"Unknown cache_type_k '{cache_type_k}', using default. "
                    f"Valid options: {list(self.GGML_TYPE_MAP.keys())}"
                )

        if cache_type_v is not None:
            type_v_lower = cache_type_v.lower()
            if type_v_lower in self.GGML_TYPE_MAP:
                ctx_params.type_v = self.GGML_TYPE_MAP[type_v_lower]
                if verbose:
                    logger.info(f"Using cache_type_v: {cache_type_v}")
            else:
                logger.warning(
                    f"Unknown cache_type_v '{cache_type_v}', using default. "
                    f"Valid options: {list(self.GGML_TYPE_MAP.keys())}"
                )

        if rope_freq_base > 0:
            ctx_params.rope_freq_base = rope_freq_base
        if rope_freq_scale > 0:
            ctx_params.rope_freq_scale = rope_freq_scale

        if seed == -1:
            # Use random seed
            import random

            seed = random.randint(0, 2**32 - 1)

        # Create context
        self._ctx = self._lib.llama_new_context_with_model(self._model, ctx_params)

        if self._ctx == ffi.NULL:
            self._lib.llama_free_model(self._model)
            raise RuntimeError("Failed to create llama context")

        # Store config
        self._n_ctx = n_ctx
        self._n_batch = n_batch
        self._n_threads = n_threads

        # Get vocab from model (new API - llama.cpp b7376+)
        self._vocab = self._lib.llama_model_get_vocab(self._model)

        # Get memory handle for KV cache operations (new API - llama.cpp b7376+)
        self._memory = self._lib.llama_get_memory(self._ctx)

        # Initialize sampler chain
        self._sampler = None

        # Initialize multimodal context if projector path provided
        self._mtmd_lib = None
        self._mtmd_ctx = ffi.NULL
        self._supports_audio = False
        self._supports_vision = False

        if mmproj_path:
            self._init_multimodal(mmproj_path, n_threads, flash_attn, verbose)

        if verbose:
            n_vocab = self._lib.llama_vocab_n_tokens(self._vocab)
            n_ctx_train = self._lib.llama_model_n_ctx_train(self._model)
            n_embd = self._lib.llama_model_n_embd(self._model)
            logger.info(
                f"Model loaded: vocab={n_vocab}, ctx_train={n_ctx_train}, embd={n_embd}"
            )

        # Warmup: run a dummy decode to compile Metal shaders and warm caches
        # This significantly reduces TTFT on the first real request
        if warmup and not embedding:
            self._warmup()

    def _warmup(self):
        """Run a warmup inference to compile GPU shaders and warm caches.

        On first inference, Metal/CUDA need to compile shaders which adds
        significant latency. Running a short warmup moves this cost to
        model loading time rather than first request time.
        """
        t_start = time.perf_counter()
        logger.debug("Running warmup inference...")

        try:
            # Clear KV cache
            self._lib.llama_memory_clear(self._memory, True)

            # Tokenize a short prompt
            warmup_text = "Hello"
            tokens = self.tokenize(warmup_text, add_special=True, parse_special=False)

            # Decode the tokens (this compiles shaders)
            if tokens:
                self._decode_batch(tokens)

            # Sample one token to warm up the sampler path
            self._create_sampler(temperature=0.7, top_p=0.95, top_k=40)
            self._sample_token()

            # Clear state for real inference
            self._lib.llama_memory_clear(self._memory, True)
            if self._sampler is not None:
                self._lib.llama_sampler_free(self._sampler)
                self._sampler = None

            t_end = time.perf_counter()
            logger.info(f"Warmup completed in {(t_end - t_start)*1000:.1f}ms")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    def _init_multimodal(
        self,
        mmproj_path: str,
        n_threads: int,
        flash_attn: bool,
        verbose: bool,
    ) -> None:
        """Initialize multimodal context for audio/vision input.

        Args:
            mmproj_path: Path to the multimodal projector file (.gguf)
            n_threads: Number of threads for encoding
            flash_attn: Whether to use flash attention
            verbose: Print verbose output
        """
        # Load mtmd library
        self._mtmd_lib = get_mtmd_lib()
        if self._mtmd_lib is None:
            raise RuntimeError(
                "Failed to load mtmd library. Multimodal support requires libmtmd. "
                "Try clearing the cache and re-downloading: "
                "python -c \"from llamafarm_llama._binary import clear_cache; clear_cache()\""
            )

        if verbose:
            logger.info(f"Loading multimodal projector: {mmproj_path}")

        # Get default params and configure
        mtmd_params = self._mtmd_lib.mtmd_context_params_default()
        mtmd_params.use_gpu = True
        mtmd_params.n_threads = n_threads
        mtmd_params.flash_attn_type = 1 if flash_attn else -1
        mtmd_params.warmup = True
        mtmd_params.print_timings = verbose

        # Initialize mtmd context with the model
        mmproj_path_bytes = mmproj_path.encode("utf-8")
        self._mtmd_ctx = self._mtmd_lib.mtmd_init_from_file(
            mmproj_path_bytes,
            self._model,
            mtmd_params,
        )

        if self._mtmd_ctx == ffi.NULL:
            raise RuntimeError(f"Failed to load multimodal projector: {mmproj_path}")

        # Query capabilities
        self._supports_audio = self._mtmd_lib.mtmd_support_audio(self._mtmd_ctx)
        self._supports_vision = self._mtmd_lib.mtmd_support_vision(self._mtmd_ctx)

        if verbose:
            caps = []
            if self._supports_audio:
                caps.append("audio")
            if self._supports_vision:
                caps.append("vision")
            logger.info(f"Multimodal capabilities: {', '.join(caps) if caps else 'none'}")

    @property
    def supports_audio(self) -> bool:
        """Whether this model supports audio input."""
        return self._supports_audio

    @property
    def supports_vision(self) -> bool:
        """Whether this model supports vision (image) input."""
        return self._supports_vision

    def create_audio_bitmap(
        self,
        audio_data: Union[bytes, List[float]],
        audio_format: str = "wav",
    ) -> "ffi.CData":
        """Create an audio bitmap for multimodal processing.

        Args:
            audio_data: Audio data as WAV bytes or PCM float samples (16kHz mono)
            audio_format: Format of audio_data - "wav" for WAV bytes, "pcm" for float samples

        Returns:
            mtmd_bitmap handle (must be freed with free_bitmap)

        Raises:
            RuntimeError: If model doesn't support audio
        """
        if not self._supports_audio:
            raise RuntimeError("Model does not support audio input")

        if audio_format == "wav":
            # Use helper that decodes WAV internally via miniaudio
            buf = ffi.new(f"unsigned char[{len(audio_data)}]", audio_data)
            bitmap = self._mtmd_lib.mtmd_helper_bitmap_init_from_buf(
                self._mtmd_ctx, buf, len(audio_data)
            )
        elif audio_format == "pcm":
            # PCM float array (16kHz mono, float32)
            if isinstance(audio_data, bytes):
                import struct
                n_samples = len(audio_data) // 4
                samples = list(struct.unpack(f"{n_samples}f", audio_data))
            else:
                samples = list(audio_data)

            samples_arr = ffi.new(f"float[{len(samples)}]", samples)
            bitmap = self._mtmd_lib.mtmd_bitmap_init_from_audio(
                len(samples), samples_arr
            )
        else:
            raise ValueError(f"Unknown audio format: {audio_format}")

        if bitmap == ffi.NULL:
            raise RuntimeError("Failed to create audio bitmap")

        return bitmap

    def free_bitmap(self, bitmap: "ffi.CData") -> None:
        """Free a bitmap created by create_audio_bitmap.

        Args:
            bitmap: Bitmap handle from create_audio_bitmap
        """
        if self._mtmd_lib is not None and bitmap != ffi.NULL:
            self._mtmd_lib.mtmd_bitmap_free(bitmap)

    def create_chat_completion_with_audio(
        self,
        messages: List[Dict],
        audio_data: Union[bytes, List[float]],
        audio_format: str = "wav",
        *,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        stream: bool = False,
        stop: Optional[List[str]] = None,
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionChunk]]:
        """Create a chat completion with audio input.

        The audio will be processed through the multimodal projector and
        inserted at the default marker position in the prompt.

        Args:
            messages: List of chat messages. Should contain the marker for audio placement.
            audio_data: Audio data as WAV bytes or PCM float samples (16kHz mono)
            audio_format: Format - "wav" for WAV bytes, "pcm" for float samples
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            min_p: Min-p sampling
            repeat_penalty: Repetition penalty
            stream: Whether to stream the response
            stop: Stop sequences

        Returns:
            ChatCompletionResponse or Iterator[ChatCompletionChunk] if streaming
        """
        if not self._supports_audio:
            raise RuntimeError("Model does not support audio input")

        # Create audio bitmap
        bitmap = self.create_audio_bitmap(audio_data, audio_format)

        # Tokenize messages with audio
        # Guard against tokenization errors to avoid leaking the bitmap
        try:
            chunks = self._tokenize_with_media(messages, [bitmap])
        except Exception:
            self.free_bitmap(bitmap)
            raise

        # Process chunks and generate response
        if stream:
            # Wrap streaming in a generator that manages bitmap lifetime
            # This ensures bitmap is freed AFTER the stream is consumed,
            # avoiding use-after-free when mtmd_encode_chunk accesses the bitmap
            def _stream_with_cleanup():
                try:
                    yield from self._stream_completion_with_chunks(
                        chunks,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        repeat_penalty=repeat_penalty,
                        stop=stop,
                    )
                finally:
                    self.free_bitmap(bitmap)

            return _stream_with_cleanup()
        else:
            try:
                return self._complete_with_chunks(
                    chunks,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop,
                )
            finally:
                self.free_bitmap(bitmap)

    def _tokenize_with_media(
        self,
        messages: List[Dict],
        bitmaps: List["ffi.CData"],
    ) -> "ffi.CData":
        """Tokenize messages with media (audio/images).

        Args:
            messages: Chat messages
            bitmaps: List of media bitmaps

        Returns:
            mtmd_input_chunks handle
        """
        # Apply chat template to get prompt text
        prompt = self._apply_chat_template(messages)

        # Insert default marker if not present
        default_marker = ffi.string(self._mtmd_lib.mtmd_default_marker()).decode("utf-8")
        if default_marker not in prompt and bitmaps:
            # Insert marker before the last user message content
            # This is a simple heuristic - models may need specific placement
            last_content = messages[-1].get("content", "")
            # Handle multimodal content (list of parts)
            if isinstance(last_content, list):
                # Extract text from text parts for the replacement
                text_parts = []
                for part in last_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                last_content_str = " ".join(text_parts)
            else:
                last_content_str = last_content or ""

            if last_content_str:
                prompt = prompt.replace(
                    last_content_str,
                    f"{default_marker}{last_content_str}"
                )

        # Create input text struct
        input_text = ffi.new("struct mtmd_input_text *")
        prompt_bytes = prompt.encode("utf-8")
        input_text.text = ffi.new("char[]", prompt_bytes)
        input_text.add_special = True
        input_text.parse_special = True

        # Create output chunks
        chunks = self._mtmd_lib.mtmd_input_chunks_init()
        if chunks == ffi.NULL:
            raise RuntimeError("Failed to create input chunks")

        # Create bitmap array
        bitmaps_arr = ffi.new(f"mtmd_bitmap *[{len(bitmaps)}]", bitmaps)

        # Tokenize with media
        result = self._mtmd_lib.mtmd_tokenize(
            self._mtmd_ctx,
            chunks,
            input_text,
            bitmaps_arr,
            len(bitmaps),
        )

        if result != 0:
            self._mtmd_lib.mtmd_input_chunks_free(chunks)
            raise RuntimeError(f"Failed to tokenize with media: error {result}")

        return chunks

    def _complete_with_chunks(
        self,
        chunks: "ffi.CData",
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repeat_penalty: float,
        stop: Optional[List[str]],
    ) -> ChatCompletionResponse:
        """Generate completion from tokenized chunks."""
        try:
            # Clear KV cache
            self._lib.llama_memory_clear(self._memory, True)

            # Process each chunk
            n_chunks = self._mtmd_lib.mtmd_input_chunks_size(chunks)
            total_tokens = 0

            for i in range(n_chunks):
                chunk = self._mtmd_lib.mtmd_input_chunks_get(chunks, i)
                chunk_type = self._mtmd_lib.mtmd_input_chunk_get_type(chunk)

                if chunk_type == 0:  # MTMD_INPUT_CHUNK_TYPE_TEXT
                    # Get tokens and decode
                    n_tokens_ptr = ffi.new("size_t *")
                    tokens_ptr = self._mtmd_lib.mtmd_input_chunk_get_tokens_text(chunk, n_tokens_ptr)
                    n_tokens = n_tokens_ptr[0]

                    if n_tokens > 0:
                        tokens = [tokens_ptr[j] for j in range(n_tokens)]
                        self._decode_batch(tokens)
                        total_tokens += n_tokens
                else:
                    # Image/audio chunk - encode and use embeddings
                    result = self._mtmd_lib.mtmd_encode_chunk(self._mtmd_ctx, chunk)
                    if result != 0:
                        raise RuntimeError(f"Failed to encode chunk: error {result}")

                    # Get embeddings and feed to model
                    embd_ptr = self._mtmd_lib.mtmd_get_output_embd(self._mtmd_ctx)
                    n_embd_tokens = self._mtmd_lib.mtmd_input_chunk_get_n_tokens(chunk)

                    # Create batch with embeddings
                    self._decode_embeddings(embd_ptr, n_embd_tokens)
                    total_tokens += n_embd_tokens

            # Create sampler and generate
            self._create_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repeat_penalty=repeat_penalty,
            )

            # Generate tokens
            generated_tokens = []
            generated_text = ""
            finish_reason = "length"

            for _ in range(max_tokens):
                token = self._sample_token()

                # Check EOS
                if self._lib.llama_vocab_is_eog(self._vocab, token):
                    finish_reason = "stop"
                    break

                generated_tokens.append(token)
                text = self.detokenize([token])
                generated_text += text

                # Check stop sequences
                if stop:
                    for s in stop:
                        if s in generated_text:
                            finish_reason = "stop"
                            break
                    if finish_reason == "stop":
                        break

                # Decode for next iteration
                self._decode_batch([token])

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self._model_path,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": total_tokens,
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": total_tokens + len(generated_tokens),
                },
            }
        finally:
            self._mtmd_lib.mtmd_input_chunks_free(chunks)
            if self._sampler is not None:
                self._lib.llama_sampler_free(self._sampler)
                self._sampler = None

    def _stream_completion_with_chunks(
        self,
        chunks: "ffi.CData",
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repeat_penalty: float,
        stop: Optional[List[str]],
    ) -> Iterator[ChatCompletionChunk]:
        """Stream completion from tokenized chunks."""
        try:
            # Clear KV cache
            self._lib.llama_memory_clear(self._memory, True)

            # Process each chunk (same as non-streaming)
            n_chunks = self._mtmd_lib.mtmd_input_chunks_size(chunks)

            for i in range(n_chunks):
                chunk = self._mtmd_lib.mtmd_input_chunks_get(chunks, i)
                chunk_type = self._mtmd_lib.mtmd_input_chunk_get_type(chunk)

                if chunk_type == 0:  # TEXT
                    n_tokens_ptr = ffi.new("size_t *")
                    tokens_ptr = self._mtmd_lib.mtmd_input_chunk_get_tokens_text(chunk, n_tokens_ptr)
                    n_tokens = n_tokens_ptr[0]

                    if n_tokens > 0:
                        tokens = [tokens_ptr[j] for j in range(n_tokens)]
                        self._decode_batch(tokens)
                else:
                    # Image/audio
                    result = self._mtmd_lib.mtmd_encode_chunk(self._mtmd_ctx, chunk)
                    if result != 0:
                        raise RuntimeError(f"Failed to encode chunk: error {result}")

                    embd_ptr = self._mtmd_lib.mtmd_get_output_embd(self._mtmd_ctx)
                    n_embd_tokens = self._mtmd_lib.mtmd_input_chunk_get_n_tokens(chunk)
                    self._decode_embeddings(embd_ptr, n_embd_tokens)

            # Create sampler
            self._create_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repeat_penalty=repeat_penalty,
            )

            # Stream generation
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            generated_text = ""

            for i in range(max_tokens):
                token = self._sample_token()

                # Check EOS
                if self._lib.llama_vocab_is_eog(self._vocab, token):
                    yield {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self._model_path,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    break

                text = self.detokenize([token])
                generated_text += text

                # Check stop sequences
                should_stop = False
                if stop:
                    for s in stop:
                        if s in generated_text:
                            should_stop = True
                            break

                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self._model_path,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": "stop" if should_stop else None,
                        }
                    ],
                }

                if should_stop:
                    break

                # Decode for next iteration
                self._decode_batch([token])
        finally:
            self._mtmd_lib.mtmd_input_chunks_free(chunks)
            if self._sampler is not None:
                self._lib.llama_sampler_free(self._sampler)
                self._sampler = None

    def _decode_embeddings(self, embd_ptr: "ffi.CData", n_tokens: int) -> None:
        """Decode embeddings through the model.

        Args:
            embd_ptr: Pointer to embedding data
            n_tokens: Number of embedding tokens
        """
        n_embd = self._lib.llama_model_n_embd(self._model)

        # Create batch with embeddings
        batch = self._lib.llama_batch_init(n_tokens, n_embd, 1)
        batch.n_tokens = n_tokens

        # Copy embeddings to batch
        for i in range(n_tokens * n_embd):
            batch.embd[i] = embd_ptr[i]

        # Set positions
        pos = self._lib.llama_n_ctx(self._ctx) - n_tokens  # Current position
        for i in range(n_tokens):
            batch.pos[i] = pos + i
            batch.n_seq_id[i] = 1
            seq_id = ffi.new("llama_seq_id[1]", [0])
            batch.seq_id[i] = seq_id
            batch.logits[i] = 0 if i < n_tokens - 1 else 1

        # Decode
        result = self._lib.llama_decode(self._ctx, batch)
        self._lib.llama_batch_free(batch)

        if result != 0:
            raise RuntimeError(f"Failed to decode embeddings: error {result}")

    def _cleanup(self):
        """Internal cleanup logic. Safe to call multiple times."""
        if getattr(self, "_closed", False):
            return
        self._closed = True

        if hasattr(self, "_sampler") and self._sampler is not None:
            self._lib.llama_sampler_free(self._sampler)
            self._sampler = None

        # Free multimodal context before llama context
        if hasattr(self, "_mtmd_ctx") and self._mtmd_ctx is not None and self._mtmd_ctx != ffi.NULL:
            if self._mtmd_lib is not None:
                self._mtmd_lib.mtmd_free(self._mtmd_ctx)
            self._mtmd_ctx = ffi.NULL
            # Reset multimodal flags to prevent use-after-free
            # If these remain True, callers checking supports_audio/supports_vision
            # would attempt to use the freed multimodal context
            self._supports_audio = False
            self._supports_vision = False

        if hasattr(self, "_ctx") and self._ctx is not None and self._ctx != ffi.NULL:
            self._lib.llama_free(self._ctx)
            self._ctx = ffi.NULL  # Mark as freed to prevent double-free

        if hasattr(self, "_model") and self._model is not None and self._model != ffi.NULL:
            self._lib.llama_free_model(self._model)
            self._model = ffi.NULL  # Mark as freed to prevent double-free

    def __del__(self):
        """Clean up resources on garbage collection."""
        self._cleanup()

    @property
    def n_ctx(self) -> int:
        """Get the context size."""
        return self._lib.llama_n_ctx(self._ctx)

    @property
    def n_vocab(self) -> int:
        """Get the vocabulary size."""
        return self._lib.llama_vocab_n_tokens(self._vocab)

    @property
    def n_embd(self) -> int:
        """Get the embedding dimension."""
        return self._lib.llama_model_n_embd(self._model)

    def tokenize(
        self,
        text: str,
        add_special: bool = True,
        parse_special: bool = False,
    ) -> List[int]:
        """
        Tokenize text into tokens.

        Args:
            text: Text to tokenize.
            add_special: Add special tokens (BOS, etc.).
            parse_special: Parse special tokens in text.

        Returns:
            List of token IDs.
        """
        text_bytes = text.encode("utf-8")
        n_tokens_max = len(text_bytes) + 16  # Estimate

        tokens = ffi.new(f"llama_token[{n_tokens_max}]")
        n_tokens = self._lib.llama_tokenize(
            self._vocab,
            text_bytes,
            len(text_bytes),
            tokens,
            n_tokens_max,
            add_special,
            parse_special,
        )

        if n_tokens < 0:
            # Need more space
            n_tokens_max = -n_tokens
            tokens = ffi.new(f"llama_token[{n_tokens_max}]")
            n_tokens = self._lib.llama_tokenize(
                self._vocab,
                text_bytes,
                len(text_bytes),
                tokens,
                n_tokens_max,
                add_special,
                parse_special,
            )

        return [tokens[i] for i in range(n_tokens)]

    def detokenize(self, tokens: List[int], remove_special: bool = False) -> str:
        """
        Convert tokens back to text.

        Args:
            tokens: List of token IDs.
            remove_special: Remove special tokens.

        Returns:
            Decoded text.
        """
        raw_bytes = self._detokenize_bytes(tokens, remove_special)
        return raw_bytes.decode("utf-8", errors="replace")

    def _detokenize_bytes(self, tokens: List[int], remove_special: bool = False) -> bytes:
        """
        Convert tokens back to raw bytes (without UTF-8 decoding).

        This is useful for streaming where we need to buffer incomplete
        UTF-8 sequences (e.g., emojis split across multiple tokens).

        Args:
            tokens: List of token IDs.
            remove_special: Remove special tokens.

        Returns:
            Raw bytes from detokenization.
        """
        if not tokens:
            return b""

        tokens_array = ffi.new(f"llama_token[{len(tokens)}]", tokens)
        buf_size = len(tokens) * 16  # Estimate
        buf = ffi.new(f"char[{buf_size}]")

        n_chars = self._lib.llama_detokenize(
            self._vocab,
            tokens_array,
            len(tokens),
            buf,
            buf_size,
            remove_special,
            False,  # unparse_special
        )

        if n_chars < 0:
            # Need more space
            buf_size = -n_chars
            buf = ffi.new(f"char[{buf_size}]")
            n_chars = self._lib.llama_detokenize(
                self._vocab,
                tokens_array,
                len(tokens),
                buf,
                buf_size,
                remove_special,
                False,
            )

        return ffi.buffer(buf, n_chars)[:]

    @staticmethod
    def _decode_utf8_streaming(data: bytes) -> tuple[str, bytes]:
        """
        Decode UTF-8 bytes, returning complete text and any incomplete trailing bytes.

        This handles the case where multi-byte UTF-8 sequences (like emojis)
        are split across multiple tokens. We decode only the complete characters
        and return incomplete trailing bytes to be buffered.

        Args:
            data: Raw bytes to decode.

        Returns:
            Tuple of (decoded_text, incomplete_trailing_bytes).
        """
        if not data:
            return "", b""

        # Try to decode everything first
        try:
            return data.decode("utf-8"), b""
        except UnicodeDecodeError:
            pass

        # Find the longest valid UTF-8 prefix
        # UTF-8 continuation bytes start with 10xxxxxx (0x80-0xBF)
        # Start bytes: 0xxxxxxx (ASCII), 110xxxxx (2-byte), 1110xxxx (3-byte), 11110xxx (4-byte)
        for i in range(len(data) - 1, max(len(data) - 4, -1), -1):
            try:
                return data[:i].decode("utf-8"), data[i:]
            except UnicodeDecodeError:
                continue

        # Nothing decodable, return all as pending
        return "", data

    def _apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply chat template to messages."""
        # Build chat message array
        n_msg = len(messages)
        chat_array = ffi.new(f"struct llama_chat_message[{n_msg}]")

        # Keep references to prevent garbage collection
        role_refs = []
        content_refs = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "user").encode("utf-8")
            # Handle None content (e.g., tool call messages may have content: null)
            raw_content = msg.get("content")
            if raw_content is None:
                # Tool call messages typically have null content, but warn for other roles
                msg_role = msg.get("role", "user")
                if msg_role not in ("assistant", "tool"):
                    logger.warning(
                        f"Message at index {i} with role '{msg_role}' has None content, "
                        "using empty string"
                    )
                content_str = ""
            elif isinstance(raw_content, list):
                # Handle multimodal messages (list of content parts)
                # Extract text from text parts, use placeholder for other types
                text_parts = []
                for part in raw_content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "")
                        if part_type == "text":
                            text_parts.append(part.get("text", ""))
                        elif part_type == "input_audio":
                            text_parts.append("[audio]")
                        elif part_type == "image_url":
                            text_parts.append("[image]")
                        # Skip unknown types
                content_str = " ".join(text_parts)
            else:
                content_str = raw_content or ""
            content = content_str.encode("utf-8")
            role_refs.append(ffi.new("char[]", role))
            content_refs.append(ffi.new("char[]", content))
            chat_array[i].role = role_refs[-1]
            chat_array[i].content = content_refs[-1]

        # Get required buffer size
        buf_size = 4096
        buf = ffi.new(f"char[{buf_size}]")

        # llama.cpp b7376+: model param removed, first arg is now template string
        # Pass NULL to use model's default template
        n_chars = self._lib.llama_chat_apply_template(
            ffi.NULL,  # Use model's default template
            chat_array,
            n_msg,
            add_generation_prompt,
            buf,
            buf_size,
        )

        if n_chars < 0:
            raise RuntimeError("Failed to apply chat template")

        if n_chars > buf_size:
            # Need more space
            buf_size = n_chars + 1
            buf = ffi.new(f"char[{buf_size}]")
            n_chars = self._lib.llama_chat_apply_template(
                ffi.NULL,
                chat_array,
                n_msg,
                add_generation_prompt,
                buf,
                buf_size,
            )

        return ffi.string(buf, n_chars).decode("utf-8")

    def _create_sampler(
        self,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
    ):
        """Create a sampler chain."""
        if self._sampler is not None:
            self._lib.llama_sampler_free(self._sampler)

        chain_params = self._lib.llama_sampler_chain_default_params()
        chain = self._lib.llama_sampler_chain_init(chain_params)

        # Add samplers to chain
        if top_k > 0:
            self._lib.llama_sampler_chain_add(
                chain, self._lib.llama_sampler_init_top_k(top_k)
            )

        if top_p < 1.0:
            self._lib.llama_sampler_chain_add(
                chain, self._lib.llama_sampler_init_top_p(top_p, 1)
            )

        if min_p > 0.0:
            self._lib.llama_sampler_chain_add(
                chain, self._lib.llama_sampler_init_min_p(min_p, 1)
            )

        if temperature > 0:
            self._lib.llama_sampler_chain_add(
                chain, self._lib.llama_sampler_init_temp(temperature)
            )

        # Add distribution sampler
        if seed is None:
            import random

            seed = random.randint(0, 2**32 - 1)

        self._lib.llama_sampler_chain_add(
            chain, self._lib.llama_sampler_init_dist(seed)
        )

        self._sampler = chain

    def _decode_batch(self, tokens: List[int]) -> bool:
        """Decode a batch of tokens, chunking if necessary.

        llama.cpp has a batch size limit (n_batch). If the token count exceeds
        this limit, we need to decode in chunks.
        """
        if not tokens:
            return True

        # Process tokens in chunks of n_batch size
        n_batch = self._n_batch
        offset = 0

        while offset < len(tokens):
            # Get the chunk for this iteration
            chunk = tokens[offset:offset + n_batch]
            chunk_size = len(chunk)

            tokens_array = ffi.new(f"llama_token[{chunk_size}]", chunk)
            batch = self._lib.llama_batch_get_one(tokens_array, chunk_size)

            result = self._lib.llama_decode(self._ctx, batch)
            if result != 0:
                logger.error(f"Failed to decode batch at offset {offset}")
                return False

            offset += chunk_size

        return True

    def _sample_token(self) -> int:
        """Sample the next token."""
        # Sample using the sampler chain
        token = self._lib.llama_sampler_sample(self._sampler, self._ctx, -1)
        self._lib.llama_sampler_accept(self._sampler, token)

        return token

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        logits_processor: Optional[Callable] = None,
        **kwargs,
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionChunk]]:
        """
        Generate a chat completion.

        Args:
            messages: List of chat messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            min_p: Min-p sampling.
            repeat_penalty: Repetition penalty.
            stop: Stop sequences.
            stream: Stream the response.
            seed: Random seed.
            logits_processor: Custom logits processor (for ThinkingBudgetProcessor).

        Returns:
            Chat completion response or stream of chunks.
        """
        t_start = time.perf_counter()

        # Apply chat template
        prompt = self._apply_chat_template(messages, add_generation_prompt=True)

        # Tokenize
        tokens = self.tokenize(prompt, add_special=False, parse_special=True)
        t_tokenize = time.perf_counter()

        if len(tokens) > self._n_ctx:
            raise ValueError(
                f"Prompt too long: {len(tokens)} tokens > {self._n_ctx} context"
            )

        # Clear KV cache
        self._lib.llama_memory_clear(self._memory, True)

        # Decode prompt (this is the main TTFT cost)
        if not self._decode_batch(tokens):
            raise RuntimeError("Failed to decode prompt")
        t_prompt = time.perf_counter()

        logger.info(
            f"[TTFT] Tokenization: {(t_tokenize - t_start)*1000:.1f}ms, "
            f"Prompt processing ({len(tokens)} tokens): {(t_prompt - t_tokenize)*1000:.1f}ms"
        )

        # Create sampler
        self._create_sampler(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )

        if stream:
            return self._stream_completion(tokens, max_tokens, stop, logits_processor, t_start)
        else:
            return self._complete(tokens, max_tokens, stop, logits_processor, t_start)

    def create_completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        logits_processor: Optional[Callable] = None,
        **kwargs,
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionChunk]]:
        """
        Generate a completion from a raw prompt string (no chat template applied).

        This is useful when you have a pre-formatted prompt, such as one rendered
        from a Jinja2 chat template with tool definitions.

        Args:
            prompt: The raw prompt string to generate from.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            min_p: Min-p sampling.
            repeat_penalty: Repetition penalty.
            stop: Stop sequences.
            stream: Stream the response.
            seed: Random seed.
            logits_processor: Custom logits processor.

        Returns:
            Completion response or stream of chunks.
        """
        t_start = time.perf_counter()

        # Tokenize directly (no chat template application)
        tokens = self.tokenize(prompt, add_special=False, parse_special=True)
        t_tokenize = time.perf_counter()

        if len(tokens) > self._n_ctx:
            raise ValueError(
                f"Prompt too long: {len(tokens)} tokens > {self._n_ctx} context"
            )

        # Clear KV cache
        self._lib.llama_memory_clear(self._memory, True)

        # Decode prompt (this is the main TTFT cost)
        if not self._decode_batch(tokens):
            raise RuntimeError("Failed to decode prompt")
        t_prompt = time.perf_counter()

        logger.info(
            f"[TTFT] Tokenization: {(t_tokenize - t_start)*1000:.1f}ms, "
            f"Prompt processing ({len(tokens)} tokens): {(t_prompt - t_tokenize)*1000:.1f}ms"
        )

        # Create sampler
        self._create_sampler(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )

        if stream:
            return self._stream_completion(tokens, max_tokens, stop, logits_processor, t_start)
        else:
            return self._complete(tokens, max_tokens, stop, logits_processor, t_start)

    def _complete(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        stop: Optional[List[str]],
        logits_processor: Optional[Callable],
        t_start: float,
    ) -> ChatCompletionResponse:
        """Generate a non-streaming completion."""
        generated_tokens = []
        finish_reason = "length"
        t_first_token = None

        for i in range(max_tokens):
            # Apply logits processor if provided
            if logits_processor is not None:
                logits = self._lib.llama_get_logits(self._ctx)
                n_vocab = self._lib.llama_vocab_n_tokens(self._vocab)
                # Convert to list for processor
                logits_list = [logits[i] for i in range(n_vocab)]
                processed = logits_processor(
                    prompt_tokens + generated_tokens, logits_list
                )
                # Write back
                for j, v in enumerate(processed):
                    logits[j] = v

            token = self._sample_token()
            generated_tokens.append(token)

            # Log TTFT on first token
            if i == 0:
                t_first_token = time.perf_counter()
                logger.info(f"[TTFT] First token: {(t_first_token - t_start)*1000:.1f}ms total")

            # Check for EOS
            if self._lib.llama_vocab_is_eog(self._vocab, token):
                finish_reason = "stop"
                break

            # Decode the new token for stop sequence check
            if stop:
                text = self.detokenize(generated_tokens)
                for s in stop:
                    if s in text:
                        finish_reason = "stop"
                        # Trim to stop sequence
                        text = text[: text.index(s)]
                        generated_tokens = self.tokenize(
                            text, add_special=False, parse_special=False
                        )
                        break
                if finish_reason == "stop":
                    break

            # Decode single token for next iteration
            if not self._decode_batch([token]):
                raise RuntimeError("Failed to decode token")

        # Build response
        t_end = time.perf_counter()
        content = self.detokenize(generated_tokens)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Log generation stats
        gen_time = t_end - t_first_token if t_first_token else t_end - t_start
        tokens_per_sec = len(generated_tokens) / gen_time if gen_time > 0 else 0
        logger.info(
            f"[Perf] Generated {len(generated_tokens)} tokens in {gen_time*1000:.1f}ms "
            f"({tokens_per_sec:.1f} tok/s)"
        )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(generated_tokens),
                "total_tokens": len(prompt_tokens) + len(generated_tokens),
            },
        }

    def _stream_completion(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        stop: Optional[List[str]],
        logits_processor: Optional[Callable],
        t_start: float,
    ) -> Iterator[ChatCompletionChunk]:
        """Generate a streaming completion."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        generated_tokens = []
        accumulated_text = ""
        # Track bytes we've already processed to compute deltas
        prev_bytes_len = 0
        # Buffer for incomplete UTF-8 byte sequences (e.g., partial emojis)
        pending_bytes = b""

        for i in range(max_tokens):
            # Apply logits processor if provided
            if logits_processor is not None:
                logits = self._lib.llama_get_logits(self._ctx)
                n_vocab = self._lib.llama_vocab_n_tokens(self._vocab)
                logits_list = [logits[j] for j in range(n_vocab)]
                processed = logits_processor(
                    prompt_tokens + generated_tokens, logits_list
                )
                for j, v in enumerate(processed):
                    logits[j] = v

            token = self._sample_token()
            generated_tokens.append(token)

            # Log TTFT on first token
            if i == 0:
                t_first_token = time.perf_counter()
                logger.info(f"[TTFT] First token: {(t_first_token - t_start)*1000:.1f}ms total")

            # Check for EOS
            if self._lib.llama_vocab_is_eog(self._vocab, token):
                # Flush any pending bytes (decode with replacement for incomplete sequences)
                if pending_bytes:
                    final_text = pending_bytes.decode("utf-8", errors="replace")
                    if final_text:
                        yield {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self._model_path,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": final_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self._model_path,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                break

            # Decode to get the new text, handling incomplete UTF-8 sequences
            # (e.g., emojis that span multiple tokens)
            all_bytes = self._detokenize_bytes(generated_tokens)
            # New bytes = all bytes minus what we've already processed
            # prev_bytes_len tracks how many bytes we've already decoded into accumulated_text
            new_bytes = all_bytes[prev_bytes_len:]
            combined_bytes = pending_bytes + new_bytes

            # Decode only complete UTF-8 sequences, buffer incomplete ones
            decoded_text, pending_bytes = self._decode_utf8_streaming(combined_bytes)
            delta = decoded_text
            accumulated_text += decoded_text
            # Update how many bytes we've fully processed (not counting pending)
            prev_bytes_len = len(all_bytes) - len(pending_bytes)

            # Check stop sequences
            finish_reason = None
            if stop:
                for s in stop:
                    if s in accumulated_text:
                        finish_reason = "stop"
                        delta = delta[: delta.index(s)] if s in delta else ""
                        break

            if delta:
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self._model_path,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": finish_reason,
                        }
                    ],
                }

            if finish_reason:
                # Flush any pending bytes before stopping
                if pending_bytes:
                    final_text = pending_bytes.decode("utf-8", errors="replace")
                    if final_text:
                        yield {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self._model_path,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": final_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self._model_path,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                break

            # Decode single token for next iteration
            if not self._decode_batch([token]):
                raise RuntimeError("Failed to decode token")
        else:
            # Max tokens reached - flush any pending bytes first
            if pending_bytes:
                final_text = pending_bytes.decode("utf-8", errors="replace")
                if final_text:
                    yield {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self._model_path,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": final_text},
                                "finish_reason": None,
                            }
                        ],
                    }
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self._model_path,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "length",
                    }
                ],
            }

    def create_embedding(
        self,
        input: Union[str, List[str]],
        **kwargs,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for input text(s).

        Args:
            input: Text or list of texts to embed.

        Returns:
            Embedding response.
        """
        if not self._embedding_mode:
            raise RuntimeError("Model not initialized in embedding mode")

        texts = [input] if isinstance(input, str) else input
        embeddings_data = []

        for idx, text in enumerate(texts):
            # Tokenize
            tokens = self.tokenize(text, add_special=True, parse_special=False)

            # Clear KV cache
            self._lib.llama_memory_clear(self._memory, True)

            # Decode
            if not self._decode_batch(tokens):
                raise RuntimeError(f"Failed to decode text {idx}")

            # Get embeddings
            emb_ptr = self._lib.llama_get_embeddings(self._ctx)
            if emb_ptr == ffi.NULL:
                # Try sequence embeddings
                emb_ptr = self._lib.llama_get_embeddings_seq(self._ctx, 0)

            if emb_ptr == ffi.NULL:
                raise RuntimeError("Failed to get embeddings")

            n_embd = self._lib.llama_model_n_embd(self._model)
            embedding = [emb_ptr[i] for i in range(n_embd)]

            # L2 normalize
            import math

            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]

            embeddings_data.append(
                {
                    "index": idx,
                    "embedding": embedding,
                    "object": "embedding",
                }
            )

        return {
            "object": "list",
            "data": embeddings_data,
            "model": self._model_path,
            "usage": {
                "prompt_tokens": sum(
                    len(self.tokenize(t, add_special=True)) for t in texts
                ),
                "total_tokens": sum(
                    len(self.tokenize(t, add_special=True)) for t in texts
                ),
            },
        }

    def reset(self):
        """Reset the context state."""
        self._lib.llama_memory_clear(self._memory, True)
        if self._sampler is not None:
            self._lib.llama_sampler_reset(self._sampler)

    def close(self):
        """Explicitly close and free resources."""
        self._cleanup()
