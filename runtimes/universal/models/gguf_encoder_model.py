"""
GGUF encoder model wrapper using llama-cpp for embeddings.

Provides the same interface as EncoderModel but uses llama-cpp for
GGUF quantized embedding models, enabling faster inference and lower memory usage.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from utils.gguf_metadata_cache import get_gguf_metadata_cached
from utils.gpu_allocator import InsufficientVRAMError, get_llama_gpu_params
from utils.model_format import get_gguf_file_path

from .base import BaseModel

if TYPE_CHECKING:
    from llamafarm_llama import Llama

logger = logging.getLogger(__name__)


class GGUFEncoderModel(BaseModel):
    """Wrapper for GGUF embedding models using llama-cpp.

    This class provides an interface compatible with EncoderModel but uses
    llama-cpp for inference with GGUF quantized embedding models. GGUF
    embedding models offer:
    - 50-75% smaller file sizes (4-bit/8-bit quantization)
    - 2-3x faster inference on Apple Silicon (Metal)
    - Significantly lower memory requirements
    - Optimized CPU inference

    The model is automatically configured for the target device (Metal/CUDA/CPU)
    and specializes in generating text embeddings.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        token: str | None = None,
        preferred_quantization: str | None = None,
    ):
        """Initialize GGUF encoder model for embeddings.

        Args:
            model_id: HuggingFace model identifier (e.g., "nomic-ai/nomic-embed-text-v1.5-GGUF")
            device: Target device ("cuda", "mps", or "cpu")
            token: Optional HuggingFace authentication token for gated models
            preferred_quantization: Optional quantization preference (e.g., "Q4_K_M", "Q8_0").
                                    If None, defaults to Q4_K_M. Only downloads the specified
                                    quantization to save disk space.
        """
        super().__init__(model_id, device, token=token)
        self.model_type = "encoder_embedding"
        self.supports_streaming = False
        self.llama: Llama | None = None
        self.preferred_quantization = preferred_quantization
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def load(self) -> None:
        """Load the GGUF embedding model using llama-cpp.

        This method:
        1. Locates the .gguf file in the HuggingFace cache
        2. Configures GPU layers based on the target device
        3. Initializes the llama-cpp Llama instance in embedding mode
        4. Runs initialization in a thread pool (blocking operation)

        Raises:
            FileNotFoundError: If no .gguf file found in model repository
            Exception: If model loading fails
        """

        logger.info(f"Loading GGUF embedding model: {self.model_id}")

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

        # Configure GPU layers for llama.cpp (uses its own GPU detection, not PyTorch's)
        from utils.device import get_gguf_gpu_layers

        n_gpu_layers = get_gguf_gpu_layers()

        # Embedding models use a small, fixed context window.  This value
        # is shared between the VRAM estimator and the Llama() constructor
        # so that the allocation decision matches actual memory usage.
        embedding_n_ctx = 512

        # GPU allocation: select optimal GPU based on free VRAM
        gpu_params = {}
        try:
            metadata = get_gguf_metadata_cached(gguf_path)
            gpu_params = get_llama_gpu_params(
                model_size_bytes=metadata.file_size_bytes,
                n_ctx=embedding_n_ctx,
                n_gpu_layers=n_gpu_layers,
                total_layers=metadata.n_layer,
                n_layer=metadata.n_layer,
                n_head_kv=metadata.n_head_kv,
                head_k_size=metadata.head_k_size,
                head_v_size=metadata.head_v_size,
            )
            if gpu_params:
                logger.info(
                    f"GPU allocation: main_gpu={gpu_params.get('main_gpu')}, "
                    f"split_mode={gpu_params.get('split_mode')}"
                )
            else:
                logger.debug("No CUDA GPUs detected, using default GPU allocation")
        except InsufficientVRAMError as e:
            if e.gpu_details:
                logger.error(f"GPU allocation failed:\n{e.gpu_details}")
            else:
                logger.error(f"GPU allocation failed: {e}")
            raise RuntimeError(str(e)) from e
        except Exception as e:
            logger.warning(f"GPU allocation failed, using defaults: {e}")

        # Load model using llama-cpp in embedding mode
        # Run in thread pool since Llama() initialization is blocking
        loop = asyncio.get_running_loop()

        def _load_model():
            try:
                from llamafarm_llama import Llama
            except ImportError as e:
                raise ImportError(
                    "llamafarm-llama is required for GGUF models but is not installed. "
                    "Install it with: pip install llamafarm-llama"
                ) from e

            # Build GPU-specific kwargs from allocation
            gpu_kwargs = {}
            if gpu_params.get("main_gpu") is not None:
                gpu_kwargs["main_gpu"] = gpu_params["main_gpu"]
            if gpu_params.get("split_mode") is not None:
                gpu_kwargs["split_mode"] = gpu_params["split_mode"]
            if gpu_params.get("tensor_split") is not None:
                gpu_kwargs["tensor_split"] = gpu_params["tensor_split"]

            return Llama(
                model_path=gguf_path,
                embedding=True,  # Enable embedding mode
                n_ctx=embedding_n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=None,  # Auto-detect optimal threads
                verbose=False,  # Disable verbose logging
                **gpu_kwargs,
            )

        self.llama = await loop.run_in_executor(self._executor, _load_model)

        logger.info(
            f"GGUF embedding model loaded successfully on {self.device} "
            f"with {n_gpu_layers} GPU layers"
        )

    async def embed(
        self, texts: list[str], normalize: bool = True
    ) -> list[list[float]]:
        """Generate embeddings for input texts.

        Args:
            texts: List of input texts
            normalize: Whether to L2 normalize embeddings (default: True)

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            AssertionError: If model not loaded

        Examples:
            >>> model = GGUFEncoderModel("nomic-ai/nomic-embed-text-v1.5-GGUF", "cpu")
            >>> await model.load()
            >>> embeddings = await model.embed(["Hello world", "How are you?"])
            >>> len(embeddings)
            2
        """
        assert self.llama is not None, "Model not loaded. Call load() first."

        # Short-circuit for empty input to avoid numpy.AxisError during normalization
        if not texts:
            return []

        # Run embedding generation in thread pool (blocking call)
        loop = asyncio.get_running_loop()

        def _generate_embeddings():
            """Generate embeddings in separate thread."""
            try:
                embeddings = []
                for text in texts:
                    # llama-cpp create_embedding returns dict with 'data' key
                    result = self.llama.create_embedding(input=text)
                    embedding = result["data"][0]["embedding"]
                    embeddings.append(embedding)
                return embeddings
            except Exception as e:
                logger.error(
                    f"Error during llama-cpp embedding generation: {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"Embedding generation failed: {e}") from e

        try:
            embeddings = await loop.run_in_executor(
                self._executor, _generate_embeddings
            )

            # Apply L2 normalization if requested
            if normalize:
                import numpy as np

                embeddings_array = np.array(embeddings)
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                # Avoid division by zero
                norms = np.maximum(norms, 1e-9)
                embeddings_array = embeddings_array / norms
                embeddings = embeddings_array.tolist()

            return embeddings
        except RuntimeError:
            # Re-raise errors from _generate_embeddings() with their original message
            raise
        except Exception as e:
            # Handle errors from normalization or other processing steps
            logger.error(f"Error during embedding post-processing: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process embeddings: {e}") from e

    async def generate(self, *args, **kwargs):
        """Not applicable for encoder models."""
        raise NotImplementedError("Encoder models do not support text generation")

    async def unload(self) -> None:
        """Unload GGUF encoder model and free resources."""
        logger.info(f"Unloading GGUF encoder model: {self.model_id}")

        # Clear llama-cpp instance
        self.llama = None

        # Shutdown thread pool executor
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True, cancel_futures=True)
            # Create new executor for potential future use
            self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info(f"GGUF encoder model unloaded: {self.model_id}")

    def __del__(self):
        """Cleanup thread pool executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
