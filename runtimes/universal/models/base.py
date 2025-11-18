"""
Base model class for all HuggingFace models (transformers & diffusers).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all model types (transformers, diffusers, etc.)."""

    def __init__(self, model_id: str, device: str, token: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self.token = token  # HuggingFace authentication token
        self.model: Optional[Any] = None
        self.tokenizer: Optional["PreTrainedTokenizerBase"] = None
        self.processor: Optional[Any] = None  # For vision/audio models
        self.feature_extractor: Optional[Any] = None  # For audio models
        self.pipe: Optional[Any] = None  # For diffusion models
        self.model_type = "unknown"
        self.supports_streaming = False

    @abstractmethod
    async def load(self) -> None:
        """Load the model and associated components."""
        pass

    async def unload(self) -> None:
        """Unload the model and free resources.

        Default implementation for transformers models. Subclasses should override
        if they need custom cleanup (e.g., GGUF models with llama-cpp-python).
        """
        logger.info(f"Unloading model: {self.model_id}")

        # Move model to CPU to free GPU memory
        if self.model is not None and hasattr(self.model, "to"):
            try:
                self.model = self.model.to("cpu")
            except Exception as e:
                logger.warning(f"Could not move model to CPU: {e}")

        # Clear references
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.feature_extractor = None
        self.pipe = None

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")

        # Clear MPS cache if available (PyTorch 2.0+)
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache")
            except Exception:
                pass

        logger.info(f"Model unloaded: {self.model_id}")

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "device": self.device,
            "supports_streaming": self.supports_streaming,
        }

    def get_dtype(self):
        """Get optimal torch dtype for the device."""
        if self.device == "cuda":
            return torch.float16
        elif self.device == "mps":
            return torch.float16
        else:
            return torch.float32

    def apply_optimizations(self):
        """Apply platform-specific optimizations."""
        if self.pipe is None:
            return

        try:
            if self.device == "mps":
                # MPS optimizations
                self.pipe.enable_attention_slicing()
                logger.info("Enabled attention slicing for MPS")
            elif self.device == "cuda":
                # CUDA optimizations
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception:
                    logger.warning("xformers not available, skipping")

                try:
                    self.pipe.enable_model_cpu_offload()
                    logger.info("Enabled model CPU offload")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not apply optimizations: {e}")
