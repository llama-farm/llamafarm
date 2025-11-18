"""
Base model class for all HuggingFace models (transformers & diffusers).
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import torch
import logging

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

    def get_model_info(self) -> Dict[str, Any]:
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
