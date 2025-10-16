"""
Base model class for all transformers models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import torch
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all model types."""

    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.model_type = "unknown"

    @abstractmethod
    async def load(self):
        """Load the model."""
        pass

    @abstractmethod
    async def generate(self, *args, **kwargs):
        """Generate output."""
        pass

    def get_torch_dtype(self):
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
