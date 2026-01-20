"""Background Removal Model using RMBG.

Removes backgrounds from images using RMBG (Remove Background),
a state-of-the-art background removal model from BRIA AI.

Usage:
    model = BackgroundRemovalModel(model_id="rmbg")
    await model.load()
    result = await model.remove_background("image.jpg")
    # Returns: {"image": "<base64 PNG with alpha>", "mask": "<optional base64 mask>"}
"""

import asyncio
import base64
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Any

from models.base import BaseModel
from utils.image_utils import load_image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default RMBG model - high quality background removal
DEFAULT_RMBG_MODEL = "briaai/RMBG-1.4"

# Allowlist of trusted models that can use trust_remote_code=True
# These models have been reviewed for security and are from trusted sources.
# SECURITY: Only add models here after reviewing their repository code.
TRUSTED_REMOTE_CODE_MODELS = frozenset([
    "briaai/RMBG-1.4",
    "briaai/RMBG-2.0",
])


class BackgroundRemovalModel(BaseModel):
    """Background removal model using RMBG.

    RMBG is a state-of-the-art background removal model that produces
    high-quality alpha masks for separating foreground from background.

    Security Note:
        This model uses `trust_remote_code=True` which allows execution of
        code from the HuggingFace model repository. Only models listed in
        TRUSTED_REMOTE_CODE_MODELS are allowed. Attempting to load other
        models will raise a SecurityError.

    Attributes:
        model_id: Unique identifier for this model instance
        device: Device to run inference on (cpu, cuda, mps)
        hf_model_name: HuggingFace model name (must be in allowlist)
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        hf_model_name: str = DEFAULT_RMBG_MODEL,
        token: str | None = None,
    ):
        """Initialize background removal model.

        Args:
            model_id: Unique identifier for this model instance
            device: Device to run on (cpu, cuda, mps)
            hf_model_name: HuggingFace model to load
            token: Optional HuggingFace token
        """
        super().__init__(model_id=model_id, device=device, token=token)
        self.hf_model_name = hf_model_name
        self.model_type = "background_removal"
        self.supports_streaming = False
        self._is_loaded = False
        self.pipe = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded and self.pipe is not None

    async def load(self) -> None:
        """Load the RMBG model pipeline.

        Raises:
            SecurityError: If the model is not in the trusted allowlist.
        """
        if self.is_loaded:
            logger.info(f"Model {self.model_id} already loaded")
            return

        # Security check: Only allow trusted models with remote code execution
        if self.hf_model_name not in TRUSTED_REMOTE_CODE_MODELS:
            raise ValueError(
                f"Security Error: Model '{self.hf_model_name}' is not in the trusted "
                f"allowlist for remote code execution. Allowed models: "
                f"{sorted(TRUSTED_REMOTE_CODE_MODELS)}"
            )

        logger.info(f"Loading RMBG background removal model: {self.hf_model_name}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

        self._is_loaded = True
        logger.info(f"RMBG model loaded: {self.model_id}")

    def _load_sync(self) -> None:
        """Synchronous model loading (run in executor)."""
        from transformers import pipeline

        # RMBG uses the image-segmentation pipeline type
        # Force float32 for MPS compatibility
        self.pipe = pipeline(
            "image-segmentation",
            model=self.hf_model_name,
            trust_remote_code=True,
            device=self.device if self.device != "mps" else "cpu",  # RMBG may have MPS issues
            torch_dtype=self.get_dtype(force_float32=True),
        )

    async def unload(self) -> None:
        """Unload the model and free resources."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        await super().unload()
        self._is_loaded = False

    async def remove_background(
        self,
        image: str | bytes | Any,
        return_mask: bool = False,
    ) -> dict[str, Any]:
        """Remove background from an image.

        Args:
            image: Image input - can be file path, base64 string, bytes, or PIL Image
            return_mask: If True, also return the alpha mask

        Returns:
            Dictionary with:
                - image: Base64-encoded PNG with transparent background
                - mask: (optional) Base64-encoded grayscale mask
                - width: Image width
                - height: Image height
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._remove_background_sync,
            image,
            return_mask,
        )

    def _remove_background_sync(
        self,
        image: str | bytes | Any,
        return_mask: bool,
    ) -> dict[str, Any]:
        """Synchronous background removal."""
        from PIL import Image

        # Load image
        pil_image = load_image(image)
        original_size = pil_image.size

        # Run the pipeline - returns list of segment results
        results = self.pipe(pil_image)

        # RMBG pipeline returns a list with mask/segment info
        # The result depends on the pipeline output format
        if isinstance(results, list) and len(results) > 0:
            # Get the mask from the first result
            result = results[0]
            if "mask" in result:
                mask = result["mask"]
            else:
                # Try to use the result directly as a mask
                mask = result.get("score", None)
                if mask is None and hasattr(result, "mask"):
                    mask = result.mask
        else:
            # Fallback: use the result directly if it's an image
            mask = results if isinstance(results, Image.Image) else None

        if mask is None:
            raise RuntimeError("Failed to get mask from RMBG pipeline")

        # Ensure mask is PIL Image and same size as original
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)

        # Resize mask to original size if needed
        if mask.size != original_size:
            mask = mask.resize(original_size, Image.Resampling.LANCZOS)

        # Convert mask to L mode (grayscale) if needed
        if mask.mode != "L":
            mask = mask.convert("L")

        # Create output image with alpha channel
        output = pil_image.convert("RGBA")
        output.putalpha(mask)

        # Encode output as base64 PNG
        output_buffer = BytesIO()
        output.save(output_buffer, format="PNG")
        output_b64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        result = {
            "image": output_b64,
            "width": original_size[0],
            "height": original_size[1],
        }

        # Optionally include mask
        if return_mask:
            mask_buffer = BytesIO()
            mask.save(mask_buffer, format="PNG")
            result["mask"] = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")

        return result

    async def remove_background_batch(
        self,
        images: list[str | bytes | Any],
        return_mask: bool = False,
    ) -> list[dict[str, Any]]:
        """Remove background from multiple images.

        Args:
            images: List of image inputs
            return_mask: If True, also return the alpha masks

        Returns:
            List of results (one per image)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []
        for image in images:
            result = await self.remove_background(image, return_mask=return_mask)
            results.append(result)
        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "hf_model_name": self.hf_model_name,
            "is_loaded": self.is_loaded,
            "output_format": "PNG with alpha channel",
        })
        return info
