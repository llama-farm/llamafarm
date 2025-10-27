"""
Diffusion model wrapper for image generation.
"""

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
import torch
from typing import List, Optional
from PIL import Image
import logging
import random

from .base import BaseModel

logger = logging.getLogger(__name__)


class DiffusionModel(BaseModel):
    """Wrapper for HuggingFace diffusion models (Stable Diffusion, FLUX, etc.)."""

    def __init__(self, model_id: str, device: str, token: Optional[str] = None):
        super().__init__(model_id, device, token=token)
        self.model_type = "diffusion"
        self.default_steps = 50
        self.default_guidance = 7.5

    async def load(self):
        """Load the diffusion model."""
        logger.info(f"Loading diffusion model: {self.model_id}")

        dtype = self.get_dtype()

        # Determine pipeline type based on model
        if "inpaint" in self.model_id.lower():
            pipeline_class = StableDiffusionInpaintPipeline
        elif "xl" in self.model_id.lower():
            pipeline_class = StableDiffusionXLPipeline
        else:
            # Try auto-detection
            pipeline_class = DiffusionPipeline

        # Load pipeline - try safetensors first, fall back to regular weights
        # Disable safety checker to avoid false positives on innocent images
        try:
            self.pipe = pipeline_class.from_pretrained(
                self.model_id,
                dtype=dtype,
                trust_remote_code=True,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False,
            )
        except (OSError, ValueError) as e:
            if "safetensors" in str(e).lower():
                logger.info("Model doesn't have safetensors, using standard weights")
                self.pipe = pipeline_class.from_pretrained(
                    self.model_id,
                    dtype=dtype,
                    trust_remote_code=True,
                    use_safetensors=False,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            else:
                raise

        self.pipe = self.pipe.to(self.device)

        # Apply optimizations
        self.apply_optimizations()

        logger.info(f"Diffusion model loaded on {self.device}")

    def _get_scheduler(self, scheduler_name: Optional[str] = None):
        """Get scheduler by name."""
        if scheduler_name is None:
            return None

        scheduler_map = {
            "ddim": DDIMScheduler,
            "pndm": PNDMScheduler,
            "lms": LMSDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "dpm++": DPMSolverMultistepScheduler,
        }

        scheduler_class = scheduler_map.get(scheduler_name.lower())
        if scheduler_class:
            return scheduler_class.from_config(self.pipe.scheduler.config)

        return None

    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        scheduler: Optional[str] = None,
    ) -> List[Image.Image]:
        """Generate images from text prompt."""

        # Set defaults
        steps = num_inference_steps or self.default_steps
        guidance = guidance_scale or self.default_guidance

        # Handle seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Set scheduler if requested
        if scheduler:
            custom_scheduler = self._get_scheduler(scheduler)
            if custom_scheduler:
                self.pipe.scheduler = custom_scheduler

        # Generate
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

        return result.images

    async def edit(
        self,
        prompt: str,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Edit/inpaint an image."""

        if not isinstance(self.pipe, StableDiffusionInpaintPipeline):
            raise ValueError("Model does not support inpainting")

        steps = num_inference_steps or self.default_steps
        guidance = guidance_scale or self.default_guidance

        # If no mask provided, inpainting won't work properly - it needs a mask
        # For now, raise an error to indicate proper mask is required
        if mask is None:
            raise ValueError(
                "Inpainting requires a mask. Please provide a mask image or use a different model/endpoint."
            )

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

        return result.images

    async def img2img(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: float = 0.75,
        seed: Optional[int] = None,
        scheduler: Optional[str] = None,
    ) -> List[Image.Image]:
        """Transform an image based on a text prompt (img2img)."""

        # Load img2img pipeline if not already loaded
        if not hasattr(self, "img2img_pipe"):
            logger.info(f"Loading img2img pipeline for {self.model_id}")
            dtype = self.get_dtype()

            try:
                self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    dtype=dtype,
                    trust_remote_code=True,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            except (OSError, ValueError) as e:
                if "safetensors" in str(e).lower():
                    self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        self.model_id,
                        dtype=dtype,
                        trust_remote_code=True,
                        use_safetensors=False,
                        safety_checker=None,
                        requires_safety_checker=False,
                    )
                else:
                    raise

            self.img2img_pipe = self.img2img_pipe.to(self.device)

            # Apply optimizations to img2img pipe
            old_pipe = self.pipe
            self.pipe = self.img2img_pipe
            self.apply_optimizations()
            self.pipe = old_pipe

        steps = num_inference_steps or self.default_steps
        guidance = guidance_scale or self.default_guidance

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Set scheduler if requested
        if scheduler:
            custom_scheduler = self._get_scheduler(scheduler)
            if custom_scheduler:
                self.img2img_pipe.scheduler = custom_scheduler

        with torch.no_grad():
            result = self.img2img_pipe(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=steps,
                guidance_scale=guidance,
                strength=strength,
                generator=generator,
            )

        return result.images
