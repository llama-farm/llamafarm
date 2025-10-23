"""
Multimodal model wrapper for vision-language tasks.
"""

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
    pipeline,
)
import torch
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import io
import base64
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class MultimodalModel(BaseModel):
    """Wrapper for HuggingFace multimodal models (BLIP, LLaVA, Florence, etc.)."""

    def __init__(self, model_id: str, device: str, task: str = "image-to-text"):
        """
        Initialize multimodal model.

        Args:
            model_id: HuggingFace model ID
            device: Target device (cuda/mps/cpu)
            task: Model task - "image-to-text", "vqa", or "visual-chat"
        """
        super().__init__(model_id, device)
        self.task = task
        self.model_type = f"multimodal_{task}"
        self.supports_streaming = False

    async def load(self):
        """Load the multimodal model."""
        logger.info(f"Loading multimodal model ({self.task}): {self.model_id}")

        dtype = self.get_dtype()

        # Try to use pipeline for supported tasks
        if self.task in ["image-to-text", "vqa"]:
            try:
                self.pipe = pipeline(
                    self.task,
                    model=self.model_id,
                    dtype=dtype,
                    device=self.device,
                )
                # Also load processor for consistency (tests expect it)
                from transformers import AutoProcessor

                self.processor = AutoProcessor.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
                logger.info(f"Loaded multimodal model via pipeline on {self.device}")
                return
            except Exception as e:
                logger.warning(f"Pipeline loading failed, trying manual: {e}")

        # Manual loading for BLIP and similar models
        if "blip" in self.model_id.lower():
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=dtype,
                trust_remote_code=True,
            )
            self.processor = BlipProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        else:
            # Generic approach
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                dtype=dtype,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Multimodal model loaded on {self.device}")

    def _decode_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """Decode image from various input formats."""
        if isinstance(image_input, Image.Image):
            return image_input

        if isinstance(image_input, str):
            # Base64 string
            if image_input.startswith("data:image"):
                image_input = image_input.split(",", 1)[1]
            image_bytes = base64.b64decode(image_input)
            return Image.open(io.BytesIO(image_bytes))

        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    async def caption(
        self,
        image: Union[str, bytes, Image.Image],
        max_length: int = 50,
        num_beams: int = 3,
    ) -> str:
        """
        Generate a caption for an image.

        Args:
            image: Image (base64, bytes, or PIL Image)
            max_length: Maximum caption length
            num_beams: Number of beams for beam search

        Returns:
            Generated caption
        """
        pil_image = self._decode_image(image)

        # Use pipeline if available
        if self.pipe:
            # ImageToTextPipeline has limited parameter support
            # Just pass the image and let it use defaults
            result = self.pipe(pil_image)
            return result[0]["generated_text"] if result else ""

        # Manual generation
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_length=max_length, num_beams=num_beams
            )

        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    async def answer_question(
        self,
        image: Union[str, bytes, Image.Image],
        question: str,
        max_length: int = 100,
    ) -> str:
        """
        Answer a question about an image (VQA).

        Args:
            image: Image (base64, bytes, or PIL Image)
            question: Question to answer
            max_length: Maximum answer length

        Returns:
            Generated answer
        """
        pil_image = self._decode_image(image)

        # Use pipeline if available
        if self.pipe:
            if self.task == "vqa":
                # VQA pipeline takes image and question
                result = self.pipe(image=pil_image, question=question)
                return result[0]["answer"] if result else ""
            elif self.task == "image-to-text":
                # Image-to-text models (like BLIP) can handle questions as prompts
                # For BLIP, we can use the processor with text conditioning
                inputs = self.processor(
                    images=pil_image, text=question, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get the model from the pipeline
                model = self.pipe.model
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_length=max_length)

                answer = self.processor.decode(output_ids[0], skip_special_tokens=True)
                return answer.strip()

        # Manual VQA (when no pipeline)
        if self.model:
            inputs = self.processor(
                images=pil_image, text=question, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_length=max_length)

            answer = self.processor.decode(output_ids[0], skip_special_tokens=True)
            return answer.strip()

        raise ValueError("No model or pipeline available for VQA")

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Union[str, bytes, Image.Image]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Multi-turn visual chat (for LLaVA-style models).

        Args:
            messages: Chat history with role and content
            images: Optional list of images to include
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Model response
        """
        if not images:
            raise ValueError("Visual chat requires at least one image")

        # Decode images
        pil_images = [self._decode_image(img) for img in images]

        # Format conversation
        # This is model-specific; LLaVA uses special tokens
        conversation = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                conversation += f"USER: {content}\n"
            elif role == "assistant":
                conversation += f"ASSISTANT: {content}\n"

        conversation += "ASSISTANT:"

        # Process
        inputs = self.processor(
            images=pil_images[0] if len(pil_images) == 1 else pil_images,
            text=conversation,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Decode response
        response = self.processor.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the new response (after last ASSISTANT:)
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()

        return response

    async def generate(
        self,
        image: Union[str, bytes, Image.Image],
        prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Generate text from image (with optional prompt/question).

        Args:
            image: Input image
            prompt: Optional text prompt or question
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if prompt:
            return await self.answer_question(image, prompt, **kwargs)
        else:
            return await self.caption(image, **kwargs)
