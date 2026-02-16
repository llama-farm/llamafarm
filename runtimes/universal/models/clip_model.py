"""CLIP-based image classification and embedding model."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from .vision_base import ClassificationModel, ClassificationResult, EmbeddingResult

if TYPE_CHECKING:
    import torch
    from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

CLIP_VARIANTS = {
    "clip-vit-base": "openai/clip-vit-base-patch32",
    "clip-vit-large": "openai/clip-vit-large-patch14",
    "siglip-base": "google/siglip-base-patch16-224",
    "siglip-large": "google/siglip-large-patch16-256",
}


class CLIPModel(ClassificationModel):
    """CLIP-based classifier with zero-shot classification and embedding support."""

    def __init__(self, model_id: str = "clip-vit-base", device: str = "auto",
                 token: str | None = None, prompt_template: str = "a photo of a {}"):
        super().__init__(model_id, device, token)
        self.prompt_template = prompt_template
        self.clip_model: AutoModel | None = None
        self.processor: AutoProcessor | None = None
        self._class_embeddings: torch.Tensor | None = None
        self._embedding_dim: int = 0
        self._cached_class_key: tuple | None = None
        self._class_lock = asyncio.Lock()

    async def load(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModel, AutoProcessor

        self.device = self._resolve_device(self.device)
        logger.info(f"Loading CLIP model {self.model_id} on {self.device}")
        start = time.perf_counter()

        hf_id = CLIP_VARIANTS.get(self.model_id, self.model_id)

        def _load():
            model = AutoModel.from_pretrained(hf_id, token=self.token)
            proc = AutoProcessor.from_pretrained(hf_id, token=self.token)
            model = model.to(self.device)
            model.eval()
            return model, proc

        self.clip_model, self.processor = await asyncio.to_thread(_load)
        self._embedding_dim = getattr(self.clip_model.config, 'projection_dim', None) or getattr(self.clip_model.config, 'hidden_size', 512)
        self._loaded = True
        logger.info(f"CLIP loaded in {(time.perf_counter() - start) * 1000:.0f}ms (dim={self._embedding_dim})")

    async def unload(self) -> None:
        self.clip_model = None
        self.processor = None
        self._class_embeddings = None
        self._loaded = False
        await super().unload()

    async def _encode_classes(self, class_names: list[str]) -> None:
        """Pre-compute text embeddings for class names."""
        import torch
        class_key = tuple(class_names)
        # Cache check: skip re-encoding if same classes
        if class_key == self._cached_class_key:
            return
        
        self.class_names = class_names
        prompts = [self.prompt_template.format(n) for n in class_names]

        def _encode():
            inputs = self.processor(text=prompts, return_tensors="pt",
                                    padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_text_features(**inputs)
                return feats / feats.norm(dim=-1, keepdim=True)

        self._class_embeddings = await asyncio.to_thread(_encode)
        self._cached_class_key = class_key

    async def classify(self, image: bytes | np.ndarray,
                       classes: list[str] | None = None,
                       top_k: int = 5) -> ClassificationResult:
        if not self._loaded:
            await self.load()
        import torch

        if classes is not None:
            if not classes:
                raise ValueError("Empty classes list provided.")
            async with self._class_lock:
                await self._encode_classes(classes)
        elif not self.class_names or self._class_embeddings is None:
            raise ValueError("No classes provided.")

        start = time.perf_counter()
        pil_image = self._image_to_pil(image)

        def _infer():
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                sim = (feats @ self._class_embeddings.T).squeeze()
                if sim.ndim == 0:
                    sim = sim.unsqueeze(0)
                return sim.softmax(dim=-1).cpu().numpy()

        probs = await asyncio.to_thread(_infer)
        inference_time = (time.perf_counter() - start) * 1000

        effective_k = min(top_k, len(self.class_names))
        top_idx = np.argsort(probs)[::-1][:effective_k]
        best = int(top_idx[0])

        return ClassificationResult(
            confidence=float(probs[best]),
            inference_time_ms=inference_time,
            model_name=self.model_id,
            class_name=self.class_names[best],
            class_id=best,
            all_scores={self.class_names[i]: float(probs[i]) for i in top_idx},
        )

    async def embed_images(self, images: list[bytes | np.ndarray]) -> EmbeddingResult:
        """Generate embeddings for images."""
        if not self._loaded:
            await self.load()
        import torch

        start = time.perf_counter()
        pil_images = [self._image_to_pil(img) for img in images]

        def _embed():
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().tolist()

        embeddings = await asyncio.to_thread(_embed)
        return EmbeddingResult(
            confidence=1.0, inference_time_ms=(time.perf_counter() - start) * 1000,
            model_name=self.model_id, embeddings=embeddings, dimensions=self._embedding_dim,
        )

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for texts."""
        if not self._loaded:
            await self.load()
        import torch

        start = time.perf_counter()

        def _embed():
            inputs = self.processor(text=texts, return_tensors="pt",
                                    padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().tolist()

        embeddings = await asyncio.to_thread(_embed)
        return EmbeddingResult(
            confidence=1.0, inference_time_ms=(time.perf_counter() - start) * 1000,
            model_name=self.model_id, embeddings=embeddings, dimensions=self._embedding_dim,
        )

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({"variant": self.model_id, "embedding_dim": self._embedding_dim,
                     "num_classes": len(self.class_names)})
        return info
