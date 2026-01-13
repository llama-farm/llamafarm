"""Language Detection Model using XLM-RoBERTa.

Provides language identification for text inputs using a pre-trained
XLM-RoBERTa model fine-tuned for language detection.

Supports 20 languages:
- Arabic, Bulgarian, German, Greek, English, Spanish, French, Hindi, Italian,
  Japanese, Dutch, Polish, Portuguese, Russian, Swahili, Thai, Turkish,
  Urdu, Vietnamese, Chinese

Usage:
    model = LanguageDetectionModel(model_id="lang-detect")
    await model.load()
    result = await model.detect("Hello world")
    # result = {"language": "en", "language_name": "English", "confidence": 0.99}
"""

import asyncio
import logging
from typing import Any

import torch

from models.base import BaseModel

logger = logging.getLogger(__name__)

# Pre-trained language detection model from HuggingFace
DEFAULT_MODEL = "papluca/xlm-roberta-base-language-detection"

# ISO 639-1 language code mapping
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


class LanguageDetectionModel(BaseModel):
    """Language detection model using XLM-RoBERTa.

    This model identifies the language of input text with high accuracy.
    It supports 20 languages and returns confidence scores.

    Attributes:
        model_id: Unique identifier for this model instance
        device: Device to run inference on (cpu, cuda, mps)
        hf_model_name: HuggingFace model name
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        hf_model_name: str = DEFAULT_MODEL,
        token: str | None = None,
    ):
        """Initialize language detection model.

        Args:
            model_id: Unique identifier for this model instance
            device: Device to run on (cpu, cuda, mps)
            hf_model_name: HuggingFace model to load
            token: Optional HuggingFace token for private models
        """
        super().__init__(model_id=model_id, device=device, token=token)
        self.hf_model_name = hf_model_name
        self.model_type = "language_detection"
        self.supports_streaming = False
        self._is_loaded = False
        self.pipeline = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded and self.pipeline is not None

    async def load(self) -> None:
        """Load the language detection model.

        Loads the model from HuggingFace Hub using the text-classification pipeline.
        """
        if self.is_loaded:
            logger.info(f"Model {self.model_id} already loaded")
            return

        logger.info(f"Loading language detection model: {self.hf_model_name}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

        self._is_loaded = True
        logger.info(f"Language detection model loaded: {self.model_id}")

    def _load_sync(self) -> None:
        """Synchronous model loading (run in executor)."""
        from transformers import pipeline

        # Use text-classification pipeline for language detection
        self.pipeline = pipeline(
            "text-classification",
            model=self.hf_model_name,
            tokenizer=self.hf_model_name,
            device=self.device if self.device != "mps" else -1,  # MPS not supported by pipeline
            token=self.token,
            torch_dtype=self.get_dtype() if self.device != "mps" else torch.float32,
        )

    async def unload(self) -> None:
        """Unload the model and free resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        await super().unload()
        self._is_loaded = False

    async def detect(
        self,
        text: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Detect the language of a text.

        Args:
            text: Text to detect language for
            top_k: Number of top predictions to return

        Returns:
            Dictionary with:
                - language: ISO 639-1 language code (e.g., "en")
                - language_name: Full language name (e.g., "English")
                - confidence: Confidence score (0-1)
                - all_scores: Dict mapping language codes to scores
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._detect_sync, text, top_k)

    def _detect_sync(self, text: str, top_k: int) -> dict[str, Any]:
        """Synchronous language detection."""
        # Truncate long texts to avoid memory issues
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]

        # Get predictions
        results = self.pipeline(text, top_k=top_k)

        # Parse results
        all_scores = {}
        for result in results:
            lang_code = result["label"]
            score = result["score"]
            all_scores[lang_code] = score

        # Get top prediction
        top_result = results[0]
        lang_code = top_result["label"]

        return {
            "language": lang_code,
            "language_name": LANGUAGE_NAMES.get(lang_code, lang_code),
            "confidence": top_result["score"],
            "all_scores": all_scores,
        }

    async def detect_batch(
        self,
        texts: list[str],
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        """Detect languages for multiple texts.

        Args:
            texts: List of texts to detect languages for
            top_k: Number of top predictions per text

        Returns:
            List of result dictionaries (same format as detect)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._detect_batch_sync, texts, top_k)

    def _detect_batch_sync(self, texts: list[str], top_k: int) -> list[dict[str, Any]]:
        """Synchronous batch language detection."""
        # Truncate long texts
        max_length = 512
        truncated_texts = [t[:max_length] if len(t) > max_length else t for t in texts]

        results = []
        for text in truncated_texts:
            preds = self.pipeline(text, top_k=top_k)

            all_scores = {}
            for pred in preds:
                all_scores[pred["label"]] = pred["score"]

            top = preds[0]
            lang_code = top["label"]

            results.append({
                "language": lang_code,
                "language_name": LANGUAGE_NAMES.get(lang_code, lang_code),
                "confidence": top["score"],
                "all_scores": all_scores,
            })

        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "hf_model_name": self.hf_model_name,
            "is_loaded": self.is_loaded,
            "supported_languages": list(LANGUAGE_NAMES.keys()),
        })
        return info
