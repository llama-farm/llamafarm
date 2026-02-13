"""
Speech emotion recognition model using Wav2Vec2.

Analyzes raw audio to detect emotional tone in speech, enabling
the LLM to understand *how* the user is speaking, not just what they say.

Supported emotions (model-dependent):
- angry, calm, disgust, fearful, happy, neutral, sad, surprised

Designed for real-time voice pipelines where emotion context can:
- Adjust LLM response tone
- Trigger empathetic responses
- Detect frustration for escalation
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)

# Pre-trained emotion recognition models
# These are Wav2Vec2 models fine-tuned on emotion datasets
# IMPORTANT: All models must have a classification head trained for emotion recognition.
# Base pre-trained models (like facebook/wav2vec2-large-xlsr-53) will NOT work.
EMOTION_MODELS = {
    # English speech emotion - 8 classes (RAVDESS dataset)
    "wav2vec2-lg-xlsr-en": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    # Alternative: superb emotion recognition (4 classes: ang, hap, neu, sad)
    "wav2vec2-base-superb": "superb/wav2vec2-base-superb-er",
    # Large SUPERB emotion recognition model (better accuracy, more compute)
    "wav2vec2-large-superb": "superb/wav2vec2-large-superb-er",
    # Robust emotion model fine-tuned on MSP-Podcast (valence/arousal/dominance dimensions)
    # Note: This model outputs dimensional emotions, not categorical
    "wav2vec2-robust-emotion": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
}

# Default model - good balance of accuracy and speed
DEFAULT_MODEL = "wav2vec2-lg-xlsr-en"

# Expected sample rate for Wav2Vec2 models
EXPECTED_SAMPLE_RATE = 16000


@dataclass
class EmotionResult:
    """Result from emotion classification on audio."""

    emotion: str
    confidence: float
    all_scores: dict[str, float]
    duration_seconds: float


class EmotionModel(BaseModel):
    """Wav2Vec2-based speech emotion recognition model.

    Analyzes raw audio waveforms to detect emotional tone.
    Works with the same 16kHz mono PCM format used by the STT pipeline.

    Usage:
        model = EmotionModel("wav2vec2-lg-xlsr-en", device="cuda")
        await model.load()
        result = await model.classify_emotion(audio_array)
        print(f"Detected: {result.emotion} ({result.confidence:.1%})")
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cpu",
        token: str | None = None,
    ):
        """Initialize emotion recognition model.

        Args:
            model_id: Model identifier (key from EMOTION_MODELS or HuggingFace ID)
            device: Target device (cuda/mps/cpu)
            token: Optional HuggingFace token for gated models
        """
        # Resolve model alias to full HuggingFace ID
        resolved_model = EMOTION_MODELS.get(model_id, model_id)
        super().__init__(resolved_model, device, token=token)

        self.model_type = "emotion"
        self.supports_streaming = False
        self._short_id = model_id

        # Model components
        self._classifier = None
        self._feature_extractor = None
        self._labels: list[str] = []

        # Thread pool for inference (avoid blocking event loop)
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def labels(self) -> list[str]:
        """Get the emotion labels this model can predict."""
        return self._labels

    async def load(self) -> None:
        """Load the emotion recognition model."""
        # Re-create executor if it was destroyed by unload()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)

        try:
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is not installed. "
                "Install with: uv pip install transformers"
            ) from e

        logger.info(f"Loading emotion model: {self.model_id} (device={self.device})")

        # Load feature extractor (handles audio preprocessing)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id,
            token=self.token,
        )

        # Load the classification model
        self._classifier = AutoModelForAudioClassification.from_pretrained(
            self.model_id,
            token=self.token,
            torch_dtype=self.get_dtype(),
        )

        # Move to device
        self._classifier = self._classifier.to(self.device)
        self._classifier.eval()

        # Extract labels from model config
        if hasattr(self._classifier.config, "id2label"):
            self._labels = [
                self._classifier.config.id2label[i]
                for i in range(len(self._classifier.config.id2label))
            ]
        else:
            # Fallback for models without id2label
            self._labels = [
                "angry",
                "calm",
                "disgust",
                "fearful",
                "happy",
                "neutral",
                "sad",
                "surprised",
            ]

        logger.info(f"Emotion model loaded with labels: {self._labels}")

    async def unload(self) -> None:
        """Unload model and free resources."""
        logger.info(f"Unloading emotion model: {self.model_id}")

        if self._classifier is not None:
            del self._classifier
            self._classifier = None

        if self._feature_extractor is not None:
            del self._feature_extractor
            self._feature_extractor = None

        self._labels = []

        # Shutdown thread pool - wait for pending tasks to complete
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Call parent cleanup for CUDA/MPS cache clearing
        await super().unload()

        logger.info(f"Emotion model unloaded: {self.model_id}")

    async def classify_emotion(
        self,
        audio: np.ndarray,
        sample_rate: int = EXPECTED_SAMPLE_RATE,
    ) -> EmotionResult:
        """Classify emotion from raw audio.

        Args:
            audio: Numpy array of audio samples (float32, mono).
                   Values should be normalized to [-1.0, 1.0].
            sample_rate: Sample rate of the audio (default 16kHz).
                        Will resample if different from expected.

        Returns:
            EmotionResult with detected emotion and confidence scores
        """
        if self._classifier is None or self._feature_extractor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Calculate duration
        duration = len(audio) / sample_rate

        def _sync_classify():
            """Run classification synchronously in thread pool."""
            import torch

            # Resample if needed
            processed_audio = audio
            if sample_rate != EXPECTED_SAMPLE_RATE:
                processed_audio = self._resample(audio, sample_rate, EXPECTED_SAMPLE_RATE)

            # Preprocess audio
            inputs = self._feature_extractor(
                processed_audio,
                sampling_rate=EXPECTED_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self._classifier(**inputs)
                logits = outputs.logits

            # Convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs = probs.cpu().numpy()[0]

            # Build results
            all_scores = {
                label: float(prob)
                for label, prob in zip(self._labels, probs, strict=True)
            }

            # Get top prediction
            top_idx = int(np.argmax(probs))
            top_emotion = self._labels[top_idx]
            top_confidence = float(probs[top_idx])

            return top_emotion, top_confidence, all_scores

        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        emotion, confidence, all_scores = await loop.run_in_executor(
            self._executor, _sync_classify
        )

        return EmotionResult(
            emotion=emotion,
            confidence=confidence,
            all_scores=all_scores,
            duration_seconds=duration,
        )

    async def classify_emotion_batch(
        self,
        audio_segments: list[np.ndarray],
        sample_rate: int = EXPECTED_SAMPLE_RATE,
    ) -> list[EmotionResult]:
        """Classify emotion for multiple audio segments.

        More efficient than calling classify_emotion repeatedly
        due to batched inference.

        Args:
            audio_segments: List of audio arrays
            sample_rate: Sample rate for all segments

        Returns:
            List of EmotionResult, one per segment
        """
        if self._classifier is None or self._feature_extractor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        def _sync_classify_batch():
            """Run batch classification synchronously."""
            import torch

            # Resample if needed
            processed_segments = []
            for seg in audio_segments:
                if sample_rate != EXPECTED_SAMPLE_RATE:
                    seg = self._resample(seg, sample_rate, EXPECTED_SAMPLE_RATE)
                processed_segments.append(seg)

            # Preprocess all segments
            inputs = self._feature_extractor(
                processed_segments,
                sampling_rate=EXPECTED_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Batch inference
            with torch.no_grad():
                outputs = self._classifier(**inputs)
                logits = outputs.logits

            # Convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs = probs.cpu().numpy()

            return probs

        loop = asyncio.get_running_loop()
        all_probs = await loop.run_in_executor(self._executor, _sync_classify_batch)

        # Build results
        results = []
        for _i, (audio, probs) in enumerate(
            zip(audio_segments, all_probs, strict=True)
        ):
            all_scores = {
                label: float(prob)
                for label, prob in zip(self._labels, probs, strict=True)
            }
            top_idx = int(np.argmax(probs))

            results.append(
                EmotionResult(
                    emotion=self._labels[top_idx],
                    confidence=float(probs[top_idx]),
                    all_scores=all_scores,
                    duration_seconds=len(audio) / sample_rate,
                )
            )

        return results

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate.

        Uses linear interpolation for simplicity. For production use with
        significant sample rate differences, consider using librosa or scipy.
        """
        if orig_sr == target_sr:
            return audio

        # Simple linear interpolation resampling
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update(
            {
                "short_id": self._short_id,
                "labels": self._labels,
                "num_emotions": len(self._labels),
                "expected_sample_rate": EXPECTED_SAMPLE_RATE,
            }
        )
        return info
