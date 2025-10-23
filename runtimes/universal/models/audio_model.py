"""
Audio model wrapper for speech-to-text and audio processing.
"""

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)
import torch
from typing import Optional, Dict, Any, Union
import io
import base64
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class AudioModel(BaseModel):
    """Wrapper for HuggingFace audio models (Whisper, Wav2Vec2, etc.)."""

    def __init__(self, model_id: str, device: str, task: str = "transcribe"):
        """
        Initialize audio model.

        Args:
            model_id: HuggingFace model ID
            device: Target device (cuda/mps/cpu)
            task: Model task - "transcribe" or "translate"
        """
        super().__init__(model_id, device)
        self.task = task
        self.model_type = f"audio_{task}"
        self.supports_streaming = False  # TODO: Implement streaming for long audio
        self.pipe = None

    async def load(self):
        """Load the audio model."""
        logger.info(f"Loading audio model ({self.task}): {self.model_id}")

        dtype = self.get_dtype()

        # Check if it's a Whisper model
        if "whisper" in self.model_id.lower():
            # Use pipeline for Whisper (simplest approach)
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                dtype=dtype,
                device=self.device,
            )
            logger.info(f"Loaded Whisper model via pipeline on {self.device}")
        else:
            # Generic approach for other speech models
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                dtype=dtype,
                trust_remote_code=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Audio model loaded on {self.device}")

    def _decode_audio(self, audio_input: Union[str, bytes]) -> bytes:
        """Decode audio from base64 string or bytes."""
        if isinstance(audio_input, bytes):
            return audio_input

        if isinstance(audio_input, str):
            # Base64 string
            if audio_input.startswith("data:audio"):
                audio_input = audio_input.split(",", 1)[1]
            return base64.b64decode(audio_input)

        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    async def transcribe(
        self,
        audio: Union[str, bytes],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data (base64 string or bytes)
            language: Optional language code (e.g., "en", "es")
            prompt: Optional prompt to guide transcription
            temperature: Sampling temperature (0 = greedy)
            return_timestamps: Whether to return word-level timestamps

        Returns:
            Transcription result with text and optional timestamps
        """
        # Decode audio
        audio_bytes = self._decode_audio(audio)

        # Prepare generation kwargs
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
        if prompt:
            generate_kwargs["prompt_ids"] = self.pipe.tokenizer.encode(prompt)
        if return_timestamps:
            generate_kwargs["return_timestamps"] = False

        # Transcribe using pipeline
        if self.pipe:
            result = self.pipe(
                audio_bytes,
                generate_kwargs=generate_kwargs,
                return_timestamps=return_timestamps,
            )

            # Format output
            output = {"text": result["text"]}
            if return_timestamps and "chunks" in result:
                output["words"] = [
                    {
                        "word": chunk["text"],
                        "start": chunk["timestamp"][0],
                        "end": chunk["timestamp"][1],
                    }
                    for chunk in result["chunks"]
                ]

            return output
        else:
            # Manual transcription (for non-pipeline models)
            # This would require implementing audio loading and processing
            raise NotImplementedError(
                "Manual transcription not yet implemented. Use Whisper models."
            )

    async def translate(
        self,
        audio: Union[str, bytes],
        target_language: str = "en",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Translate audio to target language.

        Args:
            audio: Audio data (base64 string or bytes)
            target_language: Target language code (default: "en")
            temperature: Sampling temperature

        Returns:
            Translation result
        """
        if "whisper" not in self.model_id.lower():
            raise ValueError("Translation requires Whisper model")

        # Decode audio
        audio_bytes = self._decode_audio(audio)

        # Translate
        result = self.pipe(
            audio_bytes,
            generate_kwargs={"task": "translate", "language": target_language},
        )

        return {"text": result["text"]}

    async def generate(self, *args, **kwargs):
        """Alias for transcribe()."""
        return await self.transcribe(*args, **kwargs)
