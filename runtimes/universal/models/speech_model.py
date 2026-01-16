"""
Speech-to-text model wrapper using faster-whisper.

Supports:
- Multiple Whisper model sizes (tiny, base, small, medium, large-v3, distil-large-v3)
- GPU acceleration (CUDA) and CPU inference
- Streaming transcription output (yields segments as they're transcribed)
- Word-level timestamps
- Language detection and specification
- Multiple output formats (text, json, srt, vtt, verbose_json)
"""

import asyncio
import logging
import tempfile
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .base import BaseModel

logger = logging.getLogger(__name__)

# Model size to HuggingFace model ID mapping
MODEL_SIZES = {
    "tiny": "tiny",
    "tiny.en": "tiny.en",
    "base": "base",
    "base.en": "base.en",
    "small": "small",
    "small.en": "small.en",
    "medium": "medium",
    "medium.en": "medium.en",
    "large": "large-v3",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "distil-large-v3": "distil-large-v3",
    "distil-medium.en": "distil-medium.en",
    "distil-small.en": "distil-small.en",
}

# Default model for best speed/quality tradeoff
DEFAULT_MODEL = "distil-large-v3"


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio."""

    id: int
    start: float
    end: float
    text: str
    words: list[dict] | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    text: str
    segments: list[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float


class SpeechModel(BaseModel):
    """Wrapper for faster-whisper speech-to-text models."""

    def __init__(
        self,
        model_id: str,
        device: str,
        compute_type: str | None = None,
        token: str | None = None,
    ):
        """Initialize speech model.

        Args:
            model_id: Model size/name (e.g., "large-v3", "distil-large-v3", "medium")
            device: Device to run on ("cuda", "cpu", "auto")
            compute_type: Compute type for inference. Auto-selects if None:
                         - cuda: "float16" (or "int8_float16" for lower VRAM)
                         - cpu: "int8" (fastest) or "float32" (most accurate)
            token: HuggingFace token (not typically needed for Whisper)
        """
        super().__init__(model_id, device, token=token)
        self.model_type = "speech"
        self.supports_streaming = True
        self._whisper_model = None

        # Resolve model size alias
        self._resolved_model = MODEL_SIZES.get(model_id, model_id)

        # Auto-select compute type based on device
        if compute_type is None:
            if device == "cuda":
                self.compute_type = "float16"
            elif device == "mps":
                # MPS not directly supported by faster-whisper, fall back to CPU
                self.compute_type = "int8"
                self.device = "cpu"
                logger.warning(
                    "MPS not supported by faster-whisper, falling back to CPU with int8"
                )
            else:
                self.compute_type = "int8"
        else:
            self.compute_type = compute_type

    async def load(self) -> None:
        """Load the Whisper model."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install it with: uv pip install 'universal-runtime[speech]'"
            ) from e

        logger.info(
            f"Loading Whisper model: {self._resolved_model} "
            f"(device={self.device}, compute_type={self.compute_type})"
        )

        # Load model - faster-whisper handles downloading automatically
        self._whisper_model = WhisperModel(
            self._resolved_model,
            device=self.device if self.device != "mps" else "cpu",
            compute_type=self.compute_type,
        )

        logger.info(f"Whisper model loaded: {self._resolved_model}")

    async def unload(self) -> None:
        """Unload the model and free resources."""
        logger.info(f"Unloading speech model: {self.model_id}")

        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None

        # Call parent cleanup for CUDA/MPS cache clearing
        await super().unload()

        logger.info(f"Speech model unloaded: {self.model_id}")

    async def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        vad_filter: bool = True,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float | list[float] | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file (supports mp3, wav, m4a, webm, etc.)
            language: Language code (e.g., "en", "es", "fr"). If None, auto-detected.
            task: "transcribe" for same-language output, "translate" for English output
            word_timestamps: Whether to include word-level timestamps
            initial_prompt: Optional text to condition the model
            vad_filter: Use voice activity detection to filter out silence
            beam_size: Beam size for decoding (higher = more accurate, slower)
            best_of: Number of candidates to consider (higher = more accurate, slower)
            temperature: Temperature(s) for sampling. List enables fallback on failure.
                        Defaults to [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if None.

        Returns:
            TranscriptionResult with full text, segments, and metadata
        """
        if self._whisper_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if temperature is None:
            temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        def _sync_transcribe():
            """Run transcription synchronously in thread pool."""
            segments_generator, info = self._whisper_model.transcribe(
                str(audio_path),
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                initial_prompt=initial_prompt,
                vad_filter=vad_filter,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
            )

            # Collect all segments
            segments = []
            full_text_parts = []

            for segment in segments_generator:
                words = None
                if word_timestamps and segment.words:
                    words = [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in segment.words
                    ]

                segments.append(
                    TranscriptionSegment(
                        id=segment.id,
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                        words=words,
                        avg_logprob=segment.avg_logprob,
                        no_speech_prob=segment.no_speech_prob,
                    )
                )
                full_text_parts.append(segment.text)

            return segments, full_text_parts, info

        # Run transcription in thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            segments, full_text_parts, info = await loop.run_in_executor(
                pool, _sync_transcribe
            )

        return TranscriptionResult(
            text="".join(full_text_parts).strip(),
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )

    async def transcribe_stream(
        self,
        audio_path: str | Path,
        language: str | None = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        vad_filter: bool = True,
        beam_size: int = 5,
    ) -> AsyncGenerator[TranscriptionSegment, None]:
        """Transcribe audio and yield segments as they're processed.

        This is useful for streaming responses - segments are yielded
        as soon as they're transcribed rather than waiting for completion.

        Transcription runs in a background thread while segments are yielded
        via an asyncio.Queue, enabling true streaming behavior.

        Args:
            audio_path: Path to audio file
            language: Language code (if None, auto-detected)
            task: "transcribe" or "translate"
            word_timestamps: Include word-level timestamps
            initial_prompt: Optional conditioning text
            vad_filter: Use voice activity detection
            beam_size: Beam size for decoding

        Yields:
            TranscriptionSegment objects as they're transcribed
        """
        if self._whisper_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Queue for passing segments from thread to async generator
        # None signals completion, Exception signals error
        queue: asyncio.Queue[TranscriptionSegment | None | Exception] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _sync_transcribe_stream():
            """Run transcription synchronously and push segments to queue."""
            try:
                segments_generator, _info = self._whisper_model.transcribe(
                    str(audio_path),
                    language=language,
                    task=task,
                    word_timestamps=word_timestamps,
                    initial_prompt=initial_prompt,
                    vad_filter=vad_filter,
                    beam_size=beam_size,
                )

                for segment in segments_generator:
                    words = None
                    if word_timestamps and segment.words:
                        words = [
                            {
                                "word": w.word,
                                "start": w.start,
                                "end": w.end,
                                "probability": w.probability,
                            }
                            for w in segment.words
                        ]

                    transcription_segment = TranscriptionSegment(
                        id=segment.id,
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                        words=words,
                        avg_logprob=segment.avg_logprob,
                        no_speech_prob=segment.no_speech_prob,
                    )
                    # Thread-safe put to async queue
                    loop.call_soon_threadsafe(queue.put_nowait, transcription_segment)

                # Signal completion
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as e:
                # Signal error
                loop.call_soon_threadsafe(queue.put_nowait, e)

        # Start transcription in background thread
        with ThreadPoolExecutor() as pool:
            future = pool.submit(_sync_transcribe_stream)

            # Yield segments as they arrive
            while True:
                item = await queue.get()
                if item is None:
                    # Completion signal
                    break
                if isinstance(item, Exception):
                    # Re-raise exception from thread
                    raise item
                yield item

            # Ensure thread completes (propagate any uncaught exceptions)
            future.result()

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        file_extension: str = ".wav",
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio from bytes.

        Args:
            audio_bytes: Raw audio file bytes
            file_extension: File extension hint for format detection
            **kwargs: Additional arguments passed to transcribe()

        Returns:
            TranscriptionResult
        """
        # Write to temporary file (faster-whisper requires file path)
        # Assign tmp_path before write to ensure cleanup even if write fails
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=file_extension, delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            return await self.transcribe(tmp_path, **kwargs)
        finally:
            # Clean up temp file
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    async def transcribe_bytes_stream(
        self,
        audio_bytes: bytes,
        file_extension: str = ".wav",
        **kwargs,
    ) -> AsyncGenerator[TranscriptionSegment, None]:
        """Transcribe audio bytes and yield segments as they're processed.

        Args:
            audio_bytes: Raw audio file bytes
            file_extension: File extension hint
            **kwargs: Additional arguments passed to transcribe_stream()

        Yields:
            TranscriptionSegment objects
        """
        # Write to temporary file
        # Assign tmp_path before write to ensure cleanup even if write fails
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=file_extension, delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            async for segment in self.transcribe_stream(tmp_path, **kwargs):
                yield segment
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_id": self.model_id,
            "resolved_model": self._resolved_model,
            "model_type": self.model_type,
            "device": self.device,
            "compute_type": self.compute_type,
            "supports_streaming": self.supports_streaming,
        }
