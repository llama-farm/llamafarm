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
from queue import Empty, Queue
from threading import Thread
from typing import TYPE_CHECKING, Literal

from .base import BaseModel

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Queue size limit to prevent unbounded memory growth during streaming
MAX_SEGMENT_QUEUE_SIZE = 100

# Allowed file extensions for temp files (security: prevent arbitrary file creation)
ALLOWED_AUDIO_EXTENSIONS = frozenset({
    ".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".aiff", ".mp4", ".opus"
})

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
# distil-large-v3-turbo is ~2x faster than distil-large-v3 with minimal quality loss
DEFAULT_MODEL = "distil-large-v3-turbo"


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
        # Persistent thread pool for transcription (avoids ~50-100ms overhead per request)
        self._executor = ThreadPoolExecutor(max_workers=2)

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
        # Re-create executor if it was destroyed by unload()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2)

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

        # Shutdown thread pool - wait for pending tasks to complete
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Call parent cleanup for CUDA/MPS cache clearing
        await super().unload()

        logger.info(f"Speech model unloaded: {self.model_id}")

    async def transcribe(
        self,
        audio_path: str | Path | None = None,
        audio_array: "np.ndarray | None" = None,
        language: str | None = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        vad_filter: bool = True,
        beam_size: int = 1,
        best_of: int = 1,
        temperature: float | list[float] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio from file or numpy array.

        Args:
            audio_path: Path to audio file (supports mp3, wav, m4a, webm, etc.)
            audio_array: Audio as numpy array (float32, normalized to [-1.0, 1.0])
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

        if audio_path is None and audio_array is None:
            raise ValueError("Either audio_path or audio_array must be provided")

        if audio_path is not None and audio_array is not None:
            raise ValueError("Only one of audio_path or audio_array can be provided")

        if temperature is None:
            temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Use audio_array if provided, otherwise use audio_path
        audio_input = audio_array if audio_array is not None else str(audio_path)

        def _sync_transcribe():
            """Run transcription synchronously in thread pool."""
            segments_generator, info = self._whisper_model.transcribe(
                audio_input,
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

        # Run transcription in persistent thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        segments, full_text_parts, info = await loop.run_in_executor(
            self._executor, _sync_transcribe
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
        beam_size: int = 1,
    ) -> AsyncGenerator[TranscriptionSegment, None]:
        """Transcribe audio and yield segments as they're processed.

        This enables true streaming - segments are yielded immediately as
        they're transcribed, not collected and returned at the end.

        Uses a Thread + Queue pattern for low-latency segment delivery,
        allowing downstream processing (LLM, TTS) to start on the first
        segment while transcription continues.

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

        # Queue for streaming segments from thread to async generator
        # Can contain: TranscriptionSegment, None (completion), or Exception (error)
        # Use maxsize to prevent unbounded memory growth
        segment_queue: Queue[TranscriptionSegment | None | Exception] = Queue(
            maxsize=MAX_SEGMENT_QUEUE_SIZE
        )

        def _sync_transcribe_stream():
            """Run transcription in thread and put segments in queue immediately."""
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

                    # Put segment in queue immediately for streaming
                    segment_queue.put(
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
            except Exception as e:
                logger.error(f"STT streaming error: {e}")
                # Signal error to consumer
                segment_queue.put(e)
            finally:
                # Signal completion
                segment_queue.put(None)

        # Start transcription in background thread
        thread = Thread(target=_sync_transcribe_stream, daemon=True)
        thread.start()

        # Yield segments as they arrive (true streaming)
        while True:
            try:
                # Very short timeout for low latency
                segment = segment_queue.get(timeout=0.005)
                if segment is None:
                    # Transcription complete
                    break
                if isinstance(segment, Exception):
                    # Propagate error from transcription thread
                    raise segment
                yield segment
            except Empty:
                # No segment ready yet, yield control to event loop
                await asyncio.sleep(0)

        # Ensure thread completes
        thread.join(timeout=1.0)

    async def transcribe_audio(
        self,
        audio: "np.ndarray",
        language: str | None = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        vad_filter: bool = True,
        beam_size: int = 1,
        best_of: int = 1,
        temperature: float | list[float] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio from a numpy array directly (no file I/O).

        This is the most efficient transcription method for raw PCM data,
        as it avoids writing to disk and file format parsing.

        Args:
            audio: Numpy array of audio samples (float32, mono).
                   Values should be normalized to [-1.0, 1.0].
                   Can be any sample rate (Whisper resamples to 16kHz internally).
            language: Language code (e.g., "en", "es", "fr"). If None, auto-detected.
            task: "transcribe" for same-language output, "translate" for English output
            word_timestamps: Whether to include word-level timestamps
            initial_prompt: Optional text to condition the model
            vad_filter: Use voice activity detection to filter out silence
            beam_size: Beam size for decoding (higher = more accurate, slower)
            best_of: Number of candidates to consider (higher = more accurate, slower)
            temperature: Temperature(s) for sampling. List enables fallback on failure.

        Returns:
            TranscriptionResult with full text, segments, and metadata
        """
        if self._whisper_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if temperature is None:
            temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        def _sync_transcribe():
            """Run transcription synchronously in thread pool."""
            # faster-whisper accepts numpy array directly
            segments_generator, info = self._whisper_model.transcribe(
                audio,
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

        # Run transcription in persistent thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        segments, full_text_parts, info = await loop.run_in_executor(
            self._executor, _sync_transcribe
        )

        return TranscriptionResult(
            text="".join(full_text_parts).strip(),
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )

    async def transcribe_audio_stream(
        self,
        audio: "np.ndarray",
        language: str | None = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        vad_filter: bool = True,
        beam_size: int = 1,
    ) -> AsyncGenerator[TranscriptionSegment, None]:
        """Stream transcription segments from numpy array.

        Like transcribe_audio but yields segments immediately as they're
        transcribed, enabling parallel processing downstream.

        Args:
            audio: Numpy array of audio samples (float32, mono, normalized [-1, 1]).
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

        # Queue for streaming segments from thread to async generator
        # Can contain: TranscriptionSegment, None (completion), or Exception (error)
        # Use maxsize to prevent unbounded memory growth
        segment_queue: Queue[TranscriptionSegment | None | Exception] = Queue(
            maxsize=MAX_SEGMENT_QUEUE_SIZE
        )

        def _sync_transcribe_stream():
            """Run transcription in thread and put segments in queue immediately."""
            try:
                segments_generator, _info = self._whisper_model.transcribe(
                    audio,
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

                    # Put segment in queue immediately for streaming
                    segment_queue.put(
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
            except Exception as e:
                logger.error(f"STT streaming error: {e}")
                # Signal error to consumer
                segment_queue.put(e)
            finally:
                # Signal completion
                segment_queue.put(None)

        # Start transcription in background thread
        thread = Thread(target=_sync_transcribe_stream, daemon=True)
        thread.start()

        # Yield segments as they arrive (true streaming)
        while True:
            try:
                # Very short timeout for low latency
                segment = segment_queue.get(timeout=0.005)
                if segment is None:
                    # Transcription complete
                    break
                if isinstance(segment, Exception):
                    # Propagate error from transcription thread
                    raise segment
                yield segment
            except Empty:
                # No segment ready yet, yield control to event loop
                await asyncio.sleep(0)

        # Ensure thread completes
        thread.join(timeout=1.0)

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        file_extension: str = ".wav",
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio from bytes.

        Args:
            audio_bytes: Raw audio file bytes
            file_extension: File extension hint for format detection (must be in ALLOWED_AUDIO_EXTENSIONS)
            **kwargs: Additional arguments passed to transcribe()

        Returns:
            TranscriptionResult

        Raises:
            ValueError: If file_extension is not in ALLOWED_AUDIO_EXTENSIONS.
        """
        # Validate file extension to prevent arbitrary file creation
        ext_lower = file_extension.lower()
        if ext_lower not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. "
                f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            )

        # Write to temporary file (faster-whisper requires file path)
        # Assign tmp_path before write to ensure cleanup even if write fails
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=ext_lower, delete=False
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
            file_extension: File extension hint (must be in ALLOWED_AUDIO_EXTENSIONS)
            **kwargs: Additional arguments passed to transcribe_stream()

        Yields:
            TranscriptionSegment objects

        Raises:
            ValueError: If file_extension is not in ALLOWED_AUDIO_EXTENSIONS.
        """
        # Validate file extension to prevent arbitrary file creation
        ext_lower = file_extension.lower()
        if ext_lower not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. "
                f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            )

        # Write to temporary file
        # Assign tmp_path before write to ensure cleanup even if write fails
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=ext_lower, delete=False
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
