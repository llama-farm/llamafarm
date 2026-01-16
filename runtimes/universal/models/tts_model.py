"""
Text-to-speech model wrapper supporting multiple TTS backends.

Supports:
- Kokoro TTS (default): High-quality neural TTS with 82M parameters, ~100ms TTFA on GPU
- Streaming synthesis output (yields audio chunks as they're generated)
- Multiple pre-defined voices (American/British, Male/Female)
- Speed adjustment
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread

from .base import BaseModel

logger = logging.getLogger(__name__)

# Kokoro voice presets
# Format: {voice_id: (display_name, lang_code)}
# lang_code: 'a' = American English, 'b' = British English
KOKORO_VOICES = {
    # American English voices
    "af_heart": ("Heart (American Female)", "a"),
    "af_bella": ("Bella (American Female)", "a"),
    "af_nicole": ("Nicole (American Female)", "a"),
    "af_sarah": ("Sarah (American Female)", "a"),
    "af_sky": ("Sky (American Female)", "a"),
    "am_adam": ("Adam (American Male)", "a"),
    "am_michael": ("Michael (American Male)", "a"),
    # British English voices
    "bf_emma": ("Emma (British Female)", "b"),
    "bf_isabella": ("Isabella (British Female)", "b"),
    "bm_george": ("George (British Male)", "b"),
    "bm_lewis": ("Lewis (British Male)", "b"),
}

DEFAULT_VOICE = "af_heart"
DEFAULT_SAMPLE_RATE = 24000  # Kokoro native sample rate


@dataclass
class SynthesisResult:
    """Complete synthesis result."""

    audio: bytes  # Raw PCM audio (16-bit signed integers)
    sample_rate: int
    duration: float  # Duration in seconds
    format: str  # "pcm"


@dataclass
class VoiceInfo:
    """Information about an available voice."""

    id: str
    name: str
    language: str
    model: str


class TTSModel(BaseModel):
    """Wrapper for text-to-speech models (Kokoro)."""

    def __init__(
        self,
        model_id: str = "kokoro",
        device: str = "auto",
        voice: str = DEFAULT_VOICE,
        token: str | None = None,
    ):
        """Initialize TTS model.

        Args:
            model_id: Model identifier. Currently supports "kokoro".
            device: Device to run on ("cuda", "cpu", "mps", "auto").
                    "auto" will select the best available device.
            voice: Default voice ID to use for synthesis.
            token: HuggingFace token (not typically needed for Kokoro).
        """
        super().__init__(model_id, device, token=token)
        self.model_type = "tts"
        self.supports_streaming = True
        self.voice = voice
        self._tts_pipeline = None
        self._backend = "kokoro"  # Only Kokoro supported for now

    async def load(self) -> None:
        """Load the TTS model."""
        await self._load_kokoro()

    async def _load_kokoro(self) -> None:
        """Load Kokoro TTS model."""
        try:
            from kokoro import KPipeline
        except ImportError as e:
            raise ImportError(
                "kokoro is not installed. "
                "Install it with: uv pip install 'universal-runtime[tts]'"
            ) from e

        # Pre-load spaCy model to prevent misaki from trying to download it
        # (misaki calls spacy.cli.download which fails in uv venvs without pip)
        try:
            import spacy

            # Try to load the model - if it fails, provide helpful error
            try:
                _nlp = spacy.load("en_core_web_sm")
                del _nlp  # Don't need to keep it, just verify it loads
                logger.debug("spaCy en_core_web_sm model pre-loaded successfully")
            except OSError:
                raise ImportError(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install it with: uv pip install 'universal-runtime[tts]'"
                )
        except ImportError as e:
            raise ImportError(
                "spacy is not installed. "
                "Install it with: uv pip install 'universal-runtime[tts]'"
            ) from e

        # Determine language code from default voice
        voice_info = KOKORO_VOICES.get(self.voice)
        lang_code = voice_info[1] if voice_info else "a"

        logger.info(
            f"Loading Kokoro TTS model (device={self.device}, lang={lang_code})"
        )

        # KPipeline handles model loading
        # It will download the model on first use (~82M parameters)
        self._tts_pipeline = KPipeline(lang_code=lang_code)

        logger.info("Kokoro TTS model loaded")

    async def unload(self) -> None:
        """Unload the model and free resources."""
        logger.info(f"Unloading TTS model: {self.model_id}")

        if self._tts_pipeline is not None:
            del self._tts_pipeline
            self._tts_pipeline = None

        # Call parent cleanup for CUDA/MPS cache clearing
        await super().unload()

        logger.info(f"TTS model unloaded: {self.model_id}")

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> SynthesisResult:
        """Synthesize speech from text (non-streaming).

        Args:
            text: Text to synthesize (max 4096 characters recommended).
            voice: Voice ID to use. If None, uses the model's default voice.
            speed: Speed multiplier (0.5 to 2.0). Default is 1.0.

        Returns:
            SynthesisResult with PCM audio data, sample rate, and duration.
        """
        if self._tts_pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        voice = voice or self.voice

        def _sync_synthesize():
            """Run synthesis synchronously in thread pool."""
            import numpy as np

            audio_chunks = []

            # Kokoro yields (graphemes, phonemes, audio) tuples
            # Audio may be a PyTorch tensor, so convert to numpy if needed
            for _gs, _ps, audio in self._tts_pipeline(text, voice=voice, speed=speed):
                if audio is not None:
                    # Convert tensor to numpy if needed
                    if hasattr(audio, "cpu"):
                        audio = audio.cpu().numpy()
                    audio_chunks.append(audio)

            if not audio_chunks:
                # Return empty audio if nothing generated
                return np.array([], dtype=np.float32)

            # Concatenate all chunks
            return np.concatenate(audio_chunks)

        # Run synthesis in thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            audio_array = await loop.run_in_executor(pool, _sync_synthesize)

        # Convert float32 audio to 16-bit PCM bytes
        import numpy as np

        # Clamp to [-1, 1] range and convert to int16
        audio_clamped = np.clip(audio_array, -1.0, 1.0)
        pcm_bytes = (audio_clamped * 32767).astype(np.int16).tobytes()

        duration = len(audio_array) / DEFAULT_SAMPLE_RATE if len(audio_array) > 0 else 0

        return SynthesisResult(
            audio=pcm_bytes,
            sample_rate=DEFAULT_SAMPLE_RATE,
            duration=duration,
            format="pcm",
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as they're generated.

        This enables low-latency audio playback by yielding PCM chunks
        as soon as they're synthesized rather than waiting for completion.

        Args:
            text: Text to synthesize.
            voice: Voice ID to use. If None, uses the model's default voice.
            speed: Speed multiplier (0.5 to 2.0). Default is 1.0.

        Yields:
            bytes: Raw PCM audio chunks (16-bit signed integers, 24kHz mono).
        """
        if self._tts_pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        voice = voice or self.voice

        # Use a queue to communicate between the synthesis thread and async generator
        audio_queue: Queue[bytes | None] = Queue()

        def _sync_generator():
            """Run synthesis in a thread and put chunks in the queue."""
            import numpy as np

            try:
                for _gs, _ps, audio in self._tts_pipeline(
                    text, voice=voice, speed=speed
                ):
                    if audio is not None and len(audio) > 0:
                        # Convert tensor to numpy if needed
                        if hasattr(audio, "cpu"):
                            audio = audio.cpu().numpy()
                        # Convert to PCM bytes
                        audio_clamped = np.clip(audio, -1.0, 1.0)
                        pcm_bytes = (audio_clamped * 32767).astype(np.int16).tobytes()
                        audio_queue.put(pcm_bytes)
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}")
            finally:
                # Signal completion
                audio_queue.put(None)

        # Start synthesis in background thread
        thread = Thread(target=_sync_generator, daemon=True)
        thread.start()

        # Yield chunks as they arrive
        while True:
            try:
                # Non-blocking check with small timeout
                chunk = audio_queue.get(timeout=0.01)
                if chunk is None:
                    # Synthesis complete
                    break
                yield chunk
            except Empty:
                # No chunk ready yet, yield control to event loop
                await asyncio.sleep(0)

        # Ensure thread completes
        thread.join(timeout=1.0)

    def get_voices(self) -> list[VoiceInfo]:
        """Get list of available voices.

        Returns:
            List of VoiceInfo objects describing available voices.
        """
        voices = []
        for voice_id, (name, lang_code) in KOKORO_VOICES.items():
            language = "en-US" if lang_code == "a" else "en-GB"
            voices.append(
                VoiceInfo(
                    id=voice_id,
                    name=name,
                    language=language,
                    model=self.model_id,
                )
            )
        return voices

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "backend": self._backend,
            "device": self.device,
            "default_voice": self.voice,
            "supports_streaming": self.supports_streaming,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "voices": [v.id for v in self.get_voices()],
        }
