"""
Text-to-speech model wrapper supporting multiple TTS backends.

Supports:
- Kokoro TTS (default): High-quality neural TTS with 82M parameters, ~100ms TTFA on GPU
- Chatterbox Turbo: Fast voice cloning TTS with 350M parameters, sub-200ms latency
- Pocket TTS: Lightweight CPU TTS from Kyutai with 100M parameters, ~6x realtime
- Streaming synthesis output (yields audio chunks as they're generated)
- Multiple pre-defined voices (Kokoro, Pocket) or custom voice profiles (Chatterbox)
- Speed adjustment
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from typing import Any

from .base import BaseModel

logger = logging.getLogger(__name__)


def _patch_librosa_for_float32() -> None:
    """Monkey-patch librosa.load to return float32 instead of float64.

    This is needed because Chatterbox uses librosa internally and the float64
    output causes dtype mismatches with PyTorch models (especially on MPS).
    Must be called before any Chatterbox code loads audio.
    """
    try:
        import librosa
        import numpy as np

        if hasattr(librosa, "_original_load"):
            return  # Already patched

        librosa._original_load = librosa.load

        def _load_float32(*args, **kwargs):
            y, sr = librosa._original_load(*args, **kwargs)
            return y.astype(np.float32), sr

        librosa.load = _load_float32
        logger.debug("Patched librosa.load to return float32")
    except ImportError:
        pass  # librosa not installed, patch not needed

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
DEFAULT_SAMPLE_RATE = 24000  # Both Kokoro and Chatterbox use 24kHz

# Chatterbox built-in voice presets (bundled reference audio)
# Format: {voice_id: (display_name, filename)}
# These are pre-packaged voice samples for zero-config voice cloning
CHATTERBOX_BUILTIN_VOICES = {
    # LibriSpeech samples - clear audiobook narration
    "cb_male_calm": ("Male Narrator (Calm)", "male_calm.wav"),
    "cb_female_warm": ("Female Narrator (Warm)", "female_warm.wav"),
    "cb_male_energetic": ("Male Speaker (Energetic)", "male_energetic.wav"),
    "cb_female_professional": ("Female Speaker (Professional)", "female_professional.wav"),
}

# Pocket TTS built-in voice presets
# Format: {voice_id: display_name}
# 100M parameter lightweight CPU TTS from Kyutai
POCKET_TTS_VOICES = {
    "alba": "Alba",
    "marius": "Marius",
    "javert": "Javert",
    "jean": "Jean",
    "fantine": "Fantine",
    "cosette": "Cosette",
    "eponine": "Eponine",
    "azelma": "Azelma",
}

DEFAULT_POCKET_TTS_VOICE = "alba"


def _get_builtin_voices_dir() -> str:
    """Get the path to the built-in voices directory."""
    import os
    return os.path.join(os.path.dirname(__file__), "..", "voices")


def download_builtin_voices(force: bool = False) -> dict[str, bool]:
    """Download built-in voice samples for Chatterbox TTS.

    Downloads high-quality voice samples from LibriSpeech for voice cloning.
    Requires the `soundfile` package to be installed.

    Args:
        force: If True, re-download even if files exist.

    Returns:
        Dict mapping voice IDs to success status.

    Example:
        >>> from models.tts_model import download_builtin_voices
        >>> results = download_builtin_voices()
        >>> print(results)
        {'cb_male_calm': True, 'cb_female_warm': True, ...}
    """
    import importlib.util
    import os

    voices_dir = _get_builtin_voices_dir()
    download_script = os.path.join(voices_dir, "download_voices.py")

    if not os.path.exists(download_script):
        raise FileNotFoundError(
            f"Download script not found at {download_script}. "
            "Please ensure the voices directory is intact."
        )

    # Use importlib.util for safer module loading without sys.path manipulation
    # This prevents potential code injection from malicious modules on the path
    spec = importlib.util.spec_from_file_location("download_voices", download_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {download_script}")

    download_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(download_module)

    return download_module.download_all_voices(force=force)


def check_builtin_voices() -> dict[str, bool]:
    """Check which built-in voices are available.

    Returns:
        Dict mapping voice IDs to availability (True if file exists).
    """
    import os

    voices_dir = _get_builtin_voices_dir()
    results = {}

    for voice_id, (_, filename) in CHATTERBOX_BUILTIN_VOICES.items():
        voice_path = os.path.join(voices_dir, filename)
        results[voice_id] = os.path.exists(voice_path)

    return results


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


@dataclass
class VoiceProfile:
    """Voice profile for Chatterbox voice cloning."""

    name: str
    audio_path: str
    description: str = ""


@dataclass
class ChatterboxConfig:
    """Chatterbox Turbo synthesis parameters.

    Note: exaggeration and cfg_weight are NOT supported by Turbo and will be ignored.
    Use temperature and top_k/top_p for controlling output variation.
    """

    temperature: float = 0.8  # Controls randomness (higher = more varied)
    top_k: int = 1000  # Top-k sampling
    top_p: float = 0.95  # Nucleus sampling threshold
    repetition_penalty: float = 1.2  # Penalty for repeating tokens


class TTSBackend(ABC):
    """Abstract base class for TTS backends."""

    def __init__(self, default_voice: str):
        self.default_voice = default_voice
        self._executor = ThreadPoolExecutor(max_workers=2)

    @abstractmethod
    async def load(self, device: str) -> None:
        """Load the TTS model."""
        pass

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model and free resources."""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize speech (non-streaming)."""
        pass

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks."""
        pass

    @abstractmethod
    def get_voices(self) -> list[VoiceInfo]:
        """List available voices."""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this backend supports native streaming."""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        pass

    def shutdown(self) -> None:
        """Shutdown the thread pool.

        Uses wait=True to ensure all pending tasks complete before shutdown,
        preventing task cancellation and potential resource leaks.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


class KokoroBackend(TTSBackend):
    """Kokoro TTS backend with native streaming support."""

    def __init__(self, default_voice: str = DEFAULT_VOICE):
        super().__init__(default_voice)
        self._pipeline = None

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def sample_rate(self) -> int:
        return DEFAULT_SAMPLE_RATE

    async def load(self, device: str) -> None:
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

            try:
                _nlp = spacy.load("en_core_web_sm")
                del _nlp
                logger.debug("spaCy en_core_web_sm model pre-loaded successfully")
            except OSError as e:
                raise ImportError(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install it with: uv pip install 'universal-runtime[tts]'"
                ) from e
        except ImportError as e:
            raise ImportError(
                "spacy is not installed. "
                "Install it with: uv pip install 'universal-runtime[tts]'"
            ) from e

        # Determine language code from default voice
        voice_info = KOKORO_VOICES.get(self.default_voice)
        lang_code = voice_info[1] if voice_info else "a"

        logger.info(f"Loading Kokoro TTS model (device={device}, lang={lang_code})")
        self._pipeline = KPipeline(lang_code=lang_code)
        logger.info("Kokoro TTS model loaded")

    async def unload(self) -> None:
        """Unload Kokoro model."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        self.shutdown()

    async def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize speech from text."""
        if self._pipeline is None:
            raise RuntimeError("Kokoro model not loaded. Call load() first.")

        def _sync_synthesize():
            import numpy as np

            audio_chunks = []
            for _gs, _ps, audio in self._pipeline(text, voice=voice, speed=speed):
                if audio is not None:
                    if hasattr(audio, "cpu"):
                        audio = audio.cpu().numpy()
                    audio_chunks.append(audio)

            if not audio_chunks:
                return np.array([], dtype=np.float32)
            return np.concatenate(audio_chunks)

        loop = asyncio.get_running_loop()
        audio_array = await loop.run_in_executor(self._executor, _sync_synthesize)

        import numpy as np

        audio_clamped = np.clip(audio_array, -1.0, 1.0)
        pcm_bytes = (audio_clamped * 32767).astype(np.int16).tobytes()
        duration = len(audio_array) / self.sample_rate if len(audio_array) > 0 else 0

        return SynthesisResult(
            audio=pcm_bytes,
            sample_rate=self.sample_rate,
            duration=duration,
            format="pcm",
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as they're generated."""
        if self._pipeline is None:
            raise RuntimeError("Kokoro model not loaded. Call load() first.")

        audio_queue: Queue[bytes | None] = Queue()

        def _sync_generator():
            import numpy as np

            try:
                for _gs, _ps, audio in self._pipeline(text, voice=voice, speed=speed):
                    if audio is not None and len(audio) > 0:
                        if hasattr(audio, "cpu"):
                            audio = audio.cpu().numpy()
                        audio_clamped = np.clip(audio, -1.0, 1.0)
                        pcm_bytes = (audio_clamped * 32767).astype(np.int16).tobytes()
                        audio_queue.put(pcm_bytes)
            except Exception as e:
                logger.error(f"Kokoro TTS synthesis error: {e}")
            finally:
                audio_queue.put(None)

        thread = Thread(target=_sync_generator, daemon=True)
        thread.start()

        while True:
            try:
                chunk = audio_queue.get(timeout=0.005)
                if chunk is None:
                    break
                yield chunk
            except Empty:
                await asyncio.sleep(0)

        thread.join(timeout=1.0)

    def get_voices(self) -> list[VoiceInfo]:
        """Get list of available Kokoro voices."""
        return [
            VoiceInfo(
                id=voice_id,
                name=name,
                language="en-US" if lang_code == "a" else "en-GB",
                model="kokoro",
            )
            for voice_id, (name, lang_code) in KOKORO_VOICES.items()
        ]


def _is_mlx_available() -> bool:
    """Check if MLX audio is available (macOS with Apple Silicon)."""
    import platform
    import sys

    if sys.platform != "darwin":
        return False

    # Check for Apple Silicon
    if platform.machine() not in ("arm64", "aarch64"):
        return False

    try:
        import mlx_audio.tts  # noqa: F401

        return True
    except ImportError:
        return False


class KokoroMLXBackend(TTSBackend):
    """Kokoro TTS backend using MLX for Apple Silicon.

    This backend uses the mlx-audio library which provides native MLX
    implementations optimized for Apple Silicon's unified memory and
    Neural Engine.

    Performance benefits over PyTorch/MPS:
    - Native MLX operations avoid CPU<->GPU memory copies
    - Better utilization of Apple Silicon's unified memory
    - Optimized for M-series chip architecture
    """

    def __init__(self, default_voice: str = DEFAULT_VOICE):
        super().__init__(default_voice)
        self._model = None

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def sample_rate(self) -> int:
        return DEFAULT_SAMPLE_RATE

    async def load(self, device: str) -> None:
        """Load Kokoro TTS model using MLX."""
        del device  # MLX handles device placement automatically

        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as e:
            raise ImportError(
                "mlx-audio is not installed. "
                "Install it with: uv pip install 'universal-runtime[tts-mlx]'"
            ) from e

        logger.info("Loading Kokoro TTS model with MLX backend")

        def _load_model():
            return load_model("mlx-community/Kokoro-82M-bf16")

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(self._executor, _load_model)

        # Warmup to trigger JIT compilation
        logger.info("Warming up Kokoro MLX model...")

        def _warmup():
            for _ in self._model.generate(
                text="Hello.",
                voice=self.default_voice,
                speed=1.0,
            ):
                pass

        await loop.run_in_executor(self._executor, _warmup)
        logger.info("Kokoro TTS model loaded with MLX backend")

    async def unload(self) -> None:
        """Unload Kokoro MLX model."""
        if self._model is not None:
            del self._model
            self._model = None
        self.shutdown()

    async def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize speech from text."""
        del kwargs  # Unused
        if self._model is None:
            raise RuntimeError("Kokoro MLX model not loaded. Call load() first.")

        def _sync_synthesize():
            import numpy as np

            audio_chunks = []
            for result in self._model.generate(text=text, voice=voice, speed=speed):
                if result.audio is not None:
                    # Convert MLX array to numpy
                    audio = np.array(result.audio)
                    audio_chunks.append(audio)

            if not audio_chunks:
                return np.array([], dtype=np.float32)
            return np.concatenate(audio_chunks)

        loop = asyncio.get_running_loop()
        audio_array = await loop.run_in_executor(self._executor, _sync_synthesize)

        import numpy as np

        audio_clamped = np.clip(audio_array, -1.0, 1.0)
        pcm_bytes = (audio_clamped * 32767).astype(np.int16).tobytes()
        duration = len(audio_array) / self.sample_rate if len(audio_array) > 0 else 0

        return SynthesisResult(
            audio=pcm_bytes,
            sample_rate=self.sample_rate,
            duration=duration,
            format="pcm",
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as they're generated."""
        del kwargs  # Unused
        if self._model is None:
            raise RuntimeError("Kokoro MLX model not loaded. Call load() first.")

        audio_queue: Queue[bytes | None] = Queue()

        def _sync_generator():
            import numpy as np

            try:
                for result in self._model.generate(text=text, voice=voice, speed=speed):
                    if result.audio is not None and len(result.audio) > 0:
                        # Convert MLX array to numpy then to PCM
                        audio = np.array(result.audio)
                        audio_clamped = np.clip(audio, -1.0, 1.0)
                        pcm_bytes = (audio_clamped * 32767).astype(np.int16).tobytes()
                        audio_queue.put(pcm_bytes)
            except Exception as e:
                logger.error(f"Kokoro MLX TTS synthesis error: {e}")
            finally:
                audio_queue.put(None)

        thread = Thread(target=_sync_generator, daemon=True)
        thread.start()

        while True:
            try:
                chunk = audio_queue.get(timeout=0.005)
                if chunk is None:
                    break
                yield chunk
            except Empty:
                await asyncio.sleep(0)

        thread.join(timeout=1.0)

    def get_voices(self) -> list[VoiceInfo]:
        """Get list of available Kokoro voices."""
        return [
            VoiceInfo(
                id=voice_id,
                name=name,
                language="en-US" if lang_code == "a" else "en-GB",
                model="kokoro-mlx",
            )
            for voice_id, (name, lang_code) in KOKORO_VOICES.items()
        ]


class ChatterboxTurboBackend(TTSBackend):
    """Chatterbox Turbo TTS backend with voice cloning.

    Note: Chatterbox Turbo doesn't support native streaming, so synthesize_stream()
    generates the full audio first, then yields it in chunks for API compatibility.
    """

    # Chunk size for simulated streaming (~0.5s at 24kHz)
    STREAM_CHUNK_SAMPLES = 12000

    def __init__(
        self,
        default_voice: str,
        voice_profiles: dict[str, VoiceProfile] | None = None,
        config: ChatterboxConfig | None = None,
    ):
        super().__init__(default_voice)
        self._voice_profiles = voice_profiles or {}
        self._config = config or ChatterboxConfig()
        self._model = None
        self._sample_rate = DEFAULT_SAMPLE_RATE

    @property
    def supports_streaming(self) -> bool:
        # Returns False to indicate no native streaming; we simulate it
        return False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _resolve_voice_path(self, voice: str) -> str:
        """Resolve voice name to audio file path.

        Priority:
        1. Built-in voices (cb_male_calm, etc.)
        2. User-defined voice profiles
        3. Direct file path (with path traversal protection)

        Raises:
            ValueError: If path traversal is detected in direct file paths.
        """
        import os

        # Check built-in voices first
        if voice in CHATTERBOX_BUILTIN_VOICES:
            _, filename = CHATTERBOX_BUILTIN_VOICES[voice]
            builtin_path = os.path.join(_get_builtin_voices_dir(), filename)
            if os.path.exists(builtin_path):
                return builtin_path
            else:
                logger.warning(
                    f"Built-in voice '{voice}' not found at {builtin_path}. "
                    "Voice samples may not be installed."
                )

        # Check user-defined voice profiles
        if voice in self._voice_profiles:
            profile_path = self._voice_profiles[voice].audio_path
            # Validate user-defined profile paths against traversal
            resolved = os.path.realpath(profile_path)
            if ".." in profile_path or not os.path.isabs(resolved):
                raise ValueError(
                    f"Invalid voice profile path: {profile_path}. "
                    "Path traversal is not allowed."
                )
            return profile_path

        # Direct file path - validate against path traversal attacks
        # Resolve the path and check for traversal attempts
        resolved_path = os.path.realpath(voice)

        # Check for path traversal indicators
        if ".." in voice:
            raise ValueError(
                f"Path traversal detected in voice path: {voice}. "
                "Use absolute paths or built-in voice names."
            )

        # Ensure the path exists and is a file (not a directory)
        if os.path.exists(resolved_path) and not os.path.isfile(resolved_path):
            raise ValueError(
                f"Voice path must be a file, not a directory: {voice}"
            )

        return voice

    async def load(self, device: str) -> None:
        """Load Chatterbox Turbo model."""
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
        except ImportError as e:
            raise ImportError(
                "chatterbox-tts is not installed. "
                "Install it with: uv pip install 'universal-runtime[tts-chatterbox]'"
            ) from e

        # Force CPU on MPS - Chatterbox uses librosa which requires float64,
        # but MPS doesn't support float64. CPU is fast enough for TTS.
        if device == "mps":
            logger.warning(
                "Chatterbox Turbo doesn't support MPS (requires float64). "
                "Falling back to CPU."
            )
            device = "cpu"

        logger.info(f"Loading Chatterbox Turbo TTS model (device={device})")

        # Patch librosa before loading model (must happen in main thread context)
        _patch_librosa_for_float32()

        def _load_model():
            import torch

            # Force float32 as default to avoid dtype mismatches in Chatterbox
            torch.set_default_dtype(torch.float32)

            return ChatterboxTurboTTS.from_pretrained(device=device)

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(self._executor, _load_model)

        # Get sample rate from model
        if hasattr(self._model, "sr"):
            self._sample_rate = self._model.sr

        logger.info(
            f"Chatterbox Turbo TTS model loaded (sample_rate={self._sample_rate})"
        )

    async def unload(self) -> None:
        """Unload Chatterbox model."""
        if self._model is not None:
            del self._model
            self._model = None
        self.shutdown()

    async def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize speech from text with voice cloning."""
        if self._model is None:
            raise RuntimeError("Chatterbox model not loaded. Call load() first.")

        audio_path = self._resolve_voice_path(voice)
        temperature = kwargs.get("temperature", self._config.temperature)
        top_k = kwargs.get("top_k", self._config.top_k)
        top_p = kwargs.get("top_p", self._config.top_p)
        repetition_penalty = kwargs.get("repetition_penalty", self._config.repetition_penalty)

        def _sync_synthesize():
            import numpy as np
            import torch

            # Ensure float32 default to avoid dtype mismatches
            torch.set_default_dtype(torch.float32)

            # Generate audio with Chatterbox Turbo
            wav = self._model.generate(
                text,
                audio_prompt_path=audio_path,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # Convert tensor to numpy
            if hasattr(wav, "cpu"):
                wav = wav.cpu().numpy()

            # Handle shape - Chatterbox returns [1, samples] or [samples]
            if wav.ndim > 1:
                wav = wav.squeeze()

            return wav.astype(np.float32)

        loop = asyncio.get_running_loop()
        audio_array = await loop.run_in_executor(self._executor, _sync_synthesize)

        import numpy as np

        # Normalize to [-1, 1] if needed and convert to PCM
        max_val = np.abs(audio_array).max()
        if max_val > 1.0:
            audio_array = audio_array / max_val

        audio_clamped = np.clip(audio_array, -1.0, 1.0)
        pcm_bytes = (audio_clamped * 32767).astype(np.int16).tobytes()
        duration = len(audio_array) / self.sample_rate if len(audio_array) > 0 else 0

        return SynthesisResult(
            audio=pcm_bytes,
            sample_rate=self.sample_rate,
            duration=duration,
            format="pcm",
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """Simulate streaming by generating full audio, then yielding chunks.

        Since Chatterbox Turbo doesn't support native streaming, we generate
        the complete audio first and then yield it in chunks for compatibility
        with the streaming API.
        """
        # Generate full audio
        result = await self.synthesize(text, voice, speed, **kwargs)

        # Yield in chunks (~0.5s each)
        chunk_size = self.STREAM_CHUNK_SAMPLES * 2  # 2 bytes per sample (int16)
        audio = result.audio

        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size]
            # Small delay between chunks to simulate streaming
            await asyncio.sleep(0.01)

    def get_voices(self) -> list[VoiceInfo]:
        """Get list of available voices (built-in + user profiles)."""
        import os

        voices = []

        # Add built-in voices
        for voice_id, (display_name, filename) in CHATTERBOX_BUILTIN_VOICES.items():
            # Only include if the voice file exists
            builtin_path = os.path.join(_get_builtin_voices_dir(), filename)
            if os.path.exists(builtin_path):
                voices.append(
                    VoiceInfo(
                        id=voice_id,
                        name=display_name,
                        language="en",
                        model="chatterbox-turbo",
                    )
                )

        # Add user-defined voice profiles
        for name, profile in self._voice_profiles.items():
            voices.append(
                VoiceInfo(
                    id=name,
                    name=profile.description or name,
                    language="en",
                    model="chatterbox-turbo",
                )
            )

        return voices


class PocketTTSBackend(TTSBackend):
    """Pocket TTS backend - lightweight 100M parameter CPU TTS from Kyutai.

    Features:
    - ~6x faster than real-time on Apple Silicon
    - ~200ms time-to-first-audio
    - Native streaming support
    - 8 built-in voices + custom voice cloning from WAV files
    - CPU-only (no GPU required)

    Note: Pocket TTS supports streaming output for arbitrarily long text.
    """

    def __init__(self, default_voice: str = DEFAULT_POCKET_TTS_VOICE):
        super().__init__(default_voice)
        self._model = None
        self._voice_states: dict[str, Any] = {}  # Cache loaded voice states
        self._sample_rate = DEFAULT_SAMPLE_RATE

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _resolve_voice(self, voice: str) -> str:
        """Resolve voice name - returns voice name for built-in or path for custom."""
        import os

        # Check if it's a built-in voice
        if voice in POCKET_TTS_VOICES:
            return voice

        # Check if it's a file path
        if os.path.exists(voice):
            return voice

        # Default to the voice name and let pocket_tts handle it
        logger.warning(f"Unknown voice '{voice}', passing to pocket_tts as-is")
        return voice

    async def load(self, device: str) -> None:  # noqa: ARG002
        """Load Pocket TTS model.

        Note: Pocket TTS is CPU-only by design for efficiency.
        The device parameter is ignored.
        """
        del device  # Pocket TTS is CPU-only
        try:
            from pocket_tts import TTSModel as PocketModel
        except ImportError as e:
            raise ImportError(
                "pocket-tts is not installed. "
                "Install it with: uv pip install 'universal-runtime[tts-pocket]'"
            ) from e

        logger.info("Loading Pocket TTS model (CPU-only)")

        def _load_model():
            return PocketModel.load_model()

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(self._executor, _load_model)

        # Get sample rate from model
        if hasattr(self._model, "sample_rate"):
            self._sample_rate = self._model.sample_rate

        logger.info(f"Pocket TTS model loaded (sample_rate={self._sample_rate})")

    async def unload(self) -> None:
        """Unload Pocket TTS model."""
        if self._model is not None:
            del self._model
            self._model = None
        self._voice_states.clear()
        self.shutdown()

    def _get_voice_state(self, voice: str) -> Any:
        """Get or create voice state for the given voice.

        Pocket TTS's get_state_for_audio_prompt() accepts both:
        - Built-in voice names (e.g., "alba", "marius")
        - Paths to WAV files for custom voice cloning
        """
        if voice in self._voice_states:
            return self._voice_states[voice]

        resolved = self._resolve_voice(voice)

        # get_state_for_audio_prompt handles both built-in voices and custom WAV files
        voice_state = self._model.get_state_for_audio_prompt(resolved)

        self._voice_states[voice] = voice_state
        return voice_state

    async def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize speech from text."""
        # Note: speed and kwargs are part of the TTSBackend interface but not supported
        del speed, kwargs
        if self._model is None:
            raise RuntimeError("Pocket TTS model not loaded. Call load() first.")

        def _sync_synthesize():
            import numpy as np

            voice_state = self._get_voice_state(voice)
            audio = self._model.generate_audio(voice_state, text)

            # Convert to numpy if tensor
            if hasattr(audio, "numpy"):
                audio = audio.numpy()

            return audio.astype(np.float32)

        loop = asyncio.get_running_loop()
        audio_array = await loop.run_in_executor(self._executor, _sync_synthesize)

        import numpy as np

        # Normalize and convert to PCM
        max_val = np.abs(audio_array).max()
        if max_val > 1.0:
            audio_array = audio_array / max_val

        audio_clamped = np.clip(audio_array, -1.0, 1.0)
        pcm_bytes = (audio_clamped * 32767).astype(np.int16).tobytes()
        duration = len(audio_array) / self.sample_rate if len(audio_array) > 0 else 0

        return SynthesisResult(
            audio=pcm_bytes,
            sample_rate=self.sample_rate,
            duration=duration,
            format="pcm",
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as they're generated.

        Pocket TTS supports native streaming for arbitrarily long text.
        """
        # Note: speed and kwargs are part of the TTSBackend interface but not supported
        del speed, kwargs
        if self._model is None:
            raise RuntimeError("Pocket TTS model not loaded. Call load() first.")

        audio_queue: Queue[bytes | None] = Queue()

        def _sync_generator():
            import numpy as np

            try:
                voice_state = self._get_voice_state(voice)

                # Use streaming generation if available
                if hasattr(self._model, "generate_audio_stream"):
                    for audio_chunk in self._model.generate_audio_stream(
                        voice_state, text
                    ):
                        if audio_chunk is not None and len(audio_chunk) > 0:
                            if hasattr(audio_chunk, "numpy"):
                                audio_chunk = audio_chunk.numpy()
                            audio_clamped = np.clip(audio_chunk, -1.0, 1.0)
                            pcm_bytes = (audio_clamped * 32767).astype(
                                np.int16
                            ).tobytes()
                            audio_queue.put(pcm_bytes)
                else:
                    # Fallback: generate full audio and chunk it
                    audio = self._model.generate_audio(voice_state, text)
                    if hasattr(audio, "numpy"):
                        audio = audio.numpy()
                    audio_clamped = np.clip(audio, -1.0, 1.0)
                    pcm_bytes = (audio_clamped * 32767).astype(np.int16).tobytes()

                    # Yield in ~0.5s chunks
                    chunk_size = self.sample_rate * 2  # 2 bytes per sample
                    for i in range(0, len(pcm_bytes), chunk_size):
                        audio_queue.put(pcm_bytes[i : i + chunk_size])
            except Exception as e:
                logger.error(f"Pocket TTS synthesis error: {e}")
            finally:
                audio_queue.put(None)

        thread = Thread(target=_sync_generator, daemon=True)
        thread.start()

        while True:
            try:
                chunk = audio_queue.get(timeout=0.005)
                if chunk is None:
                    break
                yield chunk
            except Empty:
                await asyncio.sleep(0)

        thread.join(timeout=1.0)

    def get_voices(self) -> list[VoiceInfo]:
        """Get list of available Pocket TTS voices."""
        return [
            VoiceInfo(
                id=voice_id,
                name=display_name,
                language="en",
                model="pocket-tts",
            )
            for voice_id, display_name in POCKET_TTS_VOICES.items()
        ]


class TTSModel(BaseModel):
    """Unified TTS model wrapper supporting multiple backends.

    Supported backends:
    - kokoro: High-quality neural TTS with built-in voices (82M params).
              Automatically uses MLX on Apple Silicon for faster inference.
    - kokoro-mlx: Explicit MLX backend for Kokoro (Apple Silicon only)
    - chatterbox-turbo: Fast voice cloning TTS (350M params, GPU)
    - pocket-tts: Lightweight CPU TTS from Kyutai (100M params, ~6x realtime)

    Example usage:
        # Kokoro (default) - auto-selects MLX on Apple Silicon
        model = TTSModel(model_id="kokoro", voice="af_heart")
        await model.load()
        result = await model.synthesize("Hello world")

        # Chatterbox Turbo with voice profiles
        profiles = {"narrator": VoiceProfile("narrator", "./voices/narrator.wav")}
        model = TTSModel(
            model_id="chatterbox-turbo",
            voice="narrator",
            voice_profiles=profiles,
        )
        await model.load()
        result = await model.synthesize("Hello world")

        # Pocket TTS (CPU-only, lightweight)
        model = TTSModel(model_id="pocket-tts", voice="alba")
        await model.load()
        result = await model.synthesize("Hello world")
    """

    def __init__(
        self,
        model_id: str = "kokoro",
        device: str = "auto",
        voice: str = DEFAULT_VOICE,
        voice_profiles: dict[str, VoiceProfile] | None = None,
        chatterbox_config: ChatterboxConfig | None = None,
        token: str | None = None,
    ):
        """Initialize TTS model.

        Args:
            model_id: Backend identifier. Options:
                - "kokoro": Auto-selects MLX on Apple Silicon, PyTorch elsewhere
                - "kokoro-mlx": Force MLX backend (Apple Silicon only)
                - "chatterbox-turbo": Voice cloning TTS
                - "pocket-tts": Lightweight CPU TTS
            device: Device to run on ("cuda", "cpu", "mps", "auto").
                    Note: pocket-tts and kokoro-mlx ignore this parameter.
            voice: Default voice ID (Kokoro/Pocket) or profile name (Chatterbox).
            voice_profiles: Voice profiles for Chatterbox voice cloning.
            chatterbox_config: Chatterbox-specific parameters.
            token: HuggingFace token (not typically needed).
        """
        super().__init__(model_id, device, token=token)
        self.model_type = "tts"
        self.voice = voice
        self._voice_profiles = voice_profiles
        self._chatterbox_config = chatterbox_config
        self._backend: TTSBackend | None = None
        # Override parent's default - TTS supports streaming by default
        self.supports_streaming = True

    def _create_backend(self) -> TTSBackend:
        """Factory method to create appropriate backend."""
        if self.model_id == "kokoro":
            # Prefer MLX backend on Apple Silicon for better performance
            if _is_mlx_available():
                logger.info("Using MLX backend for Kokoro TTS (Apple Silicon detected)")
                return KokoroMLXBackend(default_voice=self.voice)
            return KokoroBackend(default_voice=self.voice)
        elif self.model_id == "kokoro-mlx":
            # Explicit MLX backend request
            if not _is_mlx_available():
                raise RuntimeError(
                    "kokoro-mlx requires Apple Silicon and mlx-audio package. "
                    "Install with: uv pip install 'universal-runtime[tts-mlx]'"
                )
            return KokoroMLXBackend(default_voice=self.voice)
        elif self.model_id == "chatterbox-turbo":
            return ChatterboxTurboBackend(
                default_voice=self.voice,
                voice_profiles=self._voice_profiles,
                config=self._chatterbox_config,
            )
        elif self.model_id == "pocket-tts":
            return PocketTTSBackend(default_voice=self.voice)
        else:
            raise ValueError(
                f"Unknown TTS model: {self.model_id}. "
                "Supported models: kokoro, kokoro-mlx, chatterbox-turbo, pocket-tts"
            )

    async def load(self) -> None:
        """Load the TTS model."""
        self._backend = self._create_backend()
        await self._backend.load(self.device)
        # Update streaming support based on loaded backend
        self.supports_streaming = self._backend.supports_streaming

    async def unload(self) -> None:
        """Unload the model and free resources."""
        logger.info(f"Unloading TTS model: {self.model_id}")

        if self._backend is not None:
            await self._backend.unload()
            self._backend = None

        # Call parent cleanup for CUDA/MPS cache clearing
        await super().unload()

        logger.info(f"TTS model unloaded: {self.model_id}")

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize speech from text (non-streaming).

        Args:
            text: Text to synthesize (max 4096 characters recommended).
            voice: Voice ID to use. If None, uses the model's default voice.
            speed: Speed multiplier (0.5 to 2.0). Default is 1.0.
            **kwargs: Backend-specific parameters (e.g., temperature, top_k for Chatterbox).

        Returns:
            SynthesisResult with PCM audio data, sample rate, and duration.
        """
        if self._backend is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        voice = voice or self.voice
        return await self._backend.synthesize(text, voice, speed, **kwargs)

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as they're generated.

        Args:
            text: Text to synthesize.
            voice: Voice ID to use. If None, uses the model's default voice.
            speed: Speed multiplier (0.5 to 2.0). Default is 1.0.
            **kwargs: Backend-specific parameters.

        Yields:
            bytes: Raw PCM audio chunks (16-bit signed integers, 24kHz mono).
        """
        if self._backend is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        voice = voice or self.voice
        async for chunk in self._backend.synthesize_stream(text, voice, speed, **kwargs):
            yield chunk

    def get_voices(self) -> list[VoiceInfo]:
        """Get list of available voices.

        Returns:
            List of VoiceInfo objects describing available voices.
        """
        if self._backend:
            return self._backend.get_voices()

        # Fallback before load - return voices for known backends
        if self.model_id == "kokoro":
            return KokoroBackend().get_voices()
        elif self.model_id == "pocket-tts":
            return PocketTTSBackend().get_voices()
        elif self.model_id == "chatterbox-turbo" and self._voice_profiles:
            return [
                VoiceInfo(
                    id=name,
                    name=profile.description or name,
                    language="en",
                    model="chatterbox-turbo",
                )
                for name, profile in self._voice_profiles.items()
            ]
        return []

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "backend": self.model_id,
            "device": self.device,
            "default_voice": self.voice,
            "supports_streaming": self.supports_streaming,
            "sample_rate": self._backend.sample_rate if self._backend else DEFAULT_SAMPLE_RATE,
            "voices": [v.id for v in self.get_voices()],
        }
