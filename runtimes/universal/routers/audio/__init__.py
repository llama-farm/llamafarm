"""Audio router for speech-to-text transcription and translation endpoints."""

from .router import router, set_speech_loader

__all__ = ["router", "set_speech_loader"]
