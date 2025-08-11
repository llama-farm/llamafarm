"""OpenAI-Compatible Cloud API Component"""

from .openai_compatible_api import OpenAICompatibleAPI

# Backward compatibility alias
OpenAIAPI = OpenAICompatibleAPI

__all__ = ["OpenAICompatibleAPI", "OpenAIAPI"]