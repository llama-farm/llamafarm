import sys
from pathlib import Path

from .providers.base import RuntimeProvider
from .providers.lemonade_provider import LemonadeProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider

# Add repo root to path for config imports
repo_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import Model, Provider  # noqa: E402


class RuntimeService:
    """Service for resolving and managing runtime providers."""

    @staticmethod
    def get_provider(provider_enum: Provider, config: Model) -> RuntimeProvider:
        """Get provider implementation for the given provider enum.

        Args:
            provider_enum: The Provider enum value to look up

        Returns:
            The RuntimeProvider implementation for this provider

        Raises:
            ValueError: If the provider is invalid

        """
        match provider_enum:
            case Provider.openai:
                return OpenAIProvider(config)
            case Provider.ollama:
                return OllamaProvider(config)
            case Provider.lemonade:
                return LemonadeProvider(config)


runtime_service = RuntimeService()
