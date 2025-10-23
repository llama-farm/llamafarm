from config.datamodel import LlamaFarmConfig, Model, Provider

from services.model_service import ModelService

from .providers.base import RuntimeProvider
from .providers.lemonade_provider import LemonadeProvider
from .providers.universal_provider import UniversalProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider


class RuntimeService:
    """Service for resolving and managing runtime providers."""

    @staticmethod
    def get_provider(model_config: Model) -> RuntimeProvider:
        """Get provider implementation for the given provider enum.

        Args:
            provider_enum: The Provider enum value to look up

        Returns:
            The RuntimeProvider implementation for this provider

        Raises:
            ValueError: If the provider is invalid

        """
        provider_enum = model_config.provider

        match provider_enum:
            case Provider.openai:
                return OpenAIProvider(model_config=model_config)
            case Provider.ollama:
                return OllamaProvider(model_config=model_config)
            case Provider.lemonade:
                return LemonadeProvider(model_config=model_config)
            case Provider.universal:
                return UniversalProvider(model_config=model_config)


runtime_service = RuntimeService()
