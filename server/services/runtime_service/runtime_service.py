from config.datamodel import Model, Provider

from .providers.base import RuntimeProvider
from .providers.lemonade_provider import LemonadeProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .providers.router_provider import ModelResolver, RouterProvider
from .providers.universal_provider import UniversalProvider


class RuntimeService:
    """Service for resolving and managing runtime providers."""

    @staticmethod
    def get_provider(
        model_config: Model,
        all_models: list[Model] | None = None,
    ) -> RuntimeProvider:
        """Get provider implementation for the given provider enum.

        Args:
            model_config: The model configuration to create a provider for
            all_models: Optional list of all model configurations (required for router
                       provider to resolve target models)

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
            case Provider.router:
                # Router provider needs access to all models to resolve targets
                model_resolver = None
                if all_models:
                    model_resolver = ModelResolver(all_models)
                return RouterProvider(
                    model_config=model_config,
                    model_resolver=model_resolver,
                )


runtime_service = RuntimeService()
