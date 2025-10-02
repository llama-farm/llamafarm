"""Provider registry for runtime providers.

This module maintains a registry of all available runtime providers and provides
functions to register new providers and retrieve provider implementations.

The registry pattern allows new providers to be added without modifying core code,
following the Open/Closed Principle.
"""

import sys
from pathlib import Path
from typing import Dict

# Add repo root to path for config imports
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import Provider  # noqa: E402
from .base import RuntimeProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider
from .lemonade_provider import LemonadeProvider


# Global provider registry - maps Provider enum to implementation
_PROVIDER_REGISTRY: Dict[Provider, RuntimeProvider] = {
    Provider.openai: OpenAIProvider(),
    Provider.ollama: OllamaProvider(),
    Provider.lemonade: LemonadeProvider(),
}


def register_provider(provider_enum: Provider, provider_impl: RuntimeProvider) -> None:
    """Register a new provider implementation.

    This allows dynamic registration of new providers without modifying this file.
    Useful for plugins or runtime-added providers.

    Args:
        provider_enum: The Provider enum value
        provider_impl: The RuntimeProvider implementation instance

    Example:
        >>> from config.datamodel import Provider
        >>> from agents.providers import register_provider, RuntimeProvider
        >>> class MyProvider(RuntimeProvider):
        ...     # implementation
        >>> register_provider(Provider.my_provider, MyProvider())
    """
    _PROVIDER_REGISTRY[provider_enum] = provider_impl


def get_provider(provider_enum: Provider) -> RuntimeProvider:
    """Get provider implementation for the given provider enum.

    Args:
        provider_enum: The Provider enum value to look up

    Returns:
        The RuntimeProvider implementation for this provider

    Raises:
        ValueError: If the provider is not registered

    Example:
        >>> from config.datamodel import Provider
        >>> from agents.providers import get_provider
        >>> provider = get_provider(Provider.ollama)
        >>> client = provider.get_client(config)
    """
    if provider_enum not in _PROVIDER_REGISTRY:
        available = ", ".join(p.value for p in _PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported provider: {provider_enum.value}. "
            f"Available providers: {available}"
        )
    return _PROVIDER_REGISTRY[provider_enum]


def get_registered_providers() -> list[Provider]:
    """Get list of all registered providers.

    Returns:
        List of Provider enum values that are currently registered
    """
    return list(_PROVIDER_REGISTRY.keys())
