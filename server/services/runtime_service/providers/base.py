"""Base class for runtime providers."""

from abc import ABC, abstractmethod

from config.datamodel import LlamaFarmConfig, Model

from agents.base.clients.client import LFAgentClient
from services.model_service import ModelService

from .health import HealthCheckResult


class RuntimeProvider(ABC):
    """Base class for runtime providers.

    Each provider implementation must define how to:
    1. Create an OpenAI-compatible client
    2. Determine the default instructor mode
    3. Get the base URL for the provider
    4. Get the API key for the provider
    5. Check the health of the provider's runtime
    """

    def __init__(self, *, model_config: Model) -> None:
        self._model_config = model_config

    @abstractmethod
    def get_client(self) -> LFAgentClient:
        """Get compatible client for this provider.

        Args:
            config: LlamaFarm configuration containing runtime settings

        Returns:
            A compatible client for this provider
        """
        pass

    @abstractmethod
    def check_health(self) -> HealthCheckResult:
        """Check health of this provider's runtime.

        Args:
            config: LlamaFarm configuration (or temp config with model settings)
                   Provider extracts base_url, port, etc. from config.runtime

        Returns:
            HealthCheckResult with status, message, latency, and details
        """
        pass
