"""Base class for runtime providers."""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import instructor
from openai import AsyncOpenAI

from .health import HealthCheckResult

if TYPE_CHECKING:
    from config.datamodel import LlamaFarmConfig


class RuntimeProvider(ABC):
    """Base class for runtime providers.

    Each provider implementation must define how to:
    1. Create an OpenAI-compatible client
    2. Determine the default instructor mode
    3. Get the base URL for the provider
    4. Get the API key for the provider
    5. Check the health of the provider's runtime
    """

    @abstractmethod
    def get_client(
        self, config: "LlamaFarmConfig"
    ) -> instructor.client.AsyncInstructor | AsyncOpenAI:
        """Get OpenAI-compatible client for this provider.

        Args:
            config: LlamaFarm configuration containing runtime settings

        Returns:
            Either an instructor-wrapped AsyncOpenAI client (for structured output)
            or a plain AsyncOpenAI client
        """
        pass

    @abstractmethod
    def get_default_instructor_mode(self) -> instructor.Mode:
        """Get default instructor mode for this provider.

        Returns:
            The instructor.Mode that works best with this provider
        """
        pass

    @abstractmethod
    def get_base_url(self, config: "LlamaFarmConfig") -> str:
        """Get base URL for this provider.

        Args:
            config: LlamaFarm configuration containing runtime settings

        Returns:
            The base URL for the provider's API
        """
        pass

    @abstractmethod
    def get_api_key(self, config: "LlamaFarmConfig") -> Optional[str]:
        """Get API key for this provider.

        Args:
            config: LlamaFarm configuration containing runtime settings

        Returns:
            The API key to use, or None if not required
        """
        pass

    @abstractmethod
    def check_health(self, config: "LlamaFarmConfig") -> HealthCheckResult:
        """Check health of this provider's runtime.

        Args:
            config: LlamaFarm configuration (or temp config with model settings)
                   Provider extracts base_url, port, etc. from config.runtime

        Returns:
            HealthCheckResult with status, message, latency, and details
        """
        pass
