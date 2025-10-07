"""Base class for runtime providers."""

import sys
from abc import ABC, abstractmethod
from pathlib import Path

import instructor
from openai import AsyncOpenAI

from .health import HealthCheckResult

# Add repo root to path for config imports
repo_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import Model  # noqa: E402


class RuntimeProvider(ABC):
    """Base class for runtime providers.

    Each provider implementation must define how to:
    1. Create an OpenAI-compatible client
    2. Determine the default instructor mode
    3. Get the base URL for the provider
    4. Get the API key for the provider
    5. Check the health of the provider's runtime
    """

    def __init__(self, model_config: Model) -> None:
        self._model_config: Model = model_config

    @abstractmethod
    def get_client(self) -> instructor.client.AsyncInstructor | AsyncOpenAI:
        """Get OpenAI-compatible client for this provider.

        Args:
            config: LlamaFarm configuration containing runtime settings

        Returns:
            Either an instructor-wrapped AsyncOpenAI client (for structured output)
            or a plain AsyncOpenAI client
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

    @property
    @abstractmethod
    def _default_instructor_mode(self) -> instructor.Mode:
        """Return the default instructor mode for this runtime."""
        pass

    @property
    def _instructor_mode(self) -> instructor.Mode:
        """Get instructor mode for this runtime."""
        mode = self._model_config.instructor_mode
        try:
            return (
                instructor.mode.Mode[mode.upper()]
                if mode
                else self._default_instructor_mode
            )
        except (KeyError, TypeError):
            return self._default_instructor_mode
