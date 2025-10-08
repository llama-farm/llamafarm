"""OpenAI runtime provider implementation."""

import sys
import time
from pathlib import Path

import instructor
import requests
from openai import AsyncOpenAI

from .base import RuntimeProvider
from .health import HealthCheckResult

# Add repo root to path for config imports
repo_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import PromptFormat  # noqa: E402

openai_base_url = "https://api.openai.com/v1"


class OpenAIProvider(RuntimeProvider):
    """OpenAI API provider implementation."""

    @property
    def _default_instructor_mode(self) -> instructor.Mode:
        """Return the default instructor mode for this runtime."""
        return instructor.Mode.TOOLS

    @property
    def _base_url(self) -> str:
        """Get base URL for OpenAI API."""
        return self._model_config.base_url or openai_base_url

    @property
    def _api_key(self) -> str:
        """Get API key for OpenAI."""
        return self._model_config.api_key or ""

    def get_client(self) -> instructor.client.AsyncInstructor | AsyncOpenAI:
        """Get OpenAI client with optional instructor wrapping."""
        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

        if self._model_config.prompt_format == PromptFormat.structured:
            mode = self._instructor_mode
            return instructor.from_openai(client, mode=mode)
        return client

    def check_health(self) -> HealthCheckResult:
        """Check health of OpenAI API."""
        start = int(time.time() * 1000)
        base_url = self._base_url.replace("/v1", "")

        # For OpenAI, we can check if the base URL is reachable
        # A full health check would require an API key
        try:
            # Just check if we can reach the base domain
            domain = base_url.split("/v1")[0]
            _ = requests.get(domain, timeout=2.0)
            latency = int(time.time() * 1000) - start

            return HealthCheckResult(
                name="openai",
                status="reachable",
                message=f"{base_url} domain reachable (API key not verified)",
                latency_ms=latency,
                details={"base_url": base_url},
            )
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                name="openai",
                status="unhealthy",
                message=f"Timeout connecting to {base_url}",
                latency_ms=int(time.time() * 1000) - start,
                details={"base_url": base_url},
            )
        except Exception as e:
            return HealthCheckResult(
                name="openai",
                status="unhealthy",
                message=f"Error: {str(e)}",
                latency_ms=int(time.time() * 1000) - start,
                details={"base_url": base_url},
            )
