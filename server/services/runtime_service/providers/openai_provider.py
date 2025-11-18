"""OpenAI runtime provider implementation."""

import time

import requests

from agents.base.clients.client import LFAgentClient
from agents.base.clients.openai import LFAgentClientOpenAI

from .base import RuntimeProvider
from .health import HealthCheckResult

openai_base_url = "https://api.openai.com/v1"
openai_api_key = ""


class OpenAIProvider(RuntimeProvider):
    """OpenAI API provider implementation."""

    @property
    def _base_url(self) -> str:
        """Get base URL for OpenAI API."""
        return self._model_config.base_url or openai_base_url

    @property
    def _api_key(self) -> str:
        """Get API key for OpenAI."""
        return self._model_config.api_key or ""

    def get_client(self) -> LFAgentClient:
        """Get OpenAI client with optional instructor wrapping."""
        cfg_copy = self._model_config.model_copy()
        if not cfg_copy.base_url:
            cfg_copy.base_url = self._base_url
        if not cfg_copy.api_key:
            cfg_copy.api_key = self._api_key
        client = LFAgentClientOpenAI(
            model_config=cfg_copy,
        )

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
