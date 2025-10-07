"""Ollama runtime provider implementation."""

import sys
import time
from pathlib import Path

import instructor
import requests
from openai import AsyncOpenAI

from core.settings import settings

from .base import RuntimeProvider
from .health import HealthCheckResult

# Add repo root to path for config imports
repo_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import PromptFormat  # noqa: E402


class OllamaProvider(RuntimeProvider):
    """Ollama local runtime provider implementation."""

    @property
    def _default_instructor_mode(self) -> instructor.Mode:
        return instructor.Mode.MD_JSON

    @property
    def _base_url(self) -> str:
        """Get base URL for Ollama API."""
        return self._model_config.base_url or f"{settings.ollama_host}/v1"

    @property
    def _api_key(self) -> str:
        """Get API key for Ollama (usually not required)."""
        return self._model_config.api_key or settings.ollama_api_key

    def get_client(self) -> instructor.client.AsyncInstructor | AsyncOpenAI:
        """Get Ollama client with optional instructor wrapping."""
        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

        if self._model_config.prompt_format == PromptFormat.structured:
            mode = self._instructor_mode
            return instructor.from_openai(client, mode=mode)
        return client

    def check_health(self) -> HealthCheckResult:
        """Check health of Ollama runtime."""
        start = int(time.time() * 1000)
        base = self._base_url.replace("/v1", "")
        url = f"{base}/api/tags"

        try:
            resp = requests.get(url, timeout=1.0)
            latency = int(time.time() * 1000) - start

            if 200 <= resp.status_code < 300:
                data = resp.json()
                model_count = len(data.get("models", []))
                return HealthCheckResult(
                    name="ollama",
                    status="healthy",
                    message=f"{base} reachable, {model_count} model(s) available",
                    latency_ms=latency,
                    details={
                        "host": base,
                        "model_count": model_count,
                        "models": [m.get("name") for m in data.get("models", [])],
                    },
                )
            else:
                return HealthCheckResult(
                    name="ollama",
                    status="unhealthy",
                    message=f"{base} returned HTTP {resp.status_code}",
                    latency_ms=latency,
                    details={"host": base, "status_code": resp.status_code},
                )
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                name="ollama",
                status="unhealthy",
                message=f"Timeout connecting to {base}",
                latency_ms=int(time.time() * 1000) - start,
                details={"host": base},
            )
        except Exception as e:
            return HealthCheckResult(
                name="ollama",
                status="unhealthy",
                message=f"Error: {str(e)}",
                latency_ms=int(time.time() * 1000) - start,
                details={"host": base},
            )
