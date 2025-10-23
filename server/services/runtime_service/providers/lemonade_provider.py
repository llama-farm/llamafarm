"""Lemonade runtime provider implementation."""

import time

import requests

from agents.base.clients.client import LFAgentClient
from agents.base.clients.openai import LFAgentClientOpenAI
from core.settings import settings

from .base import RuntimeProvider
from .health import HealthCheckResult


class LemonadeProvider(RuntimeProvider):
    """Lemonade local runtime provider implementation."""

    @property
    def _base_url(self) -> str:
        """Get base URL for Lemonade API."""

        return (
            self._model_config.base_url
            or f"http://{settings.lemonade_host}:{settings.lemonade_port}/api/v1"
        )

    @property
    def _api_key(self) -> str:
        """Get API key for Lemonade (uses 'lemonade' as default)."""
        return self._model_config.api_key or settings.lemonade_api_key

    def get_client(self) -> LFAgentClient:
        """Get Lemonade client with optional instructor wrapping."""
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
        """Check health of Lemonade runtime."""
        start = int(time.time() * 1000)
        base = self._base_url.replace("/api/v1", "")
        url = f"{base}/api/v1/models"

        try:
            resp = requests.get(url, timeout=1.0)
            latency = int(time.time() * 1000) - start

            if 200 <= resp.status_code < 300:
                data = resp.json()
                models = data.get("data", [])
                model_ids = [m.get("id") for m in models if m.get("id")]

                return HealthCheckResult(
                    name="lemonade",
                    status="healthy",
                    message=f"{base} reachable, {len(model_ids)} model(s) loaded",
                    latency_ms=latency,
                    details={
                        "host": base,
                        "model_count": len(model_ids),
                        "models": model_ids,
                    },
                )
            else:
                return HealthCheckResult(
                    name="lemonade",
                    status="unhealthy",
                    message=f"{base} returned HTTP {resp.status_code}",
                    latency_ms=latency,
                    details={"host": base, "status_code": resp.status_code},
                )
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                name="lemonade",
                status="unhealthy",
                message=f"Timeout connecting to {base} - is Lemonade running? (nx start lemonade)",
                latency_ms=int(time.time() * 1000) - start,
                details={"host": base},
            )
        except Exception as e:
            return HealthCheckResult(
                name="lemonade",
                status="unhealthy",
                message=f"Error: {str(e)}",
                latency_ms=int(time.time() * 1000) - start,
                details={"host": base},
            )
