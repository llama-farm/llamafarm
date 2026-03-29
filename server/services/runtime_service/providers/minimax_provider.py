"""MiniMax runtime provider implementation.

MiniMax provides an OpenAI-compatible API at https://api.minimax.io/v1.
Models: MiniMax-M2.7, MiniMax-M2.7-highspeed (both 204K context).
"""

import os
import time

import requests

from agents.base.clients.client import LFAgentClient
from agents.base.clients.openai import LFAgentClientOpenAI

from .base import RuntimeProvider
from .health import HealthCheckResult

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"


class MiniMaxProvider(RuntimeProvider):
    """MiniMax API provider implementation.

    Uses the OpenAI-compatible endpoint at api.minimax.io/v1.
    Temperature is clamped to (0.0, 1.0] as required by the MiniMax API.
    """

    @property
    def _base_url(self) -> str:
        """Get base URL for MiniMax API."""
        return self._model_config.base_url or MINIMAX_BASE_URL

    @property
    def _api_key(self) -> str:
        """Get API key for MiniMax, falling back to MINIMAX_API_KEY env var."""
        return (
            self._model_config.api_key
            or os.environ.get(MINIMAX_API_KEY_ENV, "")
        )

    @staticmethod
    def _clamp_temperature(config) -> None:
        """Clamp temperature to MiniMax's accepted range (0.0, 1.0].

        MiniMax rejects temperature=0 and values >1.0.
        """
        params = config.model_api_parameters
        if params and hasattr(params, "__contains__") and "temperature" in params:
            temp = params["temperature"]
            if isinstance(temp, (int, float)):
                params["temperature"] = max(0.01, min(float(temp), 1.0))

    def get_client(self) -> LFAgentClient:
        """Get OpenAI-compatible client configured for MiniMax."""
        cfg_copy = self._model_config.model_copy()
        if not cfg_copy.base_url:
            cfg_copy.base_url = self._base_url
        if not cfg_copy.api_key:
            cfg_copy.api_key = self._api_key

        self._clamp_temperature(cfg_copy)

        client = LFAgentClientOpenAI(
            model_config=cfg_copy,
        )
        return client

    def check_health(self) -> HealthCheckResult:
        """Check health of MiniMax API."""
        start = int(time.time() * 1000)
        base_url = self._base_url

        try:
            resp = requests.get(
                base_url.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=5.0,
            )
            latency = int(time.time() * 1000) - start

            if resp.status_code == 200:
                return HealthCheckResult(
                    name="minimax",
                    status="healthy",
                    message="MiniMax API reachable and authenticated",
                    latency_ms=latency,
                    details={"base_url": base_url},
                )
            elif resp.status_code == 401:
                return HealthCheckResult(
                    name="minimax",
                    status="reachable",
                    message="MiniMax API reachable (API key not verified)",
                    latency_ms=latency,
                    details={"base_url": base_url},
                )
            else:
                return HealthCheckResult(
                    name="minimax",
                    status="reachable",
                    message=f"MiniMax API returned status {resp.status_code}",
                    latency_ms=latency,
                    details={"base_url": base_url},
                )
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                name="minimax",
                status="unhealthy",
                message=f"Timeout connecting to {base_url}",
                latency_ms=int(time.time() * 1000) - start,
                details={"base_url": base_url},
            )
        except Exception as e:
            return HealthCheckResult(
                name="minimax",
                status="unhealthy",
                message=f"Error: {str(e)}",
                latency_ms=int(time.time() * 1000) - start,
                details={"base_url": base_url},
            )
