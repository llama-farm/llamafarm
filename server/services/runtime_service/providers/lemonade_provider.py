"""Lemonade runtime provider implementation."""

import time
from collections.abc import Iterable

import requests

from agents.base.clients.client import LFAgentClient
from agents.base.clients.openai import LFAgentClientOpenAI
from core.settings import settings

from .base import RuntimeModelStatus, RuntimeProvider
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

    def get_model_runtime_status(self) -> RuntimeModelStatus:
        """Get runtime status for the configured Lemonade model."""
        base = self._base_url.replace("/api/v1", "")
        model_name = self._model_config.model

        try:
            resp = requests.get(f"{base}/api/v1/models", timeout=1.5)
            if not 200 <= resp.status_code < 300:
                return RuntimeModelStatus(
                    status="unreachable",
                    host=base,
                    runtime_message=f"{base} returned HTTP {resp.status_code}",
                )

            models = resp.json().get("data", [])
            loaded = _find_named_model(models, model_name) is not None
            return RuntimeModelStatus(
                status="loaded" if loaded else "idle",
                host=base,
                loaded=loaded,
                runtime_message=(
                    "Model is loaded in Lemonade"
                    if loaded
                    else "Runtime reachable; model not reported as loaded"
                ),
            )
        except requests.exceptions.Timeout:
            return RuntimeModelStatus(
                status="unreachable",
                host=base,
                runtime_message=f"Timeout connecting to {base}",
            )
        except Exception as e:
            return RuntimeModelStatus(
                status="unreachable",
                host=base,
                runtime_message=f"Error: {str(e)}",
            )


def _candidate_model_names(name: str) -> set[str]:
    candidates = {name}
    if ":" in name:
        base, tag = name.rsplit(":", 1)
        if tag == "latest":
            candidates.add(base)
    else:
        candidates.add(f"{name}:latest")
    return candidates


def _find_named_model(models: Iterable[dict], configured_name: str) -> dict | None:
    candidates = _candidate_model_names(configured_name)
    for model in models:
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id in candidates:
            return model
    return None
