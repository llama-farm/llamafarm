"""Ollama runtime provider implementation."""

import time
from collections.abc import Iterable

import requests

from agents.base.clients.client import LFAgentClient
from agents.base.clients.ollama import LFAgentClientOllama
from core.settings import settings

from .base import RuntimeModelStatus, RuntimeProvider, format_bytes
from .health import HealthCheckResult


class OllamaProvider(RuntimeProvider):
    """Ollama local runtime provider implementation."""

    @property
    def _base_url(self) -> str:
        """Get base URL for Ollama API."""
        return self._model_config.base_url or f"{settings.ollama_host}/v1"

    @property
    def _api_key(self) -> str:
        """Get API key for Ollama (usually not required)."""
        return self._model_config.api_key or settings.ollama_api_key

    def get_client(self) -> LFAgentClient:
        """Get Ollama client with optional instructor wrapping."""
        cfg_copy = self._model_config.model_copy()
        if not cfg_copy.base_url:
            cfg_copy.base_url = self._base_url
        if not cfg_copy.api_key:
            cfg_copy.api_key = self._api_key

        client = LFAgentClientOllama(
            model_config=cfg_copy,
        )
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

    def get_model_runtime_status(self) -> RuntimeModelStatus:
        """Get runtime status for the configured Ollama model."""
        base = self._base_url.replace("/v1", "")
        model_name = self._model_config.model

        try:
            ps_resp = requests.get(f"{base}/api/ps", timeout=1.5)
            if 200 <= ps_resp.status_code < 300:
                running_models = ps_resp.json().get("models", [])
                match = _find_named_model(running_models, model_name)
                if match:
                    gpu_bytes = _coerce_int(match.get("size_vram"))
                    # size is total model memory; size_vram is the GPU portion.
                    # Report total as memory_usage and VRAM separately as gpu_allocation.
                    memory_bytes = (
                        _coerce_int(match.get("size")) or gpu_bytes
                    )
                    details = match.get("details", {})
                    status_details = {}
                    if isinstance(details, dict):
                        if details.get("parameter_size"):
                            status_details["parameter_size"] = details["parameter_size"]
                        if details.get("quantization_level"):
                            status_details["quantization_level"] = details[
                                "quantization_level"
                            ]
                    if match.get("expires_at"):
                        status_details["expires_at"] = match["expires_at"]

                    return RuntimeModelStatus(
                        status="running",
                        host=base,
                        loaded=True,
                        running=True,
                        memory_usage_bytes=memory_bytes,
                        memory_usage_human=format_bytes(memory_bytes),
                        gpu_allocation=(
                            format_bytes(gpu_bytes)
                            if gpu_bytes and gpu_bytes > 0
                            else None
                        ),
                        runtime_message="Model is currently loaded in Ollama",
                        details=status_details,
                    )

            tags_resp = requests.get(f"{base}/api/tags", timeout=1.5)
            if 200 <= tags_resp.status_code < 300:
                installed_models = tags_resp.json().get("models", [])
                installed = _find_named_model(installed_models, model_name)
                if installed:
                    return RuntimeModelStatus(
                        status="idle",
                        host=base,
                        runtime_message="Model is installed in Ollama but not currently loaded",
                    )
                return RuntimeModelStatus(
                    status="missing",
                    host=base,
                    runtime_message="Configured model is not installed in Ollama",
                )

            return RuntimeModelStatus(
                status="unreachable",
                host=base,
                runtime_message=f"Ollama returned HTTP {tags_resp.status_code}",
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
    else:
        candidates.add(f"{name}:latest")
    return candidates


def _find_named_model(models: Iterable[dict], configured_name: str) -> dict | None:
    candidates = _candidate_model_names(configured_name)
    for model in models:
        name = model.get("name")
        if isinstance(name, str) and name in candidates:
            return model
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
