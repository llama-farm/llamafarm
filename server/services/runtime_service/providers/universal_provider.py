"""Universal Runtime provider implementation with streaming support."""

import time

import requests

from agents.base.clients.client import LFAgentClient
from agents.base.clients.openai import LFAgentClientOpenAI
from core.settings import settings

from .base import RuntimeProvider
from .health import HealthCheckResult


class UniversalProvider(RuntimeProvider):
    """Universal Runtime provider with SSE streaming support.

    This provider connects to the Universal Runtime (runtimes/universal) which:
    - Supports any HuggingFace transformer or diffusion model
    - Provides OpenAI-compatible endpoints
    - Supports SSE streaming for real-time token generation
    - Auto-detects hardware acceleration (MPS/CUDA/CPU)
    """

    @property
    def _base_url(self) -> str:
        """Get base URL for Universal Runtime API."""
        return (
            self._model_config.base_url
            or f"http://{settings.universal_host}:{settings.universal_port}/v1"
        )

    @property
    def _api_key(self) -> str:
        """Get API key for Universal Runtime."""
        return self._model_config.api_key or settings.universal_api_key

    def get_client(self) -> LFAgentClient:
        """Get Universal Runtime client with OpenAI compatibility.

        The client supports:
        - Standard chat completions
        - SSE streaming (set stream=True in requests)
        - Image generation (diffusion models)
        - Text embeddings (encoder models)
        - Audio transcription (speech-to-text)
        - Vision tasks (classification, VQA)

        Args:
            extra_body: Additional parameters to pass through

        Returns:
            LFAgentClient configured for Universal Runtime
        """
        if not self._model_config.base_url:
            self._model_config.base_url = self._base_url
        if not self._model_config.api_key:
            self._model_config.api_key = self._api_key

        client = LFAgentClientOpenAI(
            model_config=self._model_config,
        )
        return client

    def check_health(self) -> HealthCheckResult:
        """Check health of Universal Runtime.

        Verifies:
        - Runtime is reachable
        - Models are loaded
        - Device information (MPS/CUDA/CPU)
        - Response latency

        Returns:
            HealthCheckResult with status, loaded models, and device info
        """
        start = int(time.time() * 1000)
        base = self._base_url.replace("/v1", "").rstrip("/")

        # First check /health endpoint for device info
        health_url = f"{base}/health"
        models_url = f"{base}/v1/models"

        try:
            # Check health endpoint
            health_resp = requests.get(health_url, timeout=2.0)
            latency = int(time.time() * 1000) - start

            if 200 <= health_resp.status_code < 300:
                health_data = health_resp.json()
                device_info = health_data.get("device", {})
                device_type = (
                    device_info.get("type", "unknown")
                    if isinstance(device_info, dict)
                    else device_info
                )

                # Check models endpoint
                try:
                    models_resp = requests.get(models_url, timeout=1.0)
                    if 200 <= models_resp.status_code < 300:
                        models_data = models_resp.json()
                        models = models_data.get("data", [])
                        model_ids = [m.get("id") for m in models if m.get("id")]
                    else:
                        model_ids = []
                except Exception:
                    model_ids = []

                message = (
                    f"{base} reachable on {device_type}, "
                    f"{len(model_ids)} model(s) loaded"
                )
                return HealthCheckResult(
                    name="universal",
                    status="healthy",
                    message=message,
                    latency_ms=latency,
                    details={
                        "host": base,
                        "device": device_type,
                        "model_count": len(model_ids),
                        "models": model_ids,
                        "streaming_supported": True,
                        "loaded_models": health_data.get("loaded_models", []),
                    },
                )
            else:
                return HealthCheckResult(
                    name="universal",
                    status="unhealthy",
                    message=f"{base} returned HTTP {health_resp.status_code}",
                    latency_ms=latency,
                    details={"host": base, "status_code": health_resp.status_code},
                )
        except requests.exceptions.Timeout:
            port = settings.universal_port
            timeout_msg = (
                f"Timeout connecting to {base} - is Universal Runtime running? "
                f"(cd runtimes/universal && uv run uvicorn server:app --port {port})"
            )
            return HealthCheckResult(
                name="universal",
                status="unhealthy",
                message=timeout_msg,
                latency_ms=int(time.time() * 1000) - start,
                details={"host": base},
            )
        except Exception as e:
            return HealthCheckResult(
                name="universal",
                status="unhealthy",
                message=f"Error: {str(e)}",
                latency_ms=int(time.time() * 1000) - start,
                details={"host": base, "error": str(e)},
            )
