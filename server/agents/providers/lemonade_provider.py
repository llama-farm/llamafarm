"""Lemonade runtime provider implementation."""

import sys
import time
from pathlib import Path
import requests
import instructor
from openai import AsyncOpenAI

# Add repo root to path for config imports
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import LlamaFarmConfig, PromptFormat  # noqa: E402
from .base import RuntimeProvider
from .health import HealthCheckResult


class LemonadeProvider(RuntimeProvider):
    """Lemonade local runtime provider implementation."""

    def get_base_url(self, config: LlamaFarmConfig) -> str:
        """Get base URL for Lemonade API."""
        if config.runtime.base_url:
            return config.runtime.base_url

        port = 11534  # default
        if config.runtime.lemonade:
            port = config.runtime.lemonade.port or 11534

        return f"http://127.0.0.1:{port}/api/v1"

    def get_api_key(self, config: LlamaFarmConfig) -> str:
        """Get API key for Lemonade (uses 'lemonade' as default)."""
        return config.runtime.api_key or "lemonade"

    def get_default_instructor_mode(self) -> instructor.Mode:
        """Lemonade works best with MD_JSON mode for local models."""
        return instructor.Mode.MD_JSON

    def get_client(
        self, config: LlamaFarmConfig
    ) -> instructor.client.AsyncInstructor | AsyncOpenAI:
        """Get Lemonade client with optional instructor wrapping."""
        client = AsyncOpenAI(
            api_key=self.get_api_key(config),
            base_url=self.get_base_url(config),
        )

        if config.runtime.prompt_format == PromptFormat.structured:
            mode = self._determine_mode(config)
            return instructor.from_openai(client, mode=mode)
        return client

    def _determine_mode(self, config: LlamaFarmConfig) -> instructor.Mode:
        """Determine instructor mode from config or use default."""
        if config.runtime.instructor_mode:
            return instructor.mode.Mode[config.runtime.instructor_mode.upper()]
        return self.get_default_instructor_mode()

    def check_health(self, config: LlamaFarmConfig) -> HealthCheckResult:
        """Check health of Lemonade runtime."""
        start = int(time.time() * 1000)
        # Use get_base_url to extract config (handles lemonade.port, base_url, etc.)
        base = self.get_base_url(config).replace("/api/v1", "")
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
