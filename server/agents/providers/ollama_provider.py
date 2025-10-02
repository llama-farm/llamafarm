"""Ollama runtime provider implementation."""

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
from core.settings import settings
from .base import RuntimeProvider


class OllamaProvider(RuntimeProvider):
    """Ollama local runtime provider implementation."""

    def get_base_url(self, config: LlamaFarmConfig) -> str:
        """Get base URL for Ollama API."""
        return config.runtime.base_url or f"{settings.ollama_host}/v1"

    def get_api_key(self, config: LlamaFarmConfig) -> str:
        """Get API key for Ollama (usually not required)."""
        return config.runtime.api_key or settings.ollama_api_key

    def get_default_instructor_mode(self) -> instructor.Mode:
        """Ollama works best with MD_JSON mode for local models."""
        return instructor.Mode.MD_JSON

    def get_client(
        self, config: LlamaFarmConfig
    ) -> instructor.client.AsyncInstructor | AsyncOpenAI:
        """Get Ollama client with optional instructor wrapping."""
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

    def check_health(self, config: dict = None) -> dict:
        """Check health of Ollama runtime."""
        start = int(time.time() * 1000)
        base = config.get("base_url", settings.ollama_host) if config else settings.ollama_host
        url = f"{base}/api/tags"

        try:
            resp = requests.get(url, timeout=1.0)
            if 200 <= resp.status_code < 300:
                data = resp.json()
                model_count = len(data.get("models", []))
                return {
                    "name": "ollama",
                    "status": "healthy",
                    "message": f"{base} reachable, {model_count} model(s) available",
                    "latency_ms": int(time.time() * 1000) - start,
                    "details": {
                        "host": base,
                        "model_count": model_count,
                        "models": [m.get("name") for m in data.get("models", [])],
                    },
                }
            else:
                return {
                    "name": "ollama",
                    "status": "unhealthy",
                    "message": f"{base} returned HTTP {resp.status_code}",
                    "latency_ms": int(time.time() * 1000) - start,
                    "details": {"host": base, "status_code": resp.status_code},
                }
        except requests.exceptions.Timeout:
            return {
                "name": "ollama",
                "status": "unhealthy",
                "message": f"Timeout connecting to {base}",
                "latency_ms": int(time.time() * 1000) - start,
                "details": {"host": base},
            }
        except Exception as e:
            return {
                "name": "ollama",
                "status": "unhealthy",
                "message": f"Error: {str(e)}",
                "latency_ms": int(time.time() * 1000) - start,
                "details": {"host": base},
            }
