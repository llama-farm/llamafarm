"""OpenAI runtime provider implementation."""

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


class OpenAIProvider(RuntimeProvider):
    """OpenAI API provider implementation."""

    def get_base_url(self, config: LlamaFarmConfig) -> str:
        """Get base URL for OpenAI API."""
        return config.runtime.base_url or "https://api.openai.com/v1"

    def get_api_key(self, config: LlamaFarmConfig) -> str:
        """Get API key for OpenAI."""
        return config.runtime.api_key

    def get_default_instructor_mode(self) -> instructor.Mode:
        """OpenAI supports TOOLS mode (function calling)."""
        return instructor.Mode.TOOLS

    def get_client(
        self, config: LlamaFarmConfig
    ) -> instructor.client.AsyncInstructor | AsyncOpenAI:
        """Get OpenAI client with optional instructor wrapping."""
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
        """Check health of OpenAI API.

        Note: This is a basic implementation. Full health check would require
        making an authenticated request to verify API key validity.
        """
        start = int(time.time() * 1000)
        base_url = config.get("base_url", "https://api.openai.com/v1") if config else "https://api.openai.com/v1"

        # For OpenAI, we can check if the base URL is reachable
        # A full health check would require an API key
        try:
            # Just check if we can reach the base domain
            domain = base_url.split("/v1")[0]
            resp = requests.get(domain, timeout=2.0)

            return {
                "name": "openai",
                "status": "reachable",
                "message": f"{base_url} domain reachable (API key not verified)",
                "latency_ms": int(time.time() * 1000) - start,
                "details": {"base_url": base_url},
            }
        except requests.exceptions.Timeout:
            return {
                "name": "openai",
                "status": "unhealthy",
                "message": f"Timeout connecting to {base_url}",
                "latency_ms": int(time.time() * 1000) - start,
                "details": {"base_url": base_url},
            }
        except Exception as e:
            return {
                "name": "openai",
                "status": "unhealthy",
                "message": f"Error: {str(e)}",
                "latency_ms": int(time.time() * 1000) - start,
                "details": {"base_url": base_url},
            }
