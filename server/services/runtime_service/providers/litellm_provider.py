"""LiteLLM runtime provider - unified gateway for 100+ LLM providers."""

from agents.base.clients.client import LFAgentClient
from agents.base.clients.litellm import LFAgentClientLiteLLM

from .base import RuntimeProvider
from .health import HealthCheckResult


class LiteLLMProvider(RuntimeProvider):
    """LiteLLM provider - routes to 100+ LLM providers via litellm SDK."""

    def get_client(self) -> LFAgentClient:
        return LFAgentClientLiteLLM(model_config=self._model_config)

    def check_health(self) -> HealthCheckResult:
        return HealthCheckResult(
            name="litellm",
            status="reachable",
            message="LiteLLM SDK routes to provider at call time",
            latency_ms=0,
            details={"model": self._model_config.model},
        )
