"""Base class for runtime providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from config.datamodel import Model

from agents.base.clients.client import LFAgentClient

from .health import HealthCheckResult


class RuntimeProvider(ABC):
    """Base class for runtime providers.

    Each provider implementation must define how to:
    1. Create an OpenAI-compatible client
    2. Determine the default instructor mode
    3. Get the base URL for the provider
    4. Get the API key for the provider
    5. Check the health of the provider's runtime
    """

    def __init__(self, *, model_config: Model) -> None:
        self._model_config = model_config

    @abstractmethod
    def get_client(self) -> LFAgentClient:
        """Get compatible client for this provider.

        Args:
            config: LlamaFarm configuration containing runtime settings

        Returns:
            A compatible client for this provider
        """
        pass

    @abstractmethod
    def check_health(self) -> HealthCheckResult:
        """Check health of this provider's runtime.

        Args:
            config: LlamaFarm configuration (or temp config with model settings)
                   Provider extracts base_url, port, etc. from config.runtime

        Returns:
            HealthCheckResult with status, message, latency, and details
        """
        pass

    def get_model_runtime_status(self) -> "RuntimeModelStatus":
        """Describe the current runtime state for this configured model.

        Providers with richer runtime APIs should override this method.
        The default behavior preserves a useful status for providers that
        do not expose per-model loading information.
        """
        health = self.check_health()
        host = None
        if health.details:
            host = health.details.get("host") or health.details.get("base_url")

        if health.status in {"healthy", "degraded"}:
            status = "reachable"
        elif health.status == "reachable":
            status = "remote"
        else:
            status = "unreachable"

        return RuntimeModelStatus(
            status=status,
            host=host,
            runtime_message=health.message,
        )


@dataclass
class CachedModel:
    """Cached model information."""

    id: str
    name: str
    size: int
    path: str


@dataclass
class RuntimeModelStatus:
    """Structured runtime status for a configured model."""

    status: str
    host: str | None = None
    loaded: bool = False
    running: bool = False
    memory_usage_bytes: int | None = None
    memory_usage_human: str | None = None
    gpu_allocation: str | None = None
    uptime_seconds: int | None = None
    uptime_human: str | None = None
    runtime_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert runtime status to JSON-serializable output."""
        data: dict[str, Any] = {
            "runtime_status": self.status,
            "runtime_loaded": self.loaded,
            "runtime_running": self.running,
        }
        optional_fields = {
            "runtime_host": self.host,
            "memory_usage_bytes": self.memory_usage_bytes,
            "memory_usage_human": self.memory_usage_human,
            "gpu_allocation": self.gpu_allocation,
            "uptime_seconds": self.uptime_seconds,
            "uptime_human": self.uptime_human,
            "runtime_message": self.runtime_message,
            "runtime_details": self.details or None,
        }
        for key, value in optional_fields.items():
            if value is not None:
                data[key] = value
        return data


def format_bytes(value: int | float | None) -> str | None:
    """Format a byte count using binary units."""
    if value is None:
        return None
    num = float(value)
    if num < 0:
        return None

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    while num >= 1024 and unit_index < len(units) - 1:
        num /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(num)} {units[unit_index]}"
    return f"{num:.1f} {units[unit_index]}"


def format_duration(seconds: int | float | None) -> str | None:
    """Render a duration in a short human-readable form.

    Examples: 45s, 12m, 2h 5m, 1d 3h.
    """
    if seconds is None:
        return None
    total = int(seconds)
    if total < 0:
        return None
    if total < 60:
        return f"{total}s"
    minutes, _ = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, rem_minutes = divmod(minutes, 60)
    if hours < 24:
        if rem_minutes == 0:
            return f"{hours}h"
        return f"{hours}h {rem_minutes}m"
    days, rem_hours = divmod(hours, 24)
    if rem_hours == 0:
        return f"{days}d"
    return f"{days}d {rem_hours}h"
