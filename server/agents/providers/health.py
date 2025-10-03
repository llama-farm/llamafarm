"""Health check result types for runtime providers."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HealthCheckResult:
    """Structured health check result from a runtime provider.

    Attributes:
        name: Provider or component name
        status: Health status ("healthy", "degraded", or "unhealthy")
        message: Human-readable status message
        latency_ms: Response time in milliseconds
        details: Additional provider-specific details
    """

    name: str
    status: str
    message: str
    latency_ms: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API responses."""
        result = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "latency_ms": self.latency_ms,
        }
        if self.details:
            result["details"] = self.details
        return result
