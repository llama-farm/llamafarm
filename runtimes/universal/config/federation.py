"""Federation configuration for multi-node vision cascade.

Defines peer configurations for both standalone and mesh deployments.
On standalone, peers are manually configured with direct URLs.
On mesh, Atmosphere auto-discovers peers and registers them.

The cascade chain doesn't care which mode it's in -- it just calls
.detect() on the next model, whether local or remote.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PeerConfig:
    """Configuration for a remote LlamaFarm peer.

    In standalone mode, these are manually configured.
    In mesh mode, Atmosphere auto-populates them via gossip.
    """

    name: str                       # Human-readable peer name
    url: str                        # Direct URL or "atmosphere://node-id"
    models: list[str] = field(default_factory=list)  # What models they have
    gpu_vram_gb: float = 0          # For routing decisions
    priority: int = 0               # Lower = tried first in cascade
    timeout: float = 30.0           # Request timeout in seconds
    healthy: bool = True            # Last known health status
    latency_ms: float = 0.0        # Last known latency


@dataclass
class FederationConfig:
    """Federation configuration for the vision cascade.

    Controls how this LlamaFarm instance connects to and uses
    remote peers for cascade escalation.
    """

    enabled: bool = False
    peers: list[PeerConfig] = field(default_factory=list)
    auto_register_atmosphere: bool = True  # Auto-add Atmosphere peers
    health_check_interval: float = 60.0  # Seconds between health checks
    max_retries: int = 2  # Retries per peer before marking unhealthy
