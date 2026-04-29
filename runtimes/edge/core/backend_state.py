"""Process-wide backend readiness state for the edge runtime.

The edge runtime serves both HTTP clients and a Zenoh IPC bus
(local/llm/{request,response,status}) used by drone flight-control.
Flight-control needs an honest signal on local/llm/status to decide
whether to issue LLM-dependent commands. Without this state, init or
preload failures are swallowed and the heartbeat publishes "ready"
forever — flight-control then issues commands that are silently
dropped at the inference layer.

This module provides a single process-wide BACKEND_STATE object that
the lifespan code mutates as initialization progresses, and that the
Zenoh heartbeat reads to publish honest status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
import time


class Readiness(str, Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class BackendState:
    readiness: Readiness = Readiness.INITIALIZING
    reason: str = ""
    backend_initialized: bool = False
    last_transition_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    _lock: Lock = field(default_factory=Lock)

    def set(self, readiness: Readiness, reason: str = "") -> None:
        with self._lock:
            self.readiness = readiness
            self.reason = reason
            self.last_transition_ms = int(time.time() * 1000)

    def mark_backend_initialized(self) -> None:
        with self._lock:
            self.backend_initialized = True
            # Bump last_transition_ms so heartbeat consumers diffing on the
            # timestamp can detect this transition. Without this, the
            # backend_initialized flip is invisible in the published snapshot
            # to anyone watching last_transition_ms as a change marker.
            self.last_transition_ms = int(time.time() * 1000)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "readiness": self.readiness.value,
                "reason": self.reason,
                "backend_initialized": self.backend_initialized,
                "last_transition_ms": self.last_transition_ms,
            }


BACKEND_STATE = BackendState()
