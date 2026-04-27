"""
Zenoh IPC interface for the edge runtime.

Allows the orchestrator and other drone services to request LLM inference
over the Zenoh pub/sub bus (Unix socket IPC), matching the communication
pattern used by vision, comms, and flight-control.

Topics:
  local/llm/request   — subscribe: incoming inference requests (JSON)
  local/llm/response  — publish: inference results (JSON)
  local/llm/status    — publish: periodic heartbeat with model info
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time

logger = logging.getLogger("edge-runtime.zenoh")

ZENOH_ENDPOINT = os.getenv(
    "ZENOH_ENDPOINT", "unixsock-stream//run/arc/zenoh.sock"
)
ZENOH_ENABLED = os.getenv("ZENOH_ENABLED", "true").lower() in ("true", "1", "yes")

TOPIC_REQUEST = "local/llm/request"
TOPIC_RESPONSE = "local/llm/response"
TOPIC_STATUS = "local/llm/status"

STATUS_INTERVAL_S = 5.0


class ZenohIPC:
    """Manages a Zenoh session for LLM inference over IPC."""

    def __init__(self, inference_fn):
        """
        Args:
            inference_fn: async callable(request_dict) -> response content string.
                          Called for each incoming inference request.
        """
        self._inference_fn = inference_fn
        self._session = None
        self._subscriber = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []
        self._pending_futures: list[asyncio.Future] = []

    async def start(self) -> bool:
        """Open Zenoh session and start subscriber + heartbeat tasks.

        Returns True if started successfully, False on failure (graceful degradation).
        """
        if not ZENOH_ENABLED:
            logger.info("Zenoh IPC disabled (ZENOH_ENABLED=false)")
            return False

        logger.info("startup-step BEGIN: %s", "zenoh-import")
        try:
            import zenoh
        except ImportError:
            logger.warning(
                "eclipse-zenoh package not installed, Zenoh IPC unavailable"
            )
            return False
        logger.info("startup-step END: %s", "zenoh-import")

        try:
            logger.info("startup-step BEGIN: %s", "zenoh-config")
            config = zenoh.Config()
            config.insert_json5(
                "connect/endpoints",
                json.dumps([ZENOH_ENDPOINT]),
            )
            config.insert_json5("scouting/multicast/enabled", "false")
            logger.info("startup-step END: %s", "zenoh-config")

            logger.info("startup-step BEGIN: %s", "zenoh-session-open")
            self._session = zenoh.open(config)
            logger.info("Zenoh session open (endpoint=%s)", ZENOH_ENDPOINT)
            logger.info("startup-step END: %s", "zenoh-session-open")
        except Exception:
            logger.warning(
                "Failed to connect to Zenoh at %s — continuing HTTP-only",
                ZENOH_ENDPOINT,
                exc_info=True,
            )
            return False

        self._loop = asyncio.get_event_loop()
        logger.info("startup-step BEGIN: %s", "zenoh-subscriber-declare")
        self._subscriber = self._session.declare_subscriber(
            TOPIC_REQUEST, self._on_request
        )
        logger.info("Subscribed to %s", TOPIC_REQUEST)
        logger.info("startup-step END: %s", "zenoh-subscriber-declare")

        logger.info("startup-step BEGIN: %s", "zenoh-heartbeat-spawn")
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        logger.info("startup-step END: %s", "zenoh-heartbeat-spawn")
        return True

    async def stop(self):
        """Cancel background tasks and close the Zenoh session."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            # Expected: tasks were explicitly cancelled above
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

        # Cancel in-flight request handlers before closing the session
        for future in list(self._pending_futures):
            future.cancel()
        self._pending_futures.clear()

        if self._subscriber is not None:
            try:
                self._subscriber.undeclare()
            except Exception:
                logger.warning("Error undeclaring Zenoh subscriber", exc_info=True)
            self._subscriber = None

        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                logger.warning("Error closing Zenoh session", exc_info=True)
            self._session = None
            logger.info("Zenoh session closed")

    # ------------------------------------------------------------------
    # Request handler
    # ------------------------------------------------------------------

    def _on_request(self, sample):
        """Callback invoked by Zenoh subscriber on each request."""
        try:
            payload = json.loads(bytes(sample.payload))
            future = asyncio.run_coroutine_threadsafe(
                self._handle_request(payload), self._loop
            )
            self._pending_futures.append(future)

            def _remove_future(f):
                # Already cleared by stop()
                with contextlib.suppress(ValueError):
                    self._pending_futures.remove(f)

            future.add_done_callback(_remove_future)
        except Exception:
            logger.error("Error dispatching Zenoh request", exc_info=True)

    async def _handle_request(self, request: dict):
        """Process a single inference request and publish the response."""
        request_id = request.get("request_id", "unknown")
        model = request.get("model", "unknown")
        t0 = time.monotonic()

        try:
            content = await self._inference_fn(request)
            inference_ms = int((time.monotonic() - t0) * 1000)

            response = {
                "request_id": request_id,
                "model": model,
                "content": content,
                "inference_time_ms": inference_ms,
                "timestamp_ms": int(time.time() * 1000),
            }
        except Exception as exc:
            inference_ms = int((time.monotonic() - t0) * 1000)
            response = {
                "request_id": request_id,
                "model": model,
                "content": "",
                "error": "inference failed",
                "inference_time_ms": inference_ms,
                "timestamp_ms": int(time.time() * 1000),
            }
            logger.error("Inference failed for request %s: %s", request_id, exc)

        self._session.put(
            TOPIC_RESPONSE, json.dumps(response).encode()
        )

    # ------------------------------------------------------------------
    # Status heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self):
        """Publish periodic status to local/llm/status."""
        logger.info(
            "Status heartbeat started (interval=%.1fs, topic=%s)",
            STATUS_INTERVAL_S,
            TOPIC_STATUS,
        )
        try:
            while True:
                status = {
                    "service": "edge-runtime",
                    "status": "ready",
                    "timestamp_ms": int(time.time() * 1000),
                }
                self._session.put(
                    TOPIC_STATUS, json.dumps(status).encode()
                )
                await asyncio.sleep(STATUS_INTERVAL_S)
        except asyncio.CancelledError:
            raise
