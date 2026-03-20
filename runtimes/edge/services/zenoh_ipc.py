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
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> bool:
        """Open Zenoh session and start subscriber + heartbeat tasks.

        Returns True if started successfully, False on failure (graceful degradation).
        """
        if not ZENOH_ENABLED:
            logger.info("Zenoh IPC disabled (ZENOH_ENABLED=false)")
            return False

        try:
            import zenoh
        except ImportError:
            logger.warning(
                "eclipse-zenoh package not installed, Zenoh IPC unavailable"
            )
            return False

        try:
            config = zenoh.Config()
            config.insert_json5(
                "connect/endpoints",
                json.dumps([ZENOH_ENDPOINT]),
            )
            config.insert_json5("scouting/multicast/enabled", "false")

            self._session = zenoh.open(config)
            logger.info("Zenoh session open (endpoint=%s)", ZENOH_ENDPOINT)
        except Exception:
            logger.warning(
                "Failed to connect to Zenoh at %s — continuing HTTP-only",
                ZENOH_ENDPOINT,
                exc_info=True,
            )
            return False

        self._tasks.append(asyncio.create_task(self._subscribe_requests()))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        return True

    async def stop(self):
        """Cancel background tasks and close the Zenoh session."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                logger.warning("Error closing Zenoh session", exc_info=True)
            self._session = None
            logger.info("Zenoh session closed")

    # ------------------------------------------------------------------
    # Request subscriber
    # ------------------------------------------------------------------

    async def _subscribe_requests(self):
        """Subscribe to local/llm/request and dispatch inference."""
        subscriber = await self._session.declare_subscriber(TOPIC_REQUEST)
        logger.info("Subscribed to %s", TOPIC_REQUEST)

        try:
            async for sample in subscriber:
                try:
                    payload = json.loads(bytes(sample.payload))
                    await self._handle_request(payload)
                except Exception:
                    logger.error(
                        "Error handling Zenoh request", exc_info=True
                    )
        except asyncio.CancelledError:
            await subscriber.undeclare()
            raise

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
                "error": str(exc),
                "inference_time_ms": inference_ms,
                "timestamp_ms": int(time.time() * 1000),
            }
            logger.error("Inference failed for request %s: %s", request_id, exc)

        await self._session.put(
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
                await self._session.put(
                    TOPIC_STATUS, json.dumps(status).encode()
                )
                await asyncio.sleep(STATUS_INTERVAL_S)
        except asyncio.CancelledError:
            raise
