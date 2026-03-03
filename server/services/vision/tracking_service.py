"""Tracking service — proxy for vision object tracking endpoints."""

from typing import Any
from urllib.parse import quote

from server.services.universal_runtime_service import UniversalRuntimeService


class VisionTrackingService:
    """Proxy to runtime tracking endpoints."""

    @staticmethod
    async def start_tracking(payload: dict[str, Any]) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/track/start", json=payload)

    @staticmethod
    async def track_frame(payload: dict[str, Any]) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/track/frame", json=payload)

    @staticmethod
    async def stop_tracking(payload: dict[str, Any]) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/track/stop", json=payload)

    @staticmethod
    async def get_status(session_id: str) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "GET", f"/v1/vision/track/{quote(session_id, safe='')}")
