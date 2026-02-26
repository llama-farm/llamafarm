"""Drone vision service — proxy for runtime drone vision endpoints."""

from typing import Any

from server.services.universal_runtime_service import UniversalRuntimeService


class VisionDroneService:
    """Proxy to runtime drone vision endpoints."""

    @staticmethod
    async def geo_reference(payload: dict[str, Any]) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/drone/geo", json=payload,
        )

    @staticmethod
    async def list_export_profiles() -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/drone/export/profiles",
        )

    @staticmethod
    async def export_for_edge(payload: dict[str, Any]) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/drone/export", json=payload,
        )

    @staticmethod
    async def list_train_presets() -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/drone/train/presets",
        )

    @staticmethod
    async def list_sessions() -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/drone/sessions",
        )
