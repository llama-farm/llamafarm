"""Detection service — proxy to runtime /v1/vision/detect."""

from typing import Any

from server.services.universal_runtime_service import UniversalRuntimeService


class VisionDetectionService:
    """Proxy to runtime detection endpoint."""

    @staticmethod
    async def detect(
        image: str,
        model: str = "yolov8n",
        confidence_threshold: float = 0.5,
        classes: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image": image,
            "model": model,
            "confidence_threshold": confidence_threshold,
        }
        if classes:
            payload["classes"] = classes
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/detect", json=payload
        )
