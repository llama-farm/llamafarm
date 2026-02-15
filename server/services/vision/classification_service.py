"""Classification service — proxy to runtime /v1/vision/classify."""

from typing import Any

from server.services.universal_runtime_service import UniversalRuntimeService


class VisionClassificationService:
    """Proxy to runtime classification endpoint."""

    @staticmethod
    async def classify(
        image: str,
        classes: list[str],
        model: str = "clip-vit-base",
        top_k: int = 5,
    ) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/classify",
            json={"image": image, "model": model, "classes": classes, "top_k": top_k},
        )
