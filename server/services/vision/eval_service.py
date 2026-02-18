"""Eval service — proxy for vision model evaluation endpoints."""

from typing import Any

from server.services.universal_runtime_service import UniversalRuntimeService


class VisionEvalService:
    """Proxy to runtime eval endpoints."""

    @staticmethod
    async def start_eval(payload: dict[str, Any]) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/eval", json=payload)

    @staticmethod
    async def get_eval_status(job_id: str) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "GET", f"/v1/vision/eval/{job_id}")

    @staticmethod
    async def start_compare(payload: dict[str, Any]) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/eval/compare", json=payload)

    @staticmethod
    async def leaderboard(dataset: str | None = None, limit: int = 20) -> Any:
        qs = f"?limit={limit}"
        if dataset:
            qs += f"&dataset={dataset}"
        return await UniversalRuntimeService._make_request(
            "GET", f"/v1/vision/eval/leaderboard{qs}")

    @staticmethod
    async def model_history(model_name: str, limit: int = 50) -> Any:
        return await UniversalRuntimeService._make_request(
            "GET", f"/v1/vision/eval/leaderboard/{model_name}?limit={limit}")
