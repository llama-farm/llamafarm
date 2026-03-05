"""Pipeline service — proxy for streaming, training, model management."""

from typing import Any

from server.services.universal_runtime_service import UniversalRuntimeService


class VisionPipelineService:
    """Proxy to runtime pipeline endpoints."""

    # --- Detect + Classify ---

    @staticmethod
    async def detect_classify(payload: dict[str, Any]) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/detect_classify", json=payload)

    # --- Streaming ---

    @staticmethod
    async def stream_sessions() -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/stream/sessions")

    @staticmethod
    async def stream_start(config: dict[str, Any]) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/stream/start", json=config)

    @staticmethod
    async def stream_frame(session_id: str, image: str) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/stream/frame",
            json={"session_id": session_id, "image": image})

    @staticmethod
    async def stream_stop(session_id: str) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/stream/stop", json={"session_id": session_id})

    # --- Training ---

    @staticmethod
    async def train(model: str, dataset: str, task: str = "detection",
                    config: dict | None = None,
                    base_model: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "dataset": dataset, "task": task}
        if config:
            payload["config"] = config
        if base_model:
            payload["base_model"] = base_model
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/train", json=payload)

    @staticmethod
    async def train_status(job_id: str) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "GET", f"/v1/vision/train/{job_id}")

    @staticmethod
    async def train_cancel(job_id: str) -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "DELETE", f"/v1/vision/train/{job_id}")

    # --- Models ---

    @staticmethod
    async def list_models() -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/models")

    @staticmethod
    async def save_model(model_id: str, name: str,
                         description: str = "") -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/models/save",
            json={"model_id": model_id, "name": name, "description": description})

    @staticmethod
    async def load_model(model_id: str,
                         device: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model_id": model_id}
        if device:
            payload["device"] = device
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/models/load", json=payload)

    @staticmethod
    async def export_model(model_id: str, format: str,
                           quantization: str = "fp16") -> dict[str, Any]:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/models/export",
            json={"model_id": model_id, "format": format, "quantization": quantization})
