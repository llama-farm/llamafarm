"""Vision Pipeline Service - Proxies streaming, training, model, and federation requests."""

import logging
from typing import Any

from services.universal_runtime_service import UniversalRuntimeService

logger = logging.getLogger(__name__)


class VisionStreamingService:
    """Proxy for streaming vision endpoints on Universal Runtime."""

    @classmethod
    async def start_session(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/stream/start", json=request
        )

    @classmethod
    async def process_frame(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/stream/frame", json=request
        )

    @classmethod
    async def stop_session(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/stream/stop", json=request
        )

    @classmethod
    async def list_sessions(cls) -> Any:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/stream/sessions"
        )

    @classmethod
    async def get_session_stats(cls, session_id: str) -> Any:
        return await UniversalRuntimeService._make_request(
            "GET", f"/v1/vision/stream/sessions/{session_id}/stats"
        )

    @classmethod
    async def submit_correction(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/corrections", json=request
        )

    @classmethod
    async def get_replay_buffer_status(cls) -> Any:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/replay-buffer"
        )

    @classmethod
    async def clear_replay_buffer(cls) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/replay-buffer/clear"
        )


class VisionTrainingService:
    """Proxy for training and auto-training endpoints."""

    @classmethod
    async def start_training(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/train", json=request, timeout=600.0
        )

    @classmethod
    async def get_training_status(cls, job_id: str) -> Any:
        return await UniversalRuntimeService._make_request(
            "GET", f"/v1/vision/train/{job_id}"
        )

    @classmethod
    async def list_training_jobs(cls) -> Any:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/train"
        )

    @classmethod
    async def cancel_training(cls, job_id: str) -> Any:
        return await UniversalRuntimeService._make_request(
            "DELETE", f"/v1/vision/train/{job_id}"
        )

    @classmethod
    async def get_auto_train_status(cls) -> Any:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/auto-train/status"
        )

    @classmethod
    async def trigger_auto_training(cls) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/auto-train/trigger"
        )


class VisionModelService:
    """Proxy for vision model management endpoints."""

    @classmethod
    async def list_models(cls) -> Any:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/models"
        )

    @classmethod
    async def save_model(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/models/save", json=request
        )

    @classmethod
    async def load_model(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/models/load", json=request
        )

    @classmethod
    async def delete_model(cls, task: str, name: str) -> Any:
        return await UniversalRuntimeService._make_request(
            "DELETE", f"/v1/vision/models/{task}/{name}"
        )


class VisionSegmentationService:
    """Proxy for segmentation endpoint."""

    @classmethod
    async def segment(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/segment", json=request
        )


class VisionFederationService:
    """Proxy for federation endpoints."""

    @classmethod
    async def list_peers(cls) -> Any:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/federation/peers"
        )

    @classmethod
    async def register_peer(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/federation/peers", json=request
        )

    @classmethod
    async def remove_peer(cls, name: str) -> Any:
        return await UniversalRuntimeService._make_request(
            "DELETE", f"/v1/vision/federation/peers/{name}"
        )

    @classmethod
    async def federation_status(cls) -> Any:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/federation/status"
        )

    @classmethod
    async def escalate(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/federation/escalate", json=request
        )

    @classmethod
    async def list_packages(cls) -> Any:
        return await UniversalRuntimeService._make_request(
            "GET", "/v1/vision/federation/packages"
        )

    @classmethod
    async def create_package(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/federation/packages", json=request
        )

    @classmethod
    async def import_package(cls, request: dict[str, Any]) -> Any:
        return await UniversalRuntimeService._make_request(
            "POST", "/v1/vision/federation/packages/import", json=request
        )
