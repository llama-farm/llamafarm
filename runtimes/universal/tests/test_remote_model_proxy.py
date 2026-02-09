"""Tests for RemoteModelProxy.

Verifies:
- Same DetectionModel interface as local models
- HTTP round-trip with mocked server
- Timeout handling
- Health check on load
- Error handling when remote is down
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.remote_model_proxy import RemoteModelProxy
from models.vision_base import DetectionBox, DetectionResult


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock()
    return client


class TestRemoteModelProxy:
    """Tests for RemoteModelProxy."""

    @pytest.mark.asyncio
    async def test_implements_detection_interface(self):
        """RemoteModelProxy is a DetectionModel."""
        from models.vision_base import DetectionModel

        proxy = RemoteModelProxy(
            model_id="remote:gpu-1/yolov8x",
            remote_url="http://localhost:8080",
            remote_model="yolov8x",
        )

        assert isinstance(proxy, DetectionModel)
        assert proxy.task == "detection"

    @pytest.mark.asyncio
    async def test_detect_parses_response(self):
        """detect() correctly parses remote JSON response."""
        proxy = RemoteModelProxy(
            model_id="remote:test/yolov8x",
            remote_url="http://test:8080",
            remote_model="yolov8x",
        )

        # Mock the client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "detections": [
                {
                    "box": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
                    "class_name": "person",
                    "class_id": 0,
                    "confidence": 0.95,
                },
                {
                    "box": {"x1": 50, "y1": 50, "x2": 150, "y2": 150},
                    "class_name": "car",
                    "class_id": 1,
                    "confidence": 0.85,
                },
            ],
            "model": "yolov8x",
            "inference_time_ms": 45.0,
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        proxy._client = mock_client
        proxy._loaded = True

        result = await proxy.detect(b"fake_image_bytes")

        assert isinstance(result, DetectionResult)
        assert len(result.boxes) == 2
        assert result.boxes[0].class_name == "person"
        assert result.boxes[0].confidence == 0.95
        assert result.boxes[0].x1 == 100
        assert result.boxes[1].class_name == "car"
        assert result.confidence == 0.95  # max confidence

    @pytest.mark.asyncio
    async def test_detect_sends_correct_payload(self):
        """detect() sends base64 image and correct params."""
        proxy = RemoteModelProxy(
            model_id="remote:test/yolov8x",
            remote_url="http://test:8080",
            remote_model="yolov8x",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"detections": []}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        proxy._client = mock_client
        proxy._loaded = True

        await proxy.detect(
            b"test_image",
            confidence_threshold=0.3,
            classes=["person", "car"],
        )

        # Verify the call
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://test:8080/v1/vision/detect"

        payload = call_args[1]["json"]
        assert payload["model"] == "yolov8x"
        assert payload["confidence_threshold"] == 0.3
        assert payload["classes"] == ["person", "car"]
        assert "image" in payload  # Base64 encoded

    @pytest.mark.asyncio
    async def test_detect_handles_remote_error(self):
        """detect() returns empty result on remote error."""
        proxy = RemoteModelProxy(
            model_id="remote:test/yolov8x",
            remote_url="http://unreachable:8080",
            remote_model="yolov8x",
        )

        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Connection refused")
        proxy._client = mock_client
        proxy._loaded = True

        result = await proxy.detect(b"test_image")

        assert isinstance(result, DetectionResult)
        assert len(result.boxes) == 0
        assert result.confidence == 0.0
        assert proxy.is_healthy is False  # Marked unhealthy

    @pytest.mark.asyncio
    async def test_detect_without_load_raises(self):
        """detect() raises if load() wasn't called."""
        proxy = RemoteModelProxy(
            model_id="remote:test/yolov8x",
            remote_url="http://test:8080",
            remote_model="yolov8x",
        )

        with pytest.raises(RuntimeError, match="not loaded"):
            await proxy.detect(b"test_image")

    @pytest.mark.asyncio
    async def test_health_check_on_load(self):
        """load() performs health check."""
        proxy = RemoteModelProxy(
            model_id="remote:test/yolov8x",
            remote_url="http://test:8080",
            remote_model="yolov8x",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            MockClient.return_value = mock_instance

            await proxy.load()

        assert proxy._loaded is True
        assert proxy.is_healthy is True

    @pytest.mark.asyncio
    async def test_unhealthy_on_failed_health_check(self):
        """load() marks unhealthy if health check fails."""
        proxy = RemoteModelProxy(
            model_id="remote:test/yolov8x",
            remote_url="http://unreachable:8080",
            remote_model="yolov8x",
        )

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = Exception("Connection refused")
            MockClient.return_value = mock_instance

            await proxy.load()

        assert proxy._loaded is True  # Still marked loaded
        assert proxy.is_healthy is False

    @pytest.mark.asyncio
    async def test_unload_closes_client(self):
        """unload() closes the HTTP client."""
        proxy = RemoteModelProxy(
            model_id="remote:test/yolov8x",
            remote_url="http://test:8080",
            remote_model="yolov8x",
        )

        mock_client = AsyncMock()
        proxy._client = mock_client
        proxy._loaded = True

        await proxy.unload()

        mock_client.aclose.assert_called_once()
        assert proxy._loaded is False

    def test_model_info(self):
        """get_model_info() includes remote details."""
        proxy = RemoteModelProxy(
            model_id="remote:gpu-1/yolov8x",
            remote_url="http://gpu1:8080",
            remote_model="yolov8x",
            node_id="gpu-server-1",
        )

        info = proxy.get_model_info()

        assert info["remote_url"] == "http://gpu1:8080"
        assert info["remote_model"] == "yolov8x"
        assert info["node_id"] == "gpu-server-1"


class TestRemoteModelInCascade:
    """Tests that RemoteModelProxy works in the cascade chain."""

    @pytest.mark.asyncio
    async def test_cascade_with_remote_model(self):
        """Remote model works alongside local models in cascade."""
        from models.streaming_vision import (
            StreamingVisionDetector,
            StreamingConfig,
            CascadeConfig,
        )
        from models.vision_base import DetectionResult, DetectionBox

        # Track which models were called
        called = set()

        async def mock_loader(model_id):
            if model_id.startswith("remote:"):
                # Create a RemoteModelProxy-like mock
                model = MagicMock()

                async def remote_detect(image, confidence_threshold=0.5):
                    called.add(model_id)
                    return DetectionResult(
                        confidence=0.88,
                        inference_time_ms=50.0,
                        model_name=model_id,
                        boxes=[DetectionBox(
                            x1=0.1, y1=0.1, x2=0.3, y2=0.3,
                            class_name="person", class_id=0,
                            confidence=0.88,
                        )],
                    )

                model.detect = remote_detect
                return model
            else:
                model = MagicMock()

                async def local_detect(image, confidence_threshold=0.5):
                    called.add(model_id)
                    return DetectionResult(
                        confidence=0.55,
                        inference_time_ms=5.0,
                        model_name=model_id,
                        boxes=[DetectionBox(
                            x1=0.1, y1=0.1, x2=0.3, y2=0.3,
                            class_name="person", class_id=0,
                            confidence=0.55,
                        )],
                    )

                model.detect = local_detect
                return model

        detector = StreamingVisionDetector(model_loader=mock_loader)

        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cooldown_seconds=0,
            cascade=CascadeConfig(
                cascade_chain=["yolov8m", "remote:gpu-1/yolov8x"],
            ),
        )

        session = await detector.start_session(model_id="yolov8n", config=config)
        result = await detector.process_frame(session.session_id, b"test_image")

        # Primary was uncertain, yolov8m was uncertain, remote resolved it
        assert "yolov8n" in called
        assert "yolov8m" in called
        assert "remote:gpu-1/yolov8x" in called
        assert result.status == "action"
        assert result.cascade_resolved_by == "remote:gpu-1/yolov8x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
