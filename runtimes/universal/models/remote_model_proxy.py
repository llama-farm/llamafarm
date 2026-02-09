"""Remote model proxy for federated cascade.

Implements the same DetectionModel interface as YOLOModel but calls
a remote LlamaFarm instance's /v1/vision/detect endpoint. The cascade
doesn't know or care if a model is local or remote.

Standalone: configured manually with a peer URL.
Mesh: Atmosphere auto-discovers and registers proxies via gossip.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .vision_base import DetectionBox, DetectionModel, DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class RemoteModelConfig:
    """Configuration for a remote model proxy."""

    remote_url: str             # Base URL of the remote LlamaFarm instance
    remote_model: str           # Model ID on the remote instance
    timeout: float = 30.0       # Request timeout
    node_id: str = "remote"     # Atmosphere node ID or descriptive name


class RemoteModelProxy(DetectionModel):
    """Calls /v1/vision/detect on a remote LlamaFarm instance.

    Implements the same interface as YOLOModel. The cascade
    doesn't know or care if a model is local or remote.

    Example:
        ```python
        proxy = RemoteModelProxy(
            model_id="remote:gpu-server/yolov8x",
            remote_url="http://192.168.1.100:8080",
            remote_model="yolov8x",
        )
        await proxy.load()  # Verify remote is healthy

        result = await proxy.detect(image_bytes)
        # result is a DetectionResult, same as local models
        ```
    """

    def __init__(
        self,
        model_id: str,
        remote_url: str,
        remote_model: str,
        timeout: float = 30.0,
        node_id: str = "remote",
        device: str = "remote",
    ):
        super().__init__(model_id=model_id, device=device)
        self.remote_url = remote_url.rstrip("/")
        self.remote_model = remote_model
        self.timeout = timeout
        self.node_id = node_id
        self._client = None
        self._healthy = False

    async def load(self) -> None:
        """Verify remote instance is healthy."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for remote model proxy: pip install httpx")

        self._client = httpx.AsyncClient(timeout=self.timeout)

        # Health check
        try:
            response = await self._client.get(f"{self.remote_url}/health")
            self._healthy = response.status_code == 200
            self._loaded = True
            logger.info(f"Remote model {self.model_id} at {self.remote_url}: healthy={self._healthy}")
        except Exception as e:
            logger.warning(f"Remote model {self.model_id} health check failed: {e}")
            self._healthy = False
            self._loaded = True  # Mark loaded so cascade can try and handle errors

    async def unload(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._loaded = False

    async def detect(
        self,
        image: bytes | np.ndarray,
        confidence_threshold: float | None = None,
        classes: list[str] | None = None,
    ) -> DetectionResult:
        """Detect objects by calling the remote LlamaFarm instance.

        Sends the image as base64 to the remote /v1/vision/detect endpoint.
        Converts the response back into a DetectionResult.

        Args:
            image: Image as bytes or numpy array
            confidence_threshold: Minimum confidence
            classes: Filter to specific classes

        Returns:
            DetectionResult from the remote model
        """
        if self._client is None:
            raise RuntimeError("Remote model proxy not loaded. Call load() first.")

        start_time = time.time()

        # Convert image to bytes if needed
        if isinstance(image, np.ndarray):
            from PIL import Image
            import io
            pil_img = Image.fromarray(image)
            if pil_img.mode == "RGBA":
                pil_img = pil_img.convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            image_bytes = buf.getvalue()
        else:
            image_bytes = image

        # Build request
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload: dict[str, Any] = {
            "image": image_b64,
            "model": self.remote_model,
        }
        if confidence_threshold is not None:
            payload["confidence_threshold"] = confidence_threshold
        if classes:
            payload["classes"] = classes

        try:
            response = await self._client.post(
                f"{self.remote_url}/v1/vision/detect",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Remote detection failed ({self.remote_url}): {e}")
            self._healthy = False
            return DetectionResult(
                confidence=0.0,
                inference_time_ms=(time.time() - start_time) * 1000,
                model_name=self.model_id,
                boxes=[],
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # Parse response into DetectionResult
        boxes = []
        for det in data.get("detections", []):
            box_data = det.get("box", {})
            boxes.append(DetectionBox(
                x1=box_data.get("x1", 0),
                y1=box_data.get("y1", 0),
                x2=box_data.get("x2", 0),
                y2=box_data.get("y2", 0),
                class_name=det.get("class_name", ""),
                class_id=det.get("class_id", 0),
                confidence=det.get("confidence", 0),
            ))

        max_conf = max((b.confidence for b in boxes), default=0.0)

        return DetectionResult(
            confidence=max_conf,
            inference_time_ms=elapsed_ms,
            model_name=self.model_id,
            boxes=boxes,
        )

    async def infer(self, image: bytes | np.ndarray, **kwargs) -> DetectionResult:
        """Run remote detection inference."""
        return await self.detect(
            image,
            confidence_threshold=kwargs.get("confidence_threshold"),
            classes=kwargs.get("classes"),
        )

    @property
    def is_healthy(self) -> bool:
        """Whether the remote instance is reachable."""
        return self._healthy

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the remote model."""
        info = super().get_model_info()
        info.update({
            "remote_url": self.remote_url,
            "remote_model": self.remote_model,
            "node_id": self.node_id,
            "healthy": self._healthy,
        })
        return info
