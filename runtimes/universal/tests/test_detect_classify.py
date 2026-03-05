"""Tests for the detect+classify combo endpoint."""

import io
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from routers.vision.detect_classify import (
    DetectClassifyRequest,
    detect_and_classify,
    set_detect_classify_loaders,
)

# =============================================================================
# Fixtures / helpers
# =============================================================================

@dataclass
class FakeBox:
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    class_id: int
    confidence: float


@dataclass
class FakeDetResult:
    boxes: list
    confidence: float = 0.9


@dataclass
class FakeClsResult:
    class_name: str
    confidence: float
    all_scores: dict


def _make_red_png_b64() -> str:
    """Create a tiny valid PNG as base64."""
    import base64

    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_rgba_png_b64() -> str:
    """Create a tiny RGBA PNG as base64."""
    import base64

    from PIL import Image
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# =============================================================================
# Tests
# =============================================================================

class TestDetectClassifyRequest:
    def test_defaults(self):
        r = DetectClassifyRequest(image="abc", classes=["cat", "dog"])
        assert r.confidence_threshold == 0.5
        assert r.top_k == 3
        assert r.min_crop_px == 16

    def test_empty_classes_rejected(self):
        """Pydantic should accept empty list but endpoint rejects it."""
        r = DetectClassifyRequest(image="abc", classes=[])
        assert r.classes == []


@pytest.mark.asyncio
class TestDetectAndClassify:
    async def test_no_loaders(self):
        set_detect_classify_loaders(None, None)
        req = DetectClassifyRequest(image=_make_red_png_b64(), classes=["cat"])
        from fastapi import HTTPException
        with pytest.raises(HTTPException, match="not initialized"):
            await detect_and_classify(req)

    async def test_empty_classes(self):
        set_detect_classify_loaders(AsyncMock(), AsyncMock())
        req = DetectClassifyRequest(image=_make_red_png_b64(), classes=[])
        from fastapi import HTTPException
        with pytest.raises(HTTPException, match="Classes required"):
            await detect_and_classify(req)

    async def test_no_detections(self):
        det_model = AsyncMock()
        det_model.detect = AsyncMock(return_value=FakeDetResult(boxes=[]))
        load_det = AsyncMock(return_value=det_model)
        load_cls = AsyncMock()
        set_detect_classify_loaders(load_det, load_cls)

        req = DetectClassifyRequest(image=_make_red_png_b64(), classes=["cat", "dog"])
        resp = await detect_and_classify(req)
        assert resp.total_detections == 0
        assert resp.results == []

    async def test_detect_and_classify_flow(self):
        boxes = [FakeBox(x1=10, y1=10, x2=60, y2=60, class_name="obj", class_id=0, confidence=0.9)]
        det_model = AsyncMock()
        det_model.detect = AsyncMock(return_value=FakeDetResult(boxes=boxes))

        cls_model = AsyncMock()
        cls_model.classify = AsyncMock(return_value=FakeClsResult(
            class_name="cat", confidence=0.85, all_scores={"cat": 0.85, "dog": 0.15}
        ))

        set_detect_classify_loaders(AsyncMock(return_value=det_model), AsyncMock(return_value=cls_model))
        req = DetectClassifyRequest(image=_make_red_png_b64(), classes=["cat", "dog"])
        resp = await detect_and_classify(req)

        assert resp.total_detections == 1
        assert resp.classified_count == 1
        assert resp.results[0].classification == "cat"
        assert resp.results[0].detection_class == "obj"

    async def test_tiny_crop_skipped(self):
        """Crops smaller than min_crop_px should be skipped."""
        boxes = [FakeBox(x1=10, y1=10, x2=15, y2=15, class_name="obj", class_id=0, confidence=0.9)]
        det_model = AsyncMock()
        det_model.detect = AsyncMock(return_value=FakeDetResult(boxes=boxes))

        set_detect_classify_loaders(AsyncMock(return_value=det_model), AsyncMock())
        req = DetectClassifyRequest(image=_make_red_png_b64(), classes=["x"], min_crop_px=16)
        resp = await detect_and_classify(req)
        assert resp.total_detections == 1
        assert resp.classified_count == 0

    async def test_rgba_image_handled(self):
        """RGBA images should be converted to RGB before JPEG crop encoding."""
        boxes = [FakeBox(x1=10, y1=10, x2=60, y2=60, class_name="obj", class_id=0, confidence=0.9)]
        det_model = AsyncMock()
        det_model.detect = AsyncMock(return_value=FakeDetResult(boxes=boxes))

        cls_model = AsyncMock()
        cls_model.classify = AsyncMock(return_value=FakeClsResult(
            class_name="a", confidence=0.9, all_scores={"a": 0.9}
        ))

        set_detect_classify_loaders(AsyncMock(return_value=det_model), AsyncMock(return_value=cls_model))
        req = DetectClassifyRequest(image=_make_rgba_png_b64(), classes=["a"])
        resp = await detect_and_classify(req)
        assert resp.classified_count == 1  # Should not crash on RGBA
