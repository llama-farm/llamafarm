"""Tests for vision API endpoints (detection, classification, embedding, streaming, training).

Tests all vision routers with mocked model loaders.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Test image (1x1 red pixel PNG)
TEST_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


# -----------------------------------------------------------------------------
# Detection Router Tests
# -----------------------------------------------------------------------------

class TestDetectionRouter:
    """Tests for /v1/vision/detect endpoint."""

    @pytest.fixture
    def detection_client(self):
        """Create test client with mock detection loader."""
        from routers.vision.detection import router, set_detection_loader
        from models.vision_base import DetectionResult, DetectionBox
        
        app = FastAPI()
        app.include_router(router)
        
        async def mock_loader(model_id):
            mock_model = AsyncMock()
            
            async def mock_detect(image, confidence_threshold=None, classes=None):
                return DetectionResult(
                    confidence=0.95,
                    inference_time_ms=15.0,
                    model_name=model_id,
                    boxes=[
                        DetectionBox(10, 20, 100, 200, "person", 0, 0.95),
                        DetectionBox(150, 50, 300, 250, "car", 2, 0.88),
                    ],
                    class_names=["person", "car"],
                    image_width=640,
                    image_height=480,
                )
            
            mock_model.detect = mock_detect
            return mock_model
        
        set_detection_loader(mock_loader)
        return TestClient(app)

    def test_detect_with_base64(self, detection_client):
        """Test detection with base64 image."""
        response = detection_client.post(
            "/v1/vision/detect",
            json={
                "model": "yolov8n",
                "image": TEST_IMAGE_B64,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "detection"
        assert data["model"] == "yolov8n"
        assert len(data["detections"]) == 2
        assert data["detections"][0]["class"] == "person"
        assert data["detections"][0]["confidence"] == 0.95

    def test_detect_with_confidence_threshold(self, detection_client):
        """Test detection with confidence threshold."""
        response = detection_client.post(
            "/v1/vision/detect",
            json={
                "model": "yolov8n",
                "image": TEST_IMAGE_B64,
                "confidence_threshold": 0.9,
            }
        )
        
        assert response.status_code == 200

    def test_detect_with_class_filter(self, detection_client):
        """Test detection with class filter."""
        response = detection_client.post(
            "/v1/vision/detect",
            json={
                "model": "yolov8n",
                "image": TEST_IMAGE_B64,
                "classes": ["person"],
            }
        )
        
        assert response.status_code == 200

    def test_detect_missing_image(self, detection_client):
        """Test detection fails without image."""
        response = detection_client.post(
            "/v1/vision/detect",
            json={"model": "yolov8n"}
        )
        
        # Should fail validation
        assert response.status_code == 422 or response.status_code == 400

    def test_detect_loader_not_set(self):
        """Test error when loader not configured."""
        from routers.vision.detection import router, set_detection_loader
        
        set_detection_loader(None)
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        response = client.post(
            "/v1/vision/detect",
            json={"model": "yolov8n", "image": TEST_IMAGE_B64}
        )
        
        assert response.status_code == 500


# -----------------------------------------------------------------------------
# Classification Router Tests
# -----------------------------------------------------------------------------

class TestClassificationRouter:
    """Tests for /v1/vision/classify endpoint."""

    @pytest.fixture
    def classification_client(self):
        """Create test client with mock classification loader."""
        from routers.vision.classification import router, set_classification_loader
        from models.vision_base import ClassificationResult
        
        app = FastAPI()
        app.include_router(router)
        
        async def mock_loader(model_id):
            mock_model = AsyncMock()
            
            async def mock_classify(image, classes=None, top_k=5):
                return ClassificationResult(
                    confidence=0.92,
                    inference_time_ms=25.0,
                    model_name=model_id,
                    class_name=classes[0] if classes else "unknown",
                    class_id=0,
                    all_scores={c: 1.0/len(classes) for c in (classes or ["unknown"])},
                )
            
            mock_model.classify = mock_classify
            return mock_model
        
        set_classification_loader(mock_loader)
        return TestClient(app)

    def test_classify_with_classes(self, classification_client):
        """Test classification with provided classes."""
        response = classification_client.post(
            "/v1/vision/classify",
            json={
                "model": "clip-vit-base",
                "image": TEST_IMAGE_B64,
                "classes": ["cat", "dog", "bird"],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "classification"
        assert data["model"] == "clip-vit-base"
        assert data["class"] == "cat"
        assert "scores" in data

    def test_classify_top_k(self, classification_client):
        """Test classification with top_k parameter."""
        response = classification_client.post(
            "/v1/vision/classify",
            json={
                "model": "clip-vit-base",
                "image": TEST_IMAGE_B64,
                "classes": ["a", "b", "c", "d", "e"],
                "top_k": 3,
            }
        )
        
        assert response.status_code == 200

    def test_classify_missing_classes(self, classification_client):
        """Test classification fails without classes."""
        response = classification_client.post(
            "/v1/vision/classify",
            json={
                "model": "clip-vit-base",
                "image": TEST_IMAGE_B64,
            }
        )
        
        # Should fail - classes required for zero-shot
        assert response.status_code in (400, 422, 500)


# -----------------------------------------------------------------------------
# Embedding Router Tests
# -----------------------------------------------------------------------------

class TestEmbeddingRouter:
    """Tests for /v1/vision/embed endpoint."""

    @pytest.fixture
    def embedding_client(self):
        """Create test client with mock embedding loader."""
        from routers.vision.embedding import router, set_embedding_loader
        from models.vision_base import EmbeddingResult
        
        app = FastAPI()
        app.include_router(router)
        
        async def mock_loader(model_id):
            mock_model = AsyncMock()
            
            async def mock_embed_images(images):
                return EmbeddingResult(
                    confidence=1.0,
                    inference_time_ms=20.0,
                    model_name=model_id,
                    embeddings=[[0.1] * 512 for _ in images],
                    dimensions=512,
                )
            
            async def mock_embed_texts(texts):
                return EmbeddingResult(
                    confidence=1.0,
                    inference_time_ms=15.0,
                    model_name=model_id,
                    embeddings=[[0.2] * 512 for _ in texts],
                    dimensions=512,
                )
            
            mock_model.embed_images = mock_embed_images
            mock_model.embed_texts = mock_embed_texts
            return mock_model
        
        set_embedding_loader(mock_loader)
        return TestClient(app)

    def test_embed_single_image(self, embedding_client):
        """Test embedding single image."""
        response = embedding_client.post(
            "/v1/vision/embed",
            json={
                "model": "clip-vit-base",
                "images": [TEST_IMAGE_B64],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "embedding"
        assert len(data["embeddings"]) == 1
        assert data["dimensions"] == 512

    def test_embed_multiple_images(self, embedding_client):
        """Test embedding multiple images."""
        response = embedding_client.post(
            "/v1/vision/embed",
            json={
                "model": "clip-vit-base",
                "images": [TEST_IMAGE_B64, TEST_IMAGE_B64, TEST_IMAGE_B64],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["embeddings"]) == 3

    def test_embed_texts(self, embedding_client):
        """Test embedding texts."""
        response = embedding_client.post(
            "/v1/vision/embed",
            json={
                "model": "clip-vit-base",
                "texts": ["a photo of a cat", "a photo of a dog"],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["embeddings"]) == 2

    def test_embed_requires_input(self, embedding_client):
        """Test embedding fails without images or texts."""
        response = embedding_client.post(
            "/v1/vision/embed",
            json={"model": "clip-vit-base"}
        )
        
        assert response.status_code in (400, 422)


# -----------------------------------------------------------------------------
# Streaming Router Tests
# -----------------------------------------------------------------------------

class TestStreamingRouter:
    """Tests for /v1/vision/streaming/* endpoints."""

    @pytest.fixture
    def streaming_client(self):
        """Create test client with mock streaming detector."""
        from routers.vision.streaming import router
        from models.streaming_vision import (
            get_streaming_detector,
            set_streaming_model_loader,
            StreamSession,
            FrameResult,
        )
        from models.vision_base import DetectionBox, DetectionResult
        
        app = FastAPI()
        app.include_router(router)
        
        # Mock model loader
        async def mock_loader(model_id):
            mock_model = AsyncMock()
            
            async def mock_detect(image, confidence_threshold=0.5):
                return DetectionResult(
                    confidence=0.9,
                    inference_time_ms=10.0,
                    model_name=model_id,
                    boxes=[DetectionBox(10, 20, 100, 200, "person", 0, 0.9)],
                    class_names=["person"],
                    image_width=640,
                    image_height=480,
                )
            
            mock_model.detect = mock_detect
            return mock_model
        
        set_streaming_model_loader(mock_loader)
        return TestClient(app)

    def test_start_session(self, streaming_client):
        """Test starting a streaming session."""
        response = streaming_client.post(
            "/v1/vision/streaming/start",
            json={
                "model": "yolov8n",
                "target_fps": 1.0,
                "confidence_threshold": 0.7,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["model"] == "yolov8n"

    def test_process_frame(self, streaming_client):
        """Test processing a frame."""
        # Start session
        start_resp = streaming_client.post(
            "/v1/vision/streaming/start",
            json={"model": "yolov8n"}
        )
        session_id = start_resp.json()["session_id"]
        
        # Process frame
        response = streaming_client.post(
            f"/v1/vision/streaming/frame/{session_id}",
            json={"image": TEST_IMAGE_B64}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "action", "review")

    def test_stop_session(self, streaming_client):
        """Test stopping a session."""
        # Start session
        start_resp = streaming_client.post(
            "/v1/vision/streaming/start",
            json={"model": "yolov8n"}
        )
        session_id = start_resp.json()["session_id"]
        
        # Stop session
        response = streaming_client.post(
            f"/v1/vision/streaming/stop/{session_id}"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "frames_processed" in data
        assert "actions_triggered" in data

    def test_list_sessions(self, streaming_client):
        """Test listing active sessions."""
        response = streaming_client.get("/v1/vision/streaming/sessions")
        
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_invalid_session_frame(self, streaming_client):
        """Test processing frame for invalid session."""
        response = streaming_client.post(
            "/v1/vision/streaming/frame/nonexistent",
            json={"image": TEST_IMAGE_B64}
        )
        
        assert response.status_code in (400, 404)


# -----------------------------------------------------------------------------
# Training Router Tests
# -----------------------------------------------------------------------------

class TestTrainingRouter:
    """Tests for /v1/vision/training/* endpoints."""

    @pytest.fixture
    def training_client(self):
        """Create test client with mock training setup."""
        from routers.vision.training import router
        
        app = FastAPI()
        app.include_router(router)
        
        return TestClient(app)

    def test_get_training_status(self, training_client):
        """Test getting training status."""
        response = training_client.get("/v1/vision/training/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "active_jobs" in data

    def test_list_training_jobs(self, training_client):
        """Test listing training jobs."""
        response = training_client.get("/v1/vision/training/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data


# -----------------------------------------------------------------------------
# Combined Vision Router Tests
# -----------------------------------------------------------------------------

class TestCombinedVisionRouter:
    """Tests for combined vision router initialization."""

    def test_router_includes_all_subrouters(self):
        """Test that vision_router includes all expected routes."""
        from routers.vision import vision_router
        
        # Get all route paths
        routes = [r.path for r in vision_router.routes]
        
        # Check detection route exists
        assert any("/detect" in r for r in routes)
        # Check classification route exists
        assert any("/classify" in r for r in routes)
        # Check embedding route exists
        assert any("/embed" in r for r in routes)
        # Check streaming routes exist
        assert any("/streaming" in r for r in routes)

    def test_loader_setters_exist(self):
        """Test that all loader setters are exported."""
        from routers.vision import (
            set_detection_loader,
            set_classification_loader,
            set_embedding_loader,
        )
        
        # Verify they're callable
        assert callable(set_detection_loader)
        assert callable(set_classification_loader)
        assert callable(set_embedding_loader)


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------

class TestVisionErrorHandling:
    """Tests for error handling in vision endpoints."""

    def test_invalid_base64_image(self):
        """Test handling of invalid base64 image."""
        from routers.vision.detection import router, set_detection_loader
        
        app = FastAPI()
        app.include_router(router)
        
        async def mock_loader(model_id):
            mock_model = AsyncMock()
            mock_model.detect = AsyncMock(side_effect=Exception("Invalid image"))
            return mock_model
        
        set_detection_loader(mock_loader)
        client = TestClient(app)
        
        response = client.post(
            "/v1/vision/detect",
            json={"model": "yolov8n", "image": "not_valid_base64!!!"}
        )
        
        assert response.status_code == 500 or response.status_code == 400

    def test_model_loading_error(self):
        """Test handling of model loading errors."""
        from routers.vision.detection import router, set_detection_loader
        
        app = FastAPI()
        app.include_router(router)
        
        async def failing_loader(model_id):
            raise RuntimeError("Failed to load model")
        
        set_detection_loader(failing_loader)
        client = TestClient(app)
        
        response = client.post(
            "/v1/vision/detect",
            json={"model": "yolov8n", "image": TEST_IMAGE_B64}
        )
        
        assert response.status_code == 500


# -----------------------------------------------------------------------------
# Response Format Tests
# -----------------------------------------------------------------------------

class TestResponseFormats:
    """Tests for API response format compliance."""

    @pytest.fixture
    def full_client(self):
        """Create client with all routers mocked."""
        from routers.vision import vision_router
        from routers.vision.detection import set_detection_loader
        from routers.vision.classification import set_classification_loader
        from routers.vision.embedding import set_embedding_loader
        from models.vision_base import DetectionResult, DetectionBox, ClassificationResult, EmbeddingResult
        
        app = FastAPI()
        app.include_router(vision_router)
        
        async def mock_detect_loader(model_id):
            mock = AsyncMock()
            mock.detect = AsyncMock(return_value=DetectionResult(
                confidence=0.9, inference_time_ms=10, model_name=model_id,
                boxes=[DetectionBox(10, 20, 100, 200, "cat", 0, 0.9)],
                class_names=["cat"], image_width=640, image_height=480,
            ))
            return mock
        
        async def mock_classify_loader(model_id):
            mock = AsyncMock()
            mock.classify = AsyncMock(return_value=ClassificationResult(
                confidence=0.85, inference_time_ms=25, model_name=model_id,
                class_name="cat", class_id=0, all_scores={"cat": 0.85, "dog": 0.15},
            ))
            return mock
        
        async def mock_embed_loader(model_id):
            mock = AsyncMock()
            mock.embed_images = AsyncMock(return_value=EmbeddingResult(
                confidence=1.0, inference_time_ms=20, model_name=model_id,
                embeddings=[[0.1]*512], dimensions=512,
            ))
            return mock
        
        set_detection_loader(mock_detect_loader)
        set_classification_loader(mock_classify_loader)
        set_embedding_loader(mock_embed_loader)
        
        return TestClient(app)

    def test_detection_response_format(self, full_client):
        """Verify detection response matches expected schema."""
        response = full_client.post(
            "/v1/vision/detect",
            json={"model": "yolov8n", "image": TEST_IMAGE_B64}
        )
        
        data = response.json()
        
        # Required fields
        assert "object" in data
        assert "model" in data
        assert "detections" in data
        
        # Detection item format
        if data["detections"]:
            det = data["detections"][0]
            assert "class" in det
            assert "confidence" in det
            assert "box" in det
            assert all(k in det["box"] for k in ["x1", "y1", "x2", "y2"])

    def test_classification_response_format(self, full_client):
        """Verify classification response matches expected schema."""
        response = full_client.post(
            "/v1/vision/classify",
            json={"model": "clip", "image": TEST_IMAGE_B64, "classes": ["cat", "dog"]}
        )
        
        data = response.json()
        
        # Required fields
        assert "object" in data
        assert "model" in data
        assert "class" in data
        assert "confidence" in data
        assert "scores" in data

    def test_embedding_response_format(self, full_client):
        """Verify embedding response matches expected schema."""
        response = full_client.post(
            "/v1/vision/embed",
            json={"model": "clip", "images": [TEST_IMAGE_B64]}
        )
        
        data = response.json()
        
        # Required fields
        assert "object" in data
        assert "model" in data
        assert "embeddings" in data
        assert "dimensions" in data
        
        # Embedding is list of floats
        assert isinstance(data["embeddings"][0], list)
        assert isinstance(data["embeddings"][0][0], float)
