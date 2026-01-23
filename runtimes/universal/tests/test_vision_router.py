"""Tests for Vision router endpoints (OCR and document extraction)."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_ocr_model():
    """Create a mock OCR model."""
    model = MagicMock()

    # Mock OCR result
    class MockOCRResult:
        def __init__(self, text, confidence, boxes=None):
            self.text = text
            self.confidence = confidence
            self.boxes = boxes or []

    class MockBox:
        def __init__(self):
            self.x1 = 10.0
            self.y1 = 20.0
            self.x2 = 100.0
            self.y2 = 50.0
            self.text = "Hello"
            self.confidence = 0.95

    async def mock_recognize(images, languages=None, return_boxes=False):
        results = []
        for i, _ in enumerate(images):
            boxes = [MockBox()] if return_boxes else None
            results.append(MockOCRResult(f"Extracted text from image {i}", 0.95, boxes))
        return results

    model.recognize = mock_recognize
    return model


@pytest.fixture
def mock_document_model():
    """Create a mock document extraction model."""
    model = MagicMock()

    class MockField:
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.confidence = 0.9
            self.bbox = [10, 20, 100, 50]

    class MockDocumentResult:
        def __init__(self, idx, task):
            self.confidence = 0.92
            self.text = "Document content" if task == "extraction" else None
            self.fields = (
                [MockField("total", "100.00")] if task == "extraction" else None
            )
            self.answer = "The total is $100" if task == "vqa" else None
            self.classification = "invoice" if task == "classification" else None
            self.classification_scores = (
                {"invoice": 0.9, "receipt": 0.1} if task == "classification" else None
            )

    async def mock_extract(images, prompts=None):
        return [MockDocumentResult(i, "extraction") for i, _ in enumerate(images)]

    model.extract = mock_extract
    return model


@pytest.fixture
def test_app(mock_ocr_model, mock_document_model):
    """Create a test FastAPI app with the vision router."""
    from routers.vision import (
        router,
        set_document_loader,
        set_file_image_getter,
        set_ocr_loader,
    )

    app = FastAPI()
    app.include_router(router)

    # Set up mock model loaders
    async def mock_load_ocr(backend="surya", languages=None):
        return mock_ocr_model

    async def mock_load_document(model_id, task="extraction"):
        return mock_document_model

    def mock_get_file_images(file_id):
        # Return None for nonexistent file_id, or test images
        if file_id == "test_file_123":
            return ["base64_encoded_image_1"]
        return None

    set_ocr_loader(mock_load_ocr)
    set_document_loader(mock_load_document)
    set_file_image_getter(mock_get_file_images)

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestOCREndpoint:
    """Test /v1/ocr endpoint."""

    def test_ocr_with_base64_images(self, client):
        """Test POST /v1/ocr with base64 images."""
        response = client.post(
            "/v1/ocr",
            json={
                "model": "surya",
                "images": ["base64_encoded_image_1", "base64_encoded_image_2"],
                "languages": ["en"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert "text" in data["data"][0]
        assert "confidence" in data["data"][0]

    def test_ocr_with_file_id(self, client):
        """Test POST /v1/ocr with file_id."""
        response = client.post(
            "/v1/ocr",
            json={
                "model": "surya",
                "file_id": "test_file_123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1

    def test_ocr_with_invalid_file_id(self, client):
        """Test POST /v1/ocr with invalid file_id returns 400."""
        response = client.post(
            "/v1/ocr",
            json={
                "model": "surya",
                "file_id": "nonexistent_file",
            },
        )

        assert response.status_code == 400
        assert "No images found" in response.json()["detail"]

    def test_ocr_requires_images_or_file_id(self, client):
        """Test POST /v1/ocr requires either images or file_id."""
        response = client.post(
            "/v1/ocr",
            json={
                "model": "surya",
            },
        )

        assert response.status_code == 400
        assert "Either 'images' or 'file_id'" in response.json()["detail"]

    def test_ocr_with_boxes(self, client):
        """Test POST /v1/ocr with return_boxes=True."""
        response = client.post(
            "/v1/ocr",
            json={
                "model": "surya",
                "images": ["base64_image"],
                "return_boxes": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "boxes" in data["data"][0]


class TestDocumentExtractEndpoint:
    """Test /v1/documents/extract endpoint."""

    def test_document_extract_with_base64(self, client):
        """Test POST /v1/documents/extract with base64 images."""
        response = client.post(
            "/v1/documents/extract",
            json={
                "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
                "images": ["base64_document_image"],
                "task": "extraction",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["task"] == "extraction"
        assert "data" in data
        assert "fields" in data["data"][0] or "text" in data["data"][0]

    def test_document_extract_with_file_id(self, client):
        """Test POST /v1/documents/extract with file_id."""
        response = client.post(
            "/v1/documents/extract",
            json={
                "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
                "file_id": "test_file_123",
            },
        )

        assert response.status_code == 200

    def test_document_extract_requires_images_or_file_id(self, client):
        """Test POST /v1/documents/extract requires either images or file_id."""
        response = client.post(
            "/v1/documents/extract",
            json={
                "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
            },
        )

        assert response.status_code == 400


class TestRouterInitialization:
    """Test router initialization and dependency injection."""

    def test_ocr_loader_not_set_raises_error(self):
        """Test that calling OCR endpoint without setting loader raises error."""
        from routers.vision import router, set_ocr_loader

        # Reset the loader
        set_ocr_loader(None)

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/v1/ocr",
            json={
                "model": "surya",
                "images": ["base64_image"],
            },
        )

        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"].lower()

    def test_document_loader_not_set_raises_error(self):
        """Test that calling document endpoint without setting loader raises error."""
        from routers.vision import router, set_document_loader

        # Reset the loader
        set_document_loader(None)

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/v1/documents/extract",
            json={
                "model": "test-model",
                "images": ["base64_image"],
            },
        )

        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"].lower()
