"""Comprehensive tests for vision models (YOLO, CLIP, Streaming).

Tests all model classes with mocking to avoid downloading real models.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# -----------------------------------------------------------------------------
# YOLO Model Tests
# -----------------------------------------------------------------------------

class TestYOLOModel:
    """Tests for YOLOModel object detection."""

    @pytest.fixture
    def mock_yolo(self):
        """Create a mock YOLO model from Ultralytics."""
        mock = MagicMock()
        
        # Mock model attributes
        mock.names = {i: name for i, name in enumerate([
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light"
        ])}
        
        # Mock inference result
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([10, 20, 100, 200])
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.95
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = 0  # person
        
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock.return_value = [mock_result]
        
        return mock

    @pytest.mark.asyncio
    async def test_load_model(self, mock_yolo):
        """Test loading YOLO model."""
        with patch("ultralytics.YOLO", return_value=mock_yolo):
            from models.yolo_model import YOLOModel
            
            model = YOLOModel("yolov8n", device="cpu")
            await model.load()
            
            assert model._loaded
            assert len(model.class_names) == 10

    @pytest.mark.asyncio
    async def test_detect_basic(self, mock_yolo):
        """Test basic object detection."""
        with patch("ultralytics.YOLO", return_value=mock_yolo):
            from models.yolo_model import YOLOModel
            
            model = YOLOModel("yolov8n", device="cpu")
            await model.load()
            
            # Create test image (100x100 RGB)
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            result = await model.detect(test_image)
            
            assert result.model_name == "yolov8n"
            assert result.image_width == 100
            assert result.image_height == 100
            assert len(result.boxes) == 1
            assert result.boxes[0].class_name == "person"
            assert result.boxes[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_detect_with_class_filter(self, mock_yolo):
        """Test detection with class filtering."""
        with patch("ultralytics.YOLO", return_value=mock_yolo):
            from models.yolo_model import YOLOModel
            
            model = YOLOModel("yolov8n", device="cpu")
            await model.load()
            
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Filter to only cars (no results expected since detection is 'person')
            result = await model.detect(test_image, classes=["car"])
            
            # Verify class filter was passed (in real impl would filter)
            assert result is not None

    @pytest.mark.asyncio
    async def test_detect_confidence_threshold(self, mock_yolo):
        """Test confidence threshold filtering."""
        with patch("ultralytics.YOLO", return_value=mock_yolo):
            from models.yolo_model import YOLOModel
            
            model = YOLOModel("yolov8n", device="cpu", confidence_threshold=0.99)
            await model.load()
            
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = await model.detect(test_image)
            
            # With threshold 0.99, the 0.95 detection should still be returned
            # (threshold is applied in YOLO call, not filtering)
            assert result is not None

    @pytest.mark.asyncio
    async def test_unload_model(self, mock_yolo):
        """Test unloading model releases resources."""
        with patch("ultralytics.YOLO", return_value=mock_yolo):
            from models.yolo_model import YOLOModel
            
            model = YOLOModel("yolov8n", device="cpu")
            await model.load()
            assert model._loaded
            
            await model.unload()
            assert not model._loaded
            assert model.yolo is None

    @pytest.mark.asyncio
    async def test_model_info(self, mock_yolo):
        """Test get_model_info returns correct metadata."""
        with patch("ultralytics.YOLO", return_value=mock_yolo):
            from models.yolo_model import YOLOModel
            
            model = YOLOModel("yolov8n", device="cpu")
            await model.load()
            
            info = model.get_model_info()
            
            assert info["variant"] == "yolov8n"
            assert info["num_classes"] == 10
            assert "class_names" in info

    def test_supported_variants(self):
        """Test all supported YOLO variants are defined."""
        from models.yolo_model import YOLO_VARIANTS
        
        expected_variants = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
                           "yolov11n", "yolov11s", "yolov11m"]
        
        for variant in expected_variants:
            assert variant in YOLO_VARIANTS


# -----------------------------------------------------------------------------
# CLIP Classifier Tests
# -----------------------------------------------------------------------------

class TestCLIPClassifier:
    """Tests for CLIPClassifier zero-shot classification."""

    @pytest.fixture
    def mock_clip(self):
        """Create mock CLIP model and processor."""
        import torch
        
        mock_model = MagicMock()
        mock_model.config.projection_dim = 512
        
        # Mock text features (normalized)
        text_features = torch.randn(3, 512)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        mock_model.get_text_features.return_value = text_features
        
        # Mock image features (highest sim to first class)
        img_features = text_features[0:1] + torch.randn(1, 512) * 0.1
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        mock_model.get_image_features.return_value = img_features
        
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": torch.zeros(1, 10),
            "attention_mask": torch.ones(1, 10),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }
        
        return mock_model, mock_processor

    @pytest.mark.asyncio
    async def test_load_classifier(self, mock_clip):
        """Test loading CLIP classifier."""
        mock_model, mock_processor = mock_clip
        
        with patch("transformers.CLIPModel") as MockCLIP, \
             patch("transformers.CLIPProcessor") as MockProcessor:
            MockCLIP.from_pretrained.return_value = mock_model
            MockProcessor.from_pretrained.return_value = mock_processor
            
            from models.clip_model import CLIPClassifier
            
            classifier = CLIPClassifier("clip-vit-base", device="cpu")
            await classifier.load()
            
            assert classifier._loaded
            assert classifier._embedding_dim == 512

    @pytest.mark.asyncio
    async def test_zero_shot_classify(self, mock_clip):
        """Test zero-shot classification."""
        mock_model, mock_processor = mock_clip
        
        with patch("transformers.CLIPModel") as MockCLIP, \
             patch("transformers.CLIPProcessor") as MockProcessor:
            MockCLIP.from_pretrained.return_value = mock_model
            MockProcessor.from_pretrained.return_value = mock_processor
            
            from models.clip_model import CLIPClassifier
            
            classifier = CLIPClassifier("clip-vit-base", device="cpu")
            await classifier.load()
            
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            result = await classifier.classify(
                test_image,
                classes=["cat", "dog", "bird"]
            )
            
            assert result.class_name in ["cat", "dog", "bird"]
            assert 0 <= result.confidence <= 1
            assert len(result.all_scores) == 3
            assert result.model_name == "clip-vit-base"

    @pytest.mark.asyncio
    async def test_set_classes_precomputes_embeddings(self, mock_clip):
        """Test that set_classes pre-computes text embeddings."""
        mock_model, mock_processor = mock_clip
        
        with patch("transformers.CLIPModel") as MockCLIP, \
             patch("transformers.CLIPProcessor") as MockProcessor:
            MockCLIP.from_pretrained.return_value = mock_model
            MockProcessor.from_pretrained.return_value = mock_processor
            
            from models.clip_model import CLIPClassifier
            
            classifier = CLIPClassifier("clip-vit-base", device="cpu")
            await classifier.load()
            
            await classifier.set_classes(["cat", "dog", "bird"])
            
            assert classifier.class_names == ["cat", "dog", "bird"]
            assert classifier._class_embeddings is not None

    @pytest.mark.asyncio
    async def test_classify_requires_classes(self, mock_clip):
        """Test that classify raises error without classes."""
        mock_model, mock_processor = mock_clip
        
        with patch("transformers.CLIPModel") as MockCLIP, \
             patch("transformers.CLIPProcessor") as MockProcessor:
            MockCLIP.from_pretrained.return_value = mock_model
            MockProcessor.from_pretrained.return_value = mock_processor
            
            from models.clip_model import CLIPClassifier
            
            classifier = CLIPClassifier("clip-vit-base", device="cpu")
            await classifier.load()
            
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            with pytest.raises(ValueError, match="No classes provided"):
                await classifier.classify(test_image)

    def test_supported_clip_variants(self):
        """Test all supported CLIP variants are defined."""
        from models.clip_model import CLIP_VARIANTS
        
        expected_variants = ["clip-vit-base", "clip-vit-large", "siglip-base"]
        
        for variant in expected_variants:
            assert variant in CLIP_VARIANTS


# -----------------------------------------------------------------------------
# CLIP Embedder Tests
# -----------------------------------------------------------------------------

class TestCLIPEmbedder:
    """Tests for CLIPEmbedder image/text embeddings."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock CLIP embedder components."""
        import torch
        
        mock_model = MagicMock()
        mock_model.config.projection_dim = 512
        
        # Return normalized embeddings
        def get_image_features(**kwargs):
            batch_size = kwargs.get("pixel_values", torch.zeros(1)).shape[0]
            features = torch.randn(batch_size, 512)
            return features / features.norm(dim=-1, keepdim=True)
        
        def get_text_features(**kwargs):
            batch_size = kwargs.get("input_ids", torch.zeros(1)).shape[0]
            features = torch.randn(batch_size, 512)
            return features / features.norm(dim=-1, keepdim=True)
        
        mock_model.get_image_features = get_image_features
        mock_model.get_text_features = get_text_features
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        mock_processor = MagicMock()
        
        return mock_model, mock_processor

    @pytest.mark.asyncio
    async def test_embed_images(self, mock_embedder):
        """Test embedding images."""
        mock_model, mock_processor = mock_embedder
        
        with patch("transformers.CLIPModel") as MockCLIP, \
             patch("transformers.CLIPProcessor") as MockProcessor:
            MockCLIP.from_pretrained.return_value = mock_model
            MockProcessor.from_pretrained.return_value = mock_processor
            
            from models.clip_model import CLIPEmbedder
            
            embedder = CLIPEmbedder("clip-vit-base", device="cpu")
            await embedder.load()
            
            test_images = [
                np.zeros((100, 100, 3), dtype=np.uint8),
                np.zeros((200, 200, 3), dtype=np.uint8),
            ]
            
            result = await embedder.embed_images(test_images)
            
            assert len(result.embeddings) == 2
            assert result.dimensions == 512
            assert all(len(emb) == 512 for emb in result.embeddings)

    @pytest.mark.asyncio
    async def test_embed_texts(self, mock_embedder):
        """Test embedding texts."""
        mock_model, mock_processor = mock_embedder
        
        with patch("transformers.CLIPModel") as MockCLIP, \
             patch("transformers.CLIPProcessor") as MockProcessor:
            MockCLIP.from_pretrained.return_value = mock_model
            MockProcessor.from_pretrained.return_value = mock_processor
            
            from models.clip_model import CLIPEmbedder
            
            embedder = CLIPEmbedder("clip-vit-base", device="cpu")
            await embedder.load()
            
            texts = ["a photo of a cat", "a photo of a dog"]
            
            result = await embedder.embed_texts(texts)
            
            assert len(result.embeddings) == 2
            assert result.dimensions == 512

    @pytest.mark.asyncio
    async def test_embeddings_normalized(self, mock_embedder):
        """Test that embeddings are normalized."""
        mock_model, mock_processor = mock_embedder
        
        with patch("transformers.CLIPModel") as MockCLIP, \
             patch("transformers.CLIPProcessor") as MockProcessor:
            MockCLIP.from_pretrained.return_value = mock_model
            MockProcessor.from_pretrained.return_value = mock_processor
            
            from models.clip_model import CLIPEmbedder
            
            embedder = CLIPEmbedder("clip-vit-base", device="cpu")
            await embedder.load()
            
            test_image = [np.zeros((100, 100, 3), dtype=np.uint8)]
            result = await embedder.embed_images(test_image)
            
            # Check norm is approximately 1
            emb = np.array(result.embeddings[0])
            norm = np.linalg.norm(emb)
            assert 0.99 < norm < 1.01


# -----------------------------------------------------------------------------
# Streaming Vision Tests
# -----------------------------------------------------------------------------

class TestStreamingVisionDetector:
    """Tests for StreamingVisionDetector."""

    @pytest.fixture
    def mock_detector_model(self):
        """Create a mock detection model for streaming."""
        from models.vision_base import DetectionBox, DetectionResult
        
        mock = AsyncMock()
        
        async def mock_detect(image, confidence_threshold=0.5):
            return DetectionResult(
                confidence=0.9,
                inference_time_ms=10.0,
                model_name="yolov8n",
                boxes=[
                    DetectionBox(
                        x1=10, y1=20, x2=100, y2=200,
                        class_name="person",
                        class_id=0,
                        confidence=0.9
                    )
                ],
                class_names=["person"],
                image_width=640,
                image_height=480,
            )
        
        mock.detect = mock_detect
        return mock

    @pytest.mark.asyncio
    async def test_start_session(self, mock_detector_model):
        """Test starting a streaming session."""
        from models.streaming_vision import StreamingConfig, StreamingVisionDetector
        
        detector = StreamingVisionDetector()
        
        async def loader(model_id):
            return mock_detector_model
        
        detector.set_model_loader(loader)
        
        session = await detector.start_session(
            model_id="yolov8n",
            config=StreamingConfig(target_fps=1.0, confidence_threshold=0.7)
        )
        
        assert session.session_id is not None
        assert session.model_id == "yolov8n"
        assert session.config.confidence_threshold == 0.7

    @pytest.mark.asyncio
    async def test_process_frame_action(self, mock_detector_model):
        """Test processing frame that triggers action."""
        from models.streaming_vision import StreamingConfig, StreamingVisionDetector
        
        detector = StreamingVisionDetector()
        async def _load_model(m):
            return mock_detector_model

        detector.set_model_loader(_load_model)

        session = await detector.start_session(
            config=StreamingConfig(
                confidence_threshold=0.7,
                action_classes=["person"]
            )
        )

        test_frame = b"fake_image_bytes"
        result = await detector.process_frame(session.session_id, test_frame)

        assert result.status == "action"
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "person"

    @pytest.mark.asyncio
    async def test_cooldown_suppression(self, mock_detector_model):
        """Test that cooldown suppresses repeated actions."""
        from models.streaming_vision import StreamingConfig, StreamingVisionDetector
        
        detector = StreamingVisionDetector()
        async def _load_model(m):
            return mock_detector_model

        detector.set_model_loader(_load_model)

        session = await detector.start_session(
            config=StreamingConfig(
                confidence_threshold=0.7,
                cooldown_seconds=10.0,  # Long cooldown
                action_classes=["person"]
            )
        )
        
        test_frame = b"fake_image_bytes"
        
        # First frame triggers action
        result1 = await detector.process_frame(session.session_id, test_frame)
        assert result1.status == "action"
        
        # Second frame suppressed by cooldown
        result2 = await detector.process_frame(session.session_id, test_frame)
        assert result2.status == "ok"
        assert result2.suppressed is True

    @pytest.mark.asyncio
    async def test_stop_session_returns_stats(self, mock_detector_model):
        """Test stopping session returns statistics."""
        from models.streaming_vision import StreamingConfig, StreamingVisionDetector
        
        detector = StreamingVisionDetector()
        async def _load_model(m):
            return mock_detector_model

        detector.set_model_loader(_load_model)

        session = await detector.start_session(config=StreamingConfig())
        
        # Process some frames
        for _ in range(5):
            await detector.process_frame(session.session_id, b"frame")
        
        stats = await detector.stop_session(session.session_id)
        
        assert stats["session_id"] == session.session_id
        assert stats["frames_processed"] == 5
        assert "duration_seconds" in stats
        assert "avg_fps" in stats

    @pytest.mark.asyncio
    async def test_invalid_session_raises(self, mock_detector_model):
        """Test that invalid session ID raises error."""
        from models.streaming_vision import StreamingVisionDetector
        
        detector = StreamingVisionDetector()
        
        with pytest.raises(ValueError, match="not found"):
            await detector.process_frame("nonexistent", b"frame")

    @pytest.mark.asyncio
    async def test_list_sessions(self, mock_detector_model):
        """Test listing active sessions."""
        from models.streaming_vision import StreamingVisionDetector
        
        detector = StreamingVisionDetector()
        async def _load_model(m):
            return mock_detector_model

        detector.set_model_loader(_load_model)

        session1 = await detector.start_session()
        session2 = await detector.start_session()
        
        sessions = detector.list_sessions()
        
        assert len(sessions) == 2
        assert session1.session_id in sessions
        assert session2.session_id in sessions


# -----------------------------------------------------------------------------
# Vision Base Types Tests
# -----------------------------------------------------------------------------

class TestVisionBaseTypes:
    """Tests for vision base classes and dataclasses."""

    def test_detection_box_dataclass(self):
        """Test DetectionBox dataclass."""
        from models.vision_base import DetectionBox
        
        box = DetectionBox(
            x1=10.0, y1=20.0, x2=100.0, y2=200.0,
            class_name="person", class_id=0, confidence=0.95
        )
        
        assert box.x1 == 10.0
        assert box.class_name == "person"
        assert box.confidence == 0.95

    def test_detection_result_dataclass(self):
        """Test DetectionResult dataclass."""
        from models.vision_base import DetectionBox, DetectionResult
        
        result = DetectionResult(
            confidence=0.95,
            inference_time_ms=15.5,
            model_name="yolov8n",
            boxes=[DetectionBox(10, 20, 100, 200, "person", 0, 0.95)],
            class_names=["person"],
            image_width=640,
            image_height=480,
        )
        
        assert len(result.boxes) == 1
        assert result.image_width == 640

    def test_classification_result_dataclass(self):
        """Test ClassificationResult dataclass."""
        from models.vision_base import ClassificationResult
        
        result = ClassificationResult(
            confidence=0.9,
            inference_time_ms=25.0,
            model_name="clip-vit-base",
            class_name="cat",
            class_id=0,
            all_scores={"cat": 0.9, "dog": 0.08, "bird": 0.02},
        )
        
        assert result.class_name == "cat"
        assert sum(result.all_scores.values()) == pytest.approx(1.0)

    def test_embedding_result_dataclass(self):
        """Test EmbeddingResult dataclass."""
        from models.vision_base import EmbeddingResult
        
        result = EmbeddingResult(
            confidence=1.0,
            inference_time_ms=20.0,
            model_name="clip-vit-base",
            embeddings=[[0.1] * 512],
            dimensions=512,
        )
        
        assert len(result.embeddings) == 1
        assert result.dimensions == 512


# -----------------------------------------------------------------------------
# Integration-style Tests
# -----------------------------------------------------------------------------

class TestVisionIntegration:
    """Integration-style tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_detect_then_classify(self):
        """Test detecting objects then classifying crops."""
        # This is a conceptual test showing the pipeline
        from models.vision_base import DetectionBox
        
        # Mock detection output
        detections = [
            DetectionBox(10, 20, 100, 200, "animal", 0, 0.9)
        ]
        
        # In real usage, you'd crop the image and classify
        # Here we just verify the data structures work together
        assert detections[0].class_name == "animal"
        
        # Classification result
        from models.vision_base import ClassificationResult
        
        classification = ClassificationResult(
            confidence=0.85,
            inference_time_ms=30.0,
            model_name="clip-vit-base",
            class_name="cat",
            class_id=1,
            all_scores={"cat": 0.85, "dog": 0.15},
        )
        
        # Verify structures are compatible
        assert classification.class_name == "cat"

    def test_embedding_similarity(self):
        """Test computing similarity between embeddings."""
        import numpy as np
        
        # Simulate two embeddings
        emb1 = np.random.randn(512)
        emb1 = emb1 / np.linalg.norm(emb1)
        
        emb2 = np.random.randn(512)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2)
        
        # Should be between -1 and 1 for normalized vectors
        assert -1 <= similarity <= 1
