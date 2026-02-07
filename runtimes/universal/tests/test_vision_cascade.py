"""Tests for vision cascade and auto-learning functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from models.streaming_vision import (
    StreamingVisionDetector,
    StreamingConfig,
    CascadeConfig,
    StreamSession,
    FrameResult,
)
from models.vision_base import DetectionBox, DetectionResult
from vision_training.replay_buffer import ReplayBuffer


@pytest.fixture
def replay_buffer():
    """Create a test replay buffer."""
    return ReplayBuffer(max_size=100)


@pytest.fixture
def mock_detection_result():
    """Create a mock detection result factory."""
    def _create(confidence: float, class_name: str = "person"):
        return DetectionResult(
            confidence=confidence,
            inference_time_ms=10.0,
            model_name="test",
            boxes=[
                DetectionBox(
                    x1=0.1, y1=0.1, x2=0.3, y2=0.3,
                    class_name=class_name,
                    class_id=0,
                    confidence=confidence,
                )
            ],
        )
    return _create


@pytest.fixture
def mock_model_loader(mock_detection_result):
    """Create a mock model loader."""
    async def loader(model_id: str):
        model = MagicMock()
        # Secondary model returns higher confidence
        boost = 0.2 if "m" in model_id or "l" in model_id else 0.0
        
        async def detect(image, confidence_threshold=0.5):
            # Return detection with confidence based on model
            base_conf = 0.55  # Just above escalation threshold
            conf = min(1.0, base_conf + boost)
            return mock_detection_result(conf)
        
        model.detect = detect
        return model
    
    return loader


class TestStreamingVisionDetector:
    """Tests for StreamingVisionDetector."""
    
    @pytest.mark.asyncio
    async def test_start_session_without_cascade(self, mock_model_loader):
        """Test starting a session without cascade."""
        detector = StreamingVisionDetector(model_loader=mock_model_loader)
        
        session = await detector.start_session(
            model_id="yolov8n",
            config=StreamingConfig(confidence_threshold=0.7),
        )
        
        assert session.session_id is not None
        assert session.model_id == "yolov8n"
        assert session.secondary_model_id is None
    
    @pytest.mark.asyncio
    async def test_start_session_with_cascade(self, mock_model_loader):
        """Test starting a session with cascade configuration."""
        detector = StreamingVisionDetector(model_loader=mock_model_loader)
        
        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cascade=CascadeConfig(
                secondary_model_id="yolov8m",
                feedback_to_primary=True,
            ),
        )
        
        session = await detector.start_session(
            model_id="yolov8n",
            config=config,
        )
        
        assert session.secondary_model_id == "yolov8m"
    
    @pytest.mark.asyncio
    async def test_process_frame_high_confidence(self, mock_model_loader, mock_detection_result):
        """Test processing a frame with high confidence detection."""
        # Mock model that returns high confidence
        async def high_conf_loader(model_id):
            model = MagicMock()
            model.detect = AsyncMock(return_value=mock_detection_result(0.9))
            return model
        
        detector = StreamingVisionDetector(model_loader=high_conf_loader)
        
        session = await detector.start_session(
            model_id="yolov8n",
            config=StreamingConfig(confidence_threshold=0.7),
        )
        
        result = await detector.process_frame(
            session_id=session.session_id,
            image=b"test_image_data",
        )
        
        assert result.status == "action"
        assert result.confidence >= 0.7
        assert len(result.detections) > 0
    
    @pytest.mark.asyncio
    async def test_process_frame_escalation(self, mock_detection_result, replay_buffer):
        """Test that uncertain detections escalate to secondary model."""
        # Primary returns low confidence, secondary returns high
        call_count = {"primary": 0, "secondary": 0}
        
        async def cascading_loader(model_id):
            model = MagicMock()
            
            async def detect(image, confidence_threshold=0.5):
                if "n" in model_id:  # Primary
                    call_count["primary"] += 1
                    return mock_detection_result(0.55)  # Mid confidence
                else:  # Secondary
                    call_count["secondary"] += 1
                    return mock_detection_result(0.85)  # High confidence
            
            model.detect = detect
            return model
        
        detector = StreamingVisionDetector(
            model_loader=cascading_loader,
            replay_buffer=replay_buffer,
        )
        
        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cascade=CascadeConfig(
                secondary_model_id="yolov8m",
                feedback_to_primary=True,
            ),
        )
        
        session = await detector.start_session(model_id="yolov8n", config=config)
        
        result = await detector.process_frame(
            session_id=session.session_id,
            image=b"test_image",
        )
        
        assert call_count["primary"] == 1
        assert call_count["secondary"] == 1
        assert result.status == "action"
        assert result.escalated_to == "yolov8m"
        assert result.added_to_replay is True
    
    @pytest.mark.asyncio
    async def test_process_frame_review_queue(self, mock_detection_result):
        """Test that failed escalations go to review queue."""
        # Both models return low confidence
        async def low_conf_loader(model_id):
            model = MagicMock()
            model.detect = AsyncMock(return_value=mock_detection_result(0.55))
            return model
        
        detector = StreamingVisionDetector(model_loader=low_conf_loader)
        
        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cascade=CascadeConfig(
                secondary_model_id="yolov8m",
            ),
        )
        
        session = await detector.start_session(model_id="yolov8n", config=config)
        
        result = await detector.process_frame(
            session_id=session.session_id,
            image=b"test_image",
        )
        
        assert result.status == "review"
        assert result.image_id is not None
    
    @pytest.mark.asyncio
    async def test_cooldown_suppression(self, mock_detection_result):
        """Test that cooldown suppresses repeated actions."""
        async def high_conf_loader(model_id):
            model = MagicMock()
            model.detect = AsyncMock(return_value=mock_detection_result(0.9))
            return model
        
        detector = StreamingVisionDetector(model_loader=high_conf_loader)
        
        config = StreamingConfig(
            confidence_threshold=0.7,
            cooldown_seconds=1.0,
        )
        
        session = await detector.start_session(model_id="yolov8n", config=config)
        
        # First frame triggers action
        result1 = await detector.process_frame(session.session_id, b"image1")
        assert result1.status == "action"
        assert not result1.suppressed
        
        # Second frame is suppressed due to cooldown
        result2 = await detector.process_frame(session.session_id, b"image2")
        assert result2.status == "ok"
        assert result2.suppressed
    
    @pytest.mark.asyncio
    async def test_submit_correction(self, mock_model_loader, replay_buffer):
        """Test submitting a correction."""
        detector = StreamingVisionDetector(
            model_loader=mock_model_loader,
            replay_buffer=replay_buffer,
        )
        
        session = await detector.start_session(model_id="yolov8n")
        
        success = await detector.submit_correction(
            session_id=session.session_id,
            image_id="test_img_1",
            corrected_class="truck",
            original_confidence=0.4,
        )
        
        assert success is True
        assert len(replay_buffer) == 1
        
        sample = replay_buffer.get("test_img_1")
        assert sample is not None
        assert sample.label == "truck"
        assert sample.source == "correction"
    
    @pytest.mark.asyncio
    async def test_session_statistics(self, mock_detection_result, replay_buffer):
        """Test that session statistics are tracked correctly."""
        call_count = 0
        
        async def varying_loader(model_id):
            model = MagicMock()
            
            async def detect(image, confidence_threshold=0.5):
                nonlocal call_count
                call_count += 1
                # Alternate between high and mid confidence
                conf = 0.9 if call_count % 2 == 0 else 0.55
                return mock_detection_result(conf)
            
            model.detect = detect
            return model
        
        detector = StreamingVisionDetector(
            model_loader=varying_loader,
            replay_buffer=replay_buffer,
        )
        
        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cooldown_seconds=0,  # Disable for test
            cascade=CascadeConfig(
                secondary_model_id="yolov8m",
                feedback_to_primary=True,
            ),
        )
        
        session = await detector.start_session(model_id="yolov8n", config=config)
        
        # Process several frames
        for i in range(10):
            await detector.process_frame(session.session_id, f"frame_{i}".encode())
        
        stats = await detector.stop_session(session.session_id)
        
        assert stats["frames_processed"] == 10
        assert stats["model_id"] == "yolov8n"
        assert stats["secondary_model_id"] == "yolov8m"
        assert "escalations_to_secondary" in stats
        assert "secondary_success_rate" in stats


class TestReplayBuffer:
    """Tests for ReplayBuffer."""
    
    def test_add_correction(self, replay_buffer):
        """Test adding a correction."""
        sample = replay_buffer.add_correction(
            image_id="img_1",
            image_path="/path/to/image.jpg",
            corrected_label="truck",
            original_confidence=0.4,
        )
        
        assert sample.source == "correction"
        assert sample.priority == 2.0  # Corrections have higher priority
        assert len(replay_buffer) == 1
    
    def test_add_low_confidence(self, replay_buffer):
        """Test adding a low-confidence sample."""
        sample = replay_buffer.add_low_confidence(
            image_id="img_2",
            image_path="/path/to/image.jpg",
            predicted_label="car",
            confidence=0.55,
        )
        
        assert sample.source == "low_confidence"
        assert abs(sample.priority - 0.45) < 0.01  # 1 - confidence (with floating point tolerance)
        assert len(replay_buffer) == 1
    
    def test_priority_sampling(self, replay_buffer):
        """Test that priority sampling favors corrections."""
        # Add some corrections and low-confidence samples
        for i in range(5):
            replay_buffer.add_correction(
                image_id=f"correction_{i}",
                image_path=f"/path/{i}.jpg",
                corrected_label="car",
            )
        
        for i in range(5):
            replay_buffer.add_low_confidence(
                image_id=f"low_conf_{i}",
                image_path=f"/path/{i}.jpg",
                predicted_label="car",
                confidence=0.6,
            )
        
        # Sample and check that corrections appear more frequently
        samples = replay_buffer.sample(100)
        correction_count = sum(1 for s in samples if s.source == "correction")
        
        # Corrections should appear more often due to higher priority
        assert correction_count > len(samples) // 2
    
    def test_stratified_sampling(self, replay_buffer):
        """Test stratified sampling by source."""
        for i in range(10):
            replay_buffer.add_correction(
                image_id=f"correction_{i}",
                image_path=f"/path/{i}.jpg",
                corrected_label="car",
            )
        
        for i in range(10):
            replay_buffer.add_low_confidence(
                image_id=f"low_conf_{i}",
                image_path=f"/path/{i}.jpg",
                predicted_label="car",
                confidence=0.6,
            )
        
        samples = replay_buffer.sample_stratified(10, correction_ratio=0.5)
        
        corrections = [s for s in samples if s.source == "correction"]
        low_conf = [s for s in samples if s.source == "low_confidence"]
        
        assert len(corrections) == 5
        assert len(low_conf) == 5
    
    def test_max_size_eviction(self):
        """Test that oldest low-priority items are evicted at max size."""
        buffer = ReplayBuffer(max_size=5)
        
        # Add low-priority items
        for i in range(5):
            buffer.add_low_confidence(
                image_id=f"low_{i}",
                image_path=f"/path/{i}.jpg",
                predicted_label="car",
                confidence=0.6,
            )
        
        assert len(buffer) == 5
        
        # Add a high-priority correction - should evict lowest priority
        buffer.add_correction(
            image_id="correction_1",
            image_path="/path/c.jpg",
            corrected_label="truck",
        )
        
        assert len(buffer) == 5
        assert buffer.get("correction_1") is not None
    
    def test_get_stats(self, replay_buffer):
        """Test getting buffer statistics."""
        replay_buffer.add_correction("c1", "/p/1.jpg", "car")
        replay_buffer.add_correction("c2", "/p/2.jpg", "truck")
        replay_buffer.add_low_confidence("l1", "/p/3.jpg", "car", 0.5)
        
        stats = replay_buffer.get_stats()
        
        assert stats["size"] == 3
        assert stats["by_source"]["correction"] == 2
        assert stats["by_source"]["low_confidence"] == 1


class TestCascadeConfig:
    """Tests for CascadeConfig."""
    
    def test_default_config(self):
        """Test default cascade configuration."""
        config = CascadeConfig()
        
        assert config.secondary_model_id is None
        assert config.feedback_to_primary is True
        assert config.save_uncertain_images is True
    
    def test_custom_config(self):
        """Test custom cascade configuration."""
        config = CascadeConfig(
            secondary_model_id="yolov8l",
            feedback_to_primary=False,
            save_uncertain_images=False,
        )
        
        assert config.secondary_model_id == "yolov8l"
        assert config.feedback_to_primary is False
        assert config.save_uncertain_images is False


class TestTrainingTrigger:
    """Tests for training trigger callback."""
    
    @pytest.mark.asyncio
    async def test_training_trigger_callback(self, mock_detection_result, replay_buffer):
        """Test that training trigger is called when buffer fills."""
        trigger_called = False
        trigger_size = 0
        
        async def on_trigger(buffer_size: int):
            nonlocal trigger_called, trigger_size
            trigger_called = True
            trigger_size = buffer_size
        
        # Mock model that returns mid confidence (triggers escalation)
        async def mid_conf_loader(model_id):
            model = MagicMock()
            # Secondary succeeds
            conf = 0.85 if "m" in model_id else 0.55
            model.detect = AsyncMock(return_value=mock_detection_result(conf))
            return model
        
        detector = StreamingVisionDetector(
            model_loader=mid_conf_loader,
            replay_buffer=replay_buffer,
        )
        detector.set_training_trigger(on_trigger)
        
        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cooldown_seconds=0,
            cascade=CascadeConfig(
                secondary_model_id="yolov8m",
                feedback_to_primary=True,
            ),
        )
        
        session = await detector.start_session(model_id="yolov8n", config=config)
        
        # Process frames to fill buffer
        for i in range(5):
            await detector.process_frame(session.session_id, f"frame_{i}".encode())
        
        assert trigger_called is True
        assert trigger_size >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
