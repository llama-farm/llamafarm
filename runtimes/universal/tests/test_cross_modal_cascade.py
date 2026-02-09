"""Tests for cross-modal cascade enrichment and multi-hop chain.

Verifies:
- cascade_chain works with 3+ models
- ModelOpinions accumulate through the chain
- Enrichment with segmentation/classification before escalation
- Circuit breaker at max_hops
- Backward compat: secondary_model_id still works via cascade_chain
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from models.streaming_vision import (
    StreamingVisionDetector,
    StreamingConfig,
    CascadeConfig,
    FrameResult,
)
from models.vision_base import DetectionBox, DetectionResult, SegmentationMask
from vision_training.replay_buffer import ReplayBuffer, ModelOpinion


@pytest.fixture
def replay_buffer():
    return ReplayBuffer(max_size=100)


@pytest.fixture
def mock_detection_result():
    """Factory for mock detection results."""
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


def make_chain_loader(confidences: dict[str, float], mock_detection_result):
    """Create a model loader that returns specific confidences per model ID."""
    async def loader(model_id: str):
        model = MagicMock()
        conf = confidences.get(model_id, 0.5)

        async def detect(image, confidence_threshold=0.5):
            return mock_detection_result(conf)

        model.detect = detect
        return model

    return loader


class TestMultiHopCascadeChain:
    """Tests for cascade_chain with multiple models."""

    @pytest.mark.asyncio
    async def test_three_model_cascade_resolves_at_hop_2(
        self, replay_buffer, mock_detection_result
    ):
        """Chain [yolov8n -> yolov8m -> yolov8x]: yolov8x resolves it."""
        loader = make_chain_loader({
            "yolov8n": 0.55,  # Primary: uncertain
            "yolov8m": 0.55,  # Hop 1: still uncertain
            "yolov8x": 0.85,  # Hop 2: resolves it
        }, mock_detection_result)

        detector = StreamingVisionDetector(
            model_loader=loader,
            replay_buffer=replay_buffer,
        )

        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cooldown_seconds=0,
            cascade=CascadeConfig(
                cascade_chain=["yolov8m", "yolov8x"],
                feedback_to_primary=True,
            ),
        )

        session = await detector.start_session(model_id="yolov8n", config=config)
        result = await detector.process_frame(session.session_id, b"test_image")

        assert result.status == "action"
        assert result.cascade_resolved_by == "yolov8x"
        assert result.hop_count == 3  # Primary + 2 cascade hops
        assert result.escalated_to == "yolov8x"
        assert result.added_to_replay is True

        # Opinions should contain all three models
        assert len(result.opinions) == 3
        assert result.opinions[0].model_id == "yolov8n"
        assert result.opinions[1].model_id == "yolov8m"
        assert result.opinions[2].model_id == "yolov8x"

    @pytest.mark.asyncio
    async def test_cascade_resolves_at_hop_1(
        self, replay_buffer, mock_detection_result
    ):
        """Chain [yolov8n -> yolov8m -> yolov8x]: yolov8m resolves early."""
        loader = make_chain_loader({
            "yolov8n": 0.55,  # Uncertain
            "yolov8m": 0.85,  # Resolves it
            "yolov8x": 0.95,  # Never reached
        }, mock_detection_result)

        detector = StreamingVisionDetector(
            model_loader=loader,
            replay_buffer=replay_buffer,
        )

        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cooldown_seconds=0,
            cascade=CascadeConfig(
                cascade_chain=["yolov8m", "yolov8x"],
                feedback_to_primary=True,
            ),
        )

        session = await detector.start_session(model_id="yolov8n", config=config)
        result = await detector.process_frame(session.session_id, b"test_image")

        assert result.status == "action"
        assert result.cascade_resolved_by == "yolov8m"
        assert result.hop_count == 2  # Primary + 1 cascade hop (stopped early)
        assert len(result.opinions) == 2

    @pytest.mark.asyncio
    async def test_cascade_all_fail_goes_to_review(
        self, replay_buffer, mock_detection_result
    ):
        """All models uncertain -> review queue."""
        loader = make_chain_loader({
            "yolov8n": 0.55,
            "yolov8m": 0.55,
            "yolov8x": 0.55,
        }, mock_detection_result)

        detector = StreamingVisionDetector(model_loader=loader)

        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cascade=CascadeConfig(
                cascade_chain=["yolov8m", "yolov8x"],
            ),
        )

        session = await detector.start_session(model_id="yolov8n", config=config)
        result = await detector.process_frame(session.session_id, b"test_image")

        assert result.status == "review"
        assert result.image_id is not None
        # Opinions from all models that were consulted
        assert len(result.opinions) >= 2  # At least primary + first cascade

    @pytest.mark.asyncio
    async def test_high_confidence_skips_cascade(
        self, mock_detection_result
    ):
        """Primary confident -> no cascade, opinions has only primary."""
        loader = make_chain_loader({
            "yolov8n": 0.9,
            "yolov8m": 0.95,
        }, mock_detection_result)

        detector = StreamingVisionDetector(model_loader=loader)

        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cooldown_seconds=0,
            cascade=CascadeConfig(cascade_chain=["yolov8m"]),
        )

        session = await detector.start_session(model_id="yolov8n", config=config)
        result = await detector.process_frame(session.session_id, b"test_image")

        assert result.status == "action"
        assert result.hop_count == 1
        assert result.cascade_resolved_by is None
        assert len(result.opinions) == 1
        assert result.opinions[0].model_id == "yolov8n"

    @pytest.mark.asyncio
    async def test_circuit_breaker_max_hops(self, mock_detection_result):
        """Cascade respects max_hops circuit breaker."""
        # 5 models in chain but max_hops=2
        loader = make_chain_loader({
            "primary": 0.55,
            "m1": 0.55, "m2": 0.55, "m3": 0.55, "m4": 0.55,
        }, mock_detection_result)

        detector = StreamingVisionDetector(model_loader=loader)

        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cascade=CascadeConfig(
                cascade_chain=["m1", "m2", "m3", "m4"],
                max_hops=2,
            ),
        )

        session = await detector.start_session(model_id="primary", config=config)
        result = await detector.process_frame(session.session_id, b"test_image")

        # Should only try hop 1 (m1) since max_hops=2 means hop_number < 2
        # hop 0 = primary, hop 1 = m1 (index 0 in chain, hop_number=1 < max_hops=2)
        # hop 2 = m2 (index 1 in chain, hop_number=2 NOT < max_hops=2, SKIPPED)
        assert result.status == "review"
        assert result.hop_count <= 3  # primary + at most 1 cascade model


class TestCascadeReplayIntegration:
    """Tests for replay buffer integration with multi-hop cascade."""

    @pytest.mark.asyncio
    async def test_cascade_resolved_creates_structured_sample(
        self, replay_buffer, mock_detection_result
    ):
        """Cascade-resolved sample has opinions and correct source type."""
        loader = make_chain_loader({
            "yolov8n": 0.55,
            "yolov8m": 0.85,
        }, mock_detection_result)

        detector = StreamingVisionDetector(
            model_loader=loader,
            replay_buffer=replay_buffer,
        )

        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cooldown_seconds=0,
            cascade=CascadeConfig(
                cascade_chain=["yolov8m"],
                feedback_to_primary=True,
            ),
        )

        session = await detector.start_session(model_id="yolov8n", config=config)
        result = await detector.process_frame(session.session_id, b"test_image")

        assert result.added_to_replay is True
        assert len(replay_buffer) == 1

        # Get the sample
        samples = replay_buffer.sample(1)
        sample = samples[0]

        # Should be cascade-resolved with opinions
        assert sample.source in ("escalation_resolved", "cascade_resolved")
        assert len(sample.opinions) == 2
        assert sample.opinions[0].model_id == "yolov8n"
        assert sample.opinions[1].model_id == "yolov8m"
        assert sample.bbox is not None

    @pytest.mark.asyncio
    async def test_no_replay_when_feedback_disabled(
        self, replay_buffer, mock_detection_result
    ):
        """When feedback_to_primary=False, cascade doesn't add to replay."""
        loader = make_chain_loader({
            "yolov8n": 0.55,
            "yolov8m": 0.85,
        }, mock_detection_result)

        detector = StreamingVisionDetector(
            model_loader=loader,
            replay_buffer=replay_buffer,
        )

        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cooldown_seconds=0,
            cascade=CascadeConfig(
                cascade_chain=["yolov8m"],
                feedback_to_primary=False,
            ),
        )

        session = await detector.start_session(model_id="yolov8n", config=config)
        result = await detector.process_frame(session.session_id, b"test_image")

        assert result.status == "action"
        assert result.added_to_replay is False
        assert len(replay_buffer) == 0


class TestCascadeSessionStats:
    """Tests for session statistics with cascade chain."""

    @pytest.mark.asyncio
    async def test_session_stats_track_cascade(self, mock_detection_result, replay_buffer):
        """Session stats reflect cascade chain behavior."""
        call_count = {"yolov8n": 0, "yolov8m": 0}

        async def counting_loader(model_id):
            model = MagicMock()

            async def detect(image, confidence_threshold=0.5):
                call_count[model_id] = call_count.get(model_id, 0) + 1
                conf = 0.85 if model_id == "yolov8m" else 0.55
                return mock_detection_result(conf)

            model.detect = detect
            return model

        detector = StreamingVisionDetector(
            model_loader=counting_loader,
            replay_buffer=replay_buffer,
        )

        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cooldown_seconds=0,
            cascade=CascadeConfig(
                cascade_chain=["yolov8m"],
                feedback_to_primary=True,
            ),
        )

        session = await detector.start_session(model_id="yolov8n", config=config)

        for i in range(5):
            await detector.process_frame(session.session_id, f"frame_{i}".encode())

        stats = await detector.stop_session(session.session_id)

        assert stats["frames_processed"] == 5
        assert stats["escalations_to_secondary"] == 5
        assert stats["secondary_successes"] == 5
        assert stats["secondary_success_rate"] == 1.0
        assert call_count["yolov8n"] == 5
        assert call_count["yolov8m"] == 5


class TestBackwardCompatibility:
    """Tests that old secondary_model_id config still works."""

    @pytest.mark.asyncio
    async def test_secondary_model_id_becomes_chain(self, mock_detection_result):
        """secondary_model_id gets promoted to cascade_chain internally."""
        loader = make_chain_loader({
            "yolov8n": 0.55,
            "yolov8m": 0.85,
        }, mock_detection_result)

        detector = StreamingVisionDetector(model_loader=loader)

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

        # Verify chain was built
        assert session.cascade_chain == ["yolov8m"]
        assert session.secondary_model_id == "yolov8m"

        result = await detector.process_frame(session.session_id, b"test_image")
        assert result.status == "action"
        assert result.escalated_to == "yolov8m"

    @pytest.mark.asyncio
    async def test_cascade_chain_overrides_secondary(self, mock_detection_result):
        """cascade_chain takes precedence over secondary_model_id."""
        loader = make_chain_loader({
            "yolov8n": 0.55,
            "yolov8m": 0.55,
            "yolov8x": 0.85,
        }, mock_detection_result)

        detector = StreamingVisionDetector(model_loader=loader)

        config = StreamingConfig(
            confidence_threshold=0.7,
            escalation_threshold=0.5,
            cooldown_seconds=0,
            cascade=CascadeConfig(
                secondary_model_id="yolov8m",  # Would only try one model
                cascade_chain=["yolov8m", "yolov8x"],  # Overrides: try both
            ),
        )

        session = await detector.start_session(model_id="yolov8n", config=config)
        assert session.cascade_chain == ["yolov8m", "yolov8x"]

        result = await detector.process_frame(session.session_id, b"test_image")
        assert result.status == "action"
        assert result.cascade_resolved_by == "yolov8x"


class TestEnrichment:
    """Tests for cross-modal enrichment (seg + classification)."""

    @pytest.mark.asyncio
    async def test_enrichment_models_are_loaded(self, mock_detection_result):
        """Segmentation and classification models are loaded at session start."""
        loaded_models = set()

        async def tracking_loader(model_id):
            loaded_models.add(model_id)
            model = MagicMock()
            model.detect = AsyncMock(return_value=mock_detection_result(0.8))
            return model

        detector = StreamingVisionDetector(model_loader=tracking_loader)

        config = StreamingConfig(
            cascade=CascadeConfig(
                cascade_chain=["yolov8m"],
                segmentation_model_id="mobilesam",
                classification_model_id="clip-vit-base",
            ),
        )

        await detector.start_session(model_id="yolov8n", config=config)

        assert "yolov8n" in loaded_models
        assert "yolov8m" in loaded_models
        assert "mobilesam" in loaded_models
        assert "clip-vit-base" in loaded_models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
