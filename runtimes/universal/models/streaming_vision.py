"""
Streaming Vision Detector for real-time anomaly-style detection.

Provides session-based detection with:
- Configurable FPS and confidence thresholds
- Cooldown between action triggers
- Multi-model cascade (escalate low-confidence to secondary model)
- Review queue integration for uncertain detections
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from .vision_base import DetectionBox

if TYPE_CHECKING:
    from .yolo_model import YOLOModel

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming vision detection."""
    
    target_fps: float = 1.0
    confidence_threshold: float = 0.7
    escalation_threshold: float = 0.5
    action_classes: list[str] | None = None
    cooldown_seconds: float = 5.0


@dataclass
class StreamSession:
    """Active streaming session state."""
    
    session_id: str
    config: StreamingConfig
    model_id: str
    created_at: float = field(default_factory=time.time)
    last_action_time: float = 0.0
    frames_processed: int = 0
    actions_triggered: int = 0
    review_queue: list[str] = field(default_factory=list)


@dataclass 
class FrameResult:
    """Result of processing a single frame."""
    
    status: Literal["ok", "action", "review"]
    detections: list[DetectionBox] = field(default_factory=list)
    confidence: float = 0.0
    image_id: str | None = None
    suppressed: bool = False


class StreamingVisionDetector:
    """Streaming vision detector with session management.
    
    Processes frames at target FPS and triggers actions only when:
    1. Detection confidence exceeds threshold
    2. Detection class is in action_classes (if specified)
    3. Cooldown period has elapsed since last action
    
    Example:
        ```python
        detector = StreamingVisionDetector()
        
        # Start session
        session = await detector.start_session(
            model_id="yolov8n",
            config=StreamingConfig(
                action_classes=["person", "car"],
                cooldown_seconds=5.0
            )
        )
        
        # Process frames
        while streaming:
            result = await detector.process_frame(session.session_id, frame_bytes)
            if result.status == "action":
                trigger_alert(result.detections)
        
        # End session
        stats = await detector.stop_session(session.session_id)
        ```
    """
    
    def __init__(
        self,
        model_loader: Callable[[str], Any] | None = None,
    ):
        """Initialize streaming detector.
        
        Args:
            model_loader: Async function to load detection models
        """
        self._sessions: dict[str, StreamSession] = {}
        self._models: dict[str, YOLOModel] = {}
        self._model_loader = model_loader
        self._lock = asyncio.Lock()
    
    def set_model_loader(self, loader: Callable[[str], Any]) -> None:
        """Set the model loader function."""
        self._model_loader = loader
    
    async def start_session(
        self,
        model_id: str = "yolov8n",
        config: StreamingConfig | None = None,
    ) -> StreamSession:
        """Start a new streaming session.
        
        Args:
            model_id: Detection model to use
            config: Session configuration
            
        Returns:
            StreamSession with session_id
        """
        config = config or StreamingConfig()
        session_id = str(uuid.uuid4())[:8]
        
        # Ensure model is loaded
        if model_id not in self._models:
            if self._model_loader is None:
                raise RuntimeError("Model loader not configured")
            self._models[model_id] = await self._model_loader(model_id)
        
        session = StreamSession(
            session_id=session_id,
            config=config,
            model_id=model_id,
        )
        
        self._sessions[session_id] = session
        logger.info(f"Started streaming session {session_id} with model {model_id}")
        
        return session
    
    async def process_frame(
        self,
        session_id: str,
        image: bytes,
        callback: Callable[[FrameResult], Any] | None = None,
    ) -> FrameResult:
        """Process a single frame.
        
        Args:
            session_id: Active session ID
            image: Frame as bytes (JPEG/PNG)
            callback: Optional callback for action results
            
        Returns:
            FrameResult with status and detections
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        
        model = self._models.get(session.model_id)
        if model is None:
            raise RuntimeError(f"Model {session.model_id} not loaded")
        
        config = session.config
        
        # Run detection
        result = await model.detect(
            image=image,
            confidence_threshold=config.escalation_threshold,  # Use lower threshold to catch uncertain
        )
        
        session.frames_processed += 1
        
        # Filter to action classes if specified
        if config.action_classes:
            action_detections = [
                b for b in result.boxes
                if b.class_name in config.action_classes
                and b.confidence >= config.confidence_threshold
            ]
        else:
            action_detections = [
                b for b in result.boxes
                if b.confidence >= config.confidence_threshold
            ]
        
        # Check if we should trigger action
        if action_detections:
            now = time.time()
            
            # Check cooldown
            if now - session.last_action_time < config.cooldown_seconds:
                return FrameResult(
                    status="ok",
                    detections=action_detections,
                    confidence=max(d.confidence for d in action_detections),
                    suppressed=True,
                )
            
            # Trigger action
            session.last_action_time = now
            session.actions_triggered += 1
            
            frame_result = FrameResult(
                status="action",
                detections=action_detections,
                confidence=max(d.confidence for d in action_detections),
            )
            
            if callback:
                await callback(frame_result) if asyncio.iscoroutinefunction(callback) else callback(frame_result)
            
            return frame_result
        
        # Check for uncertain detections requiring review
        uncertain_detections = [
            b for b in result.boxes
            if config.escalation_threshold <= b.confidence < config.confidence_threshold
        ]
        
        if uncertain_detections:
            image_id = f"{session_id}_{session.frames_processed}"
            session.review_queue.append(image_id)
            
            return FrameResult(
                status="review",
                detections=uncertain_detections,
                confidence=max(d.confidence for d in uncertain_detections),
                image_id=image_id,
            )
        
        return FrameResult(status="ok")
    
    async def stop_session(self, session_id: str) -> dict[str, Any]:
        """Stop a streaming session and return statistics.
        
        Args:
            session_id: Session to stop
            
        Returns:
            Session statistics
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        
        duration = time.time() - session.created_at
        
        stats = {
            "session_id": session_id,
            "model_id": session.model_id,
            "frames_processed": session.frames_processed,
            "actions_triggered": session.actions_triggered,
            "review_queue_size": len(session.review_queue),
            "duration_seconds": duration,
            "avg_fps": session.frames_processed / duration if duration > 0 else 0,
        }
        
        logger.info(f"Stopped session {session_id}: {session.frames_processed} frames, {session.actions_triggered} actions")
        
        return stats
    
    def get_session(self, session_id: str) -> StreamSession | None:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def list_sessions(self) -> list[str]:
        """List active session IDs."""
        return list(self._sessions.keys())


# Global streaming detector instance
_streaming_detector: StreamingVisionDetector | None = None


def get_streaming_detector() -> StreamingVisionDetector:
    """Get or create the global streaming detector."""
    global _streaming_detector
    if _streaming_detector is None:
        _streaming_detector = StreamingVisionDetector()
    return _streaming_detector


def set_streaming_model_loader(loader: Callable[[str], Any]) -> None:
    """Set the model loader for the global streaming detector."""
    get_streaming_detector().set_model_loader(loader)
