"""
Streaming Vision Detector with Model Cascade and Auto-Learning.

Provides session-based detection with:
- Configurable FPS and confidence thresholds
- Cooldown between action triggers
- Multi-model cascade (escalate low-confidence to secondary model)
- Auto-feedback loop: secondary model success → replay buffer → retrain primary
- Review queue integration for uncertain detections
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Literal

from .vision_base import DetectionBox, DetectionResult

if TYPE_CHECKING:
    from .yolo_model import YOLOModel

logger = logging.getLogger(__name__)


@dataclass
class CascadeConfig:
    """Configuration for model cascade behavior."""
    
    secondary_model_id: str | None = None  # Fallback model for uncertain detections
    feedback_to_primary: bool = True  # Auto-add successful secondary results to replay
    save_uncertain_images: bool = True  # Save images for review queue


@dataclass
class StreamingConfig:
    """Configuration for streaming vision detection."""
    
    target_fps: float = 1.0
    confidence_threshold: float = 0.7
    escalation_threshold: float = 0.5
    action_classes: list[str] | None = None
    cooldown_seconds: float = 5.0
    
    # Cascade configuration
    cascade: CascadeConfig | None = None


@dataclass
class StreamSession:
    """Active streaming session state."""
    
    session_id: str
    config: StreamingConfig
    model_id: str
    secondary_model_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_action_time: float = 0.0
    frames_processed: int = 0
    actions_triggered: int = 0
    escalations_to_secondary: int = 0
    secondary_successes: int = 0
    review_queue_count: int = 0
    review_queue: list[str] = field(default_factory=list)


@dataclass 
class FrameResult:
    """Result of processing a single frame."""
    
    status: Literal["ok", "action", "review", "escalated"]
    detections: list[DetectionBox] = field(default_factory=list)
    confidence: float = 0.0
    image_id: str | None = None
    suppressed: bool = False
    escalated_to: str | None = None  # Model that handled escalation
    added_to_replay: bool = False  # Whether result was added to replay buffer


@dataclass
class EscalationResult:
    """Result of escalating to secondary model."""
    
    success: bool
    detections: list[DetectionBox] = field(default_factory=list)
    confidence: float = 0.0
    model_id: str = ""


class StreamingVisionDetector:
    """Streaming vision detector with model cascade and auto-learning.
    
    Implements the following flow:
    1. Primary model processes frame
    2. If confidence >= threshold → return result
    3. If confidence < escalation_threshold → send to review queue
    4. If confidence between thresholds → escalate to secondary model
       - If secondary succeeds → return result AND add to replay buffer
       - If secondary fails → send to review queue
    
    The replay buffer feeds into incremental training to improve the
    primary model over time.
    
    Example:
        ```python
        detector = StreamingVisionDetector()
        
        # Start session with cascade
        session = await detector.start_session(
            model_id="yolov8n",
            config=StreamingConfig(
                confidence_threshold=0.7,
                escalation_threshold=0.5,
                cascade=CascadeConfig(
                    secondary_model_id="yolov8m",
                    feedback_to_primary=True,
                )
            )
        )
        
        # Process frames - cascade happens automatically
        while streaming:
            result = await detector.process_frame(session.session_id, frame_bytes)
            if result.status == "action":
                trigger_alert(result.detections)
            if result.added_to_replay:
                print("Learning from this example!")
        
        # End session
        stats = await detector.stop_session(session.session_id)
        ```
    """
    
    def __init__(
        self,
        model_loader: Callable[[str], Any] | None = None,
        replay_buffer: Any | None = None,
        image_store: Any | None = None,
    ):
        """Initialize streaming detector.
        
        Args:
            model_loader: Async function to load detection models
            replay_buffer: ReplayBuffer for storing corrections/escalation successes
            image_store: ImageStore for persisting images for review
        """
        self._sessions: dict[str, StreamSession] = {}
        self._models: dict[str, YOLOModel] = {}
        self._model_loader = model_loader
        self._replay_buffer = replay_buffer
        self._image_store = image_store
        self._lock = asyncio.Lock()
        
        # Training trigger callback
        self._on_replay_buffer_threshold: Callable[[int], Any] | None = None
    
    def set_model_loader(self, loader: Callable[[str], Any]) -> None:
        """Set the model loader function."""
        self._model_loader = loader
    
    def set_replay_buffer(self, buffer: Any) -> None:
        """Set the replay buffer for auto-learning."""
        self._replay_buffer = buffer
    
    def set_image_store(self, store: Any) -> None:
        """Set the image store for review queue."""
        self._image_store = store
    
    def set_training_trigger(self, callback: Callable[[int], Any]) -> None:
        """Set callback for when replay buffer hits threshold.
        
        Callback receives current buffer size.
        """
        self._on_replay_buffer_threshold = callback
    
    async def _ensure_model_loaded(self, model_id: str) -> None:
        """Ensure a model is loaded."""
        if model_id not in self._models:
            if self._model_loader is None:
                raise RuntimeError("Model loader not configured")
            self._models[model_id] = await self._model_loader(model_id)
    
    async def start_session(
        self,
        model_id: str = "yolov8n",
        config: StreamingConfig | None = None,
    ) -> StreamSession:
        """Start a new streaming session.
        
        Args:
            model_id: Primary detection model to use
            config: Session configuration including cascade settings
            
        Returns:
            StreamSession with session_id
        """
        config = config or StreamingConfig()
        session_id = str(uuid.uuid4())[:8]
        
        # Ensure primary model is loaded
        await self._ensure_model_loaded(model_id)
        
        # Ensure secondary model is loaded if configured
        secondary_model_id = None
        if config.cascade and config.cascade.secondary_model_id:
            secondary_model_id = config.cascade.secondary_model_id
            await self._ensure_model_loaded(secondary_model_id)
        
        session = StreamSession(
            session_id=session_id,
            config=config,
            model_id=model_id,
            secondary_model_id=secondary_model_id,
        )
        
        self._sessions[session_id] = session
        logger.info(
            f"Started streaming session {session_id} with primary={model_id}, "
            f"secondary={secondary_model_id or 'none'}"
        )
        
        return session
    
    async def _escalate_to_secondary(
        self,
        session: StreamSession,
        image: bytes,
        primary_detections: list[DetectionBox],
    ) -> EscalationResult:
        """Escalate detection to secondary model.
        
        Args:
            session: Active session
            image: Frame bytes
            primary_detections: Detections from primary model
            
        Returns:
            EscalationResult with secondary model's findings
        """
        if not session.secondary_model_id:
            return EscalationResult(success=False)
        
        secondary_model = self._models.get(session.secondary_model_id)
        if secondary_model is None:
            logger.warning(f"Secondary model {session.secondary_model_id} not loaded")
            return EscalationResult(success=False)
        
        config = session.config
        
        # Run detection with secondary model (using lower threshold to catch more)
        result = await secondary_model.detect(
            image=image,
            confidence_threshold=config.escalation_threshold,
        )
        
        session.escalations_to_secondary += 1
        
        # Filter to action classes and confidence threshold
        if config.action_classes:
            confident_detections = [
                b for b in result.boxes
                if b.class_name in config.action_classes
                and b.confidence >= config.confidence_threshold
            ]
        else:
            confident_detections = [
                b for b in result.boxes
                if b.confidence >= config.confidence_threshold
            ]
        
        if confident_detections:
            session.secondary_successes += 1
            max_conf = max(d.confidence for d in confident_detections)
            return EscalationResult(
                success=True,
                detections=confident_detections,
                confidence=max_conf,
                model_id=session.secondary_model_id,
            )
        
        return EscalationResult(
            success=False,
            detections=result.boxes,
            confidence=max(d.confidence for d in result.boxes) if result.boxes else 0.0,
            model_id=session.secondary_model_id,
        )
    
    async def _add_to_replay_buffer(
        self,
        session_id: str,
        image: bytes,
        detections: list[DetectionBox],
        source: str,
        confidence: float,
    ) -> bool:
        """Add successful escalation to replay buffer for primary model training.
        
        Args:
            session_id: Session that produced this result
            image: Image bytes
            detections: Verified detections from secondary model
            source: Source identifier (e.g., "escalation", "correction")
            confidence: Detection confidence
            
        Returns:
            True if added successfully
        """
        if self._replay_buffer is None:
            logger.debug("No replay buffer configured, skipping auto-learning")
            return False
        
        try:
            # Store image if we have image store
            image_path = None
            if self._image_store:
                image_id = f"{session_id}_{int(time.time() * 1000)}"
                image_path = await self._image_store.save_image(
                    image_id=image_id,
                    image_bytes=image,
                    source=source,
                )
            
            # Add to replay buffer
            # Convert detections to label format (YOLO format: class x_center y_center width height)
            labels = []
            for det in detections:
                x_center = (det.x1 + det.x2) / 2
                y_center = (det.y1 + det.y2) / 2
                width = det.x2 - det.x1
                height = det.y2 - det.y1
                labels.append(f"{det.class_id} {x_center} {y_center} {width} {height}")
            
            sample_id = f"{session_id}_{int(time.time() * 1000)}"
            
            if source == "escalation":
                self._replay_buffer.add_low_confidence(
                    image_id=sample_id,
                    image_path=image_path or "",
                    predicted_label="\n".join(labels),
                    confidence=confidence,
                )
            else:
                self._replay_buffer.add_correction(
                    image_id=sample_id,
                    image_path=image_path or "",
                    corrected_label="\n".join(labels),
                    original_confidence=confidence,
                )
            
            # Check if we should trigger training
            buffer_size = len(self._replay_buffer)
            if self._on_replay_buffer_threshold:
                await self._on_replay_buffer_threshold(buffer_size)
            
            logger.info(f"Added to replay buffer: {sample_id} (buffer size: {buffer_size})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add to replay buffer: {e}")
            return False
    
    async def _send_to_review_queue(
        self,
        session: StreamSession,
        image: bytes,
        detections: list[DetectionBox],
        confidence: float,
    ) -> str:
        """Send uncertain image to review queue.
        
        Args:
            session: Active session
            image: Image bytes
            detections: Uncertain detections
            confidence: Best confidence score
            
        Returns:
            Image ID in review queue
        """
        image_id = f"{session.session_id}_{session.frames_processed}"
        session.review_queue.append(image_id)
        session.review_queue_count += 1
        
        if self._image_store and session.config.cascade and session.config.cascade.save_uncertain_images:
            try:
                await self._image_store.save_for_review(
                    image_id=image_id,
                    image_bytes=image,
                    detections=detections,
                    confidence=confidence,
                    model_id=session.model_id,
                    source=f"stream:{session.session_id}",
                )
            except Exception as e:
                logger.error(f"Failed to save to review queue: {e}")
        
        return image_id
    
    async def process_frame(
        self,
        session_id: str,
        image: bytes,
        callback: Callable[[FrameResult], Any] | None = None,
    ) -> FrameResult:
        """Process a single frame with cascade and auto-learning.
        
        Flow:
        1. Primary model detects
        2. If high confidence → return "action"
        3. If mid confidence → escalate to secondary
           - Secondary succeeds → "action" + add to replay buffer
           - Secondary fails → "review" queue
        4. If low confidence → "review" queue
        5. Otherwise → "ok"
        
        Args:
            session_id: Active session ID
            image: Frame as bytes (JPEG/PNG)
            callback: Optional callback for action results
            
        Returns:
            FrameResult with status, detections, and learning info
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        
        model = self._models.get(session.model_id)
        if model is None:
            raise RuntimeError(f"Model {session.model_id} not loaded")
        
        config = session.config
        session.frames_processed += 1
        
        # Run primary detection (use lower threshold to catch uncertain cases)
        result = await model.detect(
            image=image,
            confidence_threshold=config.escalation_threshold,
        )
        
        # Separate detections by confidence
        high_confidence = []
        mid_confidence = []
        low_confidence = []
        
        for box in result.boxes:
            # Filter to action classes if specified
            if config.action_classes and box.class_name not in config.action_classes:
                continue
            
            if box.confidence >= config.confidence_threshold:
                high_confidence.append(box)
            elif box.confidence >= config.escalation_threshold:
                mid_confidence.append(box)
            else:
                low_confidence.append(box)
        
        # Case 1: High confidence detections - return immediately
        if high_confidence:
            now = time.time()
            
            # Check cooldown
            if now - session.last_action_time < config.cooldown_seconds:
                return FrameResult(
                    status="ok",
                    detections=high_confidence,
                    confidence=max(d.confidence for d in high_confidence),
                    suppressed=True,
                )
            
            # Trigger action
            session.last_action_time = now
            session.actions_triggered += 1
            
            frame_result = FrameResult(
                status="action",
                detections=high_confidence,
                confidence=max(d.confidence for d in high_confidence),
            )
            
            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback(frame_result)
                else:
                    callback(frame_result)
            
            return frame_result
        
        # Case 2: Mid confidence - escalate to secondary model
        if mid_confidence and config.cascade and config.cascade.secondary_model_id:
            escalation_result = await self._escalate_to_secondary(
                session=session,
                image=image,
                primary_detections=mid_confidence,
            )
            
            if escalation_result.success:
                now = time.time()
                
                # Check cooldown
                if now - session.last_action_time < config.cooldown_seconds:
                    return FrameResult(
                        status="ok",
                        detections=escalation_result.detections,
                        confidence=escalation_result.confidence,
                        suppressed=True,
                        escalated_to=escalation_result.model_id,
                    )
                
                # Trigger action
                session.last_action_time = now
                session.actions_triggered += 1
                
                # Add to replay buffer for primary model training
                added_to_replay = False
                if config.cascade.feedback_to_primary:
                    added_to_replay = await self._add_to_replay_buffer(
                        session_id=session_id,
                        image=image,
                        detections=escalation_result.detections,
                        source="escalation",
                        confidence=escalation_result.confidence,
                    )
                
                frame_result = FrameResult(
                    status="action",
                    detections=escalation_result.detections,
                    confidence=escalation_result.confidence,
                    escalated_to=escalation_result.model_id,
                    added_to_replay=added_to_replay,
                )
                
                if callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(frame_result)
                    else:
                        callback(frame_result)
                
                return frame_result
            else:
                # Secondary also uncertain - send to review queue
                image_id = await self._send_to_review_queue(
                    session=session,
                    image=image,
                    detections=mid_confidence + escalation_result.detections,
                    confidence=max(
                        [d.confidence for d in mid_confidence] +
                        [d.confidence for d in escalation_result.detections],
                        default=0.0
                    ),
                )
                
                return FrameResult(
                    status="review",
                    detections=mid_confidence,
                    confidence=max(d.confidence for d in mid_confidence) if mid_confidence else 0.0,
                    image_id=image_id,
                    escalated_to=escalation_result.model_id,
                )
        
        # Case 3: Mid confidence but no secondary model - send to review
        if mid_confidence:
            image_id = await self._send_to_review_queue(
                session=session,
                image=image,
                detections=mid_confidence,
                confidence=max(d.confidence for d in mid_confidence),
            )
            
            return FrameResult(
                status="review",
                detections=mid_confidence,
                confidence=max(d.confidence for d in mid_confidence),
                image_id=image_id,
            )
        
        # Case 4: Only low confidence detections - send to review
        if low_confidence:
            image_id = await self._send_to_review_queue(
                session=session,
                image=image,
                detections=low_confidence,
                confidence=max(d.confidence for d in low_confidence),
            )
            
            return FrameResult(
                status="review",
                detections=low_confidence,
                confidence=max(d.confidence for d in low_confidence),
                image_id=image_id,
            )
        
        # Case 5: No detections matching our criteria
        return FrameResult(status="ok")
    
    async def submit_correction(
        self,
        session_id: str,
        image_id: str,
        corrected_class: str,
        box: DetectionBox | None = None,
        original_confidence: float = 0.0,
    ) -> bool:
        """Submit a correction for a detection (feedback from user/AI).
        
        This feeds into the replay buffer for training the primary model.
        
        Args:
            session_id: Session that produced the detection
            image_id: Image ID from review queue
            corrected_class: Correct class name
            box: Corrected bounding box (optional)
            original_confidence: Original model confidence
            
        Returns:
            True if correction was added to replay buffer
        """
        if self._replay_buffer is None:
            logger.warning("No replay buffer configured")
            return False
        
        # Get image from store if available
        image_bytes = None
        image_path = None
        if self._image_store:
            record = await self._image_store.get_image(image_id)
            if record:
                image_path = record.file_path
        
        try:
            # Create detection box if not provided
            detection = box or DetectionBox(
                x1=0.0, y1=0.0, x2=1.0, y2=1.0,  # Full image
                class_name=corrected_class,
                class_id=0,  # Will be resolved by training
                confidence=1.0,  # Human-verified
            )
            
            self._replay_buffer.add_correction(
                image_id=image_id,
                image_path=image_path or "",
                corrected_label=corrected_class,
                original_confidence=original_confidence,
            )
            
            # Mark as reviewed in image store
            if self._image_store:
                await self._image_store.mark_reviewed(
                    image_id=image_id,
                    decision="corrected",
                    corrected_class=corrected_class,
                )
            
            # Check training trigger
            buffer_size = len(self._replay_buffer)
            if self._on_replay_buffer_threshold:
                await self._on_replay_buffer_threshold(buffer_size)
            
            logger.info(f"Correction added: {image_id} → {corrected_class}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add correction: {e}")
            return False
    
    async def stop_session(self, session_id: str) -> dict[str, Any]:
        """Stop a streaming session and return statistics.
        
        Args:
            session_id: Session to stop
            
        Returns:
            Session statistics including cascade metrics
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        
        duration = time.time() - session.created_at
        
        stats = {
            "session_id": session_id,
            "model_id": session.model_id,
            "secondary_model_id": session.secondary_model_id,
            "frames_processed": session.frames_processed,
            "actions_triggered": session.actions_triggered,
            "escalations_to_secondary": session.escalations_to_secondary,
            "secondary_successes": session.secondary_successes,
            "secondary_success_rate": (
                session.secondary_successes / session.escalations_to_secondary
                if session.escalations_to_secondary > 0 else 0.0
            ),
            "review_queue_count": session.review_queue_count,
            "duration_seconds": duration,
            "avg_fps": session.frames_processed / duration if duration > 0 else 0,
        }
        
        logger.info(
            f"Stopped session {session_id}: {session.frames_processed} frames, "
            f"{session.actions_triggered} actions, "
            f"{session.escalations_to_secondary} escalations ({session.secondary_successes} successful)"
        )
        
        return stats
    
    def get_session(self, session_id: str) -> StreamSession | None:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def list_sessions(self) -> list[str]:
        """List active session IDs."""
        return list(self._sessions.keys())
    
    def get_replay_buffer_stats(self) -> dict[str, Any] | None:
        """Get replay buffer statistics."""
        if self._replay_buffer:
            return self._replay_buffer.get_stats()
        return None


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


def set_streaming_replay_buffer(buffer: Any) -> None:
    """Set the replay buffer for the global streaming detector."""
    get_streaming_detector().set_replay_buffer(buffer)


def set_streaming_image_store(store: Any) -> None:
    """Set the image store for the global streaming detector."""
    get_streaming_detector().set_image_store(store)


def set_streaming_training_trigger(callback: Callable[[int], Any]) -> None:
    """Set the training trigger callback for the global streaming detector."""
    get_streaming_detector().set_training_trigger(callback)
