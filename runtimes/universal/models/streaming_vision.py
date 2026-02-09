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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .vision_base import DetectionBox

if TYPE_CHECKING:
    from vision_training.replay_buffer import ModelOpinion

    from .yolo_model import YOLOModel

logger = logging.getLogger(__name__)


@dataclass
class CascadeConfig:
    """Configuration for model cascade behavior.

    Supports both the simple secondary_model_id (backward compatible)
    and cascade_chain for multi-hop escalation.
    """

    secondary_model_id: str | None = None  # Fallback model (backward compat)
    cascade_chain: list[str] | None = None  # Ordered list of model IDs for multi-hop
    feedback_to_primary: bool = True  # Auto-add successful secondary results to replay
    save_uncertain_images: bool = True  # Save images for review queue
    segmentation_model_id: str | None = None  # Enrich with segmentation before escalating
    classification_model_id: str | None = None  # Enrich with CLIP before escalating
    enrich_on_escalation: bool = True  # Attach seg+class enrichment on escalation
    max_hops: int = 3  # Circuit breaker


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
    cascade_chain: list[str] = field(default_factory=list)  # Resolved chain of model IDs
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
    hop_count: int = 0  # How many models were consulted
    cascade_resolved_by: str | None = None  # Which model resolved it
    opinions: list = field(default_factory=list)  # list[ModelOpinion]


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

        Builds the cascade chain from config. The chain always starts with
        the primary model. Additional models come from cascade_chain or
        secondary_model_id (backward compatible).

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

        # Build the cascade chain
        cascade_chain: list[str] = []
        secondary_model_id = None

        if config.cascade:
            if config.cascade.cascade_chain:
                # New multi-hop chain
                cascade_chain = list(config.cascade.cascade_chain)
            elif config.cascade.secondary_model_id:
                # Backward compatible: single secondary
                cascade_chain = [config.cascade.secondary_model_id]

            # Set secondary_model_id for backward compat stats
            secondary_model_id = cascade_chain[0] if cascade_chain else None

            # Load all chain models
            for chain_model_id in cascade_chain:
                await self._ensure_model_loaded(chain_model_id)

            # Load enrichment models if configured
            if config.cascade.segmentation_model_id:
                await self._ensure_model_loaded(config.cascade.segmentation_model_id)
            if config.cascade.classification_model_id:
                await self._ensure_model_loaded(config.cascade.classification_model_id)

        session = StreamSession(
            session_id=session_id,
            config=config,
            model_id=model_id,
            secondary_model_id=secondary_model_id,
            cascade_chain=cascade_chain,
        )

        self._sessions[session_id] = session
        logger.info(
            f"Started streaming session {session_id} with primary={model_id}, "
            f"cascade_chain={cascade_chain or 'none'}"
        )

        return session
    
    def _build_opinion(
        self,
        model_id: str,
        detection: DetectionBox,
        inference_time_ms: float = 0.0,
        node_id: str = "local",
    ) -> ModelOpinion:
        """Build a ModelOpinion from a detection result."""
        from vision_training.replay_buffer import ModelOpinion

        return ModelOpinion(
            model_id=model_id,
            node_id=node_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=(detection.x1, detection.y1, detection.x2, detection.y2),
            mask_polygon=detection.mask,
            inference_time_ms=inference_time_ms,
        )

    async def _enrich_detection(
        self,
        session: StreamSession,
        image: bytes,
        detection: DetectionBox,
    ) -> DetectionBox:
        """Enrich a detection with segmentation mask and/or classification.

        When a detection is uncertain, this attaches additional context
        before sending to the next model in the cascade.
        """
        cascade = session.config.cascade
        if not cascade or not cascade.enrich_on_escalation:
            return detection

        enriched = DetectionBox(
            x1=detection.x1, y1=detection.y1,
            x2=detection.x2, y2=detection.y2,
            class_name=detection.class_name,
            class_id=detection.class_id,
            confidence=detection.confidence,
            mask=detection.mask,
        )

        # Enrich with segmentation
        if cascade.segmentation_model_id:
            seg_model = self._models.get(cascade.segmentation_model_id)
            if seg_model and hasattr(seg_model, 'segment'):
                try:
                    seg_result = await seg_model.segment(
                        image=image,
                        boxes=[(detection.x1, detection.y1, detection.x2, detection.y2)],
                    )
                    if seg_result and seg_result.masks:
                        mask = seg_result.masks[0]
                        if mask.box:
                            enriched.mask = mask.box.mask
                except Exception as e:
                    logger.debug(f"Segmentation enrichment failed: {e}")

        # Enrich with classification
        if cascade.classification_model_id:
            cls_model = self._models.get(cascade.classification_model_id)
            if cls_model and hasattr(cls_model, 'classify'):
                try:
                    cls_result = await cls_model.classify(image=image)
                    if cls_result and cls_result.class_name:
                        # If CLIP is more confident, update the class name
                        if cls_result.confidence > enriched.confidence:
                            enriched.class_name = cls_result.class_name
                except Exception as e:
                    logger.debug(f"Classification enrichment failed: {e}")

        return enriched

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
        opinions: list | None = None,
        resolving_hop: int = 1,
    ) -> bool:
        """Add successful escalation to replay buffer for primary model training.

        Now carries the full ModelOpinion chain so training knows exactly
        what happened in the cascade.

        Args:
            session_id: Session that produced this result
            image: Image bytes
            detections: Verified detections from secondary/cascade model
            source: Source identifier (e.g., "escalation", "correction")
            confidence: Detection confidence
            opinions: Full list of ModelOpinions from the cascade
            resolving_hop: Which hop resolved it (1-based)

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

            sample_id = f"{session_id}_{int(time.time() * 1000)}"

            # Get best detection's bbox
            best_det = detections[0] if detections else None
            bbox = (best_det.x1, best_det.y1, best_det.x2, best_det.y2) if best_det else None
            final_label = best_det.class_name if best_det else ""

            if source == "escalation" and opinions:
                # Cascade resolved: use structured add_cascade_resolved
                self._replay_buffer.add_cascade_resolved(
                    image_id=sample_id,
                    image_path=image_path or "",
                    opinions=opinions,
                    final_label=final_label,
                    bbox=bbox,
                    resolving_hop=resolving_hop,
                )
            elif source == "correction":
                self._replay_buffer.add_correction(
                    image_id=sample_id,
                    image_path=image_path or "",
                    corrected_label=final_label,
                    original_confidence=confidence,
                    opinions=opinions,
                    bbox=bbox,
                )
            else:
                self._replay_buffer.add_low_confidence(
                    image_id=sample_id,
                    image_path=image_path or "",
                    predicted_label=final_label,
                    confidence=confidence,
                    opinions=opinions,
                    bbox=bbox,
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
        """Process a single frame with cascade chain and auto-learning.

        Flow (multi-hop cascade):
        1. Primary model (hop 0) detects
        2. If high confidence -> return "action"
        3. If mid/low confidence -> enrich with seg/classification, then
           iterate through cascade_chain (hops 1..N):
           - If any hop returns high confidence -> "action" + replay buffer
           - If all hops fail -> "review" queue
        4. No detections -> "ok"

        The cascade chain is built at session start from cascade_chain config
        or from secondary_model_id (backward compatible).

        Args:
            session_id: Active session ID
            image: Frame as bytes (JPEG/PNG)
            callback: Optional callback for action results

        Returns:
            FrameResult with status, detections, opinions, and learning info
        """

        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        model = self._models.get(session.model_id)
        if model is None:
            raise RuntimeError(f"Model {session.model_id} not loaded")

        config = session.config
        session.frames_processed += 1
        opinions: list[ModelOpinion] = []

        # ---- Hop 0: Primary model ----
        start_time = time.time()
        result = await model.detect(
            image=image,
            confidence_threshold=config.escalation_threshold,
        )
        primary_time_ms = (time.time() - start_time) * 1000

        # Separate detections by confidence
        high_confidence = []
        mid_confidence = []
        low_confidence = []

        for box in result.boxes:
            if config.action_classes and box.class_name not in config.action_classes:
                continue

            if box.confidence >= config.confidence_threshold:
                high_confidence.append(box)
            elif box.confidence >= config.escalation_threshold:
                mid_confidence.append(box)
            else:
                low_confidence.append(box)

        # Build primary opinion from best detection
        best_primary = (
            high_confidence[0] if high_confidence
            else mid_confidence[0] if mid_confidence
            else low_confidence[0] if low_confidence
            else None
        )
        if best_primary:
            opinions.append(self._build_opinion(
                model_id=session.model_id,
                detection=best_primary,
                inference_time_ms=primary_time_ms,
            ))

        # Case 1: High confidence - return immediately
        if high_confidence:
            now = time.time()

            if now - session.last_action_time < config.cooldown_seconds:
                return FrameResult(
                    status="ok",
                    detections=high_confidence,
                    confidence=max(d.confidence for d in high_confidence),
                    suppressed=True,
                    opinions=opinions,
                    hop_count=1,
                )

            session.last_action_time = now
            session.actions_triggered += 1

            frame_result = FrameResult(
                status="action",
                detections=high_confidence,
                confidence=max(d.confidence for d in high_confidence),
                opinions=opinions,
                hop_count=1,
            )

            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback(frame_result)
                else:
                    callback(frame_result)

            return frame_result

        # Case 2: Need to escalate through the cascade chain
        uncertain_detections = mid_confidence or low_confidence
        if uncertain_detections and session.cascade_chain:
            # Enrich the best detection before escalating
            best_uncertain = uncertain_detections[0]
            enriched = await self._enrich_detection(session, image, best_uncertain)

            # Walk the cascade chain
            resolved = False
            resolving_model_id = None
            resolved_detections: list[DetectionBox] = []
            resolved_confidence = 0.0

            for hop_idx, chain_model_id in enumerate(session.cascade_chain):
                hop_number = hop_idx + 1  # hop 0 is primary

                # Circuit breaker
                if hop_number >= (config.cascade.max_hops if config.cascade else 3):
                    break

                chain_model = self._models.get(chain_model_id)
                if chain_model is None:
                    logger.warning(f"Cascade model {chain_model_id} not loaded, skipping")
                    continue

                session.escalations_to_secondary += 1

                # Run detection with cascade model
                hop_start = time.time()
                hop_result = await chain_model.detect(
                    image=image,
                    confidence_threshold=config.escalation_threshold,
                )
                hop_time_ms = (time.time() - hop_start) * 1000

                # Filter to confident detections
                if config.action_classes:
                    hop_confident = [
                        b for b in hop_result.boxes
                        if b.class_name in config.action_classes
                        and b.confidence >= config.confidence_threshold
                    ]
                else:
                    hop_confident = [
                        b for b in hop_result.boxes
                        if b.confidence >= config.confidence_threshold
                    ]

                # Build opinion from this hop
                best_hop = (
                    hop_confident[0] if hop_confident
                    else hop_result.boxes[0] if hop_result.boxes
                    else None
                )
                if best_hop:
                    opinions.append(self._build_opinion(
                        model_id=chain_model_id,
                        detection=best_hop,
                        inference_time_ms=hop_time_ms,
                    ))

                if hop_confident:
                    # This hop resolved it
                    session.secondary_successes += 1
                    resolved = True
                    resolving_model_id = chain_model_id
                    resolved_detections = hop_confident
                    resolved_confidence = max(d.confidence for d in hop_confident)
                    break

            if resolved:
                now = time.time()

                if now - session.last_action_time < config.cooldown_seconds:
                    return FrameResult(
                        status="ok",
                        detections=resolved_detections,
                        confidence=resolved_confidence,
                        suppressed=True,
                        escalated_to=resolving_model_id,
                        opinions=opinions,
                        hop_count=len(opinions),
                        cascade_resolved_by=resolving_model_id,
                    )

                session.last_action_time = now
                session.actions_triggered += 1

                # Add to replay buffer with full opinion chain
                added_to_replay = False
                if config.cascade and config.cascade.feedback_to_primary:
                    added_to_replay = await self._add_to_replay_buffer(
                        session_id=session_id,
                        image=image,
                        detections=resolved_detections,
                        source="escalation",
                        confidence=resolved_confidence,
                        opinions=opinions,
                        resolving_hop=len(opinions) - 1,
                    )

                frame_result = FrameResult(
                    status="action",
                    detections=resolved_detections,
                    confidence=resolved_confidence,
                    escalated_to=resolving_model_id,
                    added_to_replay=added_to_replay,
                    opinions=opinions,
                    hop_count=len(opinions),
                    cascade_resolved_by=resolving_model_id,
                )

                if callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(frame_result)
                    else:
                        callback(frame_result)

                return frame_result
            else:
                # All hops failed -> review queue
                image_id = await self._send_to_review_queue(
                    session=session,
                    image=image,
                    detections=uncertain_detections,
                    confidence=max(d.confidence for d in uncertain_detections),
                )

                return FrameResult(
                    status="review",
                    detections=uncertain_detections,
                    confidence=max(d.confidence for d in uncertain_detections),
                    image_id=image_id,
                    escalated_to=session.cascade_chain[-1] if session.cascade_chain else None,
                    opinions=opinions,
                    hop_count=len(opinions),
                )

        # Case 3: Uncertain but no cascade chain - send to review
        if uncertain_detections:
            image_id = await self._send_to_review_queue(
                session=session,
                image=image,
                detections=uncertain_detections,
                confidence=max(d.confidence for d in uncertain_detections),
            )

            return FrameResult(
                status="review",
                detections=uncertain_detections,
                confidence=max(d.confidence for d in uncertain_detections),
                image_id=image_id,
                opinions=opinions,
                hop_count=len(opinions),
            )

        # Case 4: No detections matching our criteria
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
            # Create detection box if not provided (used for replay buffer context)
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
                bbox=(detection.x1, detection.y1, detection.x2, detection.y2),
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
        if self._replay_buffer is not None:
            return self._replay_buffer.get_stats()
        return None

    def clear_replay_buffer(self) -> None:
        """Clear the replay buffer."""
        if self._replay_buffer is not None:
            self._replay_buffer.clear()


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
