#!/usr/bin/env python3
"""
Vision Cascade + Auto-Learning Demo

Demonstrates the complete automated learning pipeline:
1. Primary model (yolov8n) processes images
2. Uncertain detections escalate to secondary model (yolov8m)
3. Secondary model successes feed back to replay buffer
4. Failed detections go to review queue
5. Corrections trigger automatic retraining

Run with:
    cd runtimes/universal && uv run python ../../examples/vision_cascade_demo.py
"""

import asyncio
import base64
import logging
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "runtimes" / "universal"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")


async def main():
    """Run the cascade demo."""
    
    print("\n" + "=" * 60)
    print("  Vision Cascade + Auto-Learning Demo")
    print("=" * 60 + "\n")
    
    # Import components
    from models.streaming_vision import (
        StreamingVisionDetector,
        StreamingConfig,
        CascadeConfig,
    )
    from vision_training.replay_buffer import ReplayBuffer
    from storage.image_store import ImageStore
    from vision_training.auto_trainer import AutoTrainer, AutoTrainConfig
    
    # Initialize components
    print("📦 Initializing components...")
    
    replay_buffer = ReplayBuffer(max_size=100)
    image_store = ImageStore()
    
    # Mock model loader for demo
    async def mock_model_loader(model_id: str):
        """Mock model loader that returns a fake detector."""
        class MockModel:
            def __init__(self, model_id: str):
                self.model_id = model_id
                # Secondary model is "better" - higher confidence
                self.confidence_boost = 0.2 if "m" in model_id or "l" in model_id else 0.0
            
            async def detect(self, image, confidence_threshold=0.5):
                """Mock detection based on image content."""
                from models.vision_base import DetectionResult, DetectionBox
                import hashlib
                
                # Use hash of image to generate consistent "detections"
                img_hash = int(hashlib.md5(image).hexdigest()[:8], 16)
                
                # Generate 0-3 detections based on hash
                num_detections = img_hash % 4
                
                boxes = []
                for i in range(num_detections):
                    # Generate confidence based on hash and model
                    base_conf = 0.3 + (((img_hash >> (i * 4)) % 10) / 10) * 0.5
                    conf = min(1.0, base_conf + self.confidence_boost)
                    
                    if conf >= confidence_threshold:
                        boxes.append(DetectionBox(
                            x1=0.1 + i * 0.2,
                            y1=0.1 + i * 0.1,
                            x2=0.3 + i * 0.2,
                            y2=0.4 + i * 0.1,
                            class_name=["person", "car", "dog", "cat"][i % 4],
                            class_id=i % 4,
                            confidence=conf,
                        ))
                
                return DetectionResult(
                    confidence=max(b.confidence for b in boxes) if boxes else 0.0,
                    inference_time_ms=10.0,
                    model_name=self.model_id,
                    boxes=boxes,
                )
        
        print(f"  ✓ Loaded model: {model_id}")
        return MockModel(model_id)
    
    # Initialize detector with cascade
    detector = StreamingVisionDetector(
        model_loader=mock_model_loader,
        replay_buffer=replay_buffer,
        image_store=image_store,
    )
    
    # Track when training would be triggered
    training_triggered = False
    
    async def on_training_trigger(buffer_size: int):
        nonlocal training_triggered
        if buffer_size >= 5:  # Low threshold for demo
            training_triggered = True
            print(f"\n🚀 TRAINING TRIGGERED! Buffer size: {buffer_size}")
    
    detector.set_training_trigger(on_training_trigger)
    
    print("  ✓ Streaming detector initialized")
    print("  ✓ Replay buffer ready (max size: 100)")
    print("  ✓ Image store ready")
    
    # Start session with cascade
    print("\n📹 Starting streaming session with cascade...")
    
    config = StreamingConfig(
        confidence_threshold=0.7,
        escalation_threshold=0.5,
        action_classes=["person", "car"],
        cooldown_seconds=0.1,  # Short for demo
        cascade=CascadeConfig(
            secondary_model_id="yolov8m",
            feedback_to_primary=True,
            save_uncertain_images=True,
        ),
    )
    
    session = await detector.start_session(
        model_id="yolov8n",
        config=config,
    )
    
    print(f"  ✓ Session started: {session.session_id}")
    print(f"  ✓ Primary model: yolov8n")
    print(f"  ✓ Secondary model: yolov8m")
    print(f"  ✓ Confidence threshold: {config.confidence_threshold}")
    print(f"  ✓ Escalation threshold: {config.escalation_threshold}")
    
    # Process some mock frames
    print("\n🎬 Processing frames...")
    print("-" * 60)
    
    results_summary = {
        "ok": 0,
        "action": 0,
        "review": 0,
        "escalated": 0,
        "added_to_replay": 0,
    }
    
    for i in range(20):
        # Generate a "frame" (just random bytes for demo)
        frame = f"frame_{i}_content_{i * 137}".encode() * 100
        
        result = await detector.process_frame(
            session_id=session.session_id,
            image=frame,
        )
        
        results_summary[result.status] += 1
        if result.added_to_replay:
            results_summary["added_to_replay"] += 1
        
        # Print interesting results
        if result.status != "ok":
            emoji = {
                "action": "🎯",
                "review": "🔍",
                "escalated": "⬆️",
            }.get(result.status, "❓")
            
            escalation_info = f" (escalated to {result.escalated_to})" if result.escalated_to else ""
            replay_info = " [LEARNED]" if result.added_to_replay else ""
            
            print(f"  Frame {i:2d}: {emoji} {result.status.upper():8s} conf={result.confidence:.2f}{escalation_info}{replay_info}")
    
    # Print summary
    print("-" * 60)
    print("\n📊 Session Summary:")
    print(f"  • Total frames: 20")
    print(f"  • Actions triggered: {results_summary['action']}")
    print(f"  • Sent to review: {results_summary['review']}")
    print(f"  • Added to replay buffer: {results_summary['added_to_replay']}")
    
    # Check replay buffer
    buffer_stats = replay_buffer.get_stats()
    print(f"\n📚 Replay Buffer:")
    print(f"  • Size: {buffer_stats['size']}")
    print(f"  • By source: {buffer_stats['by_source']}")
    print(f"  • Avg priority: {buffer_stats['avg_priority']:.2f}")
    
    # Stop session
    stats = await detector.stop_session(session.session_id)
    
    print(f"\n📈 Session Statistics:")
    print(f"  • Frames processed: {stats['frames_processed']}")
    print(f"  • Actions triggered: {stats['actions_triggered']}")
    print(f"  • Escalations: {stats['escalations_to_secondary']}")
    print(f"  • Secondary successes: {stats['secondary_successes']}")
    print(f"  • Secondary success rate: {stats['secondary_success_rate']:.1%}")
    print(f"  • Review queue: {stats['review_queue_count']}")
    
    # Simulate corrections
    print("\n✏️  Simulating human corrections...")
    
    for i in range(3):
        await detector.submit_correction(
            session_id="manual",
            image_id=f"correction_{i}",
            corrected_class="truck",
            original_confidence=0.4,
        )
    
    buffer_stats = replay_buffer.get_stats()
    print(f"  • Added 3 corrections")
    print(f"  • Buffer size now: {buffer_stats['size']}")
    print(f"  • By source: {buffer_stats['by_source']}")
    
    # Training trigger check
    if training_triggered:
        print("\n✅ Auto-training would have been triggered!")
    else:
        print("\n⏳ Auto-training threshold not yet reached")
    
    # Demo complete
    print("\n" + "=" * 60)
    print("  Demo Complete! 🎉")
    print("=" * 60)
    print("""
Key takeaways:
1. Primary model handles most detections quickly
2. Uncertain detections automatically escalate to secondary model
3. Successful secondary results feed back for training
4. Human corrections also feed the replay buffer
5. Training triggers automatically when buffer fills

The cascade pattern gives you:
• Fast inference (80% handled by small model)
• High accuracy (20% verified by large model)
• Continuous improvement (auto-learning from both)
""")


if __name__ == "__main__":
    asyncio.run(main())
