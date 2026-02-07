#!/usr/bin/env python3
"""
Demo: Streaming Detection with Cooldown

Demonstrates:
- Starting a streaming session
- Processing frames with action detection
- Cooldown behavior (suppressing repeated triggers)
- Session statistics
"""

import argparse
import base64
import sys
import time

import httpx

RUNTIME_URL = "http://localhost:11540"

# Simulated video frames (using same image repeatedly with slight variations)
SAMPLE_IMAGE_URL = "https://ultralytics.com/images/zidane.jpg"


def download_image(url: str) -> bytes:
    """Download image from URL."""
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    return response.content


def start_session(
    model: str = "yolov8n",
    target_fps: float = 1.0,
    confidence_threshold: float = 0.7,
    action_classes: list = None,
    cooldown_seconds: float = 5.0,
) -> dict:
    """Start a streaming session."""
    payload = {
        "model": model,
        "target_fps": target_fps,
        "confidence_threshold": confidence_threshold,
        "cooldown_seconds": cooldown_seconds,
    }
    if action_classes:
        payload["action_classes"] = action_classes
    
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/streaming/start",
        json=payload,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def process_frame(session_id: str, image_b64: str) -> dict:
    """Process a frame in the session."""
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/streaming/frame/{session_id}",
        json={"image": image_b64},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def stop_session(session_id: str) -> dict:
    """Stop the session and get stats."""
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/streaming/stop/{session_id}",
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def list_sessions() -> dict:
    """List active sessions."""
    response = httpx.get(
        f"{RUNTIME_URL}/v1/vision/streaming/sessions",
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def run_demo(
    num_frames: int = 10,
    cooldown: float = 3.0,
    frame_interval: float = 0.5,
):
    """Run the streaming demo."""
    print("📹 LlamaFarm Vision - Streaming Detection Demo")
    print("=" * 60)
    
    # Download sample image
    print("\n📥 Downloading sample image...")
    image_data = download_image(SAMPLE_IMAGE_URL)
    image_b64 = base64.b64encode(image_data).decode()
    print(f"  Image size: {len(image_data) / 1024:.1f} KB")
    
    # Start session
    print("\n🎬 Starting streaming session...")
    session = start_session(
        model="yolov8n",
        confidence_threshold=0.5,
        action_classes=["person"],
        cooldown_seconds=cooldown,
    )
    session_id = session["session_id"]
    print(f"  Session ID: {session_id}")
    print(f"  Model: {session['model']}")
    print(f"  Cooldown: {cooldown}s")
    
    # Show active sessions
    sessions = list_sessions()
    print(f"  Active sessions: {len(sessions['sessions'])}")
    
    # Process frames
    print(f"\n📊 Processing {num_frames} frames (interval: {frame_interval}s)...")
    print("-" * 60)
    
    actions = 0
    suppressed = 0
    reviews = 0
    
    for i in range(1, num_frames + 1):
        start_time = time.time()
        
        result = process_frame(session_id, image_b64)
        
        status = result["status"]
        confidence = result.get("confidence", 0)
        detections = result.get("detections", [])
        is_suppressed = result.get("suppressed", False)
        
        # Status indicator
        if status == "action":
            icon = "🚨"
            actions += 1
        elif status == "review":
            icon = "❓"
            reviews += 1
        else:
            if is_suppressed:
                icon = "🔇"
                suppressed += 1
            else:
                icon = "✓ "
        
        # Detection info
        det_info = ""
        if detections:
            classes = [d["class_name"] for d in detections]
            det_info = f" [{', '.join(classes)}]"
        
        print(f"  Frame {i:2d}: {icon} {status:8s} "
              f"conf={confidence:.2f}{det_info}")
        
        # Wait for next frame
        elapsed = time.time() - start_time
        if i < num_frames and frame_interval > elapsed:
            time.sleep(frame_interval - elapsed)
    
    # Stop session
    print("\n⏹️  Stopping session...")
    stats = stop_session(session_id)
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 Session Statistics")
    print("=" * 60)
    print(f"  Frames processed: {stats['frames_processed']}")
    print(f"  Actions triggered: {stats['actions_triggered']}")
    print(f"  Actions suppressed: {suppressed}")
    print(f"  Review queue: {stats['review_queue_size']}")
    print(f"  Duration: {stats['duration_seconds']:.1f}s")
    print(f"  Average FPS: {stats['avg_fps']:.2f}")
    
    # Summary
    print("\n💡 Demo Summary:")
    print(f"   • The first 'person' detection triggered an ACTION")
    print(f"   • Subsequent detections within {cooldown}s were SUPPRESSED")
    print(f"   • After cooldown expired, new ACTIONs could trigger")
    print(f"   • This prevents alert fatigue in real-time monitoring")
    
    print("\n✨ Streaming demo complete!")


def main():
    parser = argparse.ArgumentParser(description="Streaming Detection Demo")
    parser.add_argument("--frames", "-n", type=int, default=10,
                       help="Number of frames to process")
    parser.add_argument("--cooldown", "-c", type=float, default=3.0,
                       help="Cooldown between actions (seconds)")
    parser.add_argument("--interval", "-i", type=float, default=0.5,
                       help="Interval between frames (seconds)")
    args = parser.parse_args()
    
    try:
        run_demo(args.frames, args.cooldown, args.interval)
    except httpx.ConnectError:
        print("❌ Could not connect to runtime!")
        print("   Make sure the runtime is running:")
        print("   cd runtimes/universal && uv run python server.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
