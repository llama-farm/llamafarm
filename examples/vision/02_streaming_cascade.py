#!/usr/bin/env python3
"""
Step 2: Streaming Sessions with Cascade Pipeline

Demonstrates the real-time vision pipeline:
- Start a streaming session with cascade chain
- Process frames with automatic escalation
- Review the replay buffer
- Submit human corrections
- Check auto-training status

Prerequisites:
    - Universal Runtime running: nx start universal-runtime
    - Run 01_detect_and_classify.py first (to warm up models)

Run:
    cd runtimes/universal
    uv run python ../../examples/vision/02_streaming_cascade.py
"""

import asyncio
import base64
import os
import time
from io import BytesIO
from pathlib import Path

import httpx


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_runtime_url():
    if url := os.environ.get("RUNTIME_URL"):
        return url.rstrip("/")
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("RUNTIME_URL="):
                    return line.split("=", 1)[1].strip().strip('"\'').rstrip("/")
    return "http://localhost:14345"


RUNTIME_URL = get_runtime_url()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_frame(index: int, width: int = 320, height: int = 240) -> bytes:
    """Generate a unique test frame."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return b"\xff\xd8\xff\xe0" + bytes([index % 256]) * 50 + b"\xff\xd9"

    img = Image.new("RGB", (width, height), (
        100 + (index * 7) % 100,
        150 + (index * 13) % 80,
        200 - (index * 3) % 100,
    ))
    draw = ImageDraw.Draw(img)

    # Moving object
    x = 20 + (index * 30) % (width - 80)
    y = 40 + (index * 15) % (height - 80)
    draw.rectangle([(x, y), (x + 60, y + 60)], fill=(255, 0, 0))
    draw.ellipse([(x + 10, y - 20), (x + 50, y)], fill=(255, 220, 185))

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def print_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print_header("LlamaFarm Vision - Streaming Cascade Demo")

    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=60.0) as client:

        # Health check
        try:
            resp = await client.get("/health")
            resp.raise_for_status()
        except Exception as e:
            print(f"Cannot reach runtime at {RUNTIME_URL}: {e}")
            print("Start the runtime: nx start universal-runtime")
            return

        # ------------------------------------------------------------------
        # 1. Start streaming session with cascade
        # ------------------------------------------------------------------
        print_header("1. Start Streaming Session")
        print("POST /v1/vision/stream/start")
        print()

        start_resp = await client.post("/v1/vision/stream/start", json={
            "model": "yolov8n",
            "source_id": "demo-camera-1",
            "cascade": {
                "enabled": True,
                "cascade_chain": ["yolov8n", "yolov8m"],
                "confidence_threshold": 0.7,
                "enrich_on_escalation": True,
            },
        })

        if start_resp.status_code != 200:
            print(f"  Failed to start session: {start_resp.text[:200]}")
            return

        session = start_resp.json()
        session_id = session.get("session_id", "unknown")
        print(f"  Session ID: {session_id}")
        print(f"  Model: {session.get('model', 'yolov8n')}")
        print(f"  Cascade: {session.get('cascade', {})}")

        # ------------------------------------------------------------------
        # 2. Process frames
        # ------------------------------------------------------------------
        print_header("2. Process Frames (10 frames)")
        print("POST /v1/vision/stream/frame")
        print()

        results = {"action": 0, "ok": 0, "review": 0, "escalated": 0}
        total_time = 0

        for i in range(10):
            frame = generate_frame(i)
            t0 = time.perf_counter()

            resp = await client.post("/v1/vision/stream/frame", json={
                "session_id": session_id,
                "image": b64(frame),
            })

            elapsed = (time.perf_counter() - t0) * 1000
            total_time += elapsed

            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "ok")
                conf = data.get("confidence", 0)
                results[status] = results.get(status, 0) + 1

                icon = {"action": "!!", "review": "??", "escalated": "^^"}.get(status, "  ")
                print(f"  Frame {i:2d}: [{icon}] {status:10s} conf={conf:.3f}  {elapsed:.0f}ms")
            else:
                print(f"  Frame {i:2d}: ERROR {resp.status_code}")

        avg = total_time / 10
        print()
        print(f"  Average frame time: {avg:.1f}ms")
        print(f"  Results: {results}")

        # ------------------------------------------------------------------
        # 3. Check session stats
        # ------------------------------------------------------------------
        print_header("3. Session Statistics")
        print(f"GET /v1/vision/stream/sessions/{session_id}/stats")
        print()

        resp = await client.get(f"/v1/vision/stream/sessions/{session_id}/stats")
        if resp.status_code == 200:
            stats = resp.json()
            for k, v in stats.items():
                if not k.startswith("_"):
                    print(f"  {k}: {v}")
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 4. Check replay buffer
        # ------------------------------------------------------------------
        print_header("4. Replay Buffer Status")
        print("GET /v1/vision/replay-buffer")
        print()

        resp = await client.get("/v1/vision/replay-buffer")
        if resp.status_code == 200:
            buf = resp.json()
            print(f"  Size: {buf.get('size', 0)}")
            print(f"  By source: {buf.get('by_source', {})}")
            print(f"  Avg priority: {buf.get('avg_priority', 0):.2f}")
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 5. Submit a correction
        # ------------------------------------------------------------------
        print_header("5. Submit Human Correction")
        print("POST /v1/vision/corrections")
        print()

        resp = await client.post("/v1/vision/corrections", json={
            "image_id": "demo_correction_001",
            "correct_class": "truck",
            "bbox": [100.0, 150.0, 300.0, 350.0],
            "reviewed_by": "demo_user",
        })
        if resp.status_code == 200:
            print(f"  Correction submitted: {resp.json()}")
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 6. List active sessions
        # ------------------------------------------------------------------
        print_header("6. Active Sessions")
        print("GET /v1/vision/stream/sessions")
        print()

        resp = await client.get("/v1/vision/stream/sessions")
        if resp.status_code == 200:
            sessions = resp.json()
            if isinstance(sessions, list):
                print(f"  Active sessions: {len(sessions)}")
                for s in sessions:
                    print(f"    - {s.get('session_id', '?')} ({s.get('model', '?')})")
            else:
                print(f"  {sessions}")
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 7. Stop session
        # ------------------------------------------------------------------
        print_header("7. Stop Session")
        print("POST /v1/vision/stream/stop")
        print()

        resp = await client.post("/v1/vision/stream/stop", json={
            "session_id": session_id,
        })
        if resp.status_code == 200:
            final = resp.json()
            print(f"  Session stopped. Final stats:")
            for k, v in final.items():
                if not k.startswith("_"):
                    print(f"    {k}: {v}")
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 8. Auto-training status
        # ------------------------------------------------------------------
        print_header("8. Auto-Training Status")
        print("GET /v1/vision/auto-train/status")
        print()

        resp = await client.get("/v1/vision/auto-train/status")
        if resp.status_code == 200:
            status = resp.json()
            for k, v in status.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

    print_header("Done!")
    print("Streaming cascade demo complete.")
    print("Run 03_federation_and_models.py for federation and model management.")


if __name__ == "__main__":
    asyncio.run(main())
