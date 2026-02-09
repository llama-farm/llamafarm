#!/usr/bin/env python3
"""
Step 3: Federation, Model Management, and Training

Demonstrates the distributed learning pipeline:
- Model management (list, save, load)
- Federation peer management
- Training jobs
- Model packaging for distribution
- Review queue

Prerequisites:
    - Universal Runtime running: nx start universal-runtime

Run:
    cd runtimes/universal
    uv run python ../../examples/vision/03_federation_and_models.py
"""

import asyncio
import os
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


def print_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def print_json(data, indent: int = 4):
    """Pretty-print JSON-like data."""
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                print(f"{'':>{indent}}{k}:")
                print_json(v, indent + 4)
            else:
                print(f"{'':>{indent}}{k}: {v}")
    elif isinstance(data, list):
        if not data:
            print(f"{'':>{indent}}(empty)")
        for item in data[:10]:
            if isinstance(item, dict):
                summary = ", ".join(f"{k}={v}" for k, v in list(item.items())[:4])
                print(f"{'':>{indent}}- {summary}")
            else:
                print(f"{'':>{indent}}- {item}")
        if len(data) > 10:
            print(f"{'':>{indent}}... and {len(data) - 10} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print_header("LlamaFarm Vision - Federation & Model Management Demo")

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
        # 1. List vision models
        # ------------------------------------------------------------------
        print_header("1. Available Vision Models")
        print("GET /v1/vision/models")
        print()

        resp = await client.get("/v1/vision/models")
        if resp.status_code == 200:
            models = resp.json()
            if isinstance(models, list):
                print(f"  Found {len(models)} models:")
                for m in models:
                    name = m.get("name", m.get("model_id", "?"))
                    task = m.get("task", "?")
                    print(f"    - {name} ({task})")
            elif isinstance(models, dict):
                print_json(models)
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 2. Save current model
        # ------------------------------------------------------------------
        print_header("2. Save Model")
        print("POST /v1/vision/models/save")
        print()

        resp = await client.post("/v1/vision/models/save", json={
            "model": "yolov8n",
            "task": "detection",
            "name": "demo-checkpoint-v1",
        })
        if resp.status_code == 200:
            result = resp.json()
            print("  Model saved:")
            print_json(result)
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 3. Load a saved model
        # ------------------------------------------------------------------
        print_header("3. Load Saved Model")
        print("POST /v1/vision/models/load")
        print()

        resp = await client.post("/v1/vision/models/load", json={
            "task": "detection",
            "name": "demo-checkpoint-v1",
        })
        if resp.status_code == 200:
            result = resp.json()
            print("  Model loaded:")
            print_json(result)
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 4. Training jobs
        # ------------------------------------------------------------------
        print_header("4. Training Pipeline")

        # List existing jobs
        print("GET /v1/vision/train")
        print()

        resp = await client.get("/v1/vision/train")
        if resp.status_code == 200:
            jobs = resp.json()
            if isinstance(jobs, list):
                print(f"  Training jobs: {len(jobs)}")
                for job in jobs:
                    print(f"    - {job.get('job_id', '?')}: {job.get('status', '?')}")
            else:
                print_json(jobs)
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # Auto-training status
        print()
        print("GET /v1/vision/auto-train/status")
        print()

        resp = await client.get("/v1/vision/auto-train/status")
        if resp.status_code == 200:
            print("  Auto-train status:")
            print_json(resp.json())
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 5. Federation
        # ------------------------------------------------------------------
        print_header("5. Federation Peers")

        # List peers
        print("GET /v1/vision/federation/peers")
        print()

        resp = await client.get("/v1/vision/federation/peers")
        if resp.status_code == 200:
            peers = resp.json()
            if isinstance(peers, list):
                print(f"  Peers: {len(peers)}")
                for p in peers:
                    print(f"    - {p.get('name', '?')} @ {p.get('url', '?')}")
            else:
                print_json(peers)
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # Register a demo peer
        print()
        print("POST /v1/vision/federation/peers")
        print()

        resp = await client.post("/v1/vision/federation/peers", json={
            "name": "demo-gpu-server",
            "url": "http://192.168.1.100:14345",
            "models": ["yolov8x", "yolov8l"],
            "gpu_vram_gb": 24.0,
            "priority": 1,
            "timeout": 30.0,
        })
        if resp.status_code == 200:
            print("  Peer registered:")
            print_json(resp.json())
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # Federation status
        print()
        print("GET /v1/vision/federation/status")
        print()

        resp = await client.get("/v1/vision/federation/status")
        if resp.status_code == 200:
            print("  Federation status:")
            print_json(resp.json())
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 6. Model Packages
        # ------------------------------------------------------------------
        print_header("6. Model Packages")

        # List packages
        print("GET /v1/vision/federation/packages")
        print()

        resp = await client.get("/v1/vision/federation/packages")
        if resp.status_code == 200:
            packages = resp.json()
            if isinstance(packages, list):
                print(f"  Packages: {len(packages)}")
                for pkg in packages:
                    print(f"    - {pkg.get('model_id', '?')} v{pkg.get('version', '?')}")
            else:
                print_json(packages)
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # Create a package
        print()
        print("POST /v1/vision/federation/packages")
        print()

        resp = await client.post("/v1/vision/federation/packages", json={
            "model_id": "yolov8n",
            "description": "Demo detection model package",
        })
        if resp.status_code == 200:
            print("  Package created:")
            print_json(resp.json())
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 7. Cleanup demo peer
        # ------------------------------------------------------------------
        print_header("7. Cleanup")
        print("DELETE /v1/vision/federation/peers/demo-gpu-server")
        print()

        resp = await client.delete("/v1/vision/federation/peers/demo-gpu-server")
        if resp.status_code == 200:
            print("  Demo peer removed")
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

        # Delete demo model checkpoint
        print()
        print("DELETE /v1/vision/models/detection/demo-checkpoint-v1")
        print()

        resp = await client.delete("/v1/vision/models/detection/demo-checkpoint-v1")
        if resp.status_code == 200:
            print("  Demo checkpoint removed")
        else:
            print(f"  {resp.status_code}: {resp.text[:200]}")

    print_header("Demo Complete!")
    print("The vision pipeline is ready for production use.")
    print()
    print("What was demonstrated:")
    print("  - Object detection, classification, segmentation, OCR")
    print("  - Streaming sessions with cascade escalation")
    print("  - Replay buffer for automatic learning")
    print("  - Human corrections feeding the training loop")
    print("  - Model save/load/versioning")
    print("  - Federation peer management")
    print("  - Model packaging for distribution")
    print()
    print("Configure defaults in llamafarm.yaml under the 'vision' key.")
    print("See docs/website/docs/vision/ for full documentation.")


if __name__ == "__main__":
    asyncio.run(main())
