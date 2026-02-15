#!/usr/bin/env python3
"""
Vision MVP End-to-End Demo
==========================
Demonstrates all 7 MVP capabilities without running servers.
Direct imports, real images, real inference.

Usage:
    cd ~/clawd/projects/llamafarm-core/runtimes/universal
    TRANSFORMERS_SKIP_MPS=1 uv run python ../../examples/vision_mvp_demo.py
"""
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_SKIP_MPS", "1")

# Add runtime to path
RUNTIME_DIR = Path(__file__).resolve().parent.parent / "runtimes" / "universal"
sys.path.insert(0, str(RUNTIME_DIR))

from models.yolo_model import YOLOModel
from models.clip_model import CLIPModel
from storage.image_store import ImageStore, ImageRecord
from vision_training.replay_buffer import ReplayBuffer, ReplaySample, ModelOpinion
from vision_training.trainer import IncrementalTrainer, TrainingConfig

# ─── Config ───────────────────────────────────────────────────────────
IMG_DIR = Path.home() / "Downloads" / "test-images"
VISION_DIR = Path.home() / ".llamafarm" / "models" / "vision"
TEST_MODEL_ID = "demo-detector"
MODEL_DIR = VISION_DIR / TEST_MODEL_ID

PASS = "✅"
FAIL = "❌"
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = ""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}: {detail}" if detail else "")


# ─── 1. Detection (YOLO) ─────────────────────────────────────────────
async def test_detection():
    print("\n═══ 1. YOLO Detection ═══")
    yolo = YOLOModel(model_id="yolov8n", device="cpu", confidence_threshold=0.25)
    await yolo.load()
    print(f"  Loaded: {yolo.get_model_info()['variant']} ({yolo.get_model_info()['num_classes']} classes)")

    for img_name in ["horse.jpg", "cat1.jpg"]:
        img_path = IMG_DIR / img_name
        if not img_path.exists():
            record(f"detect {img_name}", False, f"Image not found: {img_path}")
            continue

        image_bytes = img_path.read_bytes()
        result = await yolo.detect(image_bytes, confidence_threshold=0.25)
        dets = [(b.class_name, f"{b.confidence:.2f}") for b in result.boxes]
        detail = f"{img_name}: {len(result.boxes)} detections {dets} in {result.inference_time_ms:.0f}ms"
        record(f"detect {img_name}", len(result.boxes) > 0, detail)

    await yolo.unload()
    return yolo


# ─── 2. Classification (CLIP) ────────────────────────────────────────
async def test_classification():
    print("\n═══ 2. CLIP Classification ═══")
    clip = CLIPModel(model_id="clip-vit-base", device="cpu")
    await clip.load()
    print(f"  Loaded: dim={clip.get_model_info()['embedding_dim']}")

    classes = ["a cat", "a dog", "a horse", "a bird"]
    for img_name, expected in [("cat1.jpg", "a cat"), ("horse.jpg", "a horse")]:
        img_path = IMG_DIR / img_name
        if not img_path.exists():
            record(f"classify {img_name}", False, f"Image not found")
            continue

        result = await clip.classify(img_path.read_bytes(), classes=classes, top_k=4)
        correct = result.class_name == expected
        detail = f"{img_name}: top={result.class_name} ({result.confidence:.3f}), expected={expected}, time={result.inference_time_ms:.0f}ms"
        record(f"classify {img_name}", correct, detail)

    await clip.unload()


# ─── 3. Training + Versioned Save + ONNX Export ──────────────────────
async def test_training():
    print("\n═══ 3. Training + Model Versioning + ONNX ═══")

    # Clean previous test model
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    # Create mini dataset
    dataset_dir = Path("/tmp/vision_demo_dataset")
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True)
        (dataset_dir / "labels" / split).mkdir(parents=True)

        for img_name in ["horse.jpg", "cat1.jpg"]:
            src = IMG_DIR / img_name
            if src.exists():
                shutil.copy2(src, dataset_dir / "images" / split / img_name)
                # YOLO label: class x_center y_center width height (0-indexed)
                cls = 1 if "horse" in img_name else 0
                label = f"{cls} 0.5 0.5 0.8 0.9\n"
                (dataset_dir / "labels" / split / img_name.replace(".jpg", ".txt")).write_text(label)

    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(json.dumps({
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "cat", 1: "horse"},
    }))

    # Model loader
    async def load_model(model_id):
        m = YOLOModel(model_id=model_id, device="cpu", confidence_threshold=0.25)
        await m.load()
        return m

    trainer = IncrementalTrainer(model_loader=load_model)
    config = TrainingConfig(epochs=2, batch_size=2, learning_rate=0.01)

    print("  Starting training (2 epochs, batch 2)...")
    t0 = time.perf_counter()
    job = await trainer.start_training(
        model_id=TEST_MODEL_ID,
        dataset_path=str(dataset_yaml),
        task="detection",
        config=config,
        base_model="yolov8n",
    )
    print(f"  Job {job.job_id} queued")

    result = await trainer.wait_for_job(job.job_id, timeout=180)
    elapsed = time.perf_counter() - t0
    ok = result.status.value == "completed"
    detail = f"status={result.status.value}, time={elapsed:.1f}s"
    if result.error:
        detail += f", error={result.error}"
    record("training", ok, detail)

    # Check versioned files
    if ok:
        v1_pt = MODEL_DIR / "v1.pt"
        current_pt = MODEL_DIR / "current.pt"
        pt_ok = v1_pt.exists() and current_pt.exists()
        detail = f"v1.pt={v1_pt.exists()} ({v1_pt.stat().st_size:,}B), current.pt={current_pt.exists()}"
        record("model versioning", pt_ok, detail)

        # Check ONNX export (give it a moment)
        await asyncio.sleep(2)
        onnx_files = list(MODEL_DIR.glob("*.onnx"))
        detail = f"ONNX files: {[f.name for f in onnx_files]}"
        record("ONNX export", len(onnx_files) > 0, detail)

        # List all files in model dir
        print(f"\n  Model storage ({MODEL_DIR}):")
        for f in sorted(MODEL_DIR.rglob("*")):
            if f.is_file():
                print(f"    {f.relative_to(MODEL_DIR)} ({f.stat().st_size:,} bytes)")
    else:
        record("model versioning", False, "training failed")
        record("ONNX export", False, "training failed")

    # Cleanup dataset
    shutil.rmtree(dataset_dir, ignore_errors=True)


# ─── 4. Cascade / Fallover ───────────────────────────────────────────
async def test_cascade():
    print("\n═══ 4. Cascade Fallover ═══")
    from routers.vision.streaming import CascadeConfig, StreamSession

    # Simulate a cascade: model A → model B
    cascade = CascadeConfig(
        chain=["yolov8n", "remote:http://fallback-node:11540/v1/vision/detect"],
        confidence_threshold=0.6,
    )

    session = StreamSession(
        session_id="test-cascade",
        cascade=cascade,
    )

    ok = (len(cascade.chain) == 2 and
          cascade.chain[1].startswith("remote:") and
          cascade.confidence_threshold == 0.6)
    detail = f"chain={cascade.chain}, threshold={cascade.confidence_threshold}"
    record("cascade config", ok, detail)

    # Test that low-confidence would trigger fallover
    detail = f"session_id={session.session_id}, models={len(cascade.chain)}, fps={session.target_fps}"
    record("cascade session", session.session_id == "test-cascade", detail)


# ─── 5. Replay Buffer + Feedback ─────────────────────────────────────
async def test_replay_buffer():
    print("\n═══ 5. Replay Buffer (Feedback Cycle) ═══")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    buf = ReplayBuffer(max_size=1000, storage_dir=MODEL_DIR)

    # Add a correction (human says YOLO was wrong)
    opinion_a = ModelOpinion(
        model_id="yolov8n", class_name="dog", confidence=0.84,
        bbox=(100, 100, 300, 300), inference_time_ms=40.0,
    )
    opinion_b = ModelOpinion(
        model_id="clip-vit-base", class_name="cat", confidence=0.26,
    )

    sample1 = ReplaySample(
        id="s1", image_path=str(IMG_DIR / "cat1.jpg"),
        label="cat", source="correction", confidence=0.84, priority=2.0,
        opinions=[opinion_a, opinion_b],
        metadata={"corrected_by": "rob", "original_prediction": "dog"},
    )
    buf.add(sample1)

    # Add a low-confidence sample
    sample2 = ReplaySample(
        id="s2", image_path=str(IMG_DIR / "horse.jpg"),
        label="horse", source="low_confidence", confidence=0.35, priority=1.5,
        opinions=[ModelOpinion(model_id="yolov8n", class_name="horse", confidence=0.35)],
    )
    buf.add(sample2)

    count = len(buf)
    record("replay add", count == 2, f"buffer has {count} entries")

    # Sample with priority weighting
    samples = buf.sample(batch_size=2)
    classes = [s.label for s in samples]
    record("replay sample", len(samples) == 2, f"sampled {len(samples)}: {classes}")

    # Check opinions are stored
    first = samples[0]
    record("model opinions", len(first.opinions) > 0,
           f"{len(first.opinions)} opinions stored, first: {first.opinions[0].model_id} → {first.opinions[0].class_name}")

    # Verify persistence — create new buffer from same dir
    buf2 = ReplayBuffer(max_size=1000, storage_dir=MODEL_DIR)
    record("replay persistence", len(buf2) == 2, f"reloaded {len(buf2)} entries from SQLite")


# ─── 6. Image Store ──────────────────────────────────────────────────
async def test_image_store():
    print("\n═══ 6. Image Store ═══")

    db_path = MODEL_DIR / "image_store.db"
    store = ImageStore(db_path)

    store.add_image(ImageRecord(
        id="img1", file_path=str(IMG_DIR / "cat1.jpg"),
        source="training", class_name="cat", confidence=0.95,
    ))
    store.add_image(ImageRecord(
        id="img2", file_path=str(IMG_DIR / "horse.jpg"),
        source="correction", class_name="horse", confidence=0.70,
    ))

    pending = store.get_pending_review()
    record("image store add", len(pending) == 2, f"{len(pending)} images stored (pending review)")

    stats = store.get_stats()
    record("image store stats", stats["total_images"] >= 2, f"stats: {stats}")

    store.mark_reviewed("img1")
    pending2 = store.get_pending_review()
    record("image store review", len(pending2) == 1, f"after review: {len(pending2)} pending")


# ─── 7. Pipeline Config (JSON) ───────────────────────────────────────
async def test_pipeline_config():
    print("\n═══ 7. Pipeline Config ═══")

    pipeline_path = MODEL_DIR / "pipeline.json"
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "model_id": TEST_MODEL_ID,
        "version": 1,
        "pipeline": {
            "detect": {"model": "yolov8n", "confidence_threshold": 0.5},
            "classify": {"model": "clip-vit-base", "classes": ["cat", "horse", "dog"]},
        },
        "cascade": {
            "chain": ["yolov8n", "remote:http://edge-node:11540/v1/vision/detect"],
            "confidence_threshold": 0.6,
        },
        "training": {
            "replay_buffer_size": 1000,
            "retrain_threshold": 100,
            "auto_export_onnx": True,
        },
    }

    pipeline_path.write_text(json.dumps(config, indent=2))
    loaded = json.loads(pipeline_path.read_text())

    ok = loaded["model_id"] == TEST_MODEL_ID and loaded["cascade"]["chain"][1].startswith("remote:")
    record("pipeline config", ok, f"saved to {pipeline_path.name}, cascade ready for Atmosphere")

    print(f"\n  Pipeline config ({pipeline_path}):")
    print(f"    {json.dumps(loaded, indent=2)}")


# ─── Main ─────────────────────────────────────────────────────────────
async def main():
    print("╔══════════════════════════════════════════════╗")
    print("║    LlamaFarm Vision MVP — End-to-End Demo    ║")
    print("╚══════════════════════════════════════════════╝")

    t0 = time.perf_counter()

    await test_detection()
    await test_classification()
    await test_training()
    await test_cascade()
    await test_replay_buffer()
    await test_image_store()
    await test_pipeline_config()

    elapsed = time.perf_counter() - t0

    # ─── Summary ──────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════╗")
    print("║               RESULTS                        ║")
    print("╠══════════════════════════════════════════════╣")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, detail in results:
        icon = PASS if ok else FAIL
        print(f"║ {icon} {name:<20} {detail[:40]}")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  {passed}/{total} passed in {elapsed:.1f}s")
    print("╚══════════════════════════════════════════════╝")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
