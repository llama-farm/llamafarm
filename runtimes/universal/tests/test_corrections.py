"""Tests for federated corrections endpoints."""

import base64
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _make_correction(label: str = "person", corrected: str = "car") -> dict:
    """Create a minimal correction item."""
    # 1x1 white JPEG
    img = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 50).decode()
    return {
        "image_base64": img,
        "original_label": label,
        "corrected_label": corrected,
        "confidence": 0.45,
    }


@pytest.fixture
def setup():
    from fastapi import FastAPI
    from routers.vision.corrections import router, set_corrections_models_dir

    app = FastAPI()
    app.include_router(router)

    tmpdir = Path(tempfile.mkdtemp())
    set_corrections_models_dir(tmpdir)

    # Create a model dir
    model_dir = tmpdir / "fleet-model"
    model_dir.mkdir()
    (model_dir / "v1.pt").write_bytes(b"weights")

    return TestClient(app), tmpdir, model_dir


class TestUpload:
    def test_single_correction(self, setup):
        client, _, _ = setup
        r = client.post("/v1/vision/models/fleet-model/corrections/upload", json={
            "device_id": "drone-001",
            "corrections": [_make_correction()],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["accepted"] == 1
        assert data["rejected"] == 0
        assert data["total_corrections"] == 1
        assert data["retrain_ready"] is False

    def test_batch_upload(self, setup):
        client, _, _ = setup
        corrections = [_make_correction(f"label-{i}", f"fix-{i}") for i in range(10)]
        r = client.post("/v1/vision/models/fleet-model/corrections/upload", json={
            "device_id": "drone-002",
            "corrections": corrections,
        })
        assert r.status_code == 200
        assert r.json()["accepted"] == 10
        assert r.json()["total_corrections"] == 10

    def test_multiple_devices(self, setup):
        client, _, _ = setup
        for device in ["drone-A", "drone-B", "drone-C"]:
            client.post("/v1/vision/models/fleet-model/corrections/upload", json={
                "device_id": device,
                "corrections": [_make_correction()],
            })
        r = client.get("/v1/vision/models/fleet-model/corrections/stats")
        assert r.json()["total_corrections"] == 3
        assert len(r.json()["by_device"]) == 3

    def test_retrain_threshold(self, setup):
        client, _, _ = setup
        # Upload 50+ corrections to hit threshold
        corrections = [_make_correction(f"l{i}", f"c{i}") for i in range(55)]
        r = client.post("/v1/vision/models/fleet-model/corrections/upload", json={
            "device_id": "drone-fleet",
            "corrections": corrections,
        })
        assert r.json()["retrain_ready"] is True

    def test_invalid_device_id(self, setup):
        client, _, _ = setup
        r = client.post("/v1/vision/models/fleet-model/corrections/upload", json={
            "device_id": "../hack",
            "corrections": [_make_correction()],
        })
        assert r.status_code == 400

    def test_model_not_found(self, setup):
        client, _, _ = setup
        r = client.post("/v1/vision/models/nonexistent/corrections/upload", json={
            "device_id": "drone-001",
            "corrections": [_make_correction()],
        })
        assert r.status_code == 404

    def test_images_saved_to_disk(self, setup):
        client, _, model_dir = setup
        client.post("/v1/vision/models/fleet-model/corrections/upload", json={
            "device_id": "drone-001",
            "corrections": [_make_correction()],
        })
        corrections_dir = model_dir / "training_data" / "corrections"
        assert corrections_dir.exists()
        assert len(list(corrections_dir.glob("*.jpg"))) == 1


class TestStats:
    def test_empty_stats(self, setup):
        client, _, _ = setup
        r = client.get("/v1/vision/models/fleet-model/corrections/stats")
        assert r.status_code == 200
        assert r.json()["total_corrections"] == 0
        assert r.json()["retrain_ready"] is False

    def test_by_label(self, setup):
        client, _, _ = setup
        corrections = [
            _make_correction("person", "car"),
            _make_correction("person", "car"),
            _make_correction("truck", "bus"),
        ]
        client.post("/v1/vision/models/fleet-model/corrections/upload", json={
            "device_id": "d1", "corrections": corrections,
        })
        stats = client.get("/v1/vision/models/fleet-model/corrections/stats").json()
        assert stats["by_label"]["car"] == 2
        assert stats["by_label"]["bus"] == 1


class TestDevices:
    def test_list_devices(self, setup):
        client, _, _ = setup
        for d in ["alpha", "beta"]:
            client.post("/v1/vision/models/fleet-model/corrections/upload", json={
                "device_id": d,
                "corrections": [_make_correction()] * 3,
            })
        r = client.get("/v1/vision/models/fleet-model/corrections/devices")
        assert r.status_code == 200
        assert r.json()["total_devices"] == 2
        device_ids = [d["device_id"] for d in r.json()["devices"]]
        assert "alpha" in device_ids
        assert "beta" in device_ids


class TestRetrain:
    def test_trigger_insufficient(self, setup):
        client, _, _ = setup
        # Upload only 5 corrections
        client.post("/v1/vision/models/fleet-model/corrections/upload", json={
            "device_id": "d1",
            "corrections": [_make_correction()] * 5,
        })
        r = client.post("/v1/vision/models/fleet-model/corrections/trigger", json={
            "min_corrections": 10,
        })
        assert r.status_code == 422
        assert "Only 5" in r.json()["detail"]

    def test_trigger_success(self, setup):
        client, _, _ = setup
        client.post("/v1/vision/models/fleet-model/corrections/upload", json={
            "device_id": "d1",
            "corrections": [_make_correction()] * 20,
        })
        r = client.post("/v1/vision/models/fleet-model/corrections/trigger", json={
            "min_corrections": 10,
            "epochs": 30,
            "imgsz": 1280,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "triggered"
        assert data["corrections_used"] == 20
        assert data["config"]["epochs"] == 30

    def test_retrain_event_logged(self, setup):
        client, _, _ = setup
        client.post("/v1/vision/models/fleet-model/corrections/upload", json={
            "device_id": "d1",
            "corrections": [_make_correction()] * 15,
        })
        client.post("/v1/vision/models/fleet-model/corrections/trigger", json={
            "min_corrections": 10,
        })
        stats = client.get("/v1/vision/models/fleet-model/corrections/stats").json()
        assert stats["last_retrain"] is not None
