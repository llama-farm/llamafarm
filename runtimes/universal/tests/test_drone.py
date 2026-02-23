"""Tests for drone vision endpoints — geo-referencing, export profiles, training presets."""

import math

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with drone router."""
    from fastapi import FastAPI
    from routers.vision.drone import router, set_drone_models_dir
    from pathlib import Path
    import tempfile

    app = FastAPI()
    app.include_router(router)

    tmpdir = Path(tempfile.mkdtemp())
    set_drone_models_dir(tmpdir)

    return TestClient(app)


class TestExportProfiles:
    def test_list_profiles(self, client):
        r = client.get("/v1/vision/drone/export/profiles")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 3
        names = [p["name"] for p in data["profiles"]]
        assert "drone-nano" in names
        assert "drone-standard" in names
        assert "drone-hd" in names

    def test_drone_nano_profile(self, client):
        r = client.get("/v1/vision/drone/export/profiles")
        profiles = {p["name"]: p for p in r.json()["profiles"]}
        nano = profiles["drone-nano"]
        assert nano["target"] == "tensorrt"
        assert nano["precision"] == "int8"
        assert nano["imgsz"] == 320

    def test_drone_hd_profile(self, client):
        r = client.get("/v1/vision/drone/export/profiles")
        profiles = {p["name"]: p for p in r.json()["profiles"]}
        hd = profiles["drone-hd"]
        assert hd["imgsz"] == 1280
        assert hd["precision"] == "fp16"


class TestTrainPresets:
    def test_list_presets(self, client):
        r = client.get("/v1/vision/drone/train/presets")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 2
        names = [p["name"] for p in data["presets"]]
        assert "aerial" in names
        assert "orthomosaic" in names

    def test_aerial_preset_config(self, client):
        r = client.get("/v1/vision/drone/train/presets")
        presets = {p["name"]: p for p in r.json()["presets"]}
        aerial = presets["aerial"]
        assert aerial["imgsz"] == 1280
        assert aerial["augmentation"]["mosaic"] == 1.0
        assert aerial["augmentation"]["degrees"] == 180.0


class TestGeoReference:
    def test_basic_nadir(self, client):
        """Detection at image center with nadir gimbal should return drone GPS."""
        r = client.post("/v1/vision/drone/geo", json={
            "detections": [{
                "x1": 1870, "y1": 1030, "x2": 1970, "y2": 1130,
                "class_name": "person", "class_id": 0, "confidence": 0.9,
            }],
            "telemetry": {
                "latitude": 30.2672,
                "longitude": -97.7431,
                "altitude_m": 100.0,
                "heading_deg": 0.0,
                "gimbal_pitch_deg": -90.0,
            },
            "camera": {
                "image_width_px": 3840,
                "image_height_px": 2160,
            },
        })
        assert r.status_code == 200
        geo = r.json()["geo_detections"]
        assert len(geo) == 1
        # Center detection should be very close to drone GPS
        assert abs(geo[0]["lat"] - 30.2672) < 0.001
        assert abs(geo[0]["lng"] - (-97.7431)) < 0.001

    def test_offset_detection(self, client):
        """Detection far from center should have different GPS than drone."""
        r = client.post("/v1/vision/drone/geo", json={
            "detections": [{
                "x1": 0, "y1": 0, "x2": 100, "y2": 100,
                "class_name": "car", "class_id": 1, "confidence": 0.8,
            }],
            "telemetry": {
                "latitude": 30.0,
                "longitude": -97.0,
                "altitude_m": 50.0,
                "heading_deg": 0.0,
                "gimbal_pitch_deg": -90.0,
            },
        })
        assert r.status_code == 200
        geo = r.json()["geo_detections"]
        assert len(geo) == 1
        # Top-left detection should be offset from center
        assert geo[0]["lat"] != 30.0 or geo[0]["lng"] != -97.0

    def test_missing_latitude(self, client):
        r = client.post("/v1/vision/drone/geo", json={
            "detections": [{"x1": 0, "y1": 0, "x2": 1, "y2": 1, "class_name": "x", "class_id": 0, "confidence": 0.5}],
            "telemetry": {"altitude_m": 100.0},
        })
        assert r.status_code == 422

    def test_missing_altitude(self, client):
        r = client.post("/v1/vision/drone/geo", json={
            "detections": [{"x1": 0, "y1": 0, "x2": 1, "y2": 1, "class_name": "x", "class_id": 0, "confidence": 0.5}],
            "telemetry": {"latitude": 30.0, "longitude": -97.0},
        })
        assert r.status_code == 422

    def test_heading_rotation(self, client):
        """Heading 90° should swap north/east offsets."""
        base = {
            "detections": [{
                "x1": 3800, "y1": 1030, "x2": 3840, "y2": 1130,  # far right
                "class_name": "x", "class_id": 0, "confidence": 0.5,
            }],
            "telemetry": {
                "latitude": 0.0, "longitude": 0.0, "altitude_m": 100.0,
                "gimbal_pitch_deg": -90.0,
            },
        }

        base["telemetry"]["heading_deg"] = 0.0
        r0 = client.post("/v1/vision/drone/geo", json=base)
        g0 = r0.json()["geo_detections"][0]

        base["telemetry"]["heading_deg"] = 90.0
        r90 = client.post("/v1/vision/drone/geo", json=base)
        g90 = r90.json()["geo_detections"][0]

        # At heading=0, rightward pixel → east offset (lng change)
        # At heading=90, rightward pixel → south offset (lat change, negative)
        assert abs(g0["lng"]) > abs(g0["lat"])  # mostly east
        assert abs(g90["lat"]) > abs(g90["lng"])  # mostly north/south


class TestSessions:
    def test_list_empty(self, client):
        r = client.get("/v1/vision/drone/sessions")
        assert r.status_code == 200
        assert r.json()["count"] == 0


class TestExportValidation:
    def test_invalid_model_id(self, client):
        r = client.post("/v1/vision/drone/export", json={
            "model_id": "../etc/passwd",
            "target": "onnx",
        })
        assert r.status_code == 400

    def test_unknown_profile(self, client):
        r = client.post("/v1/vision/drone/export", json={
            "model_id": "test-model",
            "profile": "nonexistent",
        })
        assert r.status_code == 400
        assert "nonexistent" in r.json()["detail"]

    def test_model_not_found(self, client):
        r = client.post("/v1/vision/drone/export", json={
            "model_id": "no-such-model",
            "target": "onnx",
        })
        assert r.status_code == 404
