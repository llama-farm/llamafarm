"""Tests for model versioning and deployment tracking."""

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def setup():
    """Create test client with a model directory containing versioned files."""
    from fastapi import FastAPI
    from routers.vision.deployments import router, set_deployments_models_dir

    app = FastAPI()
    app.include_router(router)

    tmpdir = Path(tempfile.mkdtemp())
    set_deployments_models_dir(tmpdir)

    # Create a model with versions
    model_dir = tmpdir / "test-drone-model"
    model_dir.mkdir()
    (model_dir / "v1.pt").write_bytes(b"fake-weights-v1")
    (model_dir / "v2.pt").write_bytes(b"fake-weights-v2-larger-file")
    (model_dir / "v2.onnx").write_bytes(b"fake-onnx-v2")
    (model_dir / "metadata.json").write_text(json.dumps({"name": "test-drone-model"}))

    return TestClient(app), tmpdir, model_dir


class TestVersions:
    def test_list_versions(self, setup):
        client, _, _ = setup
        r = client.get("/v1/vision/models/test-drone-model/versions")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 2
        versions = [v["version"] for v in data["versions"]]
        assert 1 in versions
        assert 2 in versions

    def test_list_versions_sorted(self, setup):
        client, _, _ = setup
        r = client.get("/v1/vision/models/test-drone-model/versions")
        versions = r.json()["versions"]
        assert versions[0]["version"] < versions[-1]["version"]

    def test_promote_version(self, setup):
        client, _, model_dir = setup
        r = client.post("/v1/vision/models/test-drone-model/versions/promote", json={"version": 2})
        assert r.status_code == 200
        data = r.json()
        assert data["promoted_version"] == 2
        assert (model_dir / "current.pt").exists() or (model_dir / "current.pt").is_symlink()

    def test_promote_nonexistent_version(self, setup):
        client, _, _ = setup
        r = client.post("/v1/vision/models/test-drone-model/versions/promote", json={"version": 99})
        assert r.status_code == 404

    def test_promote_marks_outdated(self, setup):
        client, _, _ = setup
        # Register a deployment at v1
        client.post("/v1/vision/models/test-drone-model/deployments", json={
            "device_id": "drone-001", "model_version": 1,
        })
        client.patch("/v1/vision/models/test-drone-model/deployments/drone-001", json={
            "status": "deployed",
        })
        # Promote to v2
        r = client.post("/v1/vision/models/test-drone-model/versions/promote", json={"version": 2})
        assert r.json()["outdated_deployments"] == 1
        # Check deployment is now outdated
        deps = client.get("/v1/vision/models/test-drone-model/deployments").json()
        assert deps["deployments"][0]["status"] == "outdated"

    def test_model_not_found(self, setup):
        client, _, _ = setup
        r = client.get("/v1/vision/models/nonexistent/versions")
        assert r.status_code == 404


class TestDeployments:
    def test_list_empty(self, setup):
        client, _, _ = setup
        r = client.get("/v1/vision/models/test-drone-model/deployments")
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_register_deployment(self, setup):
        client, _, _ = setup
        r = client.post("/v1/vision/models/test-drone-model/deployments", json={
            "device_id": "drone-001",
            "device_name": "Alpha Drone",
            "model_version": 2,
            "export_format": "onnx",
        })
        assert r.status_code == 200
        assert r.json()["device_id"] == "drone-001"
        assert r.json()["model_version"] == 2
        assert r.json()["status"] == "pending"

    def test_register_auto_version(self, setup):
        client, _, _ = setup
        r = client.post("/v1/vision/models/test-drone-model/deployments", json={
            "device_id": "drone-002",
        })
        assert r.status_code == 200
        # Should pick latest version (2)
        assert r.json()["model_version"] == 2

    def test_update_deployment(self, setup):
        client, _, _ = setup
        client.post("/v1/vision/models/test-drone-model/deployments", json={
            "device_id": "drone-001", "model_version": 1,
        })
        r = client.patch("/v1/vision/models/test-drone-model/deployments/drone-001", json={
            "status": "deployed",
            "last_inference_count": 1500,
        })
        assert r.status_code == 200
        updated = r.json()["updated"]
        assert updated["status"] == "deployed"
        assert updated["last_inference_count"] == 1500

    def test_update_nonexistent(self, setup):
        client, _, _ = setup
        r = client.patch("/v1/vision/models/test-drone-model/deployments/nope", json={
            "status": "deployed",
        })
        assert r.status_code == 404

    def test_delete_deployment(self, setup):
        client, _, _ = setup
        client.post("/v1/vision/models/test-drone-model/deployments", json={
            "device_id": "drone-001", "model_version": 1,
        })
        r = client.delete("/v1/vision/models/test-drone-model/deployments/drone-001")
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        # Verify gone
        deps = client.get("/v1/vision/models/test-drone-model/deployments").json()
        assert deps["total"] == 0

    def test_delete_nonexistent(self, setup):
        client, _, _ = setup
        r = client.delete("/v1/vision/models/test-drone-model/deployments/nope")
        assert r.status_code == 404

    def test_status_summary(self, setup):
        client, _, _ = setup
        client.post("/v1/vision/models/test-drone-model/deployments", json={
            "device_id": "d1", "model_version": 1,
        })
        client.post("/v1/vision/models/test-drone-model/deployments", json={
            "device_id": "d2", "model_version": 2,
        })
        client.patch("/v1/vision/models/test-drone-model/deployments/d2", json={
            "status": "deployed",
        })
        deps = client.get("/v1/vision/models/test-drone-model/deployments").json()
        summary = deps["status_summary"]
        assert summary.get("pending", 0) == 1
        assert summary.get("deployed", 0) == 1

    def test_invalid_device_id(self, setup):
        client, _, _ = setup
        r = client.post("/v1/vision/models/test-drone-model/deployments", json={
            "device_id": "../etc/passwd", "model_version": 1,
        })
        assert r.status_code == 400

    def test_persistence(self, setup):
        """Deployment data persists across reloads."""
        client, tmpdir, model_dir = setup
        client.post("/v1/vision/models/test-drone-model/deployments", json={
            "device_id": "drone-persist", "model_version": 1,
        })
        # Verify deployment.json exists
        assert (model_dir / "deployment.json").exists()
        data = json.loads((model_dir / "deployment.json").read_text())
        assert "drone-persist" in data["deployments"]
