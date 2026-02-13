"""Tests for ONNX export pipeline: endpoint, auto-export, quantization mapping."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Export endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_export_model():
    """Create a mock model with an export() method."""
    model = AsyncMock()
    model.export = AsyncMock(return_value="/tmp/exports/yolov8n.onnx")
    return model


@pytest.fixture
def export_app(mock_export_model, tmp_path):
    """FastAPI app with the models router wired up."""
    from routers.vision.models import (
        _load_model_fn,
        _VISION_MODELS_DIR,
        router,
        set_model_export_loader,
        set_vision_models_dir,
    )

    # Save original global state
    import routers.vision.models as _models_mod
    original_loader = _models_mod._load_model_fn
    original_models_dir = _models_mod._VISION_MODELS_DIR

    async def mock_loader(model_id: str):
        return mock_export_model

    set_model_export_loader(mock_loader)
    set_vision_models_dir(tmp_path / "models")

    app = FastAPI()
    app.include_router(router)
    yield app

    # Restore original global state
    _models_mod._load_model_fn = original_loader
    _models_mod._VISION_MODELS_DIR = original_models_dir


@pytest.fixture
def export_client(export_app):
    return TestClient(export_app)


def test_export_endpoint_onnx_fp16(export_client, mock_export_model, tmp_path):
    """Test ONNX export with fp16 quantization."""
    # Create a fake export file so stat() works
    exports_dir = tmp_path / "models" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    fake_onnx = Path("/tmp/exports/yolov8n.onnx")
    fake_onnx.parent.mkdir(parents=True, exist_ok=True)
    fake_onnx.write_bytes(b"\x00" * 1024)

    try:
        resp = export_client.post("/v1/vision/models/export", json={
            "model_id": "yolov8n",
            "format": "onnx",
            "quantization": "fp16",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == "onnx"
        assert data["export_path"] == "/tmp/exports/yolov8n.onnx"
        assert data["size_mb"] >= 0
        assert data["export_time_seconds"] >= 0

        # Verify half=True was passed for fp16
        mock_export_model.export.assert_called_once()
        call_kwargs = mock_export_model.export.call_args
        assert call_kwargs.kwargs.get("half") is True
        assert call_kwargs.kwargs.get("int8", False) is False
    finally:
        fake_onnx.unlink(missing_ok=True)


def test_export_endpoint_int8(export_client, mock_export_model):
    """Test int8 quantization maps to int8=True."""
    fake_onnx = Path("/tmp/exports/yolov8n.onnx")
    fake_onnx.parent.mkdir(parents=True, exist_ok=True)
    fake_onnx.write_bytes(b"\x00" * 512)

    try:
        resp = export_client.post("/v1/vision/models/export", json={
            "model_id": "yolov8n",
            "format": "onnx",
            "quantization": "int8",
        })
        assert resp.status_code == 200

        call_kwargs = mock_export_model.export.call_args
        assert call_kwargs.kwargs.get("int8") is True
        assert call_kwargs.kwargs.get("half", False) is False
    finally:
        fake_onnx.unlink(missing_ok=True)


def test_export_endpoint_fp32(export_client, mock_export_model):
    """Test fp32 quantization passes neither half nor int8."""
    fake_onnx = Path("/tmp/exports/yolov8n.onnx")
    fake_onnx.parent.mkdir(parents=True, exist_ok=True)
    fake_onnx.write_bytes(b"\x00" * 2048)

    try:
        resp = export_client.post("/v1/vision/models/export", json={
            "model_id": "yolov8n",
            "format": "onnx",
            "quantization": "fp32",
        })
        assert resp.status_code == 200

        call_kwargs = mock_export_model.export.call_args
        assert call_kwargs.kwargs.get("half", False) is False
        assert call_kwargs.kwargs.get("int8", False) is False
    finally:
        fake_onnx.unlink(missing_ok=True)


def test_export_endpoint_no_loader(tmp_path):
    """Test export fails gracefully when loader not initialized."""
    from routers.vision.models import (
        router,
        set_model_export_loader,
        set_vision_models_dir,
    )

    set_model_export_loader(None)
    set_vision_models_dir(tmp_path / "models")

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.post("/v1/vision/models/export", json={
        "model_id": "yolov8n",
        "format": "onnx",
        "quantization": "fp16",
    })
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Auto-export tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_export_after_promotion():
    """Test that _auto_export is called after successful validation gate promotion."""
    from vision_training.auto_trainer import AutoTrainConfig, AutoTrainer

    mock_buffer = MagicMock()
    mock_buffer.__len__ = MagicMock(return_value=100)
    mock_trainer = AsyncMock()
    mock_gate = AsyncMock()

    config = AutoTrainConfig(
        auto_export_onnx=True,
        export_quantization="fp16",
        export_formats=["onnx"],
    )

    auto_trainer = AutoTrainer(
        replay_buffer=mock_buffer,
        trainer=mock_trainer,
        config=config,
        validation_gate=mock_gate,
    )

    # Mock the _auto_export method directly
    auto_trainer._auto_export = AsyncMock(return_value=["/tmp/model.onnx"])

    # Simulate what _wait_for_completion does after promotion
    mock_job = MagicMock()
    mock_job.status.value = "completed"
    mock_job.metrics = {"model_path": "/tmp/model.pt"}
    mock_trainer.wait_for_job = AsyncMock(return_value=mock_job)

    mock_validation = MagicMock()
    mock_validation.passed = True
    mock_validation.improvement = 0.05
    mock_validation.reason = "Better mAP"
    mock_gate.validate_candidate = AsyncMock(return_value=mock_validation)

    mock_version = MagicMock()
    mock_version.version = 2
    mock_gate.promote = AsyncMock(return_value=mock_version)

    await auto_trainer._wait_for_completion("job-1", 50)

    # Verify auto-export was called
    auto_trainer._auto_export.assert_called_once_with("/tmp/model.pt")


@pytest.mark.asyncio
async def test_auto_export_disabled():
    """Test that auto-export is skipped when disabled."""
    from vision_training.auto_trainer import AutoTrainConfig, AutoTrainer

    mock_buffer = MagicMock()
    mock_buffer.__len__ = MagicMock(return_value=100)
    mock_trainer = AsyncMock()
    mock_gate = AsyncMock()

    config = AutoTrainConfig(auto_export_onnx=False)

    auto_trainer = AutoTrainer(
        replay_buffer=mock_buffer,
        trainer=mock_trainer,
        config=config,
        validation_gate=mock_gate,
    )

    auto_trainer._auto_export = AsyncMock()

    mock_job = MagicMock()
    mock_job.status.value = "completed"
    mock_job.metrics = {"model_path": "/tmp/model.pt"}
    mock_trainer.wait_for_job = AsyncMock(return_value=mock_job)

    mock_validation = MagicMock()
    mock_validation.passed = True
    mock_validation.improvement = 0.05
    mock_validation.reason = "Better mAP"
    mock_gate.validate_candidate = AsyncMock(return_value=mock_validation)

    mock_version = MagicMock()
    mock_version.version = 2
    mock_gate.promote = AsyncMock(return_value=mock_version)

    await auto_trainer._wait_for_completion("job-1", 50)

    auto_trainer._auto_export.assert_not_called()


@pytest.mark.asyncio
async def test_auto_export_failure_is_non_fatal():
    """Test that auto-export failure doesn't crash the training pipeline."""
    from vision_training.auto_trainer import AutoTrainConfig, AutoTrainer

    mock_buffer = MagicMock()
    mock_buffer.__len__ = MagicMock(return_value=100)
    mock_trainer = AsyncMock()
    mock_gate = AsyncMock()

    config = AutoTrainConfig(auto_export_onnx=True)

    auto_trainer = AutoTrainer(
        replay_buffer=mock_buffer,
        trainer=mock_trainer,
        config=config,
        validation_gate=mock_gate,
    )

    # Make auto-export raise an exception
    auto_trainer._auto_export = AsyncMock(side_effect=RuntimeError("Export failed"))

    mock_job = MagicMock()
    mock_job.status.value = "completed"
    mock_job.metrics = {"model_path": "/tmp/model.pt"}
    mock_trainer.wait_for_job = AsyncMock(return_value=mock_job)

    mock_validation = MagicMock()
    mock_validation.passed = True
    mock_validation.improvement = 0.05
    mock_validation.reason = "Better mAP"
    mock_gate.validate_candidate = AsyncMock(return_value=mock_validation)

    mock_version = MagicMock()
    mock_version.version = 2
    mock_gate.promote = AsyncMock(return_value=mock_version)

    # Should NOT raise -- export failure is non-fatal
    await auto_trainer._wait_for_completion("job-1", 50)


# ---------------------------------------------------------------------------
# CLIP export tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clip_export_only_onnx():
    """Test that CLIP export rejects non-ONNX formats."""
    from models.clip_model import CLIPEmbedder

    embedder = CLIPEmbedder("clip-vit-base")

    with pytest.raises(ValueError, match="CLIP export only supports ONNX"):
        await embedder.export(format="coreml", output_path="/tmp/clip_export")


@pytest.mark.asyncio
async def test_clip_export_onnx():
    """Test CLIP ONNX export with mocked optimum."""
    from models.clip_model import CLIPEmbedder

    embedder = CLIPEmbedder("clip-vit-base")

    with tempfile.TemporaryDirectory() as tmpdir:
        mock_ort_model = MagicMock()
        mock_tokenizer_inst = MagicMock()

        # Build mock modules with proper __spec__ so transformers doesn't choke
        mock_optimum = MagicMock()
        mock_optimum.__spec__ = MagicMock()
        mock_optimum_onnx = MagicMock()
        mock_optimum_onnx.__spec__ = MagicMock()
        mock_optimum_onnx.ORTModelForFeatureExtraction.from_pretrained.return_value = mock_ort_model
        mock_optimum.onnxruntime = mock_optimum_onnx

        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

        saved_modules = {}
        modules_to_mock = {
            "optimum": mock_optimum,
            "optimum.onnxruntime": mock_optimum_onnx,
        }
        for mod_name, mock_mod in modules_to_mock.items():
            saved_modules[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = mock_mod

        try:
            with patch("transformers.AutoTokenizer", mock_auto_tokenizer):
                result = await embedder.export(format="onnx", output_path=tmpdir)

                assert result.endswith("model.onnx")
                mock_optimum_onnx.ORTModelForFeatureExtraction.from_pretrained.assert_called_once()
                mock_ort_model.save_pretrained.assert_called_once()
                mock_tokenizer_inst.save_pretrained.assert_called_once()
        finally:
            for mod_name, original in saved_modules.items():
                if original is None:
                    sys.modules.pop(mod_name, None)
                else:
                    sys.modules[mod_name] = original


# ---------------------------------------------------------------------------
# AutoTrainConfig defaults
# ---------------------------------------------------------------------------


def test_auto_train_config_defaults():
    """Test that AutoTrainConfig has correct export defaults."""
    from vision_training.auto_trainer import AutoTrainConfig

    config = AutoTrainConfig()

    assert config.auto_export_onnx is True
    assert config.export_quantization == "fp16"
    assert config.export_formats == ["onnx"]
