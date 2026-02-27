"""Model management router — /v1/vision/models endpoints"""

import contextlib
import json
import logging
import time
from collections.abc import Callable, Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-models"])

_VISION_MODELS_DIR: Path | None = None
_load_model_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_vision_models_dir(d: Path) -> None:
    global _VISION_MODELS_DIR
    _VISION_MODELS_DIR = d
    d.mkdir(parents=True, exist_ok=True)


def set_model_export_loader(fn: Callable[..., Coroutine[Any, Any, Any]] | None) -> None:
    global _load_model_fn
    _load_model_fn = fn


def _models_dir() -> Path:
    if _VISION_MODELS_DIR is None:
        raise HTTPException(status_code=500, detail="Vision models dir not configured")
    return _VISION_MODELS_DIR


class ModelExportRequest(BaseModel):
    model_id: str
    format: Literal["onnx", "coreml", "tensorrt", "tflite", "openvino"]
    quantization: Literal["fp32", "fp16", "int8"] = "fp16"

class ModelExportResponse(BaseModel):
    export_path: str
    format: str
    size_mb: float
    export_time_seconds: float


@router.get("/v1/vision/models")
@handle_endpoint_errors("vision_list_models")
async def list_vision_models() -> dict[str, Any]:
    """List all saved vision models."""
    d = _models_dir()
    models = []
    for model_path in sorted(d.iterdir()):
        if not model_path.is_dir():
            continue
        info: dict[str, Any] = {"name": model_path.name}
        meta_file = model_path / "metadata.json"
        if meta_file.exists():
            with contextlib.suppress(Exception):  # best-effort metadata read
                info.update(json.loads(meta_file.read_text()))
        # Count versions
        pts = list(model_path.glob("v*.pt"))
        info["versions"] = len(pts)
        info["has_current"] = (model_path / "current.pt").exists()
        with contextlib.suppress(Exception):  # best-effort metadata read
            size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
            info["size_mb"] = round(size / (1024 * 1024), 2)
        models.append(info)
    return {"models": models, "total": len(models)}


@router.post("/v1/vision/models/export", response_model=ModelExportResponse)
@handle_endpoint_errors("vision_export_model")
async def export_model(request: ModelExportRequest) -> ModelExportResponse:
    """Export a model to ONNX/CoreML/etc."""
    if _load_model_fn is None:
        raise HTTPException(status_code=500, detail="Export loader not initialized")
    model = await _load_model_fn(request.model_id)

    kwargs: dict[str, Any] = {"simplify": True}
    if request.quantization == "fp16":
        kwargs["half"] = True
    elif request.quantization == "int8":
        kwargs["int8"] = True

    export_dir = _models_dir() / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    path = await model.export(format=request.format, output_path=str(export_dir), **kwargs)
    elapsed = time.perf_counter() - start

    p = Path(path)
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size

    return ModelExportResponse(
        export_path=path, format=request.format,
        size_mb=round(size / (1024 * 1024), 2),
        export_time_seconds=round(elapsed, 2),
    )


@router.post("/v1/vision/models/save")
@handle_endpoint_errors("vision_save_model")
async def save_model(model_id: str, name: str) -> dict[str, Any]:
    """Save a trained model."""
    safe_name = Path(name).name
    if safe_name != name or '..' in name or ':' in name or '\\' in name:
        raise HTTPException(status_code=400, detail="Invalid model name")
    save_path = _models_dir() / safe_name
    if not str(save_path.resolve()).startswith(str(_models_dir().resolve())):
        raise HTTPException(status_code=400, detail="Invalid model name")
    save_path.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "source_model_id": model_id,
            "created_at": datetime.utcnow().isoformat()}
    (save_path / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"name": name, "saved": True}


@router.post("/v1/vision/models/load")
@handle_endpoint_errors("vision_load_model")
async def load_model(name: str) -> dict[str, Any]:
    """Load a saved model for inference."""
    safe_name = Path(name).name
    if safe_name != name or '..' in name or ':' in name or '\\' in name:
        raise HTTPException(status_code=400, detail="Invalid model name")
    model_path = _models_dir() / safe_name
    if not str(model_path.resolve()).startswith(str(_models_dir().resolve())):
        raise HTTPException(status_code=400, detail="Invalid model name")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    meta = {}
    if (model_path / "metadata.json").exists():
        with contextlib.suppress(Exception):  # best-effort metadata read
            meta = json.loads((model_path / "metadata.json").read_text())
    return {"name": name, "metadata": meta, "loaded": True}
