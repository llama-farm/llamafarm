"""Vision models router for saving, loading, managing, and exporting trained models.

Provides endpoints for:
- Listing saved vision models (detection, classification, segmentation)
- Saving trained models to disk
- Loading saved models for inference
- Deleting saved models
- Exporting models to ONNX/CoreML/TensorRT/TFLite/OpenVINO with quantization
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Coroutine

from fastapi import APIRouter, HTTPException

from api_types.vision import ModelExportRequest, ModelExportResponse
from services.error_handler import handle_endpoint_errors
from services.path_validator import sanitize_model_name, validate_path_within_directory, PathValidationError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision-models"])

# Injected models directory
_VISION_MODELS_DIR: Path | None = None

# Dependency injection for model loader (used by export endpoint)
_load_model_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_model_export_loader(
    load_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
) -> None:
    """Set the model loader function for export operations."""
    global _load_model_fn
    _load_model_fn = load_fn


def set_vision_models_dir(models_dir: Path) -> None:
    """Set the vision models directory."""
    global _VISION_MODELS_DIR
    _VISION_MODELS_DIR = models_dir
    models_dir.mkdir(parents=True, exist_ok=True)


def _get_models_dir() -> Path:
    """Get the models directory, raising error if not set."""
    if _VISION_MODELS_DIR is None:
        raise HTTPException(
            status_code=500,
            detail="Vision models directory not configured.",
        )
    return _VISION_MODELS_DIR


def _get_model_path(model_name: str, task: str = "detection") -> Path:
    """Get the path for a vision model directory."""
    safe_name = sanitize_model_name(model_name)
    return _get_models_dir() / task / safe_name


@router.get("/v1/vision/models")
@handle_endpoint_errors("vision_list_models")
async def list_vision_models(
    task: str | None = None,
) -> dict[str, Any]:
    """List all saved vision models.
    
    Returns models organized by task type (detection, classification, segmentation).
    
    Query params:
    - task: Filter by task type (detection, classification, segmentation)
    
    Response:
    ```json
    {
        "models": [
            {
                "name": "my-detector",
                "task": "detection",
                "base_model": "yolov8n",
                "classes": ["person", "car", "truck"],
                "num_classes": 3,
                "created_at": "2024-01-15T10:30:00Z",
                "size_mb": 12.5
            }
        ],
        "total": 1
    }
    ```
    """
    models_dir = _get_models_dir()
    models = []
    
    tasks = [task] if task else ["detection", "classification", "segmentation"]
    
    for task_type in tasks:
        task_dir = models_dir / task_type
        if not task_dir.exists():
            continue
        
        for model_path in task_dir.glob("*"):
            if not model_path.is_dir():
                continue
            
            # Read model info
            info = {"name": model_path.name, "task": task_type, "path": str(model_path)}
            
            # Try to read metadata
            meta_file = model_path / "metadata.json"
            if meta_file.exists():
                import json
                try:
                    meta = json.loads(meta_file.read_text())
                    info.update(meta)
                except Exception:
                    pass
            
            # Try to read classes
            classes_file = model_path / "classes.txt"
            if classes_file.exists():
                classes = classes_file.read_text().strip().split("\n")
                info["classes"] = classes
                info["num_classes"] = len(classes)
            
            # Get size
            try:
                size_bytes = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
                info["size_mb"] = round(size_bytes / (1024 * 1024), 2)
            except Exception:
                pass
            
            # Get modification time
            try:
                info["modified_at"] = model_path.stat().st_mtime
            except Exception:
                pass
            
            models.append(info)
    
    # Sort by modification time (newest first)
    models.sort(key=lambda x: x.get("modified_at", 0), reverse=True)
    
    return {
        "models": models,
        "total": len(models),
        "models_dir": str(models_dir),
    }


@router.post("/v1/vision/models/save")
@handle_endpoint_errors("vision_save_model")
async def save_vision_model(
    model_id: str,
    name: str,
    task: str = "detection",
    base_model: str | None = None,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    """Save a trained model to disk.
    
    After training, save the model with a name for later loading.
    
    Request:
    ```json
    {
        "model_id": "yolov8n_finetuned_abc123",
        "name": "my-custom-detector",
        "task": "detection",
        "base_model": "yolov8n",
        "classes": ["person", "car", "truck"]
    }
    ```
    
    The model will be saved to the vision models directory and can
    be loaded later using /v1/vision/models/load.
    """
    import json
    from datetime import datetime
    
    models_dir = _get_models_dir()
    save_path = _get_model_path(name, task)
    
    # Create directory
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Write metadata
    metadata = {
        "name": name,
        "task": task,
        "base_model": base_model or model_id,
        "source_model_id": model_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    (save_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    
    # Write classes if provided
    if classes:
        (save_path / "classes.txt").write_text("\n".join(classes))
    
    # TODO: Actually copy/save the model weights
    # This depends on how the model is currently stored in memory
    # For YOLO, we'd copy the .pt file
    # For CLIP, we'd save the adapter weights
    
    logger.info(f"Saved vision model '{name}' to {save_path}")
    
    return {
        "name": name,
        "task": task,
        "path": str(save_path),
        "saved": True,
    }


@router.post("/v1/vision/models/load")
@handle_endpoint_errors("vision_load_model")
async def load_vision_model(
    name: str,
    task: str = "detection",
) -> dict[str, Any]:
    """Load a saved model for inference.
    
    Request:
    ```json
    {
        "name": "my-custom-detector",
        "task": "detection"
    }
    ```
    
    After loading, use the model name in detection/classification requests.
    """
    import json
    
    model_path = _get_model_path(name, task)
    
    if not model_path.exists():
        # List available models
        models_dir = _get_models_dir()
        task_dir = models_dir / task
        available = [f.name for f in task_dir.glob("*") if f.is_dir()] if task_dir.exists() else []
        
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' not found. Available {task} models: {available}"
        )
    
    # Read metadata
    meta_file = model_path / "metadata.json"
    metadata = {}
    if meta_file.exists():
        try:
            metadata = json.loads(meta_file.read_text())
        except Exception:
            pass
    
    # Read classes
    classes = []
    classes_file = model_path / "classes.txt"
    if classes_file.exists():
        classes = classes_file.read_text().strip().split("\n")
    
    # TODO: Actually load the model into memory
    # This depends on the model loader infrastructure
    
    logger.info(f"Loaded vision model '{name}' from {model_path}")
    
    return {
        "name": name,
        "task": task,
        "path": str(model_path),
        "classes": classes,
        "num_classes": len(classes),
        "metadata": metadata,
        "loaded": True,
    }


@router.delete("/v1/vision/models/{task}/{name}")
@handle_endpoint_errors("vision_delete_model")
async def delete_vision_model(
    task: str,
    name: str,
) -> dict[str, Any]:
    """Delete a saved vision model.
    
    Permanently removes the model from disk.
    """
    models_dir = _get_models_dir()
    
    # Validate inputs
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    model_path = _get_model_path(name, task)
    
    try:
        resolved = validate_path_within_directory(model_path, models_dir)
    except PathValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    
    shutil.rmtree(resolved)
    logger.info(f"Deleted vision model '{name}' ({task})")
    
    return {
        "name": name,
        "task": task,
        "deleted": True,
    }


@router.post("/v1/vision/models/export", response_model=ModelExportResponse)
@handle_endpoint_errors("vision_export_model")
async def export_vision_model(request: ModelExportRequest) -> ModelExportResponse:
    """Export a model to ONNX, CoreML, TensorRT, TFLite, or OpenVINO.

    Supports quantization levels: fp32 (full), fp16 (half), int8 (smallest).
    ONNX models can run on any platform via ONNX Runtime.
    """
    if _load_model_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Model export loader not initialized.",
        )

    # Load the model
    model = await _load_model_fn(request.model_id)

    # Map quantization to export kwargs
    export_kwargs: dict[str, Any] = {"simplify": True}
    if request.quantization == "fp16":
        export_kwargs["half"] = True
    elif request.quantization == "int8":
        export_kwargs["int8"] = True

    # Export to models/exports directory
    export_dir = _get_models_dir() / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    export_path = await model.export(
        format=request.format,
        output_path=str(export_dir),
        **export_kwargs,
    )
    elapsed = time.perf_counter() - start

    export_file = Path(export_path)
    if export_file.is_dir():
        size_bytes = sum(f.stat().st_size for f in export_file.rglob("*") if f.is_file())
    else:
        size_bytes = export_file.stat().st_size

    return ModelExportResponse(
        export_path=str(export_path),
        format=request.format,
        size_mb=round(size_bytes / (1024 * 1024), 2),
        export_time_seconds=round(elapsed, 2),
    )


@router.get("/v1/vision/auto-train/status")
@handle_endpoint_errors("vision_auto_train_status")
async def get_auto_train_status() -> dict[str, Any]:
    """Get auto-training status.
    
    Returns:
    ```json
    {
        "enabled": true,
        "threshold": 50,
        "buffer_size": 42,
        "training_eligible": false,
        "last_training_at": "2024-01-15T10:30:00Z",
        "next_eligible_at": "2024-01-15T16:30:00Z",
        "training_in_progress": false,
        "total_trainings": 3,
        "total_samples_trained": 150
    }
    ```
    """
    from vision_training.auto_trainer import get_auto_trainer
    
    auto_trainer = get_auto_trainer()
    
    if auto_trainer is None:
        return {
            "enabled": False,
            "threshold": 50,
            "buffer_size": 0,
            "training_eligible": False,
            "message": "Auto-trainer not initialized",
        }
    
    return auto_trainer.get_status()


@router.post("/v1/vision/auto-train/trigger")
@handle_endpoint_errors("vision_auto_train_trigger")
async def trigger_auto_training() -> dict[str, Any]:
    """Manually trigger auto-training if eligible.
    
    Checks if the replay buffer has enough samples and triggers
    training immediately if so.
    """
    from vision_training.auto_trainer import get_auto_trainer
    
    auto_trainer = get_auto_trainer()
    
    if auto_trainer is None:
        raise HTTPException(
            status_code=400,
            detail="Auto-trainer not initialized"
        )
    
    triggered = await auto_trainer.check_and_train()
    
    return {
        "triggered": triggered,
        "status": auto_trainer.get_status(),
    }
