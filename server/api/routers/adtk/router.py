"""
ADTK Router - Endpoints for time series anomaly detection.

Provides access to:
- Specialized time series anomaly detectors (Level Shift, Seasonal, Persist, etc.)
- Unsupervised detection for unlabeled time series data

ADTK (Anomaly Detection Toolkit) excels at time-series-specific anomalies
that general-purpose detectors miss.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.universal_runtime_service import UniversalRuntimeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/adtk", tags=["adtk"])


# =============================================================================
# Request/Response Types
# =============================================================================


class ADTKFitRequest(BaseModel):
    """Request to fit an ADTK detector."""

    model: str = Field(default="default", description="Model identifier")
    detector: str = Field(
        default="level_shift",
        description="Detector type (level_shift, seasonal, persist, volatility_shift, threshold, interquartile_range)",
    )
    data: list[dict[str, Any]] = Field(
        ...,
        description="Time series data as list of {timestamp, value} dicts",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Detector-specific parameters (e.g., c, window, side)",
    )
    overwrite: bool = Field(
        default=True,
        description="Overwrite existing model if it exists",
    )
    description: str | None = Field(
        default=None,
        description="Optional model description",
    )


class ADTKDetectRequest(BaseModel):
    """Request to detect anomalies in time series data."""

    model: str | None = Field(
        default=None,
        description="Model name. If None, uses default detector settings.",
    )
    detector: str = Field(
        default="level_shift",
        description="Detector type to use (ignored if model is specified)",
    )
    data: list[dict[str, Any]] = Field(
        ...,
        description="Time series data as list of {timestamp, value} dicts",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Detector-specific parameters (e.g., c, window, side)",
    )


class ADTKLoadRequest(BaseModel):
    """Request to load an ADTK model."""

    model: str = Field(..., description="Model identifier")


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/detectors")
async def list_detectors() -> dict[str, Any]:
    """List available ADTK detector types.

    Returns all supported detectors with descriptions and parameters:
    - level_shift: Detects sudden level changes in time series
    - seasonal: Detects deviations from seasonal patterns
    - persist: Detects sudden spikes/drops that quickly revert
    - volatility_shift: Detects changes in variance/volatility
    - threshold: Simple threshold-based detection
    - interquartile_range: Statistical outlier detection

    Example response:
    ```json
    {
        "object": "list",
        "detectors": [
            {
                "name": "level_shift",
                "description": "Detects sudden level changes in time series",
                "parameters": {"c": 6.0, "side": "both", "window": 5}
            }
        ]
    }
    ```
    """
    return await UniversalRuntimeService.adtk_list_detectors()


@router.post("/fit")
async def fit_detector(request: ADTKFitRequest) -> dict[str, Any]:
    """Fit an ADTK detector on time series data.

    Trains a detector to learn patterns in the data, enabling
    anomaly detection on new observations.

    Auto-saves after fitting and returns the saved path.

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "detector": "level_shift",
        "data": [
            {"timestamp": "2024-01-01T00:00:00", "value": 100.0},
            {"timestamp": "2024-01-01T01:00:00", "value": 102.0},
            {"timestamp": "2024-01-01T02:00:00", "value": 98.0}
        ],
        "params": {"c": 6.0, "window": 5}
    }
    ```

    Response includes:
    - model: Model identifier
    - detector: Detector type used
    - saved_path: Path where model was saved
    - training_time_ms: Time to fit the model
    """
    return await UniversalRuntimeService.adtk_fit(request.model_dump())


@router.post("/detect")
async def detect_anomalies(request: ADTKDetectRequest) -> dict[str, Any]:
    """Detect anomalies in time series data.

    Can use a saved model or default detector settings for ad-hoc detection.

    Example with saved model:
    ```json
    {
        "model": "sensor-detector",
        "data": [
            {"timestamp": "2024-01-02T00:00:00", "value": 100.0},
            {"timestamp": "2024-01-02T01:00:00", "value": 500.0},
            {"timestamp": "2024-01-02T02:00:00", "value": 102.0}
        ]
    }
    ```

    Example with ad-hoc detection (no model):
    ```json
    {
        "detector": "spike",
        "data": [
            {"timestamp": "2024-01-02T00:00:00", "value": 100.0},
            {"timestamp": "2024-01-02T01:00:00", "value": 500.0}
        ],
        "params": {"c": 1.5}
    }
    ```

    Response includes:
    - model: Model identifier (null if ad-hoc)
    - detector: Detector type used
    - anomalies: List of detected anomaly timestamps with scores
    - detection_time_ms: Time to run detection
    """
    return await UniversalRuntimeService.adtk_detect(request.model_dump())


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List all saved ADTK models.

    Returns models with metadata:
    - name: Model identifier
    - detector: Detector type
    - created_at: Creation timestamp
    - description: Model description (if set)
    """
    return await UniversalRuntimeService.adtk_list_models()


@router.post("/load")
async def load_model(request: ADTKLoadRequest) -> dict[str, Any]:
    """Load a saved ADTK model.

    Supports "-latest" suffix to load the most recent version:
    ```json
    {
        "model": "sensor-detector-latest"
    }
    ```
    """
    return await UniversalRuntimeService.adtk_load(request.model_dump())


@router.delete("/models/{model_name}")
async def delete_model(model_name: str) -> dict[str, Any]:
    """Delete a saved ADTK model.

    Args:
        model_name: Model identifier to delete
    """
    # Validate to prevent path traversal
    if "/" in model_name or "\\" in model_name or ".." in model_name:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")

    return await UniversalRuntimeService.adtk_delete(model_name)
