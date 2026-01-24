"""
Drift Router - Endpoints for data drift detection.

Monitors production data for drift from training distributions:
- KS test: Kolmogorov-Smirnov test for univariate numeric drift
- MMD: Maximum Mean Discrepancy for multivariate drift
- Chi-squared: Chi-squared test for categorical drift

Powered by Alibi Detect.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.universal_runtime_service import UniversalRuntimeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drift", tags=["drift"])


# =============================================================================
# Request/Response Types
# =============================================================================


class DriftFitRequest(BaseModel):
    """Request to fit a drift detector."""

    model: str | None = Field(
        default=None,
        description="Model identifier (auto-generated if not provided)",
    )
    detector: str = Field(
        default="ks",
        description="Detector type: ks (numeric), mmd (multivariate), chi_squared (categorical)",
    )
    reference_data: list[list[float]] = Field(
        ...,
        description="Reference data to learn the baseline distribution",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Names for each feature column",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Detector-specific parameters (e.g., p_val)",
    )
    overwrite: bool = Field(
        default=True,
        description="Overwrite existing model if it exists",
    )
    description: str | None = Field(
        default=None,
        description="Optional model description",
    )


class DriftDetectRequest(BaseModel):
    """Request to detect drift in new data."""

    model: str = Field(..., description="Model identifier")
    data: list[list[float]] = Field(
        ...,
        description="New data to check for drift",
    )


class DriftLoadRequest(BaseModel):
    """Request to load a drift model."""

    model: str = Field(..., description="Model identifier (supports -latest suffix)")


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/detectors")
async def list_detectors() -> dict[str, Any]:
    """List available drift detector types.

    Returns:
    - ks: Kolmogorov-Smirnov test (univariate numeric)
    - mmd: Maximum Mean Discrepancy (multivariate)
    - chi_squared: Chi-squared test (categorical)

    Each detector includes:
    - name: Detector identifier
    - description: What the detector does
    - multivariate: Whether it handles multiple features together
    - default_params: Default configuration
    """
    return await UniversalRuntimeService.drift_list_detectors()


@router.post("/fit")
async def fit_detector(request: DriftFitRequest) -> dict[str, Any]:
    """Fit a drift detector on reference data.

    Trains a detector to learn the baseline distribution from your
    training data. This baseline is then used to detect drift in
    production data.

    Auto-saves after fitting and returns the saved path.

    Example request:
    ```json
    {
        "model": "feature-drift-detector",
        "detector": "ks",
        "reference_data": [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9]
        ],
        "feature_names": ["feature_a", "feature_b", "feature_c"]
    }
    ```

    Response includes:
    - model: Model identifier
    - detector: Detector type used
    - saved_path: Path where model was saved
    - reference_size: Number of reference samples
    - n_features: Number of features
    """
    return await UniversalRuntimeService.drift_fit(request.model_dump())


@router.post("/detect")
async def detect_drift(request: DriftDetectRequest) -> dict[str, Any]:
    """Check for drift in new data.

    Compares new data against the reference distribution
    to detect if significant drift has occurred.

    Example request:
    ```json
    {
        "model": "feature-drift-detector",
        "data": [
            [5.0, 6.0, 7.0],
            [5.1, 6.1, 7.1]
        ]
    }
    ```

    Response includes:
    - model: Model identifier
    - result.is_drift: Whether drift was detected (boolean)
    - result.p_value: Statistical p-value
    - result.threshold: Detection threshold used
    - result.distance: Distance measure (detector-specific)
    - detection_time_ms: Time to run detection
    """
    return await UniversalRuntimeService.drift_detect(request.model_dump())


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List all saved drift models.

    Returns models with metadata:
    - name: Model identifier
    - detector: Detector type
    - created_at: Creation timestamp
    - description: Model description (if set)
    - is_fitted: Whether model is fitted
    - reference_size: Number of reference samples
    """
    return await UniversalRuntimeService.drift_list_models()


@router.get("/status/{model_name}")
async def get_status(model_name: str) -> dict[str, Any]:
    """Get status of a drift detector.

    Returns current state including:
    - model: Model identifier
    - detector: Detector type
    - is_fitted: Whether model is fitted
    - reference_size: Number of reference samples
    - detection_count: Total detections run
    - drift_count: Number of times drift was detected
    - last_detection: Result of last detection (if any)
    """
    return await UniversalRuntimeService.drift_status(model_name)


@router.post("/reset/{model_name}")
async def reset_detector(model_name: str) -> dict[str, Any]:
    """Reset a drift detector.

    Clears reference data and statistics but keeps the model.
    Useful for retraining on new baseline data.
    """
    return await UniversalRuntimeService.drift_reset(model_name)


@router.post("/load")
async def load_model(request: DriftLoadRequest) -> dict[str, Any]:
    """Load a saved drift model.

    Supports "-latest" suffix to load the most recent version:
    ```json
    {
        "model": "feature-drift-detector-latest"
    }
    ```
    """
    return await UniversalRuntimeService.drift_load(request.model_dump())


@router.delete("/models/{model_name}")
async def delete_model(model_name: str) -> dict[str, Any]:
    """Delete a saved drift model.

    Args:
        model_name: Model identifier to delete
    """
    # Validate to prevent path traversal
    if "/" in model_name or "\\" in model_name or ".." in model_name:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")

    return await UniversalRuntimeService.drift_delete(model_name)
