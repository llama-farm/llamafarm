"""TimeSeries API router with 9 endpoints.

Endpoints:
- Forecasting: POST /v1/timeseries/forecast, POST /v1/timeseries/forecast/batch
- Change points: POST /v1/timeseries/changepoints, POST /v1/timeseries/changepoints/batch
- Drift detection: POST /v1/streaming/drift/detect, POST /v1/streaming/drift/create,
                   POST /v1/streaming/drift/update/{id}, GET /v1/streaming/drift/state/{id},
                   DELETE /v1/streaming/drift/{id}
"""

import logging

from fastapi import APIRouter, HTTPException

from .service import timeseries_service
from .types import (
    ChangePointBatchRequest,
    ChangePointRequest,
    DriftDetectionRequest,
    DriftDetectorCreateRequest,
    TimeSeriesForecastBatchRequest,
    TimeSeriesForecastRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# Time-Series Forecasting
# =============================================================================


@router.post("/v1/timeseries/forecast")
async def forecast_timeseries(request: TimeSeriesForecastRequest):
    """
    Generate time-series forecasts using Chronos-Bolt.

    Chronos-Bolt is a transformer-based time-series forecasting model that
    produces probabilistic forecasts with confidence intervals.

    Example request:
    ```json
    {
        "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "horizon": 7,
        "quantiles": [0.1, 0.5, 0.9]
    }
    ```

    Response:
    ```json
    {
        "object": "timeseries_forecast",
        "forecasts": [
            {"step": 1, "point": 8.0, "lower": 7.5, "upper": 8.5},
            {"step": 2, "point": 9.0, "lower": 8.3, "upper": 9.7},
            ...
        ],
        "horizon": 7,
        "input_length": 7
    }
    ```
    """
    try:
        if len(request.values) < 3:
            raise HTTPException(
                status_code=400,
                detail="At least 3 historical values are required",
            )

        if request.horizon < 1:
            raise HTTPException(
                status_code=400,
                detail="Horizon must be at least 1",
            )

        result = await timeseries_service.forecast(
            request.values,
            horizon=request.horizon,
            quantiles=request.quantiles,
            num_samples=request.num_samples,
            model_name=request.model,
        )

        return {
            "object": "timeseries_forecast",
            "forecasts": result["forecasts"],
            "horizon": result["horizon"],
            "input_length": result["input_length"],
            "quantiles": result["quantiles"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in forecast_timeseries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/timeseries/forecast/batch")
async def forecast_timeseries_batch(request: TimeSeriesForecastBatchRequest):
    """
    Generate forecasts for multiple time-series.

    Returns forecasts for each series in the batch.
    """
    try:
        if not request.series:
            raise HTTPException(
                status_code=400,
                detail="At least one time-series is required",
            )

        for i, s in enumerate(request.series):
            if len(s) < 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Series {i} has fewer than 3 values",
                )

        results = await timeseries_service.forecast_batch(
            request.series,
            horizon=request.horizon,
            quantiles=request.quantiles,
            num_samples=request.num_samples,
            model_name=request.model,
        )

        return {
            "object": "timeseries_forecast_batch",
            "results": results,
            "total_series": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in forecast_timeseries_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Change Point Detection
# =============================================================================


@router.post("/v1/timeseries/changepoints")
async def detect_changepoints(request: ChangePointRequest):
    """
    Detect change points in a time-series using ruptures.

    Change points are locations where the statistical properties of the
    time-series (mean, variance, trend) change significantly.

    Algorithms:
    - pelt: Optimal algorithm with linear complexity (default)
    - binseg: Binary segmentation (fast but approximate)
    - window: Sliding window (good for trend changes)
    - bottomup: Bottom-up segmentation

    Models (cost functions):
    - rbf: Radial basis function (default, general purpose)
    - l1: L1 norm (robust to outliers)
    - l2: L2 norm (sensitive to mean shifts)
    - normal: Normal distribution
    - ar: Autoregressive model

    Example request:
    ```json
    {
        "values": [1, 1, 1, 1, 5, 5, 5, 5, 2, 2, 2, 2],
        "algorithm": "pelt",
        "model": "rbf"
    }
    ```

    Response:
    ```json
    {
        "object": "changepoint_detection",
        "change_points": [4, 8],
        "n_segments": 3,
        "segment_boundaries": [
            {"start": 0, "end": 4},
            {"start": 4, "end": 8},
            {"start": 8, "end": 12}
        ]
    }
    ```
    """
    try:
        from utils.changepoint_detector import SUPPORTED_ALGORITHMS, SUPPORTED_MODELS

        if request.algorithm.lower() not in SUPPORTED_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {request.algorithm}. Choose from {SUPPORTED_ALGORITHMS}",
            )

        if request.model.lower() not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}. Choose from {SUPPORTED_MODELS}",
            )

        if len(request.values) < request.min_size * 2:
            raise HTTPException(
                status_code=400,
                detail=f"Signal too short. Need at least {request.min_size * 2} points.",
            )

        result = timeseries_service.detect_changepoints(
            request.values,
            n_changepoints=request.n_changepoints,
            penalty=request.penalty,
            algorithm=request.algorithm,
            model=request.model,
            min_size=request.min_size,
        )

        return {
            "object": "changepoint_detection",
            "change_points": result["change_points"],
            "n_segments": result["n_segments"],
            "segment_boundaries": result["segment_boundaries"],
            "signal_length": result["signal_length"],
            "algorithm": result["algorithm"],
            "model": result["model"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_changepoints: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/timeseries/changepoints/batch")
async def detect_changepoints_batch(request: ChangePointBatchRequest):
    """
    Detect change points in multiple time-series.

    Returns change point results for each series in the batch.
    """
    try:
        from utils.changepoint_detector import SUPPORTED_ALGORITHMS, SUPPORTED_MODELS

        if not request.series:
            raise HTTPException(
                status_code=400,
                detail="At least one time-series is required",
            )

        if request.algorithm.lower() not in SUPPORTED_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {request.algorithm}. Choose from {SUPPORTED_ALGORITHMS}",
            )

        if request.model.lower() not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}. Choose from {SUPPORTED_MODELS}",
            )

        for i, s in enumerate(request.series):
            if len(s) < request.min_size * 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Series {i} too short. Need at least {request.min_size * 2} points.",
                )

        results = timeseries_service.detect_changepoints_batch(
            request.series,
            n_changepoints=request.n_changepoints,
            penalty=request.penalty,
            algorithm=request.algorithm,
            model=request.model,
            min_size=request.min_size,
        )

        return {
            "object": "changepoint_detection_batch",
            "results": results,
            "total_series": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_changepoints_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Concept Drift Detection
# =============================================================================


@router.post("/v1/streaming/drift/detect")
async def detect_drift(request: DriftDetectionRequest):
    """
    Detect concept drift in a data stream using River.

    Concept drift occurs when the statistical properties of data change over time,
    indicating that an ML model may need retraining.

    Algorithms:
    - adwin: ADaptive WINdowing - detects changes in mean (default)
    - page_hinkley: Page-Hinkley test - detects changes in mean
    - kswin: Kolmogorov-Smirnov Windowing - detects distribution changes
    - ddm: Drift Detection Method - monitors error rate

    Example request:
    ```json
    {
        "values": [1.0, 1.1, 1.0, 0.9, 1.0, 5.0, 5.1, 4.9, 5.0, 5.1],
        "algorithm": "adwin"
    }
    ```

    Response:
    ```json
    {
        "object": "drift_detection",
        "drift_detected": true,
        "drift_points": [6],
        "total_processed": 10
    }
    ```
    """
    try:
        from utils.drift_detector import SUPPORTED_ALGORITHMS

        if request.algorithm.lower() not in SUPPORTED_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {request.algorithm}. Choose from {SUPPORTED_ALGORITHMS}",
            )

        if len(request.values) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 values are required for drift detection",
            )

        result = timeseries_service.detect_drift(
            request.values,
            algorithm=request.algorithm,
            delta=request.delta,
            threshold=request.threshold,
            alpha=request.alpha,
            window_size=request.window_size,
        )

        return {
            "object": "drift_detection",
            "drift_detected": result["drift_detected"],
            "drift_points": result["drift_points"],
            "total_processed": result["total_processed"],
            "algorithm": result["algorithm"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_drift: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/streaming/drift/create")
async def create_drift_detector(request: DriftDetectorCreateRequest):
    """
    Create a stateful drift detector for streaming updates.

    Returns a detector_id that can be used for subsequent update calls.
    """
    try:
        from utils.drift_detector import SUPPORTED_ALGORITHMS

        if request.algorithm.lower() not in SUPPORTED_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {request.algorithm}. Choose from {SUPPORTED_ALGORITHMS}",
            )

        result = timeseries_service.create_drift_detector(
            algorithm=request.algorithm,
            detector_id=request.detector_id,
            delta=request.delta,
            threshold=request.threshold,
            alpha=request.alpha,
            window_size=request.window_size,
        )

        return {
            "object": "drift_detector",
            "detector_id": result["detector_id"],
            "algorithm": result["algorithm"],
            "status": result["status"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_drift_detector: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/streaming/drift/update/{detector_id}")
async def update_drift_detector(detector_id: str, value: float):
    """
    Update a drift detector with a new value.

    Args:
        detector_id: ID of the detector (from create endpoint)
        value: New data point

    Returns drift detection result.
    """
    try:
        result = timeseries_service.update_drift_detector(detector_id, value)

        return {
            "object": "drift_update",
            "detector_id": result["detector_id"],
            "drift_detected": result["drift_detected"],
            "index": result["index"],
            "value": result["value"],
        }

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error in update_drift_detector: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v1/streaming/drift/state/{detector_id}")
async def get_drift_detector_state(detector_id: str):
    """
    Get the current state of a drift detector.
    """
    try:
        result = timeseries_service.get_drift_detector_state(detector_id)

        return {
            "object": "drift_detector_state",
            **result,
        }

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error in get_drift_detector_state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/v1/streaming/drift/{detector_id}")
async def delete_drift_detector(detector_id: str):
    """
    Delete a drift detector.
    """
    try:
        result = timeseries_service.delete_drift_detector(detector_id)

        return {
            "object": "drift_detector",
            "detector_id": result["detector_id"],
            "status": result["status"],
        }

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error in delete_drift_detector: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
