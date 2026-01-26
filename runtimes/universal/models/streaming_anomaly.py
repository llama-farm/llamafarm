"""Streaming anomaly detection with auto-rolling retraining.

Implements the Tick-Tock pattern for real-time anomaly detection:
- Tick: Fast inference on current model
- Tock: Background retraining on accumulated data

Usage:
    detector = StreamingAnomalyDetector(
        model_id="fraud-detector",
        backend="ecod",
        min_samples=50,
        retrain_interval=100,
        window_size=1000,
    )

    # Stream data points
    for transaction in transactions:
        result = await detector.process(transaction)
        if result.is_anomaly:
            print(f"Anomaly detected: {result.score}")
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import polars as pl

from utils.polars_buffer import PolarsBuffer

logger = logging.getLogger(__name__)


class DetectorStatus(str, Enum):
    """Status of the streaming detector."""

    COLLECTING = "collecting"  # Cold start - collecting initial data
    READY = "ready"  # Model trained and ready for inference
    RETRAINING = "retraining"  # Background retraining in progress


@dataclass
class StreamingResult:
    """Result from processing a single data point."""

    index: int
    score: float | None  # None during cold start
    is_anomaly: bool | None  # None during cold start
    raw_score: float | None
    status: DetectorStatus
    samples_collected: int
    samples_until_ready: int  # 0 when ready
    model_version: int


@dataclass
class StreamingBatchResult:
    """Result from processing a batch of data points."""

    results: list[StreamingResult]
    status: DetectorStatus
    samples_collected: int
    model_version: int
    processing_time_ms: float


class StreamingAnomalyDetector:
    """Auto-rolling streaming anomaly detector.

    Implements the Tick-Tock pattern:
    - Fast inference using current trained model
    - Automatic retraining after retrain_interval new samples
    - Sliding window maintains recent data for retraining

    Thread Safety:
        This class is designed for async usage within a single event loop.
        For multi-threaded access, external synchronization is required.
    """

    def __init__(
        self,
        model_id: str,
        backend: str = "ecod",
        min_samples: int = 50,
        retrain_interval: int = 100,
        window_size: int = 1000,
        contamination: float = 0.1,
        threshold: float = 0.5,
        normalization: str = "standardization",
        rolling_windows: list[int] | None = None,
        include_lags: bool = False,
        lag_periods: list[int] | None = None,
        **backend_kwargs: Any,
    ):
        """Initialize the streaming detector.

        Args:
            model_id: Unique identifier for this detector
            backend: PyOD backend to use (ecod, hbos, isolation_forest, etc.)
            min_samples: Minimum samples before first model training
            retrain_interval: Retrain after this many new samples
            window_size: Sliding window size for data buffer
            contamination: Expected proportion of outliers
            threshold: Score threshold for anomaly classification
            normalization: Score normalization method
            rolling_windows: Optional rolling window sizes for feature computation
            include_lags: Whether to include lag features
            lag_periods: Lag periods for lag features
            **backend_kwargs: Additional arguments for PyOD backend
        """
        self.model_id = model_id
        self.backend = backend
        self.min_samples = min_samples
        self.retrain_interval = retrain_interval
        self.window_size = window_size
        self.contamination = contamination
        self.threshold = threshold
        self.normalization = normalization
        self.rolling_windows = rolling_windows
        self.include_lags = include_lags
        self.lag_periods = lag_periods
        self.backend_kwargs = backend_kwargs

        # Data buffer (Polars as substrate - automatic, internal)
        self._buffer = PolarsBuffer(window_size=window_size)

        # Model state
        self._detector = None
        self._model_version = 0
        self._feature_columns: list[str] | None = None
        self._status = DetectorStatus.COLLECTING
        self._samples_since_retrain = 0
        self._total_processed = 0

        # Normalization stats (updated during training)
        self._score_mean = 0.0
        self._score_std = 1.0

        # Retraining lock
        self._retraining = False

    @property
    def status(self) -> DetectorStatus:
        """Current detector status."""
        return self._status

    @property
    def model_version(self) -> int:
        """Current model version (increments on retrain)."""
        return self._model_version

    @property
    def samples_collected(self) -> int:
        """Total samples in buffer."""
        return self._buffer.size

    @property
    def is_ready(self) -> bool:
        """Whether detector is ready for inference."""
        return self._status in (DetectorStatus.READY, DetectorStatus.RETRAINING)

    async def process(
        self,
        data: dict[str, Any] | list[float],
        index: int = 0,
    ) -> StreamingResult:
        """Process a single data point.

        Args:
            data: Feature dict or numeric list
            index: Optional index for tracking

        Returns:
            StreamingResult with score and status
        """
        # Convert to dict if list
        if isinstance(data, list):
            data = {f"f{i}": v for i, v in enumerate(data)}

        # Add to buffer
        self._buffer.append(data)
        self._total_processed += 1
        self._samples_since_retrain += 1

        # Cold start - collecting initial data
        if self._status == DetectorStatus.COLLECTING:
            if self._buffer.size >= self.min_samples:
                await self._train_initial_model()

            return StreamingResult(
                index=index,
                score=None,
                is_anomaly=None,
                raw_score=None,
                status=self._status,
                samples_collected=self._buffer.size,
                samples_until_ready=max(0, self.min_samples - self._buffer.size),
                model_version=self._model_version,
            )

        # Score the new data point
        score, raw_score, is_anomaly = await self._score_point(data)

        # Check if retraining needed
        if self._samples_since_retrain >= self.retrain_interval and not self._retraining:
            # Trigger background retrain
            asyncio.create_task(self._retrain_model())

        return StreamingResult(
            index=index,
            score=score,
            is_anomaly=is_anomaly,
            raw_score=raw_score,
            status=self._status,
            samples_collected=self._buffer.size,
            samples_until_ready=0,
            model_version=self._model_version,
        )

    async def process_batch(
        self,
        data: list[dict[str, Any]] | list[list[float]],
    ) -> StreamingBatchResult:
        """Process a batch of data points.

        Args:
            data: List of feature dicts or numeric lists

        Returns:
            StreamingBatchResult with all scores
        """
        start_time = time.perf_counter()
        results = []

        for i, point in enumerate(data):
            result = await self.process(point, index=i)
            results.append(result)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return StreamingBatchResult(
            results=results,
            status=self._status,
            samples_collected=self._buffer.size,
            model_version=self._model_version,
            processing_time_ms=elapsed_ms,
        )

    async def _train_initial_model(self) -> None:
        """Train the initial model (cold start complete)."""
        logger.info(f"Training initial model for {self.model_id} with {self._buffer.size} samples")

        try:
            await self._fit_detector()
            self._model_version = 1
            self._samples_since_retrain = 0
            self._status = DetectorStatus.READY
            logger.info(f"Initial model trained for {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to train initial model: {e}")
            raise

    async def _retrain_model(self) -> None:
        """Retrain model in background (non-blocking)."""
        if self._retraining:
            return

        self._retraining = True
        self._status = DetectorStatus.RETRAINING
        logger.info(f"Starting background retrain for {self.model_id}")

        try:
            await self._fit_detector()
            self._model_version += 1
            self._samples_since_retrain = 0
            logger.info(f"Retrained model {self.model_id} to version {self._model_version}")
        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")
        finally:
            self._retraining = False
            self._status = DetectorStatus.READY

    async def _fit_detector(self) -> None:
        """Fit the PyOD detector on buffer data.

        Uses rolling features if configured, otherwise uses raw numeric data.
        """
        from models.pyod_backend import (
            create_detector,
            fit_detector,
            get_decision_scores,
        )

        # Get data from buffer - optionally with rolling features
        if self.rolling_windows:
            # get_features uses lazy evaluation with SIMD + parallel execution
            # fill_null(0.0) handles cold start - no data is dropped
            df = self._buffer.get_features(
                rolling_windows=self.rolling_windows,
                include_lags=self.include_lags,
                lag_periods=self.lag_periods,
                fill_null_value=0.0,  # Cold start handling for derived features
            )
            # Select only numeric columns for the model
            numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
            # Fill nulls in base numeric columns too (handles missing data in input)
            df = df.with_columns([pl.col(c).fill_null(0.0) for c in numeric_cols])
            X = df.select(numeric_cols).to_numpy()
        else:
            X = self._buffer.get_numpy()

        if len(X) == 0:
            raise ValueError("No data in buffer")

        # Create and fit detector
        self._detector = create_detector(
            self.backend,
            contamination=self.contamination,
            **self.backend_kwargs,
        )
        fit_detector(self._detector, X)

        # Update normalization stats
        scores = get_decision_scores(self._detector, X)
        self._score_mean = float(np.mean(scores))
        self._score_std = float(np.std(scores)) if np.std(scores) > 0 else 1.0

        # Store feature column names for inference
        if self.rolling_windows:
            self._feature_columns = numeric_cols
        else:
            self._feature_columns = None

    async def _score_point(
        self,
        data: dict[str, Any],
    ) -> tuple[float, float, bool]:
        """Score a single data point.

        Uses rolling features if configured, matching the training data format.

        Returns:
            (normalized_score, raw_score, is_anomaly)
        """
        from models.pyod_backend import get_decision_scores

        # Get feature vector - must match training format
        if self.rolling_windows and self._feature_columns:
            # Get the latest row with rolling features computed
            df = self._buffer.get_features(
                rolling_windows=self.rolling_windows,
                include_lags=self.include_lags,
                lag_periods=self.lag_periods,
                fill_null_value=0.0,  # Cold start handling for derived features
            )
            # Get the last row (most recent, which includes the data point we just added)
            # Fill nulls in base columns and select only feature columns
            latest = df.tail(1).with_columns(
                [pl.col(c).fill_null(0.0) for c in self._feature_columns]
            ).select(self._feature_columns)
            X = latest.to_numpy()
        else:
            # Convert to numpy array (raw numeric values)
            numeric_values = [v for v in data.values() if isinstance(v, (int, float))]
            X = np.array([numeric_values])

        # Get raw score
        raw_scores = get_decision_scores(self._detector, X)
        raw_score = float(raw_scores[0])

        # Normalize score
        if self.normalization == "standardization":
            # Sigmoid normalization to [0, 1]
            z = (raw_score - self._score_mean) / self._score_std
            score = 1 / (1 + np.exp(-z))
        elif self.normalization == "zscore":
            score = (raw_score - self._score_mean) / self._score_std
        else:  # raw
            score = raw_score

        is_anomaly = score > self.threshold

        return float(score), raw_score, is_anomaly

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics."""
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "status": self._status.value,
            "model_version": self._model_version,
            "samples_collected": self._buffer.size,
            "total_processed": self._total_processed,
            "samples_since_retrain": self._samples_since_retrain,
            "min_samples": self.min_samples,
            "retrain_interval": self.retrain_interval,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "is_ready": self.is_ready,
        }

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._buffer.clear()
        self._detector = None
        self._model_version = 0
        self._feature_columns = None
        self._status = DetectorStatus.COLLECTING
        self._samples_since_retrain = 0
        self._total_processed = 0
        self._score_mean = 0.0
        self._score_std = 1.0
        self._retraining = False


# Session manager for multiple streaming detectors
class StreamingDetectorManager:
    """Manages multiple streaming detector instances.

    Provides session-based access to detectors, creating them on demand
    and caching them for subsequent requests.
    """

    def __init__(self):
        self._detectors: dict[str, StreamingAnomalyDetector] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        model_id: str,
        backend: str = "ecod",
        **kwargs: Any,
    ) -> StreamingAnomalyDetector:
        """Get existing detector or create new one.

        Args:
            model_id: Unique detector identifier
            backend: PyOD backend for new detector
            **kwargs: Additional arguments for new detector

        Returns:
            StreamingAnomalyDetector instance
        """
        async with self._lock:
            if model_id not in self._detectors:
                self._detectors[model_id] = StreamingAnomalyDetector(
                    model_id=model_id,
                    backend=backend,
                    **kwargs,
                )
                logger.info(f"Created streaming detector: {model_id}")

            return self._detectors[model_id]

    async def get(self, model_id: str) -> StreamingAnomalyDetector | None:
        """Get existing detector by ID."""
        return self._detectors.get(model_id)

    async def delete(self, model_id: str) -> bool:
        """Delete a detector."""
        async with self._lock:
            if model_id in self._detectors:
                del self._detectors[model_id]
                logger.info(f"Deleted streaming detector: {model_id}")
                return True
            return False

    async def list_detectors(self) -> list[dict[str, Any]]:
        """List all active detectors with their stats."""
        return [d.get_stats() for d in self._detectors.values()]

    async def clear_all(self) -> int:
        """Clear all detectors. Returns count deleted."""
        async with self._lock:
            count = len(self._detectors)
            self._detectors.clear()
            return count


# Global manager instance
_manager: StreamingDetectorManager | None = None


def get_streaming_manager() -> StreamingDetectorManager:
    """Get the global streaming detector manager."""
    global _manager
    if _manager is None:
        _manager = StreamingDetectorManager()
    return _manager
