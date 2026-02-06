"""
Anomaly detection model wrapper using PyOD.

All anomaly detection is powered by PyOD, providing a unified interface
to 12+ algorithms including legacy backends and new fast algorithms.

Supported Backends:
------------------
Legacy (backward compatible):
- isolation_forest: Tree-based ensemble method, fast and effective
- one_class_svm: Support vector machine for outlier detection
- local_outlier_factor: Density-based anomaly detection
- autoencoder: Neural network for complex patterns

Fast (parameter-free or minimal tuning):
- ecod: Empirical Cumulative Distribution (recommended default)
- hbos: Histogram-based Outlier Score (fastest)
- copod: Copula-Based Outlier Detection

Distance-based:
- knn: K-Nearest Neighbors outlier detection
- mcd: Minimum Covariance Determinant

Clustering:
- cblof: Clustering-Based Local Outlier Factor

Ensemble:
- suod: Scalable Unsupervised Outlier Detection

Streaming:
- loda: Lightweight Online Detector of Anomalies

Score Normalization Methods:
---------------------------
The `normalization` parameter controls how raw anomaly scores are transformed:

1. "standardization" (default):
   - Applies sigmoid transformation using median and IQR from training data
   - Produces scores in 0-1 range (0.5 = normal, approaching 1.0 = anomalous)
   - Default threshold: 0.5

2. "zscore":
   - Z-score normalization: (score - mean) / std
   - Scores represent standard deviations from the training mean
   - Default threshold: 2.0

3. "raw":
   - No normalization, returns PyOD-native scores
   - Higher values = more anomalous for all backends
   - Default threshold: detector.threshold_

Security Notes:
- Model loading is restricted to ANOMALY_MODELS_DIR
- Path traversal attacks are prevented
- Serialization uses joblib for all PyOD models
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .base import BaseModel
from .pyod_backend import (
    AnomalyBackendType,
    create_detector,
    fit_detector,
    get_all_backends,
    get_backends_response,
    get_decision_scores,
    is_valid_backend,
)
from utils.safe_home import get_data_dir

logger = logging.getLogger(__name__)

# Safe directory for anomaly models - uses standard LlamaFarm data directory
# ~/.llamafarm/models/anomaly/ (or LF_DATA_DIR/models/anomaly/)
# Only files within this directory can be loaded - prevents path traversal attacks
_LF_DATA_DIR = get_data_dir()
ANOMALY_MODELS_DIR = (_LF_DATA_DIR / "models" / "anomaly").resolve()


def _validate_model_path(model_path: Path) -> Path:
    """Validate that model path is within the safe directory.

    Raises:
        ValueError: If path is outside the safe directory or uses path traversal
    """
    path_str = str(model_path)
    if ".." in path_str:
        raise ValueError(
            f"Security error: Model path '{model_path}' contains '..' "
            "Path traversal is not allowed."
        )

    resolved_path = model_path.resolve()

    try:
        resolved_path.relative_to(ANOMALY_MODELS_DIR)
    except ValueError:
        raise ValueError(
            f"Security error: Model path '{model_path}' is outside the allowed "
            f"directory '{ANOMALY_MODELS_DIR}'. Path traversal is not allowed."
        ) from None

    return resolved_path


# Re-export backend type for external use
AnomalyBackend = AnomalyBackendType

NormalizationMethod = Literal["standardization", "zscore", "raw"]


@dataclass
class AnomalyScore:
    """Anomaly score for a single data point."""

    index: int
    score: float  # Normalized score (0-1 for standardization)
    is_anomaly: bool
    raw_score: float  # PyOD-native score


@dataclass
class FitResult:
    """Result from fitting an anomaly detector."""

    samples_fitted: int
    training_time_ms: float
    model_params: dict[str, Any]


class AnomalyModel(BaseModel):
    """Wrapper for PyOD anomaly detection models.

    Provides a unified interface to all PyOD algorithms with:
    - Consistent score normalization
    - Save/load functionality
    - Feature scaling
    - Thread-safe async operations
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        backend: AnomalyBackend = "isolation_forest",
        contamination: float = 0.1,
        threshold: float | None = None,
        normalization: NormalizationMethod = "standardization",
        **backend_params: Any,
    ):
        """Initialize anomaly detection model.

        Args:
            model_id: Model identifier for caching/saving
            device: Target device (mostly ignored, PyOD uses CPU)
            backend: Anomaly detection algorithm
            contamination: Expected proportion of anomalies (0.0 to 0.5)
            threshold: Custom anomaly threshold (auto-determined if None)
            normalization: Score normalization method
            **backend_params: Additional backend-specific parameters
        """
        super().__init__(model_id, device)

        if not is_valid_backend(backend):
            raise ValueError(
                f"Unknown backend: {backend}. Available: {get_all_backends()}"
            )

        self.backend = backend
        self.contamination = contamination
        self._threshold = threshold
        self.normalization = normalization
        self._backend_params = backend_params
        self.model_type = f"anomaly_{backend}"
        self.supports_streaming = False

        # PyOD detector
        self._detector = None
        self._scaler = None
        self._is_fitted = False

        # Normalization statistics
        self._norm_median = None
        self._norm_iqr = None
        self._norm_mean = None
        self._norm_std = None

    @property
    def threshold(self) -> float:
        """Get anomaly threshold based on normalization method."""
        if self._threshold is not None:
            return self._threshold
        if self.normalization == "zscore":
            return 2.0
        elif self.normalization == "raw":
            # Use detector's learned threshold if available
            if self._detector is not None and hasattr(self._detector, "threshold_"):
                return self._detector.threshold_
            return 0.0
        else:  # standardization
            return 0.5

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    async def load(self) -> None:
        """Load or initialize the anomaly detection model."""
        logger.info(f"Loading anomaly model: {self.backend}")

        model_path = Path(self.model_id)
        # Validate path BEFORE checking existence to prevent path traversal attacks
        if model_path.suffix in (".pkl", ".joblib"):
            try:
                validated_path = _validate_model_path(model_path)
                if validated_path.exists():
                    await self._load_pretrained(validated_path)
                    return
            except ValueError as e:
                logger.warning(f"Path validation failed, initializing fresh model: {e}")

        await self._initialize_backend()

        logger.info(f"Anomaly model initialized: {self.backend}")

    async def _load_pretrained(self, model_path: Path) -> None:
        """Load a pre-trained model from disk."""
        logger.info(f"Loading pretrained model from: {model_path}")

        try:
            import joblib
            data = joblib.load(model_path)
        except ImportError:
            import pickle
            with open(model_path, "rb") as f:
                data = pickle.load(f)  # noqa: S301

        self._detector = data.get("detector")
        self._scaler = data.get("scaler")
        self._threshold = data.get("threshold")
        self._norm_median = data.get("norm_median")
        self._norm_iqr = data.get("norm_iqr")
        self._norm_mean = data.get("norm_mean")
        self._norm_std = data.get("norm_std")
        self.normalization = data.get("normalization", "standardization")
        self.backend = data.get("backend", self.backend)
        self._is_fitted = True

    async def _initialize_backend(self) -> None:
        """Initialize a fresh PyOD detector."""
        self._detector = create_detector(
            backend=self.backend,
            contamination=self.contamination,
            **self._backend_params,
        )

        # Initialize scaler (RobustScaler is resilient to outliers)
        from sklearn.preprocessing import RobustScaler
        self._scaler = RobustScaler()

    async def fit(
        self,
        data: list[list[float]] | np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        use_executor: bool = True,
    ) -> FitResult:
        """Fit the anomaly detector on training data.

        Args:
            data: Training data (assumed to be mostly normal)
            epochs: Training epochs (autoencoder only)
            batch_size: Batch size (autoencoder only)
            use_executor: If True, run training in thread pool

        Returns:
            FitResult with training statistics
        """
        if use_executor:
            from services.training_executor import run_in_executor
            return await run_in_executor(
                self._fit_sync, data, epochs, batch_size
            )
        else:
            return self._fit_sync(data, epochs, batch_size)

    def _fit_sync(
        self,
        data: list[list[float]] | np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> FitResult:
        """Synchronous fit for thread pool execution."""
        import time

        start_time = time.time()

        # Convert to numpy
        X = np.array(data) if not isinstance(data, np.ndarray) else data
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Fit scaler and transform
        X_scaled = self._scaler.fit_transform(X)

        # Reinitialize detector if autoencoder (needs input dims)
        if self.backend == "autoencoder":
            self._detector = create_detector(
                backend=self.backend,
                contamination=self.contamination,
                epochs=epochs,
                batch_size=batch_size,
                **self._backend_params,
            )

        # Ensure detector is initialized
        if self._detector is None:
            raise RuntimeError(
                f"Detector not initialized for backend '{self.backend}'. "
                "This may indicate an unsupported backend or initialization error."
            )

        # Fit PyOD detector
        fit_detector(self._detector, X_scaled)
        self._is_fitted = True

        # Compute normalization statistics from training data
        raw_scores = get_decision_scores(self._detector, X_scaled)

        # Standardization stats
        self._norm_median = float(np.median(raw_scores))
        self._norm_iqr = float(
            np.percentile(raw_scores, 75) - np.percentile(raw_scores, 25)
        )
        if self._norm_iqr == 0:
            self._norm_iqr = float(np.std(raw_scores)) or 1.0

        # Z-score stats
        self._norm_mean = float(np.mean(raw_scores))
        self._norm_std = float(np.std(raw_scores))
        if self._norm_std == 0:
            self._norm_std = 1.0

        # Auto-determine threshold if not set
        if self._threshold is None:
            normalized_scores = self._normalize_scores(raw_scores)
            self._threshold = float(
                np.percentile(normalized_scores, (1 - self.contamination) * 100)
            )

        training_time = (time.time() - start_time) * 1000

        return FitResult(
            samples_fitted=len(X),
            training_time_ms=training_time,
            model_params={
                "backend": self.backend,
                "contamination": self.contamination,
                "threshold": self._threshold,
                "input_dim": X.shape[1] if len(X.shape) > 1 else 1,
            },
        )

    async def score(
        self,
        data: list[list[float]] | np.ndarray,
        threshold: float | None = None,
    ) -> list[AnomalyScore]:
        """Score data points for anomalies.

        Args:
            data: Data points to score
            threshold: Override default threshold

        Returns:
            List of AnomalyScore objects
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call fit() first or load a pre-trained model."
            )

        # Convert to numpy
        X = np.array(data) if not isinstance(data, np.ndarray) else data
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Scale data
        X_scaled = self._scaler.transform(X)

        # Get raw scores from PyOD
        raw_scores = get_decision_scores(self._detector, X_scaled)

        # Normalize scores
        normalized_scores = self._normalize_scores(raw_scores)

        # Determine anomalies
        thresh = threshold if threshold is not None else self.threshold

        results = []
        for i, (raw, norm) in enumerate(
            zip(raw_scores, normalized_scores, strict=True)
        ):
            results.append(
                AnomalyScore(
                    index=i,
                    score=float(norm),
                    is_anomaly=bool(norm > thresh),
                    raw_score=float(raw),
                )
            )

        return results

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores based on configured method."""
        if self.normalization == "raw":
            return scores

        elif self.normalization == "zscore":
            if self._norm_mean is not None and self._norm_std is not None:
                mean, std = self._norm_mean, self._norm_std
            else:
                mean = np.mean(scores)
                std = np.std(scores) or 1.0
            return (scores - mean) / std

        else:  # standardization
            if self._norm_median is not None and self._norm_iqr is not None:
                median, iqr = self._norm_median, self._norm_iqr
            else:
                median = np.median(scores)
                iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
                if iqr == 0:
                    iqr = np.std(scores) or 1.0

            normalized = (scores - median) / (2 * iqr) if iqr > 0 else scores - median
            normalized = np.clip(normalized, -700, 700)  # Prevent overflow
            return 1 / (1 + np.exp(-normalized))

    async def detect(
        self,
        data: list[list[float]] | np.ndarray,
        threshold: float | None = None,
    ) -> list[AnomalyScore]:
        """Detect anomalies (returns only anomalous points)."""
        all_scores = await self.score(data, threshold)
        return [s for s in all_scores if s.is_anomaly]

    async def save(self, path: str) -> None:
        """Save the fitted model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Nothing to save.")

        model_path = Path(path)

        data = {
            "detector": self._detector,
            "scaler": self._scaler,
            "backend": self.backend,
            "threshold": self._threshold,
            "normalization": self.normalization,
            "norm_median": self._norm_median,
            "norm_iqr": self._norm_iqr,
            "norm_mean": self._norm_mean,
            "norm_std": self._norm_std,
        }

        try:
            import joblib
            joblib.dump(data, model_path.with_suffix(".joblib"))
        except ImportError:
            import pickle
            with open(model_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(data, f)

        logger.info(f"Anomaly model saved to {path}")

    async def unload(self) -> None:
        """Unload the model and free resources."""
        self._detector = None
        self._scaler = None
        self._norm_median = None
        self._norm_iqr = None
        self._norm_mean = None
        self._norm_std = None
        self._is_fitted = False
        await super().unload()

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model."""
        info = super().get_model_info()
        info.update({
            "backend": self.backend,
            "contamination": self.contamination,
            "threshold": self._threshold,
            "normalization": self.normalization,
            "is_fitted": self._is_fitted,
        })
        return info


# Export for convenience
def get_available_backends() -> dict[str, Any]:
    """Get all available anomaly detection backends with metadata."""
    return get_backends_response()
