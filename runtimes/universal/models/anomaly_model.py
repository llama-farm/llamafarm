"""
Anomaly detection model wrapper.

Supports multiple backends:
- autoencoder: Neural network trained on normal data, detects anomalies by reconstruction error
- isolation_forest: Tree-based ensemble method, fast and effective
- one_class_svm: Support vector machine for outlier detection
- local_outlier_factor: Density-based anomaly detection

Designed for:
- IoT sensor anomaly detection
- Network intrusion detection
- Fraud detection
- Manufacturing quality control

Score Normalization Methods:
---------------------------
The `normalization` parameter controls how raw anomaly scores are transformed:

1. "standardization" (default):
   - Applies sigmoid transformation using median and IQR from training data
   - Produces scores in 0-1 range (0.5 = normal, approaching 1.0 = anomalous)
   - Robust to outliers in training data
   - Default threshold: 0.5
   - Best for: General use, when you want bounded interpretable scores

2. "zscore":
   - Z-score normalization: (score - mean) / std
   - Scores represent standard deviations from the training mean
   - Interpretable: 2.0 = 2 std devs (unusual), 3.0 = rare, 4.0+ = extreme
   - Default threshold: 2.0
   - Best for: When you want scores that map to statistical significance

3. "raw":
   - No normalization, returns backend-native scores
   - Ranges vary by backend:
     * isolation_forest: ~-0.5 to 0.5 (higher = more anomalous)
     * one_class_svm: unbounded real numbers (higher = more anomalous)
     * local_outlier_factor: ~1 to 10+ (higher = more anomalous)
   - Default threshold: 0.0 (you should set your own based on the backend)
   - Best for: Debugging, or when you understand the backend's native scale

Example usage:
    # Z-score: flag anything > 3 standard deviations
    model = AnomalyModel("my_model", "cpu", normalization="zscore", threshold=3.0)

    # Raw scores for debugging
    model = AnomalyModel("my_model", "cpu", normalization="raw")
    scores = await model.score(data)
    print([s.raw_score for s in scores])  # Examine native backend scores

Security Notes:
- Model loading is restricted to a designated safe directory (ANOMALY_MODELS_DIR)
- Path traversal attacks are prevented by validating paths are within the safe directory
- Pickle/joblib deserialization is only performed on trusted files from the safe directory
"""

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)

# Safe directory for anomaly models - uses standard LlamaFarm data directory
# ~/.llamafarm/models/anomaly/ (or LF_DATA_DIR/models/anomaly/)
# Only files within this directory can be loaded - prevents path traversal attacks
_LF_DATA_DIR = Path(os.environ.get("LF_DATA_DIR", Path.home() / ".llamafarm"))
ANOMALY_MODELS_DIR = (_LF_DATA_DIR / "models" / "anomaly").resolve()


def _validate_model_path(model_path: Path) -> Path:
    """Validate that model path is within the safe directory.

    Security: This function prevents path traversal attacks by ensuring
    the model path resolves to a location within ANOMALY_MODELS_DIR.

    Raises:
        ValueError: If path is outside the safe directory or uses path traversal
    """
    # Check for path traversal patterns
    path_str = str(model_path)
    if ".." in path_str:
        raise ValueError(
            f"Security error: Model path '{model_path}' contains '..' "
            "Path traversal is not allowed."
        )

    # Resolve to absolute path
    resolved_path = model_path.resolve()

    # Verify the resolved path is within the safe directory
    try:
        resolved_path.relative_to(ANOMALY_MODELS_DIR)
    except ValueError:
        raise ValueError(
            f"Security error: Model path '{model_path}' is outside the allowed "
            f"directory '{ANOMALY_MODELS_DIR}'. Path traversal is not allowed."
        ) from None

    return resolved_path


AnomalyBackend = Literal[
    "autoencoder", "isolation_forest", "one_class_svm", "local_outlier_factor"
]

NormalizationMethod = Literal["standardization", "zscore", "raw"]
"""Score normalization methods for anomaly detection.

- "standardization" (default): Sigmoid normalization using median and IQR.
    Produces scores in 0-1 range where 0.5 is "normal" and values approaching
    1.0 are increasingly anomalous. Robust to outliers in training data.
    Recommended threshold: 0.5-0.7

- "zscore": Z-score normalization using mean and standard deviation.
    Scores represent standard deviations from the mean of training scores.
    A score of 2.0 means "2 standard deviations above normal."
    Recommended threshold: 2.0 (unusual), 3.0 (rare), 4.0 (extreme)

- "raw": No normalization, returns backend-specific raw scores.
    Useful for debugging or when you understand the backend's native scale.
    Note: Raw score ranges vary by backend:
      - isolation_forest: ~-0.5 to 0.5 (higher = more anomalous)
      - one_class_svm: unbounded (higher = more anomalous)
      - local_outlier_factor: ~1 to 10+ (higher = more anomalous)
    Threshold must be set based on backend-specific knowledge.
"""


@dataclass
class AnomalyScore:
    """Anomaly score for a single data point."""

    index: int
    score: float  # Higher = more anomalous (normalized 0-1)
    is_anomaly: bool
    raw_score: float  # Backend-specific raw score


@dataclass
class FitResult:
    """Result from fitting an anomaly detector."""

    samples_fitted: int
    training_time_ms: float
    model_params: dict[str, Any]


class AnomalyModel(BaseModel):
    """Wrapper for anomaly detection models.

    Supports both pre-trained models and on-the-fly training.

    Backends:
    - autoencoder: Best for complex patterns, requires training
    - isolation_forest: Fast, works well out of the box
    - one_class_svm: Good for small datasets
    - local_outlier_factor: Density-based, good for clustering anomalies

    Usage patterns:
    1. Pre-trained: Load a saved model and score new data
    2. Fit-then-score: Train on normal data, then detect anomalies
    3. Online: Incrementally update the model (isolation_forest only)
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        backend: AnomalyBackend = "isolation_forest",
        contamination: float = 0.1,
        threshold: float | None = None,
        normalization: NormalizationMethod = "standardization",
    ):
        """Initialize anomaly detection model.

        Args:
            model_id: Model identifier (for caching) or path to pre-trained model
            device: Target device (cpu recommended for sklearn models)
            backend: Anomaly detection backend
            contamination: Expected proportion of anomalies (0.0 to 0.5)
            threshold: Custom anomaly threshold (auto-determined if None).
                Default thresholds by normalization method:
                - standardization: 0.5 (scores 0-1, higher = anomaly)
                - zscore: 2.0 (standard deviations from mean)
                - raw: None (must be set based on backend)
            normalization: Score normalization method. See NormalizationMethod docs.
                - "standardization": Sigmoid 0-1 range (default)
                - "zscore": Standard deviations from mean
                - "raw": No normalization, backend-native scores
        """
        super().__init__(model_id, device)
        self.backend = backend
        self.contamination = contamination
        self._threshold = threshold
        self.normalization = normalization
        self.model_type = f"anomaly_{backend}"
        self.supports_streaming = False

        # Backend-specific model
        self._detector = None
        self._scaler = None  # For normalizing input data
        self._is_fitted = False

        # For autoencoder
        self._encoder = None
        self._decoder = None

        # Normalization statistics (computed during fit, used during score)
        # For standardization (sigmoid)
        self._norm_median = None
        self._norm_iqr = None
        # For zscore
        self._norm_mean = None
        self._norm_std = None

    @property
    def threshold(self) -> float:
        """Get anomaly threshold based on normalization method."""
        if self._threshold is not None:
            return self._threshold
        # Default thresholds by normalization method
        if self.normalization == "zscore":
            return 2.0  # 2 standard deviations
        elif self.normalization == "raw":
            return 0.0  # No sensible default for raw, user should set explicitly
        else:  # standardization
            return 0.5

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    async def load(self) -> None:
        """Load or initialize the anomaly detection model."""
        logger.info(f"Loading anomaly model: {self.backend}")

        # Check if model_id is a path to a pre-trained model
        model_path = Path(self.model_id)
        if model_path.exists() and model_path.suffix in (".pkl", ".joblib", ".pt"):
            # Validate path is within safe directory before loading
            try:
                validated_path = _validate_model_path(model_path)
                await self._load_pretrained(validated_path)
            except ValueError as e:
                logger.error(f"Security validation failed: {e}")
                raise
        else:
            await self._initialize_backend()

        logger.info(f"Anomaly model initialized: {self.backend}")

    async def _load_pretrained(self, model_path: Path) -> None:
        """Load a pre-trained model from disk.

        Security Note: This method should only be called with validated paths
        from _validate_model_path() to prevent path traversal attacks.
        The deserialization is considered safe because:
        1. Paths are validated to be within ANOMALY_MODELS_DIR
        2. Only administrators can place files in this directory
        3. The model files are created by the save() method of this class
        """
        logger.info(f"Loading pretrained model from validated path: {model_path}")

        if model_path.suffix == ".pt":
            # PyTorch autoencoder
            import torch

            # Note: weights_only=False is required for loading nn.Module objects
            # Security is ensured by path validation above
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            self._encoder = checkpoint.get("encoder")
            self._decoder = checkpoint.get("decoder")
            self._threshold = checkpoint.get("threshold", 0.5)
            self._scaler = checkpoint.get("scaler")
            self._norm_median = checkpoint.get("norm_median")
            self._norm_iqr = checkpoint.get("norm_iqr")
            self._norm_mean = checkpoint.get("norm_mean")
            self._norm_std = checkpoint.get("norm_std")
            self.normalization = checkpoint.get("normalization", "standardization")
            self._is_fitted = True
        else:
            # Sklearn model (pickle or joblib)
            # Security is ensured by path validation - only trusted files
            # from ANOMALY_MODELS_DIR can be loaded
            try:
                import joblib

                data = joblib.load(model_path)
            except ImportError:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)  # noqa: S301 - path is validated

            self._detector = data.get("detector")
            self._scaler = data.get("scaler")
            self._threshold = data.get("threshold", 0.5)
            self._norm_median = data.get("norm_median")
            self._norm_iqr = data.get("norm_iqr")
            self._norm_mean = data.get("norm_mean")
            self._norm_std = data.get("norm_std")
            self.normalization = data.get("normalization", "standardization")
            self._is_fitted = True

    async def _initialize_backend(self) -> None:
        """Initialize a fresh anomaly detection backend."""
        if self.backend == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            self._detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
            )

        elif self.backend == "one_class_svm":
            from sklearn.svm import OneClassSVM

            self._detector = OneClassSVM(
                nu=self.contamination,
                kernel="rbf",
                gamma="scale",
            )

        elif self.backend == "local_outlier_factor":
            from sklearn.neighbors import LocalOutlierFactor

            self._detector = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,  # Enable predict() for new data
                n_jobs=-1,
            )

        elif self.backend == "autoencoder":
            # Will be created during fit() based on input dimensions
            pass

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Initialize scaler for data normalization
        # Using RobustScaler instead of StandardScaler because:
        # - StandardScaler uses mean and std, which are sensitive to outliers
        # - In anomaly detection, training data often contains outliers
        # - RobustScaler uses median and IQR, making it resilient to outliers
        from sklearn.preprocessing import RobustScaler

        self._scaler = RobustScaler()

    async def fit(
        self,
        data: list[list[float]] | np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        use_executor: bool = True,
    ) -> FitResult:
        """Fit the anomaly detector on normal data.

        This method offloads CPU-bound training to a thread pool to avoid
        blocking the async event loop.

        Args:
            data: Training data (assumed to be mostly normal)
            epochs: Training epochs (autoencoder only)
            batch_size: Batch size (autoencoder only)
            use_executor: If True, run training in thread pool (default: True)

        Returns:
            FitResult with training statistics
        """
        if use_executor and self.backend != "autoencoder":
            # Offload sklearn training to thread pool
            from services.training_executor import run_in_executor

            return await run_in_executor(self._fit_sync, data, epochs, batch_size)
        else:
            # Autoencoder uses torch which has its own threading
            # or user explicitly requested no executor
            return self._fit_sync(data, epochs, batch_size)

    def _fit_sync(
        self,
        data: list[list[float]] | np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> FitResult:
        """Synchronous fit method for thread pool execution.

        This is the actual training logic, separated from the async wrapper
        to allow execution in a thread pool.

        Args:
            data: Training data (assumed to be mostly normal)
            epochs: Training epochs (autoencoder only)
            batch_size: Batch size (autoencoder only)

        Returns:
            FitResult with training statistics
        """
        import time

        start_time = time.time()

        # Convert to numpy array
        X = np.array(data) if not isinstance(data, np.ndarray) else data

        # Handle 1D input (e.g., single feature time series)
        # Must match the handling in score() to avoid dimension mismatches
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Fit scaler and transform data
        X_scaled = self._scaler.fit_transform(X)

        if self.backend == "autoencoder":
            # Autoencoder uses PyTorch - synchronous CPU/GPU-bound
            self._fit_autoencoder(X_scaled, epochs, batch_size)
        else:
            # Sklearn models - CPU-bound synchronous fitting
            self._detector.fit(X_scaled)

        self._is_fitted = True

        # Compute and store normalization statistics from training data
        # These are used during scoring to ensure consistent normalization
        raw_scores = self._compute_raw_scores_sync(X_scaled)

        # Statistics for standardization (sigmoid) method
        self._norm_median = float(np.median(raw_scores))
        self._norm_iqr = float(
            np.percentile(raw_scores, 75) - np.percentile(raw_scores, 25)
        )
        if self._norm_iqr == 0:
            # Fallback to std if IQR is 0
            self._norm_iqr = float(np.std(raw_scores)) or 1.0

        # Statistics for zscore method
        self._norm_mean = float(np.mean(raw_scores))
        self._norm_std = float(np.std(raw_scores))
        if self._norm_std == 0:
            self._norm_std = 1.0  # Prevent division by zero

        # Auto-determine threshold if not set
        # Threshold is computed based on normalization method
        if self._threshold is None:
            normalized_scores = self._normalize_scores(raw_scores)
            # Set threshold at (1 - contamination) percentile of normalized scores
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

    def _compute_raw_scores_sync(self, X: np.ndarray) -> np.ndarray:
        """Synchronous version of _compute_raw_scores for thread pool execution."""
        if self.backend == "autoencoder":
            # Autoencoder scoring is synchronous PyTorch
            return self._autoencoder_scores_sync(X)

        elif self.backend == "isolation_forest":
            # IsolationForest: negative score = more anomalous
            return -self._detector.score_samples(X)

        elif self.backend == "one_class_svm":
            # OneClassSVM: negative distance = more anomalous
            return -self._detector.decision_function(X)

        elif self.backend == "local_outlier_factor":
            # LOF: negative score = more anomalous
            return -self._detector.score_samples(X)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _fit_autoencoder(
        self,
        X: np.ndarray,
        epochs: int,
        batch_size: int,
        patience: int = 10,
        validation_split: float = 0.1,
    ) -> None:
        """Fit autoencoder model with early stopping.

        This is a synchronous method because PyTorch operations don't benefit
        from async - they're CPU/GPU-bound. The method is called from _fit_sync
        which may be running in a thread pool.

        Args:
            X: Scaled training data
            epochs: Maximum training epochs
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before stopping (default: 10)
            validation_split: Fraction of data to use for validation (default: 0.1)
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        input_dim = X.shape[1]
        hidden_dim = max(input_dim // 2, 8)
        latent_dim = max(hidden_dim // 2, 4)

        # Define encoder
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        # Define decoder
        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Move to device
        self._encoder = self._encoder.to(self.device)
        self._decoder = self._decoder.to(self.device)

        # Split data into training and validation sets
        # Ensure we have at least 2 samples for validation split
        n_samples = X.shape[0]
        if n_samples < 2:
            # With only 1 sample, skip validation and use all data for training
            X_train = X
            X_val = X  # Use same data for validation (no early stopping benefit)
            use_validation = False
        else:
            # Ensure at least 1 training sample remains after validation split
            n_val = min(max(1, int(n_samples * validation_split)), n_samples - 1)
            indices = np.random.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            X_train = X[train_indices]
            X_val = X[val_indices]
            use_validation = True

        # Prepare data loaders
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        train_dataset = TensorDataset(X_train_tensor)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Training setup
        optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=0.001,
        )
        criterion = nn.MSELoss()

        # Early stopping state
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_encoder_state = None
        best_decoder_state = None

        self._encoder.train()
        self._decoder.train()

        for epoch in range(epochs):
            # Training phase
            epoch_loss = 0.0
            for (batch,) in train_dataloader:
                optimizer.zero_grad()
                encoded = self._encoder(batch)
                decoded = self._decoder(encoded)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_dataloader)

            # Validation phase (only if we have proper validation data)
            if use_validation:
                self._encoder.eval()
                self._decoder.eval()
                with torch.no_grad():
                    val_encoded = self._encoder(X_val_tensor)
                    val_decoded = self._decoder(val_encoded)
                    val_loss = criterion(val_decoded, X_val_tensor).item()
                self._encoder.train()
                self._decoder.train()

                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Save best model state
                    best_encoder_state = {
                        k: v.clone() for k, v in self._encoder.state_dict().items()
                    }
                    best_decoder_state = {
                        k: v.clone() for k, v in self._decoder.state_dict().items()
                    }
                else:
                    epochs_without_improvement += 1

                if (epoch + 1) % 20 == 0:
                    logger.debug(
                        f"Epoch {epoch + 1}/{epochs}, "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Patience: {patience - epochs_without_improvement}/{patience}"
                    )

                # Early stopping check
                if epochs_without_improvement >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"no improvement for {patience} epochs. "
                        f"Best val loss: {best_val_loss:.4f}"
                    )
                    break
            else:
                # Without validation, just log training progress
                if (epoch + 1) % 20 == 0:
                    logger.debug(
                        f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}"
                    )

        # Restore best model state
        if best_encoder_state is not None:
            self._encoder.load_state_dict(best_encoder_state)
        if best_decoder_state is not None:
            self._decoder.load_state_dict(best_decoder_state)

        self._encoder.eval()
        self._decoder.eval()

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

        # Convert to numpy array
        X = np.array(data) if not isinstance(data, np.ndarray) else data

        # Handle 1D input
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Scale data
        X_scaled = self._scaler.transform(X)

        # Get raw scores
        raw_scores = await self._compute_raw_scores(X_scaled)

        # Normalize scores to 0-1 range (higher = more anomalous)
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

    async def _compute_raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute raw anomaly scores."""
        if self.backend == "autoencoder":
            return await self._autoencoder_scores(X)

        elif self.backend == "isolation_forest":
            # IsolationForest: negative score = more anomalous
            return -self._detector.score_samples(X)

        elif self.backend == "one_class_svm":
            # OneClassSVM: negative distance = more anomalous
            return -self._detector.decision_function(X)

        elif self.backend == "local_outlier_factor":
            # LOF: negative score = more anomalous
            return -self._detector.score_samples(X)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _autoencoder_scores_sync(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error scores for autoencoder (synchronous).

        This is the actual implementation used by both async and sync code paths.
        """
        import torch

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            encoded = self._encoder(X_tensor)
            decoded = self._decoder(encoded)
            # MSE per sample
            reconstruction_error = torch.mean((X_tensor - decoded) ** 2, dim=1)

        return reconstruction_error.cpu().numpy()

    async def _autoencoder_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error scores for autoencoder (async wrapper).

        This async wrapper exists for API compatibility. The actual work
        is done synchronously since PyTorch operations don't benefit from async.
        """
        return self._autoencoder_scores_sync(X)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores based on the configured normalization method.

        Methods:
        - "standardization": Sigmoid normalization to 0-1 range using median/IQR.
          Scores near 0.5 are normal, approaching 1.0 are anomalous.
        - "zscore": Z-score normalization using mean/std.
          Scores represent standard deviations from normal (2.0 = unusual, 3.0 = rare).
        - "raw": No normalization, returns backend-native scores.

        Uses stored normalization statistics from training data if available,
        ensuring consistent normalization between training and inference.
        """
        if self.normalization == "raw":
            # No normalization - return raw scores as-is
            return scores

        elif self.normalization == "zscore":
            # Z-score normalization: (score - mean) / std
            if self._norm_mean is not None and self._norm_std is not None:
                mean = self._norm_mean
                std = self._norm_std
            else:
                # Fallback to computing from current data (during fit)
                mean = np.mean(scores)
                std = np.std(scores)
                if std == 0:
                    std = 1.0
            return (scores - mean) / std

        else:  # standardization (default)
            # Sigmoid normalization using median/IQR
            if self._norm_median is not None and self._norm_iqr is not None:
                median = self._norm_median
                iqr = self._norm_iqr
            else:
                # Fallback to computing from current data (during fit)
                median = np.median(scores)
                iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
                if iqr == 0:
                    iqr = np.std(scores) or 1.0

            normalized = (scores - median) / (2 * iqr) if iqr > 0 else scores - median

            # Clip to prevent numerical overflow in np.exp
            # np.exp(-x) overflows for x < -709, so we clip to safe range
            normalized = np.clip(normalized, -700, 700)

            # Apply sigmoid to get 0-1 range
            return 1 / (1 + np.exp(-normalized))

    async def detect(
        self,
        data: list[list[float]] | np.ndarray,
        threshold: float | None = None,
    ) -> list[AnomalyScore]:
        """Detect anomalies in data (alias for score with anomaly filtering).

        Returns only data points classified as anomalies.
        """
        all_scores = await self.score(data, threshold)
        return [s for s in all_scores if s.is_anomaly]

    async def save(self, path: str) -> None:
        """Save the fitted model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Nothing to save.")

        model_path = Path(path)

        # Common fields for all backends
        common_fields = {
            "threshold": self._threshold,
            "scaler": self._scaler,
            "normalization": self.normalization,
            # Standardization stats
            "norm_median": self._norm_median,
            "norm_iqr": self._norm_iqr,
            # Z-score stats
            "norm_mean": self._norm_mean,
            "norm_std": self._norm_std,
        }

        if self.backend == "autoencoder":
            import torch

            torch.save(
                {
                    "encoder": self._encoder,
                    "decoder": self._decoder,
                    **common_fields,
                },
                model_path.with_suffix(".pt"),
            )
        else:
            try:
                import joblib

                joblib.dump(
                    {"detector": self._detector, **common_fields},
                    model_path.with_suffix(".joblib"),
                )
            except ImportError:
                with open(model_path.with_suffix(".pkl"), "wb") as f:
                    pickle.dump(
                        {"detector": self._detector, **common_fields},
                        f,
                    )

        logger.info(f"Anomaly model saved to {path}")

    async def unload(self) -> None:
        """Unload the model and free resources."""
        self._detector = None
        self._scaler = None
        self._encoder = None
        self._decoder = None
        self._norm_median = None
        self._norm_iqr = None
        self._norm_mean = None
        self._norm_std = None
        self._is_fitted = False
        await super().unload()

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model."""
        info = super().get_model_info()
        info.update(
            {
                "backend": self.backend,
                "contamination": self.contamination,
                "threshold": self._threshold,
                "normalization": self.normalization,
                "is_fitted": self._is_fitted,
            }
        )
        return info
