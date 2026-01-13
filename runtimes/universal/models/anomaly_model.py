"""
Anomaly detection model wrapper.

Supports multiple backends:
- autoencoder: Neural network trained on normal data, detects anomalies by reconstruction error
- vae: Variational Autoencoder with probabilistic latent space, anomaly score = negative ELBO
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

Non-Blocking Training:
- The fit() method uses run_in_executor() to offload CPU-bound sklearn/torch
  training to a thread pool, preventing blocking of the async event loop
- This allows health checks and other requests to be served during training
"""

import asyncio
import logging
import os
import pickle
import time
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
    "autoencoder", "vae", "isolation_forest", "one_class_svm", "local_outlier_factor",
    "copod", "hbos", "ecod",  # PyOD backends
]

ScalerType = Literal["standard", "robust"]
"""Input data scaler type for preprocessing training and inference data.

- "standard": StandardScaler - uses mean and standard deviation for centering
    and scaling. Best for data with normal distribution and few outliers.

- "robust": RobustScaler - uses median and IQR (interquartile range) for
    centering and scaling. Much more robust to outliers in training data.
    Recommended when training data may contain outliers/anomalies.

Note: The scaler type affects how input features are normalized before being
passed to the anomaly detection backend. Using RobustScaler can significantly
improve anomaly detection when the training data is contaminated with outliers.
"""

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
        scaler_type: ScalerType = "robust",
        validation_split: float = 0.1,
        patience: int = 10,
        min_delta: float = 1e-4,
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
            scaler_type: Input data scaler type. See ScalerType docs.
                - "robust": RobustScaler using median/IQR (default, recommended)
                - "standard": StandardScaler using mean/std
            validation_split: Fraction of training data for validation (autoencoder/VAE only).
                Used for early stopping. Default: 0.1 (10% holdout)
            patience: Number of epochs with no improvement before stopping (autoencoder/VAE).
                Default: 10 epochs
            min_delta: Minimum change in validation loss to qualify as improvement.
                Default: 1e-4
        """
        super().__init__(model_id, device)
        self.backend = backend
        self.contamination = contamination
        self._threshold = threshold
        self.normalization = normalization
        self.scaler_type = scaler_type
        self.validation_split = validation_split
        self.patience = patience
        self.min_delta = min_delta
        self.model_type = f"anomaly_{backend}"
        self.supports_streaming = False

        # Backend-specific model
        self._detector = None
        self._scaler = None  # For normalizing input data
        self._is_fitted = False
        self._training_data = None  # Store training data for SHAP explanations

        # For autoencoder/VAE
        self._encoder = None
        self._decoder = None
        # VAE-specific: mu and logvar layers for reparameterization
        self._mu_layer = None
        self._logvar_layer = None

        # Training metrics
        self._best_val_loss = None
        self._epochs_trained = None
        self._early_stopped = False

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
            # PyTorch autoencoder or VAE
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
            # Load scaler_type with backward compatibility (default to "standard" for old models)
            self.scaler_type = checkpoint.get("scaler_type", "standard")
            self._norm_median = checkpoint.get("norm_median")
            self._norm_iqr = checkpoint.get("norm_iqr")
            self._norm_mean = checkpoint.get("norm_mean")
            self._norm_std = checkpoint.get("norm_std")
            self.normalization = checkpoint.get("normalization", "standardization")

            # VAE-specific layers
            if checkpoint.get("backend") == "vae" or "mu_layer" in checkpoint:
                self._mu_layer = checkpoint.get("mu_layer")
                self._logvar_layer = checkpoint.get("logvar_layer")
                self.backend = "vae"

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
            # Load scaler_type with backward compatibility (default to "standard" for old models)
            self.scaler_type = data.get("scaler_type", "standard")
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

        elif self.backend in ("autoencoder", "vae"):
            # Will be created during fit() based on input dimensions
            pass

        # PyOD backends
        elif self.backend == "copod":
            from pyod.models.copod import COPOD

            self._detector = COPOD(contamination=self.contamination)

        elif self.backend == "hbos":
            from pyod.models.hbos import HBOS

            self._detector = HBOS(contamination=self.contamination)

        elif self.backend == "ecod":
            from pyod.models.ecod import ECOD

            self._detector = ECOD(contamination=self.contamination)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Initialize scaler for data normalization based on scaler_type
        self._scaler = self._create_scaler()

    def _create_scaler(self):
        """Create the appropriate scaler based on scaler_type.

        Returns:
            StandardScaler or RobustScaler instance

        Note:
            RobustScaler uses median and IQR for centering/scaling, making it
            much more robust to outliers in the training data. This is the
            recommended choice when training data may contain anomalies.

            StandardScaler uses mean and std, which can be heavily influenced
            by outliers. Use only when training data is clean.
        """
        if self.scaler_type == "robust":
            from sklearn.preprocessing import RobustScaler

            return RobustScaler()
        else:  # standard
            from sklearn.preprocessing import StandardScaler

            return StandardScaler()

    async def fit(
        self,
        data: list[list[float]] | np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> FitResult:
        """Fit the anomaly detector on normal data.

        This method offloads CPU-bound training to a thread pool to avoid
        blocking the async event loop. Health checks and other requests
        can be served during training.

        Args:
            data: Training data (assumed to be mostly normal)
            epochs: Training epochs (autoencoder only)
            batch_size: Batch size (autoencoder only)

        Returns:
            FitResult with training statistics
        """
        from utils.training_executor import get_training_executor

        # Convert to numpy array (must be done before offloading)
        X = np.array(data) if not isinstance(data, np.ndarray) else data

        # Handle 1D input (e.g., single feature time series)
        # Must match the handling in score() to avoid dimension mismatches
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Offload CPU-bound training to thread pool
        loop = asyncio.get_running_loop()
        executor = get_training_executor()

        fit_result = await loop.run_in_executor(
            executor,
            self._fit_sync,
            X,
            epochs,
            batch_size,
        )

        return fit_result

    async def fit_from_file(
        self,
        file_ref,  # FileReference from utils.streaming_data
        batch_size: int = 1000,
        epochs: int = 100,
        auto_cleanup: bool = False,
    ) -> FitResult:
        """Fit the anomaly detector from a streaming file reference.

        Memory-efficient training for large datasets. The file is read in
        batches rather than loading the entire dataset into memory.

        For sklearn backends (isolation_forest, one_class_svm, local_outlier_factor),
        the data is accumulated in batches and fit at once (sklearn models
        don't support incremental fitting for anomaly detection).

        For neural network backends (autoencoder, vae), the data is processed
        in batches during training epochs.

        Args:
            file_ref: FileReference from StreamingDataLoader.upload_file()
            batch_size: Number of rows to process at a time
            epochs: Training epochs (autoencoder/vae only)
            auto_cleanup: If True, clean up temp file after training

        Returns:
            FitResult with training statistics
        """
        from utils.streaming_data import StreamingDataLoader
        from utils.training_executor import get_training_executor

        logger.info(
            f"Starting streaming fit from {file_ref.file_path.name} "
            f"({file_ref.row_count} rows, batch_size={batch_size})"
        )

        # For sklearn backends, we need to accumulate data into memory
        # (sklearn anomaly detectors don't support partial_fit)
        # For large datasets, this may still OOM but we load incrementally
        # which can be helpful for file I/O patterns
        loop = asyncio.get_running_loop()
        executor = get_training_executor()

        # Collect all batches first (streaming from file)
        all_batches = []
        async for batch in file_ref.iter_batches(batch_size=batch_size):
            all_batches.append(batch)

        # Concatenate into single array
        X = np.vstack(all_batches)
        del all_batches  # Free memory

        # Use existing fit logic
        fit_result = await loop.run_in_executor(
            executor,
            self._fit_sync,
            X,
            epochs,
            batch_size,
        )

        # Auto cleanup if requested
        if auto_cleanup:
            loader = StreamingDataLoader()
            await loader.cleanup_file(file_ref.file_id)

        logger.info(
            f"Streaming fit complete: {fit_result.samples_fitted} samples in "
            f"{fit_result.training_time_ms:.1f}ms"
        )

        return fit_result

    def _fit_sync(
        self,
        X: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> FitResult:
        """Synchronous training logic (runs in thread pool).

        This method contains all the CPU-bound training code that would
        otherwise block the async event loop.

        Args:
            X: Preprocessed numpy array of training data
            epochs: Training epochs (autoencoder only)
            batch_size: Batch size (autoencoder only)

        Returns:
            FitResult with training statistics
        """
        start_time = time.time()

        # Fit scaler and transform data
        X_scaled = self._scaler.fit_transform(X)

        # Store scaled training data for SHAP explanations (limit to 1000 samples)
        max_samples = min(1000, len(X_scaled))
        if len(X_scaled) > max_samples:
            indices = np.random.choice(len(X_scaled), max_samples, replace=False)
            self._training_data = X_scaled[indices].copy()
        else:
            self._training_data = X_scaled.copy()

        if self.backend == "autoencoder":
            self._fit_autoencoder_sync(X_scaled, epochs, batch_size)
        elif self.backend == "vae":
            self._fit_vae_sync(X_scaled, epochs, batch_size)
        else:
            # Sklearn models - this is the blocking call
            self._detector.fit(X_scaled)

        self._is_fitted = True

        # Compute raw scores for normalization statistics
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
        """Compute raw anomaly scores synchronously.

        This is the sync version called from _fit_sync in the thread pool.
        """
        if self.backend == "autoencoder":
            return self._autoencoder_scores_sync(X)

        elif self.backend == "vae":
            return self._vae_scores_sync(X)

        elif self.backend == "isolation_forest":
            # IsolationForest: negative score = more anomalous
            return -self._detector.score_samples(X)

        elif self.backend == "one_class_svm":
            # OneClassSVM: negative distance = more anomalous
            return -self._detector.decision_function(X)

        elif self.backend == "local_outlier_factor":
            # LOF: negative score = more anomalous
            return -self._detector.score_samples(X)

        # PyOD backends: use decision_function (higher = more anomalous)
        elif self.backend in ("copod", "hbos", "ecod"):
            return self._detector.decision_function(X)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _autoencoder_scores_sync(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error scores synchronously."""
        import torch

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            encoded = self._encoder(X_tensor)
            decoded = self._decoder(encoded)
            # MSE per sample
            reconstruction_error = torch.mean((X_tensor - decoded) ** 2, dim=1)

        return reconstruction_error.cpu().numpy()

    def _fit_autoencoder_sync(
        self, X: np.ndarray, epochs: int, batch_size: int
    ) -> None:
        """Fit autoencoder model with early stopping (runs in thread pool).

        Uses validation holdout and early stopping to prevent overfitting.
        Training stops when validation loss hasn't improved for `patience` epochs.
        Best weights are restored on early stop.
        """
        import copy

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

        # Split data for validation
        n_samples = len(X)
        n_val = max(1, int(n_samples * self.validation_split))
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train = torch.FloatTensor(X[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X[val_idx]).to(self.device)

        train_dataset = TensorDataset(X_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training setup
        optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=0.001,
        )
        criterion = nn.MSELoss()

        # Early stopping state
        best_val_loss = float("inf")
        best_encoder_state = None
        best_decoder_state = None
        epochs_without_improvement = 0

        self._encoder.train()
        self._decoder.train()

        for epoch in range(epochs):
            # Training phase
            epoch_loss = 0.0
            for (batch,) in train_loader:
                optimizer.zero_grad()
                encoded = self._encoder(batch)
                decoded = self._decoder(encoded)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)

            # Validation phase
            self._encoder.eval()
            self._decoder.eval()
            with torch.no_grad():
                val_encoded = self._encoder(X_val)
                val_decoded = self._decoder(val_encoded)
                val_loss = criterion(val_decoded, X_val).item()
            self._encoder.train()
            self._decoder.train()

            # Early stopping check
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                best_encoder_state = copy.deepcopy(self._encoder.state_dict())
                best_decoder_state = copy.deepcopy(self._decoder.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % 20 == 0:
                logger.debug(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

            if epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best val loss: {best_val_loss:.4f}"
                )
                self._early_stopped = True
                break

        # Restore best weights
        if best_encoder_state is not None:
            self._encoder.load_state_dict(best_encoder_state)
            self._decoder.load_state_dict(best_decoder_state)

        self._best_val_loss = best_val_loss
        self._epochs_trained = epoch + 1
        self._encoder.eval()
        self._decoder.eval()

    def _fit_vae_sync(
        self, X: np.ndarray, epochs: int, batch_size: int
    ) -> None:
        """Fit Variational Autoencoder with early stopping.

        VAE learns a probabilistic latent space. Anomaly score is the negative
        ELBO (Evidence Lower Bound): reconstruction loss + KL divergence.
        Higher scores indicate samples that are unlikely under the learned
        distribution, i.e., anomalies.

        The VAE uses the reparameterization trick: z = mu + eps * exp(0.5 * logvar)
        where eps ~ N(0, 1). This allows backpropagation through the sampling.
        """
        import copy

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        input_dim = X.shape[1]
        hidden_dim = max(input_dim // 2, 8)
        latent_dim = max(hidden_dim // 2, 4)

        # Define encoder (outputs mu and logvar instead of direct latent)
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Separate heads for mu and logvar
        self._mu_layer = nn.Linear(hidden_dim, latent_dim)
        self._logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Define decoder
        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Move to device
        self._encoder = self._encoder.to(self.device)
        self._mu_layer = self._mu_layer.to(self.device)
        self._logvar_layer = self._logvar_layer.to(self.device)
        self._decoder = self._decoder.to(self.device)

        # Split data for validation
        n_samples = len(X)
        n_val = max(1, int(n_samples * self.validation_split))
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train = torch.FloatTensor(X[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X[val_idx]).to(self.device)

        train_dataset = TensorDataset(X_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training setup
        all_params = (
            list(self._encoder.parameters())
            + list(self._mu_layer.parameters())
            + list(self._logvar_layer.parameters())
            + list(self._decoder.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=0.001)

        # Early stopping state
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        def vae_loss(x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            """Compute ELBO loss = reconstruction + KL divergence."""
            # Reconstruction loss (MSE)
            recon_loss = torch.mean(torch.sum((x - recon) ** 2, dim=1))
            # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            return recon_loss + kl_loss

        def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            """Reparameterization trick: z = mu + eps * std."""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        self._encoder.train()
        self._decoder.train()

        for epoch in range(epochs):
            # Training phase
            epoch_loss = 0.0
            for (batch,) in train_loader:
                optimizer.zero_grad()
                h = self._encoder(batch)
                mu = self._mu_layer(h)
                logvar = self._logvar_layer(h)
                z = reparameterize(mu, logvar)
                recon = self._decoder(z)
                loss = vae_loss(batch, recon, mu, logvar)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)

            # Validation phase
            self._encoder.eval()
            self._decoder.eval()
            with torch.no_grad():
                h_val = self._encoder(X_val)
                mu_val = self._mu_layer(h_val)
                logvar_val = self._logvar_layer(h_val)
                z_val = reparameterize(mu_val, logvar_val)
                recon_val = self._decoder(z_val)
                val_loss = vae_loss(X_val, recon_val, mu_val, logvar_val).item()
            self._encoder.train()
            self._decoder.train()

            # Early stopping check
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                best_state = {
                    "encoder": copy.deepcopy(self._encoder.state_dict()),
                    "mu_layer": copy.deepcopy(self._mu_layer.state_dict()),
                    "logvar_layer": copy.deepcopy(self._logvar_layer.state_dict()),
                    "decoder": copy.deepcopy(self._decoder.state_dict()),
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % 20 == 0:
                logger.debug(
                    f"VAE Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

            if epochs_without_improvement >= self.patience:
                logger.info(
                    f"VAE early stopping at epoch {epoch + 1}. "
                    f"Best val loss: {best_val_loss:.4f}"
                )
                self._early_stopped = True
                break

        # Restore best weights
        if best_state is not None:
            self._encoder.load_state_dict(best_state["encoder"])
            self._mu_layer.load_state_dict(best_state["mu_layer"])
            self._logvar_layer.load_state_dict(best_state["logvar_layer"])
            self._decoder.load_state_dict(best_state["decoder"])

        self._best_val_loss = best_val_loss
        self._epochs_trained = epoch + 1
        self._encoder.eval()
        self._decoder.eval()

    def _vae_scores_sync(self, X: np.ndarray) -> np.ndarray:
        """Compute negative ELBO scores for VAE (higher = more anomalous).

        For anomaly detection, we use the reconstruction error plus KL divergence
        as the anomaly score. Points with high negative ELBO are unlikely under
        the learned distribution.
        """
        import torch

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            h = self._encoder(X_tensor)
            mu = self._mu_layer(h)
            logvar = self._logvar_layer(h)
            # Use mean for scoring (no sampling noise)
            z = mu
            recon = self._decoder(z)
            # Reconstruction error per sample
            recon_error = torch.sum((X_tensor - recon) ** 2, dim=1)
            # KL divergence per sample
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            # Total negative ELBO (higher = more anomalous)
            neg_elbo = recon_error + kl_div

        return neg_elbo.cpu().numpy()

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

        elif self.backend == "vae":
            return await self._vae_scores(X)

        elif self.backend == "isolation_forest":
            # IsolationForest: negative score = more anomalous
            return -self._detector.score_samples(X)

        elif self.backend == "one_class_svm":
            # OneClassSVM: negative distance = more anomalous
            return -self._detector.decision_function(X)

        elif self.backend == "local_outlier_factor":
            # LOF: negative score = more anomalous
            return -self._detector.score_samples(X)

        # PyOD backends: use decision_function (higher = more anomalous)
        elif self.backend in ("copod", "hbos", "ecod"):
            return self._detector.decision_function(X)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    async def _autoencoder_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error scores for autoencoder."""
        import torch

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            encoded = self._encoder(X_tensor)
            decoded = self._decoder(encoded)
            # MSE per sample
            reconstruction_error = torch.mean((X_tensor - decoded) ** 2, dim=1)

        return reconstruction_error.cpu().numpy()

    async def _vae_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute negative ELBO scores for VAE (async wrapper).

        Higher scores indicate samples that are unlikely under the learned
        distribution, i.e., anomalies.
        """
        return self._vae_scores_sync(X)

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
            "scaler_type": self.scaler_type,
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
        elif self.backend == "vae":
            import torch

            torch.save(
                {
                    "encoder": self._encoder,
                    "decoder": self._decoder,
                    "mu_layer": self._mu_layer,
                    "logvar_layer": self._logvar_layer,
                    "backend": "vae",
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
        self._training_data = None
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
                "scaler_type": self.scaler_type,
                "is_fitted": self._is_fitted,
            }
        )
        return info

    def get_sklearn_model(self) -> Any:
        """Get the underlying sklearn model for SHAP explanations.

        Returns:
            The sklearn detector (IsolationForest, OneClassSVM, etc.)
            or None if using autoencoder/VAE backend.
        """
        if self.backend in ("autoencoder", "vae"):
            return None
        return self._detector

    def get_training_data(self) -> np.ndarray | None:
        """Get a sample of the training data for SHAP background.

        Returns:
            Numpy array of scaled training data (max 1000 samples)
            or None if model hasn't been trained.
        """
        return self._training_data
