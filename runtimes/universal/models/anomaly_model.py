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
    ):
        """Initialize anomaly detection model.

        Args:
            model_id: Model identifier (for caching) or path to pre-trained model
            device: Target device (cpu recommended for sklearn models)
            backend: Anomaly detection backend
            contamination: Expected proportion of anomalies (0.0 to 0.5)
            threshold: Custom anomaly threshold (auto-determined if None)
        """
        super().__init__(model_id, device)
        self.backend = backend
        self.contamination = contamination
        self._threshold = threshold
        self.model_type = f"anomaly_{backend}"
        self.supports_streaming = False

        # Backend-specific model
        self._detector = None
        self._scaler = None  # For normalizing input data
        self._is_fitted = False

        # For autoencoder
        self._encoder = None
        self._decoder = None

    @property
    def threshold(self) -> float:
        """Get anomaly threshold."""
        return self._threshold or 0.5

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
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()

    async def fit(
        self,
        data: list[list[float]] | np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> FitResult:
        """Fit the anomaly detector on normal data.

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
            await self._fit_autoencoder(X_scaled, epochs, batch_size)
        else:
            # Sklearn models
            self._detector.fit(X_scaled)

        self._is_fitted = True

        # Auto-determine threshold if not set
        # Threshold is computed on normalized scores (0-1 range) for consistency
        if self._threshold is None:
            raw_scores = await self._compute_raw_scores(X_scaled)
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

    async def _fit_autoencoder(
        self, X: np.ndarray, epochs: int, batch_size: int
    ) -> None:
        """Fit autoencoder model."""
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

        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training
        optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=0.001,
        )
        criterion = nn.MSELoss()

        self._encoder.train()
        self._decoder.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in dataloader:
                optimizer.zero_grad()
                encoded = self._encoder(batch)
                decoded = self._decoder(encoded)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                logger.debug(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}"
                )

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

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range."""
        # Use sigmoid-like normalization
        # Center around median and scale by IQR
        median = np.median(scores)
        iqr = np.percentile(scores, 75) - np.percentile(scores, 25)

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

        if self.backend == "autoencoder":
            import torch

            torch.save(
                {
                    "encoder": self._encoder,
                    "decoder": self._decoder,
                    "threshold": self._threshold,
                    "scaler": self._scaler,
                },
                model_path.with_suffix(".pt"),
            )
        else:
            try:
                import joblib

                joblib.dump(
                    {
                        "detector": self._detector,
                        "scaler": self._scaler,
                        "threshold": self._threshold,
                    },
                    model_path.with_suffix(".joblib"),
                )
            except ImportError:
                with open(model_path.with_suffix(".pkl"), "wb") as f:
                    pickle.dump(
                        {
                            "detector": self._detector,
                            "scaler": self._scaler,
                            "threshold": self._threshold,
                        },
                        f,
                    )

        logger.info(f"Anomaly model saved to {path}")

    async def unload(self) -> None:
        """Unload the model and free resources."""
        self._detector = None
        self._scaler = None
        self._encoder = None
        self._decoder = None
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
                "is_fitted": self._is_fitted,
            }
        )
        return info
