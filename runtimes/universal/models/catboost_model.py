"""CatBoost Model for Classification and Regression.

This module provides CatBoost gradient boosting with unique features:
- Native categorical support: No one-hot encoding needed
- Incremental learning: Update model without full retrain
- GPU acceleration: Fast training on NVIDIA GPUs
- Ordered boosting: Reduces overfitting

Use Cases:
- Tabular data classification
- Real-time model updates with streaming data
- Handling mixed numeric/categorical features
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Import CatBoost conditionally to avoid import errors
try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None
    CatBoostRegressor = None
    Pool = None


# Default configuration
DEFAULT_ITERATIONS = 1000
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DEPTH = 6
DEFAULT_TASK_TYPE = "CPU"  # Can be "GPU" if available


def get_catboost_info() -> dict[str, Any]:
    """Get information about CatBoost availability and capabilities.

    Returns:
        Dictionary with CatBoost status and supported features
    """
    if not CATBOOST_AVAILABLE:
        return {
            "available": False,
            "error": "CatBoost not installed",
        }

    # Check for GPU support
    gpu_available = False
    try:
        # Try to detect CUDA
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass
    except Exception:
        pass

    return {
        "available": True,
        "gpu_available": gpu_available,
        "model_types": ["classifier", "regressor"],
        "features": [
            f for f in [
                "native_categorical",
                "incremental_learning",
                "gpu_acceleration" if gpu_available else None,
                "ordered_boosting",
            ] if f is not None
        ],
    }


class CatBoostModel:
    """CatBoost gradient boosting model.

    Supports both classification and regression with:
    - Native categorical feature handling
    - Incremental learning (update without full retrain)
    - GPU acceleration when available
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        model_type: str = "classifier",
        iterations: int = DEFAULT_ITERATIONS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        depth: int = DEFAULT_DEPTH,
        task_type: str = DEFAULT_TASK_TYPE,
        cat_features: list[int] | list[str] | None = None,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize CatBoost model.

        Args:
            model_id: Unique identifier for the model
            device: Device to use (cpu or cuda)
            model_type: Type of model ("classifier" or "regressor")
            iterations: Number of boosting iterations
            learning_rate: Learning rate for gradient descent
            depth: Depth of individual trees
            task_type: Compute task type ("CPU" or "GPU")
            cat_features: Indices or names of categorical features
            random_state: Random seed for reproducibility
            **kwargs: Additional CatBoost parameters
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Run: uv add catboost")

        self.model_id = model_id
        self.device = device
        self.model_type = model_type
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.task_type = task_type
        self.cat_features = cat_features
        self.random_state = random_state
        self.extra_params = kwargs

        self._model = None
        self._is_loaded = False
        self._is_fitted = False
        self._classes = None
        self._feature_names = None
        self._n_features = None

    async def load(self) -> None:
        """Initialize the CatBoost model."""
        start_time = time.time()

        # Build parameters
        params = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "random_seed": self.random_state,
            "verbose": False,
            "allow_writing_files": False,
            **self.extra_params,
        }

        # Set task type (CPU or GPU)
        if self.task_type == "GPU" and self.device != "cpu":
            params["task_type"] = "GPU"
        else:
            params["task_type"] = "CPU"

        # Create model based on type
        if self.model_type == "classifier":
            self._model = CatBoostClassifier(**params)
        elif self.model_type == "regressor":
            self._model = CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self._is_loaded = True
        load_time_ms = (time.time() - start_time) * 1000
        logger.info(f"CatBoost {self.model_type} initialized in {load_time_ms:.2f}ms")

    async def fit(
        self,
        X: np.ndarray | list,
        y: np.ndarray | list,
        feature_names: list[str] | None = None,
        eval_set: tuple | None = None,
        early_stopping_rounds: int | None = None,
    ) -> dict[str, Any]:
        """Train the model on data.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels
            feature_names: Optional feature names
            eval_set: Optional (X_val, y_val) for validation
            early_stopping_rounds: Stop if no improvement for N rounds

        Returns:
            Training statistics
        """
        if not self._is_loaded:
            await self.load()

        start_time = time.time()

        # Convert to numpy if needed
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        # Store metadata
        self._n_features = X.shape[1]
        self._feature_names = feature_names or [f"feature_{i}" for i in range(self._n_features)]

        # Create Pool for better categorical handling
        pool = Pool(
            data=X,
            label=y,
            cat_features=self.cat_features,
            feature_names=self._feature_names,
        )

        # Build fit kwargs
        fit_kwargs = {"use_best_model": False}
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_pool = Pool(
                data=np.array(X_val),
                label=np.array(y_val),
                cat_features=self.cat_features,
                feature_names=self._feature_names,
            )
            fit_kwargs["eval_set"] = eval_pool
            fit_kwargs["use_best_model"] = True

        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        # Run training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._model.fit(pool, **fit_kwargs))

        self._is_fitted = True
        if self.model_type == "classifier":
            self._classes = self._model.classes_.tolist()

        fit_time_ms = (time.time() - start_time) * 1000

        # Get best iteration if early stopping was used
        best_iteration = None
        if hasattr(self._model, "best_iteration_") and self._model.best_iteration_ is not None:
            best_iteration = self._model.best_iteration_

        logger.info(f"CatBoost fit completed in {fit_time_ms:.2f}ms")

        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "samples_fitted": len(y),
            "n_features": self._n_features,
            "iterations": self._model.tree_count_,
            "best_iteration": best_iteration,
            "classes": self._classes,
            "fit_time_ms": fit_time_ms,
        }

    async def update(
        self,
        X: np.ndarray | list,
        y: np.ndarray | list,
        sample_weight: np.ndarray | list | None = None,
    ) -> dict[str, Any]:
        """Incrementally update the model with new data.

        Uses CatBoost's init_model parameter for warm starting.

        Args:
            X: New training features
            y: New training labels
            sample_weight: Optional sample weights

        Returns:
            Update statistics
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before updating. Call fit() first.")

        start_time = time.time()

        # Convert to numpy if needed
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        # Create Pool
        pool = Pool(
            data=X,
            label=y,
            cat_features=self.cat_features,
            feature_names=self._feature_names,
            weight=sample_weight,
        )

        # Create new model with same parameters but warm start from current model
        params = {
            "iterations": max(100, self.iterations // 10),  # Fewer iterations for update
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "random_seed": self.random_state,
            "verbose": False,
            "allow_writing_files": False,
            "task_type": self._model.get_param("task_type"),
            **self.extra_params,
        }

        if self.model_type == "classifier":
            new_model = CatBoostClassifier(**params)
        else:
            new_model = CatBoostRegressor(**params)

        # Fit with init_model for warm start
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: new_model.fit(pool, init_model=self._model)
        )

        # Replace old model
        old_tree_count = self._model.tree_count_
        self._model = new_model
        new_tree_count = self._model.tree_count_

        update_time_ms = (time.time() - start_time) * 1000
        logger.info(f"CatBoost incremental update in {update_time_ms:.2f}ms")

        return {
            "model_id": self.model_id,
            "samples_added": len(y),
            "trees_before": old_tree_count,
            "trees_after": new_tree_count,
            "update_time_ms": update_time_ms,
        }

    async def predict(
        self,
        X: np.ndarray | list,
    ) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features to predict (n_samples, n_features)

        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.array(X) if not isinstance(X, np.ndarray) else X

        # Run prediction in thread pool
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(None, lambda: self._model.predict(X))

        return predictions

    async def predict_proba(
        self,
        X: np.ndarray | list,
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features to predict (n_samples, n_features)

        Returns:
            Predicted probabilities (n_samples, n_classes)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.model_type != "classifier":
            raise ValueError("predict_proba only available for classifiers")

        X = np.array(X) if not isinstance(X, np.ndarray) else X

        # Run prediction in thread pool
        loop = asyncio.get_event_loop()
        probabilities = await loop.run_in_executor(
            None,
            lambda: self._model.predict_proba(X)
        )

        return probabilities

    async def get_feature_importance(
        self,
        importance_type: str = "FeatureImportance",
    ) -> list[tuple[str, float]]:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance ("FeatureImportance", "PredictionValuesChange", etc.)

        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importance = self._model.get_feature_importance(type=importance_type)
        feature_names = self._feature_names or [f"feature_{i}" for i in range(len(importance))]

        # Sort by importance
        pairs = list(zip(feature_names, importance, strict=True))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        return pairs

    async def save(self, path: str | Path) -> str:
        """Save model to disk.

        Args:
            path: Path to save model

        Returns:
            Path where model was saved
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        model_data = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "cat_features": self.cat_features,
            "random_state": self.random_state,
            "classes": self._classes,
            "feature_names": self._feature_names,
            "n_features": self._n_features,
            "extra_params": self.extra_params,
        }

        # Use joblib for the full model + metadata
        save_data = {
            "metadata": model_data,
            "model": self._model,
        }

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: joblib.dump(save_data, path))

        logger.info(f"CatBoost model saved to {path}")
        return str(path)

    async def load_from_path(self, path: str | Path) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        loop = asyncio.get_event_loop()
        save_data = await loop.run_in_executor(None, lambda: joblib.load(path))

        # Restore metadata
        metadata = save_data["metadata"]
        self.model_id = metadata["model_id"]
        self.model_type = metadata["model_type"]
        self.iterations = metadata["iterations"]
        self.learning_rate = metadata["learning_rate"]
        self.depth = metadata["depth"]
        self.cat_features = metadata["cat_features"]
        self.random_state = metadata["random_state"]
        self._classes = metadata.get("classes")
        self._feature_names = metadata.get("feature_names")
        self._n_features = metadata.get("n_features")
        self.extra_params = metadata.get("extra_params", {})

        # Restore model
        self._model = save_data["model"]
        self._is_loaded = True
        self._is_fitted = True

        logger.info(f"CatBoost model loaded from {path}")

    async def unload(self) -> None:
        """Free model resources."""
        self._model = None
        self._is_loaded = False
        self._is_fitted = False
        logger.info(f"CatBoost model {self.model_id} unloaded")

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    @property
    def classes(self) -> list | None:
        """Get class labels for classifier."""
        return self._classes

    @property
    def n_features(self) -> int | None:
        """Get number of features."""
        return self._n_features

    @property
    def feature_names(self) -> list[str] | None:
        """Get feature names."""
        return self._feature_names
