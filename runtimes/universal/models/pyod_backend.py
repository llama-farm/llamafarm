"""
Unified PyOD backend adapter for anomaly detection.

This module provides a single interface to ALL anomaly detection algorithms,
consolidating both legacy sklearn-based backends and new PyOD algorithms.

Legacy Backend Mapping (Backward Compatible):
- isolation_forest → PyOD IForest
- one_class_svm → PyOD OCSVM
- local_outlier_factor → PyOD LOF
- autoencoder → PyOD AutoEncoder

New PyOD Backends:
- ecod: Empirical Cumulative Distribution (fast, parameter-free)
- hbos: Histogram-based Outlier Score (fastest)
- knn: K-Nearest Neighbors outlier detection
- copod: Copula-Based Outlier Detection (fast, parameter-free)
- cblof: Clustering-Based Local Outlier Factor
- suod: Scalable Unsupervised Outlier Detection (ensemble)
- loda: Lightweight Online Detector of Anomalies
- mcd: Minimum Covariance Determinant

All backends share a common interface:
- fit(X): Train on data
- decision_function(X): Return anomaly scores (higher = more anomalous)
- predict(X): Return binary labels (1 = anomaly)

Reference: https://pyod.readthedocs.io/
"""

import logging
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Backend Type Definitions
# =============================================================================

# Legacy backend names (for backward compatibility)
LegacyBackend = Literal["isolation_forest", "one_class_svm", "local_outlier_factor", "autoencoder"]

# New PyOD backend names
PyODBackend = Literal["ecod", "hbos", "knn", "copod", "cblof", "suod", "loda", "mcd"]

# All supported backend names
AnomalyBackendType = Literal[
    # Legacy (mapped to PyOD)
    "isolation_forest", "one_class_svm", "local_outlier_factor", "autoencoder",
    # New PyOD backends
    "ecod", "hbos", "knn", "copod", "cblof", "suod", "loda", "mcd",
]

# =============================================================================
# Backend Registry
# =============================================================================

# Maps backend names to PyOD module paths
# Format: "backend_name": "pyod.models.module.ClassName"
BACKEND_REGISTRY: dict[str, str] = {
    # Legacy backends (mapped to PyOD equivalents)
    "isolation_forest": "pyod.models.iforest.IForest",
    "one_class_svm": "pyod.models.ocsvm.OCSVM",
    "local_outlier_factor": "pyod.models.lof.LOF",
    "autoencoder": "pyod.models.auto_encoder.AutoEncoder",
    # New PyOD backends
    "ecod": "pyod.models.ecod.ECOD",
    "hbos": "pyod.models.hbos.HBOS",
    "knn": "pyod.models.knn.KNN",
    "copod": "pyod.models.copod.COPOD",
    "cblof": "pyod.models.cblof.CBLOF",
    "suod": "pyod.models.suod.SUOD",
    "loda": "pyod.models.loda.LODA",
    "mcd": "pyod.models.mcd.MCD",
}

# Backend metadata for /v1/anomaly/backends endpoint
BACKEND_INFO: dict[str, dict[str, Any]] = {
    # Legacy backends
    "isolation_forest": {
        "name": "Isolation Forest",
        "description": "Tree-based ensemble method. Fast and effective default choice.",
        "category": "legacy",
        "speed": "fast",
        "memory": "medium",
        "parameters": ["n_estimators", "max_samples", "contamination"],
        "best_for": "General purpose, high-dimensional data",
    },
    "one_class_svm": {
        "name": "One-Class SVM",
        "description": "Support vector machine for outlier detection. Good for small datasets.",
        "category": "legacy",
        "speed": "slow",
        "memory": "high",
        "parameters": ["kernel", "nu", "gamma"],
        "best_for": "Small datasets, well-defined boundaries",
    },
    "local_outlier_factor": {
        "name": "Local Outlier Factor",
        "description": "Density-based anomaly detection. Good for clustered anomalies.",
        "category": "legacy",
        "speed": "medium",
        "memory": "high",
        "parameters": ["n_neighbors", "contamination"],
        "best_for": "Clustered data, local anomalies",
    },
    "autoencoder": {
        "name": "AutoEncoder",
        "description": "Neural network that learns to reconstruct normal data. Best for complex patterns.",
        "category": "deep_learning",
        "speed": "slow",
        "memory": "high",
        "parameters": ["hidden_neurons", "epochs", "batch_size", "contamination"],
        "best_for": "Complex patterns, large datasets",
    },
    # Fast backends (parameter-free or minimal tuning)
    "ecod": {
        "name": "ECOD",
        "description": "Empirical Cumulative Distribution. Fast and parameter-free.",
        "category": "fast",
        "speed": "fast",
        "memory": "low",
        "parameters": ["contamination"],
        "best_for": "General purpose, no tuning needed",
    },
    "hbos": {
        "name": "HBOS",
        "description": "Histogram-based Outlier Score. Fastest algorithm available.",
        "category": "fast",
        "speed": "very_fast",
        "memory": "low",
        "parameters": ["n_bins", "contamination"],
        "best_for": "Speed-critical applications, high dimensions",
    },
    "copod": {
        "name": "COPOD",
        "description": "Copula-Based Outlier Detection. Fast and parameter-free.",
        "category": "fast",
        "speed": "fast",
        "memory": "low",
        "parameters": ["contamination"],
        "best_for": "Interpretable results, no tuning needed",
    },
    # Distance-based backends
    "knn": {
        "name": "KNN",
        "description": "K-Nearest Neighbors outlier detection. Distance-based.",
        "category": "distance",
        "speed": "medium",
        "memory": "high",
        "parameters": ["n_neighbors", "method", "contamination"],
        "best_for": "Distance-based anomalies",
    },
    "mcd": {
        "name": "MCD",
        "description": "Minimum Covariance Determinant. Robust covariance estimation.",
        "category": "distance",
        "speed": "medium",
        "memory": "medium",
        "parameters": ["contamination"],
        "best_for": "Multivariate Gaussian data",
    },
    # Clustering-based backends
    "cblof": {
        "name": "CBLOF",
        "description": "Clustering-Based Local Outlier Factor. Combines clustering with outlier detection.",
        "category": "clustering",
        "speed": "medium",
        "memory": "medium",
        "parameters": ["n_clusters", "contamination"],
        "best_for": "Grouped/clustered data",
    },
    # Ensemble backends
    "suod": {
        "name": "SUOD",
        "description": "Scalable Unsupervised Outlier Detection. Ensemble of multiple algorithms.",
        "category": "ensemble",
        "speed": "slow",
        "memory": "high",
        "parameters": ["base_estimators", "contamination"],
        "best_for": "Most robust detection, combining multiple methods",
    },
    # Online/Streaming backends
    "loda": {
        "name": "LODA",
        "description": "Lightweight Online Detector of Anomalies. Good for streaming data.",
        "category": "streaming",
        "speed": "fast",
        "memory": "low",
        "parameters": ["n_bins", "n_random_cuts", "contamination"],
        "best_for": "Streaming data, online detection",
    },
}


def get_all_backends() -> list[str]:
    """Get list of all available backend names."""
    return list(BACKEND_REGISTRY.keys())


def get_backend_info(backend: str | None = None) -> dict[str, Any]:
    """Get metadata for a specific backend or all backends.

    Args:
        backend: Backend name, or None to get all backends

    Returns:
        Backend info dict, or dict of all backends if backend is None
    """
    if backend is None:
        return BACKEND_INFO.copy()
    if backend not in BACKEND_INFO:
        raise ValueError(f"Unknown backend: {backend}. Available: {get_all_backends()}")
    return BACKEND_INFO[backend].copy()


def is_valid_backend(backend: str) -> bool:
    """Check if a backend name is valid."""
    return backend in BACKEND_REGISTRY


def is_legacy_backend(backend: str) -> bool:
    """Check if a backend is a legacy (pre-PyOD) backend name."""
    return backend in ("isolation_forest", "one_class_svm", "local_outlier_factor", "autoencoder")


# =============================================================================
# Detector Creation
# =============================================================================

def create_detector(
    backend: AnomalyBackendType,
    contamination: float = 0.1,
    **kwargs: Any,
) -> Any:
    """Create a PyOD detector instance.

    Args:
        backend: Backend type (legacy or new PyOD name)
        contamination: Expected proportion of outliers (0.0 to 0.5)
        **kwargs: Additional backend-specific parameters

    Returns:
        Initialized PyOD detector (not fitted)

    Raises:
        ValueError: If backend is not supported
        ImportError: If PyOD is not installed
    """
    if backend not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: {get_all_backends()}"
        )

    # Import the detector class dynamically
    module_path = BACKEND_REGISTRY[backend]
    module_name, class_name = module_path.rsplit(".", 1)

    try:
        import importlib
        module = importlib.import_module(module_name)
        detector_class = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"PyOD is required for backend '{backend}'. "
            f"Install with: pip install pyod"
        ) from e

    # Create detector with appropriate parameters based on backend
    detector = _create_detector_with_params(
        detector_class, backend, contamination, **kwargs
    )

    logger.info(f"Created PyOD detector: {backend} (contamination={contamination})")
    return detector


def _create_detector_with_params(
    detector_class: type,
    backend: str,
    contamination: float,
    **kwargs: Any,
) -> Any:
    """Create detector instance with backend-specific parameters."""

    # Common parameters
    common_params = {"contamination": contamination}

    # Backend-specific parameter handling
    if backend == "isolation_forest":
        return detector_class(
            **common_params,
            n_estimators=kwargs.get("n_estimators", 100),
            max_samples=kwargs.get("max_samples", "auto"),
            random_state=kwargs.get("random_state", 42),
            n_jobs=kwargs.get("n_jobs", -1),
        )

    elif backend == "one_class_svm":
        # OCSVM uses 'nu' instead of 'contamination' internally
        # but PyOD's OCSVM accepts contamination
        return detector_class(
            **common_params,
            kernel=kwargs.get("kernel", "rbf"),
            gamma=kwargs.get("gamma", "auto"),
            nu=kwargs.get("nu", contamination),  # nu ≈ contamination
        )

    elif backend == "local_outlier_factor":
        return detector_class(
            **common_params,
            n_neighbors=kwargs.get("n_neighbors", 20),
            n_jobs=kwargs.get("n_jobs", -1),
        )

    elif backend == "autoencoder":
        # AutoEncoder has more parameters
        # PyOD uses hidden_neuron_list and epoch_num (not hidden_neurons/epochs)
        # Accept friendly names and map to PyOD parameter names
        hidden_neuron_list = kwargs.get(
            "hidden_neuron_list",
            kwargs.get("hidden_neurons", [64, 32])
        )
        return detector_class(
            **common_params,
            hidden_neuron_list=hidden_neuron_list,
            epoch_num=kwargs.get("epochs", kwargs.get("epoch_num", 100)),
            batch_size=kwargs.get("batch_size", 32),
            preprocessing=kwargs.get("preprocessing", True),
            verbose=kwargs.get("verbose", 0),
            random_state=kwargs.get("random_state", 42),
        )

    elif backend == "ecod":
        # ECOD is parameter-free
        return detector_class(**common_params)

    elif backend == "hbos":
        return detector_class(
            **common_params,
            n_bins=kwargs.get("n_bins", 10),
        )

    elif backend == "copod":
        # COPOD is parameter-free
        return detector_class(**common_params)

    elif backend == "knn":
        return detector_class(
            **common_params,
            n_neighbors=kwargs.get("n_neighbors", 5),
            method=kwargs.get("method", "largest"),
            n_jobs=kwargs.get("n_jobs", -1),
        )

    elif backend == "cblof":
        return detector_class(
            **common_params,
            n_clusters=kwargs.get("n_clusters", 8),
            random_state=kwargs.get("random_state", 42),
        )

    elif backend == "suod":
        # SUOD is an ensemble - can accept base estimators
        base_estimators = kwargs.get("base_estimators")
        if base_estimators is None:
            # Default ensemble: fast + accurate mix
            from pyod.models.copod import COPOD
            from pyod.models.hbos import HBOS
            from pyod.models.iforest import IForest
            base_estimators = [
                IForest(contamination=contamination),
                HBOS(contamination=contamination),
                COPOD(contamination=contamination),
            ]
        return detector_class(
            base_estimators=base_estimators,
            contamination=contamination,
            n_jobs=kwargs.get("n_jobs", -1),
        )

    elif backend == "loda":
        return detector_class(
            **common_params,
            n_bins=kwargs.get("n_bins", 10),
            n_random_cuts=kwargs.get("n_random_cuts", 100),
        )

    elif backend == "mcd":
        return detector_class(
            **common_params,
            random_state=kwargs.get("random_state", 42),
        )

    else:
        # Fallback: pass contamination plus any user-provided kwargs
        # This ensures custom parameters aren't silently ignored
        return detector_class(**common_params, **kwargs)


# =============================================================================
# Detector Operations
# =============================================================================

def fit_detector(detector: Any, X: np.ndarray) -> None:
    """Fit a PyOD detector on training data.

    Args:
        detector: PyOD detector instance
        X: Training data (n_samples, n_features)

    Raises:
        ValueError: If X is empty or has invalid shape
    """
    if X is None or len(X) == 0:
        raise ValueError("Training data cannot be empty")
    if len(X.shape) != 2:
        raise ValueError(f"Training data must be 2D, got shape {X.shape}")
    if X.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples for training, got {X.shape[0]}")

    detector.fit(X)
    logger.debug(f"Fitted detector on {X.shape[0]} samples, {X.shape[1]} features")


def get_decision_scores(detector: Any, X: np.ndarray) -> np.ndarray:
    """Get decision scores from a fitted PyOD detector.

    PyOD's decision_function() returns scores where higher values
    indicate more anomalous samples.

    Args:
        detector: Fitted PyOD detector
        X: Data to score (n_samples, n_features)

    Returns:
        Anomaly scores (n_samples,) - higher = more anomalous
    """
    return detector.decision_function(X)


def get_predictions(detector: Any, X: np.ndarray) -> np.ndarray:
    """Get binary predictions from a fitted PyOD detector.

    Args:
        detector: Fitted PyOD detector
        X: Data to predict (n_samples, n_features)

    Returns:
        Binary labels (n_samples,) - 0 = normal, 1 = anomaly
    """
    return detector.predict(X)


def get_threshold(detector: Any) -> float:
    """Get the anomaly threshold from a fitted detector.

    Args:
        detector: Fitted PyOD detector

    Returns:
        Threshold value (samples with scores > threshold are anomalies)
    """
    return detector.threshold_


# =============================================================================
# Serialization
# =============================================================================

def serialize_detector(detector: Any, backend: str) -> dict[str, Any]:
    """Prepare a detector for saving.

    Args:
        detector: Fitted PyOD detector
        backend: Backend name

    Returns:
        Dictionary ready for joblib.dump()
    """
    return {
        "detector": detector,
        "backend": backend,
        "backend_type": "pyod",
    }


def get_backends_response() -> dict[str, Any]:
    """Get formatted response for /v1/anomaly/backends endpoint.

    Returns:
        API response with all backend information
    """
    backends_list = []
    for name, info in BACKEND_INFO.items():
        backends_list.append({
            "backend": name,
            "name": info["name"],
            "description": info["description"],
            "category": info["category"],
            "speed": info["speed"],
            "memory": info["memory"],
            "parameters": info["parameters"],
            "best_for": info["best_for"],
            "is_legacy": is_legacy_backend(name),
        })

    # Sort by category, then by name
    category_order = ["legacy", "fast", "distance", "clustering", "ensemble", "streaming", "deep_learning"]
    backends_list.sort(key=lambda x: (category_order.index(x["category"]), x["backend"]))

    return {
        "object": "list",
        "data": backends_list,
        "total": len(backends_list),
        "categories": list(set(b["category"] for b in backends_list)),
    }
