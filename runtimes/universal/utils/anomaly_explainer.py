"""SHAP-based Anomaly Explanations.

Provides human-readable explanations for why data points are flagged as anomalies.
Uses SHAP (SHapley Additive exPlanations) to compute feature importance.

Usage:
    explainer = AnomalyExplainer(model, background_data)
    explanations = explainer.explain(anomalous_points)
    # Returns: [{"feature": "cpu", "importance": 0.82, "value": 95.0, "direction": "high"}, ...]
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyExplainer:
    """SHAP-based explainer for anomaly detection models.

    Uses SHAP KernelExplainer for model-agnostic explanations of
    why data points are classified as anomalies.

    Attributes:
        model_fn: Function that returns anomaly scores (higher = more anomalous)
        background_data: Reference data for SHAP calculations
        feature_names: Names of features (optional)
    """

    def __init__(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        background_data: np.ndarray,
        feature_names: list[str] | None = None,
        n_background_samples: int = 100,
    ):
        """Initialize anomaly explainer.

        Args:
            model_fn: Function that takes data and returns anomaly scores
            background_data: Reference data for SHAP (represents "normal" data)
            feature_names: Optional names for each feature
            n_background_samples: Number of background samples to use
        """
        import shap

        self.model_fn = model_fn
        self.feature_names = feature_names

        # Subsample background data if too large
        if len(background_data) > n_background_samples:
            indices = np.random.choice(
                len(background_data),
                n_background_samples,
                replace=False,
            )
            background_data = background_data[indices]

        self.background_data = background_data

        # Create SHAP explainer
        self.explainer = shap.KernelExplainer(model_fn, background_data)

    def explain(
        self,
        data: np.ndarray,
        nsamples: int = 100,
    ) -> list[dict[str, Any]]:
        """Explain anomaly scores for given data points.

        Args:
            data: Data points to explain (n_samples, n_features)
            nsamples: Number of samples for SHAP approximation

        Returns:
            List of explanations, one per data point.
            Each explanation is a list of feature contributions.
        """
        # Get SHAP values
        shap_values = self.explainer.shap_values(data, nsamples=nsamples)

        # Handle case where shap_values is a list (for multi-class)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        explanations = []
        for i, row_shap in enumerate(shap_values):
            # Build explanation for this data point
            features = []
            for j, shap_val in enumerate(row_shap):
                feature_name = self.feature_names[j] if self.feature_names else f"feature_{j}"
                feature_value = data[i, j]

                # Determine if feature contributes to anomaly (positive SHAP = more anomalous)
                direction = "high" if shap_val > 0 else "low"

                features.append({
                    "feature": feature_name,
                    "shap_value": float(shap_val),
                    "importance": float(abs(shap_val)),
                    "value": float(feature_value),
                    "direction": direction,
                })

            # Sort by absolute importance (most important first)
            features.sort(key=lambda x: x["importance"], reverse=True)

            explanations.append({
                "features": features,
                "top_contributors": features[:3] if len(features) >= 3 else features,
                "total_shap": float(sum(abs(f["shap_value"]) for f in features)),
            })

        return explanations

    def explain_single(
        self,
        data_point: np.ndarray,
        nsamples: int = 100,
    ) -> dict[str, Any]:
        """Explain a single data point.

        Args:
            data_point: Single data point (1D array)
            nsamples: Number of samples for SHAP approximation

        Returns:
            Explanation dictionary
        """
        data = data_point.reshape(1, -1)
        return self.explain(data, nsamples=nsamples)[0]

    def get_feature_summary(
        self,
        explanations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Get summary statistics across all explanations.

        Args:
            explanations: List of explanations from explain()

        Returns:
            Summary with average importance per feature
        """
        if not explanations:
            return {"features": [], "total_explained": 0}

        # Aggregate importance by feature
        feature_importance: dict[str, list[float]] = {}
        for exp in explanations:
            for feat in exp["features"]:
                name = feat["feature"]
                if name not in feature_importance:
                    feature_importance[name] = []
                feature_importance[name].append(feat["importance"])

        # Compute averages
        summary = []
        for name, importances in feature_importance.items():
            summary.append({
                "feature": name,
                "mean_importance": float(np.mean(importances)),
                "max_importance": float(np.max(importances)),
                "count": len(importances),
            })

        # Sort by mean importance
        summary.sort(key=lambda x: x["mean_importance"], reverse=True)

        return {
            "features": summary,
            "total_explained": len(explanations),
        }


def create_explainer_for_sklearn(
    model: Any,
    background_data: np.ndarray,
    feature_names: list[str] | None = None,
    n_background_samples: int = 100,
) -> AnomalyExplainer:
    """Create an explainer for sklearn-style anomaly detection models.

    Works with IsolationForest, LocalOutlierFactor, OneClassSVM, etc.

    Args:
        model: Trained sklearn anomaly detection model
        background_data: Reference data
        feature_names: Optional feature names
        n_background_samples: Number of background samples

    Returns:
        AnomalyExplainer instance
    """
    # sklearn anomaly detectors return negative scores for anomalies
    # We negate to make higher = more anomalous (convention for SHAP)
    if hasattr(model, "decision_function"):
        def model_fn(x: np.ndarray) -> np.ndarray:
            # More negative = more anomalous, so negate
            return -model.decision_function(x)
    elif hasattr(model, "score_samples"):
        def model_fn(x: np.ndarray) -> np.ndarray:
            # More negative = more anomalous, so negate
            return -model.score_samples(x)
    else:
        raise ValueError("Model must have decision_function or score_samples method")

    return AnomalyExplainer(
        model_fn=model_fn,
        background_data=background_data,
        feature_names=feature_names,
        n_background_samples=n_background_samples,
    )
