"""SHAP Explainability Module.

This module provides explainability for ML predictions using SHAP
(SHapley Additive exPlanations). SHAP values show the contribution
of each feature to a prediction.

Key Features:
- Tree explainer for tree-based models (IForest, CatBoost, XGBoost)
- Linear explainer for linear models
- Kernel explainer for any model (model-agnostic)
- Natural language narrative generation

Use Cases:
- ML model debugging
- Regulatory compliance
- Building trust with stakeholders
- Generating LLM-ready explanations
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    shap = None
    _HAS_SHAP = False

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Contribution of a single feature to the prediction."""

    feature: str
    value: float  # Actual feature value
    shap_value: float  # SHAP contribution
    direction: str  # "increases" or "decreases"


@dataclass
class SHAPExplanation:
    """SHAP explanation for a single sample."""

    sample_index: int
    base_value: float  # Expected value (average prediction)
    prediction: float  # Actual prediction
    contributions: list[FeatureContribution]


@dataclass
class NarrativeExplanation:
    """Natural language explanation."""

    summary: str  # One sentence summary
    details: list[str]  # Per-feature explanations


# Supported explainer types — lazy to avoid import-time shap access
def _get_explainer_types() -> dict:
    if not _HAS_SHAP:
        return {}
    return {
        "tree": {
            "description": "Fast explainer for tree-based models (IForest, CatBoost, XGBoost)",
            "class": shap.TreeExplainer,
            "supported_models": ["isolation_forest", "catboost", "xgboost", "random_forest", "lightgbm"],
        },
        "linear": {
            "description": "Fast explainer for linear models",
            "class": shap.LinearExplainer,
            "supported_models": ["linear", "logistic", "ridge", "lasso"],
        },
        "kernel": {
            "description": "Model-agnostic explainer (slower but works with any model)",
            "class": shap.KernelExplainer,
            "supported_models": ["any"],
        },
    }

EXPLAINER_TYPES = _get_explainer_types()


def get_explainer_types() -> list[dict[str, Any]]:
    """Get information about available explainer types.

    Returns:
        List of explainer info dictionaries
    """
    return [
        {
            "name": name,
            "description": info["description"],
            "supported_models": info["supported_models"],
        }
        for name, info in EXPLAINER_TYPES.items()
    ]


def auto_detect_explainer_type(model: Any) -> str:
    """Auto-detect the best explainer type for a model.

    Args:
        model: The model to explain

    Returns:
        Explainer type name ("tree", "linear", "kernel")
    """
    model_type = type(model).__name__.lower()

    # Check for tree-based models
    tree_indicators = ["forest", "tree", "boost", "catboost", "xgb", "lgb", "isolation"]
    if any(ind in model_type for ind in tree_indicators):
        return "tree"

    # Check for linear models
    linear_indicators = ["linear", "logistic", "ridge", "lasso", "sgd"]
    if any(ind in model_type for ind in linear_indicators):
        return "linear"

    # Default to kernel (model-agnostic)
    return "kernel"


class SHAPExplainer:
    """SHAP-based model explainer.

    Provides SHAP value computation and natural language narrative
    generation for ML model predictions.
    """

    def __init__(
        self,
        model: Any,
        explainer_type: str | None = None,
        feature_names: list[str] | None = None,
        background_data: np.ndarray | None = None,
    ):
        """Initialize SHAP explainer.

        Args:
            model: The model to explain
            explainer_type: Type of explainer ("tree", "linear", "kernel", or None for auto-detect)
            feature_names: Optional feature names
            background_data: Background data for Kernel SHAP (required for kernel type)
        """
        if not _HAS_SHAP:
            raise ImportError("shap is required for SHAP explanations: pip install shap")
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self._explainer = None
        self._base_value = None

        # Auto-detect explainer type if not specified
        if explainer_type is None:
            self.explainer_type = auto_detect_explainer_type(model)
            logger.info(f"Auto-detected explainer type: {self.explainer_type}")
        else:
            if explainer_type not in EXPLAINER_TYPES:
                raise ValueError(
                    f"Unknown explainer type: {explainer_type}. "
                    f"Valid types: {list(EXPLAINER_TYPES.keys())}"
                )
            self.explainer_type = explainer_type

    async def load(self) -> None:
        """Initialize the SHAP explainer."""
        start_time = time.time()

        try:
            if self.explainer_type == "tree":
                self._explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == "linear":
                if self.background_data is None:
                    raise ValueError("Background data required for linear explainer")
                self._explainer = shap.LinearExplainer(self.model, self.background_data)
            elif self.explainer_type == "kernel":
                if self.background_data is None:
                    raise ValueError("Background data required for kernel explainer")
                # For kernel, we need a prediction function
                if hasattr(self.model, "predict"):
                    predict_fn = self.model.predict
                elif hasattr(self.model, "decision_function"):
                    predict_fn = self.model.decision_function
                else:
                    raise ValueError("Model must have predict or decision_function method")
                # Use a sample of background data for efficiency
                bg_sample = shap.sample(self.background_data, min(100, len(self.background_data)))
                self._explainer = shap.KernelExplainer(predict_fn, bg_sample)

            load_time_ms = (time.time() - start_time) * 1000
            logger.info(f"SHAP {self.explainer_type} explainer loaded in {load_time_ms:.2f}ms")

        except Exception as e:
            logger.error(f"Failed to load SHAP explainer: {e}")
            raise

    async def explain(
        self,
        data: np.ndarray | list[list[float]],
        top_k: int = 5,
    ) -> list[SHAPExplanation]:
        """Compute SHAP values for given data.

        Args:
            data: Data to explain (n_samples x n_features)
            top_k: Number of top features to include in explanation

        Returns:
            List of SHAP explanations, one per sample
        """
        if self._explainer is None:
            raise RuntimeError("Explainer not loaded. Call load() first.")

        # Convert to numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        # Ensure 2D
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        try:
            # Compute SHAP values
            shap_values = self._explainer.shap_values(data)

            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For multi-class, use the first class or sum
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values

            # Get base value
            base_value = self._explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0]
            base_value = float(base_value)

            # Build explanations
            explanations = []
            n_features = data.shape[1]
            feature_names = self.feature_names or [f"feature_{i}" for i in range(n_features)]

            for i in range(len(data)):
                sample_shap = shap_values[i] if len(shap_values.shape) > 1 else shap_values
                sample_data = data[i]

                # Sort by absolute SHAP value
                abs_shap = np.abs(sample_shap)
                top_indices = np.argsort(abs_shap)[::-1][:top_k]

                contributions = []
                for idx in top_indices:
                    shap_val = float(sample_shap[idx])
                    contributions.append(
                        FeatureContribution(
                            feature=feature_names[idx],
                            value=float(sample_data[idx]),
                            shap_value=shap_val,
                            direction="increases" if shap_val > 0 else "decreases",
                        )
                    )

                # Compute prediction from SHAP values
                prediction = base_value + float(np.sum(sample_shap))

                explanations.append(
                    SHAPExplanation(
                        sample_index=i,
                        base_value=base_value,
                        prediction=prediction,
                        contributions=contributions,
                    )
                )

            return explanations

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            raise

    async def get_feature_importance(
        self,
        data: np.ndarray | list[list[float]],
    ) -> list[tuple[str, float]]:
        """Get global feature importance from SHAP values.

        Args:
            data: Data to compute importance over

        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        if self._explainer is None:
            raise RuntimeError("Explainer not loaded. Call load() first.")

        # Convert to numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        # Compute SHAP values
        shap_values = self._explainer.shap_values(data)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Compute mean absolute SHAP value per feature
        importance = np.mean(np.abs(shap_values), axis=0)

        n_features = data.shape[1]
        feature_names = self.feature_names or [f"feature_{i}" for i in range(n_features)]

        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        return [(feature_names[i], float(importance[i])) for i in sorted_indices]

    async def generate_narrative(
        self,
        explanation: SHAPExplanation,
        context: dict[str, Any] | None = None,
    ) -> NarrativeExplanation:
        """Generate natural language narrative from SHAP explanation.

        Args:
            explanation: SHAP explanation to narrate
            context: Optional context (feature statistics, thresholds, etc.)

        Returns:
            Natural language explanation
        """
        contributions = explanation.contributions

        # Sort by absolute contribution
        sorted_contribs = sorted(contributions, key=lambda x: abs(x.shap_value), reverse=True)

        # Generate summary
        top_feature = sorted_contribs[0] if sorted_contribs else None
        if top_feature:
            if abs(explanation.prediction - explanation.base_value) > 0.1:
                summary = (
                    f"The prediction is significantly {'higher' if explanation.prediction > explanation.base_value else 'lower'} "
                    f"than average, primarily due to {top_feature.feature}."
                )
            else:
                summary = "The prediction is close to the average, with balanced feature contributions."
        else:
            summary = "No significant feature contributions found."

        # Generate per-feature details
        details = []
        for contrib in sorted_contribs:
            magnitude = abs(contrib.shap_value)
            if magnitude < 0.01:
                continue

            # Describe the contribution
            if magnitude > 0.3:
                strength = "strongly"
            elif magnitude > 0.1:
                strength = "moderately"
            else:
                strength = "slightly"

            # Add context if available
            context_info = ""
            if context and contrib.feature in context:
                feat_context = context[contrib.feature]
                if "mean" in feat_context and feat_context["mean"] != 0:
                    ratio = contrib.value / feat_context["mean"]
                    if ratio > 2:
                        context_info = f" ({ratio:.1f}x higher than average)"
                    elif ratio > 0 and ratio < 0.5:
                        context_info = f" ({1/ratio:.1f}x lower than average)"

            detail = (
                f"{contrib.feature} (value={contrib.value:.2f}{context_info}) "
                f"{strength} {contrib.direction} the prediction (contribution: {contrib.shap_value:+.3f})"
            )
            details.append(detail)

        return NarrativeExplanation(summary=summary, details=details)

    async def unload(self) -> None:
        """Free explainer resources."""
        self._explainer = None
        logger.info("SHAP explainer unloaded")
