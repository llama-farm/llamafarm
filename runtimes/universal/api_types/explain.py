"""Pydantic models for SHAP explainability endpoints.

SHAP (SHapley Additive exPlanations) provides interpretable explanations
for ML model predictions by computing feature contributions.

Use Cases:
- Debugging ML predictions
- Regulatory compliance
- Building user trust
- Generating LLM-ready explanations
"""


from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """Contribution of a single feature to the prediction."""

    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Actual feature value")
    shap_value: float = Field(..., description="SHAP contribution to prediction")
    direction: str = Field(..., description="'increases' or 'decreases' the prediction")


class SHAPExplanation(BaseModel):
    """SHAP explanation for a single sample."""

    sample_index: int = Field(..., description="Index of the sample in the input")
    base_value: float = Field(..., description="Expected value (average prediction)")
    prediction: float = Field(..., description="Actual prediction value")
    contributions: list[FeatureContribution] = Field(
        ..., description="Top feature contributions sorted by importance"
    )


class NarrativeExplanation(BaseModel):
    """Natural language explanation."""

    summary: str = Field(..., description="One sentence summary of the prediction")
    details: list[str] = Field(
        default_factory=list, description="Per-feature explanations in natural language"
    )


class SHAPExplainRequest(BaseModel):
    """Request to generate SHAP explanation."""

    model_type: str = Field(
        ...,
        description="Type of model to explain (anomaly, classifier, etc.)",
    )
    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    data: list[list[float]] = Field(
        ...,
        description="Data points to explain (n_samples x n_features)",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Optional feature names",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top features to include in explanation",
    )
    generate_narrative: bool = Field(
        default=True,
        description="Whether to generate natural language narrative",
    )
    background_samples: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of background samples for Kernel SHAP",
    )


class SHAPExplainResponse(BaseModel):
    """Response from SHAP explanation."""

    model_type: str = Field(..., description="Type of model explained")
    model_id: str = Field(..., description="Model identifier")
    explainer_type: str = Field(..., description="SHAP explainer type used (tree, linear, kernel)")
    explanations: list[SHAPExplanation] = Field(
        ..., description="SHAP explanations for each sample"
    )
    narrative: NarrativeExplanation | None = Field(
        default=None,
        description="Natural language narrative (if requested)",
    )
    explain_time_ms: float = Field(..., description="Time to generate explanations")


class FeatureImportance(BaseModel):
    """Global feature importance from SHAP values."""

    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Mean absolute SHAP value")


class FeatureImportanceRequest(BaseModel):
    """Request to get feature importance."""

    model_type: str = Field(
        ...,
        description="Type of model (anomaly, classifier, etc.)",
    )
    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    data: list[list[float]] = Field(
        ...,
        description="Data to compute importance over",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Optional feature names",
    )


class FeatureImportanceResponse(BaseModel):
    """Response with global feature importance."""

    model_type: str = Field(..., description="Type of model")
    model_id: str = Field(..., description="Model identifier")
    importances: list[FeatureImportance] = Field(
        ..., description="Feature importances sorted by importance"
    )
    compute_time_ms: float = Field(..., description="Time to compute importance")


class ExplainerInfo(BaseModel):
    """Information about an explainer type."""

    name: str = Field(..., description="Explainer type name")
    description: str = Field(..., description="Human-readable description")
    supported_models: list[str] = Field(..., description="Model types this explainer supports")


class ExplainersResponse(BaseModel):
    """Response listing available explainers."""

    explainers: list[ExplainerInfo] = Field(..., description="Available explainer types")


# Context for narrative generation
class FeatureContext(BaseModel):
    """Context information for a feature (for narrative generation)."""

    mean: float | None = Field(default=None, description="Feature mean value")
    std: float | None = Field(default=None, description="Feature standard deviation")
    min: float | None = Field(default=None, description="Feature minimum value")
    max: float | None = Field(default=None, description="Feature maximum value")
    description: str | None = Field(default=None, description="Human-readable feature description")


class NarrativeContext(BaseModel):
    """Context for generating rich narratives."""

    features: dict[str, FeatureContext] = Field(
        default_factory=dict,
        description="Context for each feature",
    )
    prediction_context: str | None = Field(
        default=None,
        description="Context about what the prediction means (e.g., 'anomaly score', 'probability')",
    )
