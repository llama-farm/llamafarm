"""Model configuration service for multi-model support.

This service handles model resolution and provides utilities
for working with multi-model configurations.
"""

import sys
from pathlib import Path
from typing import Any

# Add repo root to path for config imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import LlamaFarmConfig, Provider, PromptFormat  # noqa: E402
from core.logging import FastAPIStructLogger  # noqa: E402

logger = FastAPIStructLogger(__name__)


class ModelConfig:
    """Typed wrapper for model configuration from dict."""

    def __init__(self, data: dict[str, Any]):
        self.name: str = data["name"]
        self.description: str | None = data.get("description")
        # Provider might already be a Provider enum from Pydantic
        provider_val = data["provider"]
        self.provider: Provider = provider_val if isinstance(provider_val, Provider) else Provider(provider_val)
        self.model: str = data["model"]
        self.base_url: str | None = data.get("base_url")
        self.api_key: str | None = data.get("api_key")
        self.instructor_mode: str | None = data.get("instructor_mode")
        # PromptFormat might already be an enum from Pydantic
        prompt_fmt = data.get("prompt_format")
        self.prompt_format: PromptFormat | None = (
            prompt_fmt if isinstance(prompt_fmt, PromptFormat)
            else (PromptFormat(prompt_fmt) if prompt_fmt else None)
        )
        self.model_api_parameters: dict[str, Any] | None = data.get("model_api_parameters")
        self.provider_config: dict[str, Any] | None = data.get("provider_config")


class ModelService:
    """Service for resolving and managing model configurations."""

    @staticmethod
    def get_model_config(
        project_config: LlamaFarmConfig, model_name: str | None = None
    ) -> ModelConfig:
        """Get model configuration by name, falling back to default.

        Args:
            project_config: Project configuration
            model_name: Optional model name to select (from API request)

        Returns:
            ModelConfig for the selected model

        Raises:
            ValueError: If model_name doesn't exist or no default configured
        """
        if not project_config.runtime.models:
            raise ValueError("No models configured in runtime")

        # If no model name provided, use default_model or first model
        if not model_name:
            model_name = project_config.runtime.default_model
            if not model_name:
                # No default_model set, use first model
                if project_config.runtime.models:
                    model_data = project_config.runtime.models[0]  # type: ignore
                    logger.debug(
                        "No default_model set, using first model",
                        model_name=model_data.name,  # type: ignore
                    )
                    return ModelConfig(model_data.model_dump())  # type: ignore
                raise ValueError("No models configured")

        # Find model by name in list
        model_data = None
        for model in project_config.runtime.models:  # type: ignore
            if model.name == model_name:  # type: ignore
                model_data = model
                break

        if not model_data:
            available = ", ".join([m.name for m in project_config.runtime.models])  # type: ignore
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")

        logger.debug("Resolved model configuration", model_name=model_name)
        return ModelConfig(model_data.model_dump())  # type: ignore

    @staticmethod
    def list_models(project_config: LlamaFarmConfig) -> list[dict]:
        """List all available models with metadata.

        Args:
            project_config: Project configuration

        Returns:
            List of model metadata dicts with id, description, provider, model, is_default
        """
        if not project_config.runtime.models:
            return []

        default_model_name = project_config.runtime.default_model

        models = []
        for model_config in project_config.runtime.models:  # type: ignore
            models.append(
                {
                    "id": model_config.name,  # type: ignore
                    "description": model_config.description or "",  # type: ignore
                    "provider": model_config.provider.value if model_config.provider else "",  # type: ignore
                    "model": model_config.model,  # type: ignore
                    "is_default": model_config.name == default_model_name,  # type: ignore
                }
            )

        return models
