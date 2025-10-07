"""Model configuration service for multi-model support.

This service handles model resolution and provides utilities
for working with multi-model configurations.
"""

import sys
from pathlib import Path

from pydantic import BaseModel, Field

from core.logging import FastAPIStructLogger  # noqa: E402

# Add repo root to path for config imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import LlamaFarmConfig, Model  # noqa: E402

logger = FastAPIStructLogger(__name__)


class ModelService:
    """Service for resolving and managing model configurations."""

    @staticmethod
    def get_model(
        project_config: LlamaFarmConfig, model_name: str | None = None
    ) -> Model:
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

        model_config: Model | None = None

        # If no model name provided, use default_model or first model
        if not model_name:
            model_name = project_config.runtime.default_model
            if not model_name:
                # No default_model set, use first model
                if project_config.runtime.models:
                    model_config = project_config.runtime.models[0]
                    logger.debug(
                        "No default_model set, using first model",
                        model_name=model_config.name,
                    )
                    return model_config
                raise ValueError("No models configured")

        # Find model by name in list
        for model in project_config.runtime.models:
            if model.name == model_name:
                model_config = model
                break

        if not model_config:
            available = ", ".join([m.name for m in project_config.runtime.models])  # type: ignore
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")

        logger.debug("Resolved model configuration", model_name=model_name)
        return model_config

    @staticmethod
    def list_models(project_config: LlamaFarmConfig) -> list[Model]:
        """List all available models with metadata.

        Args:
            project_config: Project configuration

        Returns:
            List of model metadata dicts with id, description, provider, model, is_default
        """
        return project_config.runtime.models or []
