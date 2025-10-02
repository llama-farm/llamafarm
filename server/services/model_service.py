"""Model configuration service for multi-model support.

This service handles model resolution, normalization of legacy configs,
and provides utilities for working with multi-model configurations.
"""

import sys
from pathlib import Path
from typing import Any

# Add repo root to path for config imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import LlamaFarmConfig, Provider, PromptFormat, Lemonade  # noqa: E402
from core.logging import FastAPIStructLogger  # noqa: E402

logger = FastAPIStructLogger(__name__)


class ModelConfig:
    """Typed wrapper for model configuration from dict."""

    def __init__(self, data: dict[str, Any]):
        self.description: str | None = data.get("description")
        self.provider: Provider = Provider(data["provider"])
        self.model: str = data["model"]
        self.base_url: str | None = data.get("base_url")
        self.api_key: str | None = data.get("api_key")
        self.huggingface_token: str | None = data.get("huggingface_token")
        self.instructor_mode: str | None = data.get("instructor_mode")
        self.prompt_format: PromptFormat | None = (
            PromptFormat(data["prompt_format"]) if data.get("prompt_format") else None
        )
        self.model_api_parameters: dict[str, Any] | None = data.get("model_api_parameters")
        self.lemonade: Lemonade | None = (
            Lemonade(**data["lemonade"]) if data.get("lemonade") else None
        )


class ModelService:
    """Service for resolving and managing model configurations."""

    @staticmethod
    def normalize_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
        """Normalize config dict to support both legacy and multi-model formats.

        Converts legacy single-model configs to multi-model format internally.
        Works on raw dicts before Pydantic validation.

        Args:
            config_dict: Original configuration dictionary

        Returns:
            Normalized configuration dict with models dict populated
        """
        runtime = config_dict.get("runtime", {})

        # Already has multi-model config
        if runtime.get("models") and len(runtime["models"]) > 0:
            # Ensure default_model is set
            if not runtime.get("default_model"):
                runtime["default_model"] = list(runtime["models"].keys())[0]
                logger.debug(
                    "Auto-set default_model to first model",
                    default_model=runtime["default_model"],
                )
            config_dict["runtime"] = runtime
            return config_dict

        # Legacy format: convert to multi-model
        if runtime.get("provider") and runtime.get("model"):
            logger.info(
                "Converting legacy single-model config to multi-model format",
                provider=runtime["provider"],
                model=runtime["model"],
            )

            # Create a "default" model from legacy config
            default_model = {
                "provider": runtime["provider"],
                "model": runtime["model"],
            }

            # Copy all optional legacy fields
            for field in [
                "base_url",
                "api_key",
                "huggingface_token",
                "instructor_mode",
                "prompt_format",
                "model_api_parameters",
                "lemonade",
            ]:
                if field in runtime and runtime[field] is not None:
                    default_model[field] = runtime[field]

            runtime["models"] = {"default": default_model}
            runtime["default_model"] = "default"
            config_dict["runtime"] = runtime

        return config_dict

    @staticmethod
    def normalize_config(config: LlamaFarmConfig) -> LlamaFarmConfig:
        """Normalize config to support both legacy and multi-model formats.

        Converts legacy single-model configs to multi-model format internally.

        Args:
            config: Original configuration

        Returns:
            Normalized configuration with models dict populated
        """
        # Already has multi-model config
        if config.runtime.models and len(config.runtime.models) > 0:
            # Ensure default_model is set
            if not config.runtime.default_model:
                config.runtime.default_model = list(config.runtime.models.keys())[0]
                logger.debug(
                    "Auto-set default_model to first model",
                    default_model=config.runtime.default_model,
                )
            return config

        # Legacy format: convert to multi-model
        if config.runtime.provider and config.runtime.model:
            logger.info(
                "Converting legacy single-model config to multi-model format",
                provider=config.runtime.provider.value,
                model=config.runtime.model,
            )

            # Create a "default" model from legacy config
            default_model = {
                "provider": config.runtime.provider.value,
                "model": config.runtime.model,
            }

            # Copy all optional legacy fields
            if config.runtime.base_url:
                default_model["base_url"] = config.runtime.base_url
            if config.runtime.api_key:
                default_model["api_key"] = config.runtime.api_key
            if config.runtime.huggingface_token:
                default_model["huggingface_token"] = config.runtime.huggingface_token
            if config.runtime.instructor_mode:
                default_model["instructor_mode"] = config.runtime.instructor_mode
            if config.runtime.prompt_format:
                default_model["prompt_format"] = config.runtime.prompt_format.value
            if config.runtime.model_api_parameters:
                default_model["model_api_parameters"] = config.runtime.model_api_parameters
            if config.runtime.lemonade:
                default_model["lemonade"] = {
                    "backend": config.runtime.lemonade.backend.value
                    if config.runtime.lemonade.backend
                    else "onnx",
                    "port": config.runtime.lemonade.port,
                    "context_size": config.runtime.lemonade.context_size,
                    "model_path": config.runtime.lemonade.model_path,
                }

            config.runtime.models = {"default": default_model}  # type: ignore
            config.runtime.default_model = "default"

        return config

    @staticmethod
    def get_model_config(
        project_config: LlamaFarmConfig, model_name: str | None = None
    ) -> ModelConfig:
        """Get model configuration by name, falling back to default.

        Args:
            project_config: Project configuration (should be normalized first)
            model_name: Optional model name to select (from API request)

        Returns:
            ModelConfig for the selected model

        Raises:
            ValueError: If model_name doesn't exist or no default configured
        """
        # Use requested model or fall back to default
        selected_model = model_name or project_config.runtime.default_model

        if not selected_model:
            raise ValueError("No model specified and no default_model configured")

        if not project_config.runtime.models:
            raise ValueError("No models configured in runtime")

        if selected_model not in project_config.runtime.models:
            available = ", ".join(project_config.runtime.models.keys())
            raise ValueError(f"Model '{selected_model}' not found. Available: {available}")

        model_data = project_config.runtime.models[selected_model]
        logger.debug("Resolved model configuration", model_name=selected_model)

        return ModelConfig(model_data)  # type: ignore

    @staticmethod
    def list_models(project_config: LlamaFarmConfig) -> list[dict]:
        """List all available models with metadata.

        Args:
            project_config: Project configuration (should be normalized first)

        Returns:
            List of model metadata dicts with id, description, provider, model, is_default
        """
        if not project_config.runtime.models:
            return []

        models = []
        for name, config_data in project_config.runtime.models.items():
            models.append(
                {
                    "id": name,
                    "description": config_data.get("description", ""),  # type: ignore
                    "provider": config_data.get("provider", ""),  # type: ignore
                    "model": config_data.get("model", ""),  # type: ignore
                    "is_default": name == project_config.runtime.default_model,
                }
            )

        return models
