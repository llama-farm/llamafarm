"""Model configuration service for multi-model support.

This service handles model resolution and provides utilities
for working with multi-model configurations.
"""

import time
from collections.abc import AsyncIterator
from threading import Lock

from config.datamodel import LlamaFarmConfig, Model, Provider  # noqa: E402
from server.services.runtime_service.runtime_service import runtime_service
from server.services.runtime_service.providers.base import (
    CachedModel,
    format_duration,
)
from server.services.runtime_service.providers.universal_provider import (
    UniversalProvider,  # noqa: E402
)

from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger(__name__)


# Tracks monotonic timestamps of when each (model_alias, provider, host) was
# first observed as loaded/running, so we can report uptime for providers that
# don't expose the value themselves (Ollama, Universal, Lemonade). The tracker
# resets when the server restarts or when a model is no longer reported active,
# which is the intended behavior and is documented in lf-models.md.
_uptime_tracker: dict[tuple[str, str, str | None], float] = {}
_uptime_lock = Lock()


def _record_uptime(key: tuple[str, str, str | None], active: bool) -> int | None:
    """Return uptime seconds for `key` if active, clearing it otherwise."""
    now = time.monotonic()
    with _uptime_lock:
        if active:
            first_seen = _uptime_tracker.setdefault(key, now)
            return int(now - first_seen)
        _uptime_tracker.pop(key, None)
        return None


def _reset_uptime_tracker() -> None:
    """Test helper: clear the module-level uptime tracker."""
    with _uptime_lock:
        _uptime_tracker.clear()


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
            List of model metadata dicts with id, description, provider,
            model, is_default
        """
        return project_config.runtime.models or []

    @staticmethod
    def list_model_runtime_statuses(project_config: LlamaFarmConfig) -> dict[str, dict]:
        """Collect runtime status for each configured model.

        Returns a dict keyed by the configured model alias so API callers can
        merge the status fields into their existing model payload.
        """
        statuses: dict[str, dict] = {}
        for model in project_config.runtime.models or []:
            provider_name = getattr(model.provider, "value", str(model.provider))
            try:
                provider = runtime_service.get_provider(model)
                status = provider.get_model_runtime_status()
            except Exception as e:  # pragma: no cover - defensive fallback
                logger.warning(
                    "Failed to collect model runtime status",
                    model_name=model.name,
                    provider=provider_name,
                    error=str(e),
                )
                _record_uptime((model.name, provider_name, None), active=False)
                statuses[model.name] = {
                    "runtime_status": "unknown",
                    "runtime_loaded": False,
                    "runtime_running": False,
                    "runtime_message": f"Failed to inspect runtime state: {e}",
                }
                continue

            active = bool(status.loaded or status.running)
            tracked_seconds = _record_uptime(
                (model.name, provider_name, status.host), active=active
            )
            if active and status.uptime_seconds is None and tracked_seconds is not None:
                status.uptime_seconds = tracked_seconds
                if status.uptime_human is None:
                    status.uptime_human = format_duration(tracked_seconds)

            statuses[model.name] = status.to_dict()
        return statuses

    @staticmethod
    def list_cached_models(
        provider: Provider = Provider.universal,
    ) -> list[CachedModel]:
        """List all cached models with metadata.

        Args:
            project_config: Project configuration

        Returns:
            List of cached model metadata dicts with id, name, size, path
        """

        match provider:
            case Provider.universal:
                return UniversalProvider.list_cached_models()
            case _:
                raise ValueError(f"Unsupported provider: {provider.value}")

    @staticmethod
    async def download_model(
        provider: Provider, model_name: str
    ) -> AsyncIterator[dict]:
        """Download/cache a model for the given provider and model name."""
        match provider:
            case Provider.universal:
                async for evt in UniversalProvider.download_model(model_name):
                    yield evt
            case _:
                raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def delete_model(provider: Provider, model_name: str) -> dict:
        """Delete a cached model for the given provider and model name.

        Args:
            provider: The model provider
            model_name: The model identifier to delete

        Returns:
            Dict with deleted model info including freed space

        Raises:
            ValueError: If provider is not supported
        """
        match provider:
            case Provider.universal:
                return UniversalProvider.delete_model(model_name)
            case _:
                raise ValueError(f"Unsupported provider: {provider}")
