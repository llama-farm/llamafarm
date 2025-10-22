from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from config.datamodel import Model, Provider

from core.settings import settings
from services import runtime_service
from services.model_service import ModelService


def _now_ms() -> int:
    return int(time.time() * 1000)


def _check_server() -> dict:
    start = _now_ms()
    try:
        # If this code runs, server is up; do a cheap no-op
        return {
            "name": "server",
            "status": "healthy",
            "message": "FastAPI process responding",
            "latency_ms": _now_ms() - start,
        }
    except Exception as e:  # pragma: no cover - defensive
        return {
            "name": "server",
            "status": "unhealthy",
            "message": f"Server internal error: {e}",
            "latency_ms": _now_ms() - start,
        }


def _check_storage() -> dict:
    start = _now_ms()
    try:
        data_dir = Path(settings.lf_data_dir)
        projects_dir = data_dir / "projects"
        data_dir.mkdir(parents=True, exist_ok=True)
        projects_dir.mkdir(parents=True, exist_ok=True)

        # Writability check
        test_file = projects_dir / ".health_write_test"
        try:
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
        except Exception as e:
            return {
                "name": "storage",
                "status": "degraded",
                "message": f"Projects directory not writable: {e}",
                "latency_ms": _now_ms() - start,
            }

        return {
            "name": "storage",
            "status": "healthy",
            "message": f"{projects_dir} exists and writable",
            "latency_ms": _now_ms() - start,
        }
    except Exception as e:
        return {
            "name": "storage",
            "status": "unhealthy",
            "message": f"Storage check failed: {e}",
            "latency_ms": _now_ms() - start,
        }


def _check_ollama() -> dict:
    """Check Ollama runtime health using provider registry."""
    # Create minimal config with one model for health check
    model_config_dict = {
        "name": "ollama-health",
        "provider": Provider.ollama,
        "model": "health-check",
    }
    model_config = Model.model_validate(model_config_dict)

    provider = runtime_service.get_provider(Provider.ollama, model_config)
    result = provider.check_health()
    return result.to_dict()


def _check_seed_project() -> dict:
    """Validate the project seed runtime is reachable and model is present using provider registry."""
    start = _now_ms()

    # Load seed project config for ModelService
    try:
        from config.helpers.loader import load_config

        seed_path = (
            Path(__file__).resolve().parents[1]
            / "seeds"
            / "project_seed"
            / "llamafarm.yaml"
        )

        if not seed_path.exists():
            return {
                "name": "project",
                "status": "unhealthy",
                "message": f"Seed file not found at {seed_path}",
                "latency_ms": _now_ms() - start,
                "runtime": {"provider": None, "model": None},
            }

        # Use the proper config loader that includes URL rewriting for Docker
        project_config = load_config(config_path=seed_path, validate=False)

        # Use ModelService to get the correct model config
        model_config = ModelService.get_model(project_config, model_name=None)

        # Get provider and check health
        provider_impl = runtime_service.get_provider(
            model_config.provider, model_config
        )
        health_result = provider_impl.check_health()

        # Enhance with model validation for Ollama
        if (
            model_config.provider == Provider.ollama
            and health_result.status == "healthy"
        ):
            models = health_result.details.get("models", [])
            present = model_config.model in models
            status = "healthy" if present else "unhealthy"
            message = (
                "Model available"
                if present
                else f"Model '{model_config.model}' not found; run: ollama pull {model_config.model}"
            )

            return {
                "name": "project",
                "status": status,
                "message": message,
                "latency_ms": _now_ms() - start,
                "runtime": {
                    "provider": model_config.provider.value,
                    "host": health_result.details.get("host"),
                    "model": model_config.model,
                },
            }

        # For other providers, use health check result directly
        return {
            "name": "seed:project",
            "status": health_result.status,
            "message": health_result.message,
            "latency_ms": _now_ms() - start,
            "runtime": {
                "provider": model_config.provider.value,
                "model": model_config.model,
                **health_result.details,
            },
        }

    except Exception as e:
        return {
            "name": "project",
            "status": "unhealthy",
            "message": f"Failed to check provider health: {e}",
            "latency_ms": _now_ms() - start,
            "runtime": {"provider": None, "model": None},
        }


def _check_rag_service() -> dict:
    """Check RAG service health using cached status with background updates."""
    start = _now_ms()
    try:
        from services.rag_health_cache import get_rag_health_cache

        # Get cached health status (non-blocking)
        cache = get_rag_health_cache()
        health_data = cache.get_cached_health()

        # Convert to health service format
        status = health_data.get("status", "unhealthy")
        message = health_data.get("message", "Unknown RAG status")

        # Add cache metadata to message if available
        cache_age = health_data.get("cache_age_seconds", -1)
        source = health_data.get("source", "unknown")

        if source == "cache" and cache_age >= 0:
            if cache_age < 60:
                message += f" (checked: {cache_age}s ago)"
            else:
                message += f" (checked: {cache_age // 60}m ago)"
        elif source == "immediate_check":
            message += " (immediate check)"

        # Add detailed information if available
        details = {}
        if "worker_id" in health_data:
            details["worker_id"] = health_data["worker_id"]
        if "checks" in health_data:
            details["checks"] = health_data["checks"]
        if "metrics" in health_data:
            details["metrics"] = health_data["metrics"]

        result = {
            "name": "rag-service",
            "status": status,
            "message": message,
            "latency_ms": _now_ms() - start,
        }

        if details:
            result["details"] = details

        return result

    except Exception as e:
        return {
            "name": "rag-service",
            "status": "unhealthy",
            "message": f"RAG health cache error: {e}",
            "latency_ms": _now_ms() - start,
        }


def compute_overall_status(components: list[dict], seeds: list[dict]) -> str:
    order = {"healthy": 0, "degraded": 1, "unhealthy": 2}
    worst = 0

    # Only consider non-RAG components for overall status
    # RAG service status is included in response but doesn't affect overall health
    for c in components + seeds:
        # Skip RAG service and Project seed when computing overall status
        if c.get("name", "") in ["rag-service", "seed:project"]:
            continue
        worst = max(worst, order.get(c.get("status", "unhealthy"), 2))

    return next((k for k, v in order.items() if v == worst), "unhealthy")


def health_summary() -> dict[str, Any]:
    """Compute health summary. Keep checks quick; small timeouts only."""
    components: list[dict] = []
    seeds: list[dict] = []

    components.append(_check_server())
    components.append(_check_storage())
    components.append(_check_ollama())
    components.append(_check_rag_service())

    seeds.append(_check_seed_project())

    status = compute_overall_status(components, seeds)
    summary_parts = [f"{c['name']}={c['status']}" for c in components + seeds]
    return {
        "status": status,
        "summary": ", ".join(summary_parts),
        "components": components,
        "seeds": seeds,
        "timestamp": int(time.time()),
    }
