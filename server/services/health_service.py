from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests  # type: ignore

from core.settings import settings


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


def _load_seed_runtime_model() -> tuple[str | None, str]:
    """Return (model, message). Reads the project seed YAML and extracts runtime.model.

    If missing or invalid, returns (None, reason).
    """
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - environment specific
        return None, "PyYAML not installed"

    seed_path = (
        Path(__file__).resolve().parents[1]
        / "seeds"
        / "project_seed"
        / "llamafarm.yaml"
    )
    if not seed_path.exists():
        return None, f"Seed file not found at {seed_path}"

    try:
        data = yaml.safe_load(seed_path.read_text(encoding="utf-8")) or {}
        runtime = (data or {}).get("runtime") or {}
        model = (runtime or {}).get("model")
        if not model or not isinstance(model, str):
            return None, "runtime.model missing in seed"
        return model, "ok"
    except Exception as e:
        return None, f"Failed to parse seed YAML: {e}"


def _check_ollama() -> dict:
    start = _now_ms()
    base = settings.ollama_host.rstrip("/")
    url = f"{base}/api/tags"
    try:
        resp = requests.get(url, timeout=1.0)
        if 200 <= resp.status_code < 300:
            return {
                "name": "ollama",
                "status": "healthy",
                "message": f"{base} reachable",
                "latency_ms": _now_ms() - start,
                "details": {"host": base},
            }
        return {
            "name": "ollama",
            "status": "unhealthy",
            "message": f"{base} returned {resp.status_code}",
            "latency_ms": _now_ms() - start,
            "details": {"host": base},
        }
    except Exception as e:
        return {
            "name": "ollama",
            "status": "unhealthy",
            "message": f"{base} unreachable: {e}",
            "latency_ms": _now_ms() - start,
            "details": {"host": base},
        }


def _check_seed_project() -> dict:
    """Validate the project seed runtime: Ollama is reachable and model is present."""
    start = _now_ms()
    model, reason = _load_seed_runtime_model()
    base = settings.ollama_host.rstrip("/")
    url = f"{base}/api/tags"

    if model is None:
        return {
            "name": "project",
            "status": "unhealthy",
            "message": reason,
            "latency_ms": _now_ms() - start,
            "runtime": {"host": base, "model": None},
        }

    # check tags for model
    try:
        resp = requests.get(url, timeout=1.5)
        ok = 200 <= resp.status_code < 300
        if not ok:
            return {
                "name": "project",
                "status": "unhealthy",
                "message": f"Ollama tags returned {resp.status_code}",
                "latency_ms": _now_ms() - start,
                "runtime": {"host": base, "model": model},
            }
        data = resp.json() if resp.content else {}
        tags = {item.get("name") for item in (data.get("models") or [])}
        present = model in tags
        status = "healthy" if present else "unhealthy"
        message = (
            "Model available"
            if present
            else f"Model '{model}' not found; run: ollama pull {model}"
        )
        return {
            "name": "project",
            "status": status,
            "message": message,
            "latency_ms": _now_ms() - start,
            "runtime": {"host": base, "model": model},
        }
    except Exception as e:
        return {
            "name": "project",
            "status": "unhealthy",
            "message": f"Failed to query Ollama tags: {e}",
            "latency_ms": _now_ms() - start,
            "runtime": {"host": base, "model": model},
        }


def _check_celery() -> dict:
    start = _now_ms()
    try:
        from core.celery.celery import app as celery_app  # type: ignore

        try:
            if replies := celery_app.control.ping(timeout=0.5):
                return {
                    "name": "celery",
                    "status": "healthy",
                    "message": f"{len(replies)} worker(s) responding",
                    "latency_ms": _now_ms() - start,
                }
            return {
                "name": "celery",
                "status": "degraded",
                "message": "No workers replied to ping",
                "latency_ms": _now_ms() - start,
            }
        except Exception as e:
            return {
                "name": "celery",
                "status": "degraded",
                "message": f"Celery ping failed: {e}",
                "latency_ms": _now_ms() - start,
            }
    except Exception:
        # Celery not configured/importable
        return {
            "name": "celery",
            "status": "healthy",
            "message": "Celery not configured",
            "latency_ms": _now_ms() - start,
        }


def compute_overall_status(components: list[dict], seeds: list[dict]) -> str:
    order = {"healthy": 0, "degraded": 1, "unhealthy": 2}
    worst = 0
    for c in components + seeds:
        worst = max(worst, order.get(c.get("status", "unhealthy"), 2))
    return next((k for k, v in order.items() if v == worst), "unhealthy")


def health_summary() -> dict[str, Any]:
    """Compute health summary. Keep checks quick; small timeouts only."""
    components: list[dict] = []
    seeds: list[dict] = []

    components.append(_check_server())
    components.append(_check_storage())
    components.append(_check_ollama())
    components.append(_check_celery())

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
