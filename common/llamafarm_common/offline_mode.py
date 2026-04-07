"""
LlamaFarm offline-mode bootstrap.

This module is imported very early by `common.llamafarm_common.__init__` so
that any `huggingface_hub`, `transformers`, or `datasets` import that happens
later in the process sees a correctly-propagated `HF_HUB_OFFLINE` environment
variable.

The module does NOT import `huggingface_hub` or any heavy ML library. It only
reads + sets environment variables and uses `logging` (the stdlib) plus an
optional `structlog` lookup for richer output when available.

Environment variables read:
    LLAMAFARM_OFFLINE
        Truthy values (`1`, `true`, `yes`, `on` — case-insensitive) switch the
        runtime into strict offline mode. In strict offline mode:
          - `common.llamafarm_common.model_utils` raises immediately instead
            of calling the HuggingFace Hub API.
          - `llamafarm_llama._binary` raises instead of downloading from
            GitHub Releases.
          - `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are propagated so
            transitive calls through `huggingface_hub`/`transformers` also
            honor offline mode.

    LLAMAFARM_MODEL_DIR
        Filesystem path to a flat on-device model layout
        (`<root>/<alias>/<files>`). When set, the runtime inspects this
        location before falling back to the HuggingFace cache. See
        `common.llamafarm_common.model_dir` for the resolver implementation.

Environment variables written (when `LLAMAFARM_OFFLINE` is truthy):
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

logger = logging.getLogger(__name__)

# Truthy string values accepted for offline-mode env vars. Matches the set that
# huggingface_hub itself accepts for HF_HUB_OFFLINE (plus `yes` for friendliness).
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSY: frozenset[str] = frozenset({"0", "false", "no", "off", ""})


def _as_bool(value: str | None) -> bool:
    """Interpret an environment-variable string as a boolean.

    Returns True for truthy tokens (case-insensitive), False otherwise.
    `None` and unknown values return False.
    """
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def is_offline() -> bool:
    """Return True if strict offline mode is currently active.

    Consults the live environment so tests can toggle the var with
    `monkeypatch.setenv`/`delenv` between assertions.
    """
    return _as_bool(os.environ.get("LLAMAFARM_OFFLINE"))


def model_dir() -> str | None:
    """Return the configured `LLAMAFARM_MODEL_DIR`, or None if unset/empty."""
    value = os.environ.get("LLAMAFARM_MODEL_DIR")
    if value is None or value.strip() == "":
        return None
    return value


def propagate_hf_env() -> None:
    """Propagate `LLAMAFARM_OFFLINE` into `HF_HUB_OFFLINE` + `TRANSFORMERS_OFFLINE`.

    This MUST run before any import of `huggingface_hub`, `transformers`, or
    `datasets` for the propagation to have effect, because those libraries
    read their offline env vars once at module-load time.

    Idempotent: safe to call repeatedly. When `LLAMAFARM_OFFLINE` is set but
    the companion vars are explicitly set to falsy, we override them and log a
    warning so the operator can correct their deployment.
    """
    if not is_offline():
        return

    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        existing = os.environ.get(var)
        if existing is not None and existing.strip().lower() in _FALSY - {""}:
            logger.warning(
                "LLAMAFARM_OFFLINE=1 but %s=%r; overriding to 1 to enforce strict offline mode",
                var,
                existing,
            )
        if existing != "1":
            os.environ[var] = "1"


def log_startup_mode() -> None:
    """Emit a single info-level log line describing the resolved mode.

    Safe to call more than once: the actual log emission happens exactly once
    per process. Uses `structlog` if it is already imported into the process
    (we do NOT force an import to avoid pulling a heavy dep into `common`);
    otherwise falls back to stdlib `logging`.

    Idempotency is tracked via a function attribute rather than a module
    global so static analysis tools can follow the assignment.
    """
    if getattr(log_startup_mode, "_done", False):
        return
    log_startup_mode._done = True  # type: ignore[attr-defined]

    mode = "offline" if is_offline() else "online"
    md = model_dir()

    # Prefer structlog when it is already available, for structured fields.
    sl = _find_structlog_logger()
    if sl is not None:
        sl.info(
            "llamafarm_offline_mode",
            mode=mode,
            model_dir=md,
            hf_hub_offline=os.environ.get("HF_HUB_OFFLINE"),
            transformers_offline=os.environ.get("TRANSFORMERS_OFFLINE"),
        )
        return

    logger.info(
        "LlamaFarm runtime starting: mode=%s model_dir=%s hf_hub_offline=%s transformers_offline=%s",
        mode,
        md or "<unset>",
        os.environ.get("HF_HUB_OFFLINE") or "<unset>",
        os.environ.get("TRANSFORMERS_OFFLINE") or "<unset>",
    )


def _find_structlog_logger():
    """Return a structlog logger if structlog is importable, else None.

    Imported lazily to avoid a hard dependency on structlog from `common`.
    """
    try:
        import structlog  # type: ignore

        return structlog.get_logger(__name__)
    except Exception:
        return None


def reset_for_tests() -> None:
    """Reset the idempotent startup-log flag. Tests only."""
    log_startup_mode._done = False  # type: ignore[attr-defined]


def raise_offline_error(
    *,
    alias: str,
    tried_paths: Iterable[str],
    fix_command: str,
    extra: str | None = None,
) -> None:
    """Raise a structured `FileNotFoundError` for missing-model failures.

    The raised message is a multi-line string containing the model alias, the
    filesystem paths that were tried, and the specific `lf` CLI command that
    would make the missing file available. This format is specified in the
    runtime-offline capability spec.
    """
    lines = [f"Model {alias!r} not available in offline mode."]
    for p in tried_paths:
        lines.append(f"  Tried: {p}")
    lines.append(f"  To fix: {fix_command}")
    if extra:
        lines.append(f"  Note: {extra}")
    raise FileNotFoundError("\n".join(lines))


def raise_offline_binary_error(
    *,
    target: str,
    tried_paths: Iterable[str],
) -> None:
    """Raise a structured error for a missing llama.cpp binary in offline mode.

    Points at `lf runtime binary pull` with the target that would make the
    binary available.
    """
    lines = [f"llama.cpp binary not available in offline mode for {target}."]
    for p in tried_paths:
        lines.append(f"  Tried: {p}")
    lines.append(f"  To fix: run 'lf runtime binary pull --platform {target}' on a host with")
    lines.append("          internet, then sync the binary directory to this host.")
    raise FileNotFoundError("\n".join(lines))


# ---------------------------------------------------------------------------
# Module import-time bootstrap
# ---------------------------------------------------------------------------
# Propagate env vars as soon as this module is imported. We rely on
# `common.llamafarm_common.__init__` importing this module before any
# `model_utils`/`huggingface_hub` import.

propagate_hf_env()
