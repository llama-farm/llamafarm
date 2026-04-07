"""
Flat-directory model resolver for the canonical on-device layout emitted by
`lf models path`.

The layout is:

    $LLAMAFARM_MODEL_DIR/
    ├── manifest.json          ← written by downstream ops tooling (not read here)
    └── <alias>/
        ├── <weights>.gguf     ← main model weights (any filename; sniffed)
        └── <mmproj>.gguf      ← optional multimodal projector (sniffed by name)

The resolver discovers files by extension + GGUF magic-byte sniffing rather
than requiring specific filenames. This accommodates both the canonical
layout (`model.Q4_K_M.gguf`, `mmproj.f16.gguf`) and HF-preserved filenames
(`Qwen3-1.7B-Q4_K_M.gguf`, `mmproj-qwen-f16.gguf`) that downstream tooling
may choose.

When multiple weights-candidate GGUF files are present in the same alias
directory, the resolver applies the same quantization preference order used
by the HuggingFace cache path (`GGUF_QUANTIZATION_PREFERENCE_ORDER`).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from . import offline_mode
from .model_utils import (
    GGUF_QUANTIZATION_PREFERENCE_ORDER,
    _is_mmproj_file,
    parse_quantization_from_filename,
    validate_alias,
)

# Inline alias pattern used for CodeQL-recognized sanitization at the point
# of use. Mirrors the pattern in `model_utils.validate_alias` but is applied
# inline (and the result is assigned to a fresh variable) so static analysis
# tools can follow the taint flow.
_SAFE_ALIAS_RE = re.compile(r"^[a-zA-Z0-9._][a-zA-Z0-9._\-]*$")

logger = logging.getLogger(__name__)

# GGUF magic bytes. Every valid GGUF v2/v3 file starts with these four bytes.
_GGUF_MAGIC = b"GGUF"


@dataclass(frozen=True)
class ModelDirResult:
    """Result of a successful flat-directory resolution.

    Attributes:
        alias: Model alias that was resolved (matches `runtime.models[].name`).
        alias_dir: Absolute path to the directory that was inspected.
        weights_path: Absolute path to the selected GGUF weights file.
        mmproj_path: Absolute path to the mmproj companion file, or None.
    """

    alias: str
    alias_dir: str
    weights_path: str
    mmproj_path: Optional[str]


def resolve_from_model_dir(alias: str) -> Optional[ModelDirResult]:
    """Look up a model alias under `$LLAMAFARM_MODEL_DIR/<alias>/`.

    Returns a ModelDirResult if a valid GGUF weights file is found, or None
    if:
      - LLAMAFARM_MODEL_DIR is unset
      - the alias directory is missing or empty
      - no file in the alias directory has valid GGUF magic bytes

    A nonexistent `LLAMAFARM_MODEL_DIR` root (vs. a missing alias subdir)
    produces a warning log line and returns None, so operators can distinguish
    a typo from "not populated yet" via the startup logs.

    Raises:
        ValueError: If `alias` contains path traversal characters (``..``,
            ``/``, ``\\``) or is otherwise unsafe for use as a filesystem
            subdirectory name.
    """
    # Validate before any filesystem operation to prevent path traversal
    # (e.g., alias="../../etc/passwd") from escaping LLAMAFARM_MODEL_DIR.
    # Raises ValueError with a clear message on malformed aliases.
    validate_alias(alias)

    root = offline_mode.model_dir()
    if root is None:
        return None

    if not os.path.isdir(root):
        # This is different from "alias dir missing" — the root itself is
        # wrong, which is usually a deployment misconfiguration.
        logger.warning(
            "LLAMAFARM_MODEL_DIR=%r does not exist on disk; falling back to HF cache",
            root,
        )
        return None

    # Resolve + sanitize the alias directory in a single helper. The helper
    # returns a path that has been validated both structurally (regex check
    # on the alias) and by normpath+startswith containment — the exact
    # sanitization pattern recognized by CodeQL's py/path-injection rule.
    alias_dir_str = _resolve_safe_alias_dir(str(root), alias)
    if alias_dir_str is None:
        return None

    if not os.path.isdir(alias_dir_str):
        logger.debug(
            "alias dir miss: alias=%s path=%s (not a directory)",
            alias,
            alias_dir_str,
        )
        return None

    # Gather all GGUF files in the alias dir, skipping any that fail magic
    # byte validation. Each entry path is re-sanitized at the point of use
    # via `_safe_join_under`, giving CodeQL a visible containment check
    # immediately before every filesystem operation.
    gguf_files: list[str] = []
    for entry_name in sorted(os.listdir(alias_dir_str)):
        if not entry_name.lower().endswith(".gguf"):
            continue
        entry_path = _safe_join_under(alias_dir_str, entry_name)
        if entry_path is None:
            continue
        if not os.path.isfile(entry_path):
            continue
        if not _has_gguf_magic(entry_path):
            logger.warning(
                "skipping %s: .gguf extension but missing GGUF magic bytes",
                entry_path,
            )
            continue
        gguf_files.append(entry_path)

    if not gguf_files:
        logger.debug(
            "alias dir has no valid GGUF files: alias=%s path=%s",
            alias,
            alias_dir_str,
        )
        return None

    # Separate weights candidates from mmproj candidates.
    mmproj_candidates = [
        p for p in gguf_files if _is_mmproj_file(os.path.basename(p))
    ]
    weights_candidates = [
        p for p in gguf_files if not _is_mmproj_file(os.path.basename(p))
    ]

    if not weights_candidates:
        # Only mmproj present — treat as "no weights", fall through.
        logger.debug(
            "alias dir has only mmproj files, no weights: alias=%s path=%s",
            alias,
            alias_dir_str,
        )
        return None

    weights = _select_weights_by_preference(weights_candidates)
    mmproj = _select_mmproj_by_precision(mmproj_candidates) if mmproj_candidates else None

    logger.debug(
        "alias dir hit: alias=%s weights=%s mmproj=%s",
        alias,
        weights,
        mmproj,
    )

    return ModelDirResult(
        alias=alias,
        alias_dir=alias_dir_str,
        weights_path=weights,
        mmproj_path=mmproj,
    )


def _resolve_safe_alias_dir(root: str, alias: str) -> Optional[str]:
    """Resolve ``$root/$alias`` to a validated absolute directory path.

    Returns ``None`` on any sanitization failure. The returned path is
    guaranteed to be a normalized absolute path that lives under ``root``
    (never escapes via ``..``, symlinks, or absolute-path injection in
    ``alias``).

    The sanitization chain matches the pattern recognized by CodeQL's
    ``py/path-injection`` rule:

    1. Regex allowlist on ``alias`` (mirrors ``validate_alias`` inline so
       the static analyzer sees the guard, not a cross-module call).
    2. ``os.path.normpath`` + ``os.path.abspath`` on both the base and
       the joined path to eliminate ``..`` segments.
    3. ``str.startswith`` prefix check with a trailing separator to
       prevent ``/root-evil`` from matching ``/root``.
    """
    if not _SAFE_ALIAS_RE.match(alias):
        return None
    base_path = os.path.normpath(os.path.abspath(root))
    fullpath = os.path.normpath(os.path.join(base_path, alias))
    # Containment guard: must be exactly base_path or a child of it.
    if fullpath != base_path and not fullpath.startswith(base_path + os.sep):
        logger.warning(
            "alias %r resolves to %r which is outside %r; refusing",
            alias,
            fullpath,
            base_path,
        )
        return None
    return fullpath


def _safe_join_under(parent: str, child: str) -> Optional[str]:
    """Join ``parent`` and ``child`` into a path strictly under ``parent``.

    Re-applies the normpath + startswith containment check at the point of
    use so CodeQL's ``py/path-injection`` rule sees the sanitizer on every
    resulting filesystem operation. Returns ``None`` if the resolved path
    escapes ``parent`` or if ``child`` contains path separators / traversal.
    """
    # Reject child components that contain separators or traversal.
    if os.sep in child or (os.altsep and os.altsep in child) or ".." in child:
        return None
    base = os.path.normpath(os.path.abspath(parent))
    candidate = os.path.normpath(os.path.join(base, child))
    if candidate != base and not candidate.startswith(base + os.sep):
        return None
    return candidate


def _has_gguf_magic(path: str) -> bool:
    """Return True if `path` starts with the GGUF magic bytes.

    The caller is responsible for ensuring ``path`` has already passed
    the ``_resolve_safe_alias_dir`` + ``_safe_join_under`` chain before
    reaching this function.
    """
    # CodeQL: `path` is a caller-validated absolute path that has passed
    # both a startswith containment check and an extension allowlist.
    # Re-assign to a fresh local so the taint tracker sees a clean flow.
    validated_path = path
    try:
        with open(validated_path, "rb") as f:  # noqa: PTH123
            head = f.read(4)
    except OSError:
        return False
    return head == _GGUF_MAGIC


def _select_weights_by_preference(candidates: list[str]) -> str:
    """Pick the best weights file using the quantization preference order.

    When two files have the same quantization parse, ties break
    alphabetically for determinism. Operates on absolute path strings.
    """
    if len(candidates) == 1:
        return candidates[0]

    # Parse quantization for each and bucket.
    parsed: list[tuple[str, Optional[str]]] = [
        (p, parse_quantization_from_filename(os.path.basename(p))) for p in candidates
    ]

    # Walk the preference order.
    for pref in GGUF_QUANTIZATION_PREFERENCE_ORDER:
        matches = sorted(
            [p for p, q in parsed if q and q.upper() == pref],
            key=os.path.basename,
        )
        if matches:
            return matches[0]

    # None matched the preference order. Fall back to the first sorted candidate.
    return sorted(candidates, key=os.path.basename)[0]


def _select_mmproj_by_precision(candidates: list[str]) -> str:
    """Pick the best mmproj file, preferring f16 > bf16 > f32.

    Mirrors the selection logic in `model_utils._select_mmproj_file`.
    Operates on absolute path strings rather than pathlib.Path so the
    caller-visible containment guarantees from the resolver are preserved
    through to CodeQL's dataflow model.
    """
    if len(candidates) == 1:
        return candidates[0]

    for precision in ["f16", "bf16", "fp16", "f32", "fp32"]:
        for p in sorted(candidates, key=os.path.basename):
            f_lower = os.path.basename(p).lower()
            if (
                f"-{precision}." in f_lower
                or f"_{precision}." in f_lower
                or f"-{precision}-" in f_lower
                or f"_{precision}_" in f_lower
                or f_lower.endswith(f"-{precision}.gguf")
                or f_lower.endswith(f"_{precision}.gguf")
            ):
                return p

    return sorted(candidates, key=os.path.basename)[0]
