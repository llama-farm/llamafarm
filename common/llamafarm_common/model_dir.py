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

    # Collect GGUF files from the alias directory in a single helper that
    # performs all filesystem operations internally, after an inline
    # sanitization of the user-provided alias. This keeps every tainted-
    # path use inside a single basic block that also contains the
    # sanitizer guard, matching the pattern CodeQL's py/path-injection
    # rule recognizes.
    alias_dir_str, gguf_files = _list_gguf_files_in_alias_dir(str(root), alias)
    if alias_dir_str is None:
        return None

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


def _list_gguf_files_in_alias_dir(
    root: str, alias: str
) -> tuple[Optional[str], list[str]]:
    """Do all filesystem work for a single alias directory in one place.

    This helper exists to keep every filesystem operation on the
    alias-directory path inside the same basic block as the sanitizer
    guards, so CodeQL's ``py/path-injection`` dataflow rule recognizes
    the containment check as protecting each use.

    Sanitization chain (applied in order):

    1. Regex allowlist on ``alias`` — mirrors ``validate_alias`` inline
       so the static analyzer sees the guard rather than a cross-module
       call.
    2. Normalize the base path with ``os.path.normpath(os.path.abspath)``.
    3. Compute the alias directory via ``normpath(join(base, alias))``.
    4. ``startswith`` containment guard: the normalized path must equal
       the base or start with ``base + os.sep``. This matches the
       pattern shown in CodeQL's py/path-injection docs.
    5. Each per-file use re-applies a basename regex check + a fresh
       ``os.path.join(safe_dir, safe_name)`` inline inside the guarded
       block.

    Returns a tuple ``(safe_alias_dir, gguf_paths)``. ``safe_alias_dir``
    is ``None`` on any sanitization failure. ``gguf_paths`` is a list of
    absolute paths to validated GGUF files (each has matching magic
    bytes); it may be empty if no files match.
    """
    # Step 1: validate alias via inline regex so CodeQL sees the guard.
    if not _SAFE_ALIAS_RE.match(alias):
        return None, []

    # Step 2: normalize the base path.
    base_path = os.path.normpath(os.path.abspath(root))

    # Step 3: compute candidate alias dir, normalize.
    alias_dir = os.path.normpath(os.path.join(base_path, alias))

    # Step 4: containment guard. Must equal base or be a child of base.
    if alias_dir != base_path and not alias_dir.startswith(base_path + os.sep):
        logger.warning(
            "alias %r resolves to %r which is outside %r; refusing",
            alias,
            alias_dir,
            base_path,
        )
        return None, []

    # Step 5: every filesystem operation below is inside the guarded
    # block. The `alias_dir == base_path or alias_dir.startswith(...)`
    # condition above is the sanitizer; CodeQL's dataflow tracks it as
    # protecting the subsequent uses within this same basic block.
    if not os.path.isdir(alias_dir):
        return alias_dir, []

    gguf_paths: list[str] = []
    try:
        entry_names = sorted(os.listdir(alias_dir))
    except OSError:
        return alias_dir, []

    # Regex for acceptable GGUF filenames. Matches the pattern used by
    # main's `get_gguf_file_path` for `GGUF_MODELS_DIR` lookups.
    _SAFE_GGUF_FILENAME = re.compile(r"^[a-zA-Z0-9_.\-]+\.gguf$")

    for entry_name in entry_names:
        if not entry_name.lower().endswith(".gguf"):
            continue
        # Inline basename sanitizer: a fresh regex guard that CodeQL
        # recognizes as cutting the taint chain for this iteration.
        if not _SAFE_GGUF_FILENAME.match(entry_name):
            continue
        safe_name = entry_name  # reassign post-validation
        # Compute the per-file candidate path and re-verify containment
        # inline. Even though alias_dir is already validated, repeating
        # the startswith check at the point of use gives CodeQL a
        # visible guard against the subsequent `open(candidate)` call.
        candidate = os.path.normpath(os.path.join(alias_dir, safe_name))
        if candidate != alias_dir and not candidate.startswith(alias_dir + os.sep):
            continue
        if not (
            (candidate == alias_dir or candidate.startswith(alias_dir + os.sep))
            and os.path.isfile(candidate)
        ):
            continue
        # Read magic bytes inline so the open() is in the same block as
        # the startswith guard above.
        if not (
            candidate == alias_dir or candidate.startswith(alias_dir + os.sep)
        ):
            continue
        try:
            with open(candidate, "rb") as f:
                head = f.read(4)
        except OSError:
            continue
        if head != _GGUF_MAGIC:
            logger.warning(
                "skipping %s: .gguf extension but missing GGUF magic bytes",
                candidate,
            )
            continue
        gguf_paths.append(candidate)

    return alias_dir, gguf_paths


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
