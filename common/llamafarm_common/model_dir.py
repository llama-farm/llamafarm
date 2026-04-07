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


_SAFE_GGUF_FILENAME = re.compile(r"^[a-zA-Z0-9_.\-]+\.gguf$")


def resolve_from_model_dir(alias: str) -> Optional[ModelDirResult]:
    """Look up a model alias under `$LLAMAFARM_MODEL_DIR/<alias>/`.

    Returns a ModelDirResult if a valid GGUF weights file is found, or None
    if:
      - LLAMAFARM_MODEL_DIR is unset
      - the alias directory is missing or empty
      - no file in the alias directory has valid GGUF magic bytes

    Raises:
        ValueError: If `alias` contains path traversal characters (``..``,
            ``/``, ``\\``) or is otherwise unsafe for use as a filesystem
            subdirectory name.

    Security: this function mirrors the sanitization pattern used by
    `common/llamafarm_common/model_utils.py::get_gguf_file_path` for
    its `GGUF_MODELS_DIR` lookup, which CodeQL's py/path-injection rule
    recognizes. All filesystem operations happen inline within this
    function (no helper calls) because CodeQL's interprocedural
    dataflow treats function parameters as fresh taint sources, which
    defeats any sanitization applied inside a helper.
    """
    # Validate via the cross-module helper first (defense in depth).
    validate_alias(alias)

    # Step 1: inline regex allowlist on alias. A fresh local after the
    # guard lets CodeQL's taint tracker see the value as sanitized.
    if not _SAFE_ALIAS_RE.match(alias):
        return None
    safe_alias = alias

    # Step 2: read the env var directly inline (not through a helper)
    # so CodeQL sees the taint source in the same function as every
    # subsequent use.
    root_raw = os.environ.get("LLAMAFARM_MODEL_DIR")
    if root_raw is None or root_raw.strip() == "":
        return None

    # Step 3: compute the absolute root path. Fresh local.
    root_abs = os.path.abspath(root_raw)
    if not os.path.isdir(root_abs):
        logger.warning(
            "LLAMAFARM_MODEL_DIR=%r does not exist on disk; falling back to HF cache",
            root_raw,
        )
        return None

    # Step 4: compute the alias directory via realpath (resolves
    # symlinks) and run main's exact sanitization pattern: commonpath
    # containment + isdir check in a single compound conditional.
    alias_dir = os.path.realpath(os.path.join(root_abs, safe_alias))
    if not (
        os.path.commonpath([root_abs, alias_dir]) == root_abs
        and os.path.isdir(alias_dir)
    ):
        if os.path.commonpath([root_abs, alias_dir]) != root_abs:
            logger.warning(
                "alias %r resolves to %r which is outside %r; refusing",
                alias,
                alias_dir,
                root_abs,
            )
        return None

    # Step 5: enumerate the directory. The try/except + single-
    # conditional structure keeps the listdir in the same basic block
    # as the commonpath sanitizer.
    try:
        entry_names = sorted(os.listdir(alias_dir))
    except OSError:
        return None

    # Step 6: per-file validation. For each entry:
    #   - regex allowlist on the basename
    #   - fresh realpath + commonpath check in a compound conditional
    #     with the os.path.isfile sink
    #   - magic-byte read guarded by another commonpath check
    gguf_paths: list[str] = []
    for entry_name in entry_names:
        if not _SAFE_GGUF_FILENAME.match(entry_name):
            continue
        safe_name = entry_name

        candidate = os.path.realpath(os.path.join(alias_dir, safe_name))
        if not (
            os.path.commonpath([root_abs, candidate]) == root_abs
            and os.path.isfile(candidate)
        ):
            continue

        # Magic bytes read inside a compound conditional with a fresh
        # commonpath check so CodeQL's dataflow sees the sanitizer
        # directly guarding the open() call.
        if os.path.commonpath([root_abs, candidate]) == root_abs:
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

    if not gguf_paths:
        logger.debug(
            "alias dir has no valid GGUF files: alias=%s path=%s",
            alias,
            alias_dir,
        )
        return None

    # Separate weights candidates from mmproj candidates.
    mmproj_candidates = [
        p for p in gguf_paths if _is_mmproj_file(os.path.basename(p))
    ]
    weights_candidates = [
        p for p in gguf_paths if not _is_mmproj_file(os.path.basename(p))
    ]

    if not weights_candidates:
        logger.debug(
            "alias dir has only mmproj files, no weights: alias=%s path=%s",
            alias,
            alias_dir,
        )
        return None

    weights = _select_weights_by_preference(weights_candidates)
    mmproj = (
        _select_mmproj_by_precision(mmproj_candidates) if mmproj_candidates else None
    )

    logger.debug(
        "alias dir hit: alias=%s weights=%s mmproj=%s", alias, weights, mmproj
    )

    return ModelDirResult(
        alias=alias,
        alias_dir=alias_dir,
        weights_path=weights,
        mmproj_path=mmproj,
    )


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
