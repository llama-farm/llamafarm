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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import offline_mode
from .model_utils import (
    GGUF_QUANTIZATION_PREFERENCE_ORDER,
    _is_mmproj_file,
    parse_quantization_from_filename,
)

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
    """
    root = offline_mode.model_dir()
    if root is None:
        return None

    root_path = Path(root)
    if not root_path.exists():
        # This is different from "alias dir missing" — the root itself is
        # wrong, which is usually a deployment misconfiguration.
        logger.warning(
            "LLAMAFARM_MODEL_DIR=%r does not exist on disk; falling back to HF cache",
            root,
        )
        return None

    alias_dir = root_path / alias
    if not alias_dir.exists() or not alias_dir.is_dir():
        logger.debug("alias dir miss: alias=%s path=%s (not a directory)", alias, alias_dir)
        return None

    # Gather all GGUF files in the alias dir, skipping any that fail magic
    # byte validation.
    gguf_files: list[Path] = []
    for entry in sorted(alias_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() != ".gguf":
            continue
        if not _has_gguf_magic(entry):
            logger.warning(
                "skipping %s: .gguf extension but missing GGUF magic bytes",
                entry,
            )
            continue
        gguf_files.append(entry)

    if not gguf_files:
        logger.debug("alias dir has no valid GGUF files: alias=%s path=%s", alias, alias_dir)
        return None

    # Separate weights candidates from mmproj candidates.
    mmproj_candidates = [p for p in gguf_files if _is_mmproj_file(p.name)]
    weights_candidates = [p for p in gguf_files if not _is_mmproj_file(p.name)]

    if not weights_candidates:
        # Only mmproj present — treat as "no weights", fall through.
        logger.debug(
            "alias dir has only mmproj files, no weights: alias=%s path=%s",
            alias,
            alias_dir,
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
        alias_dir=str(alias_dir),
        weights_path=str(weights),
        mmproj_path=str(mmproj) if mmproj else None,
    )


def _has_gguf_magic(path: Path) -> bool:
    """Return True if `path` starts with the GGUF magic bytes."""
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except OSError:
        return False
    return head == _GGUF_MAGIC


def _select_weights_by_preference(candidates: list[Path]) -> Path:
    """Pick the best weights file using the quantization preference order.

    When two files have the same quantization parse, ties break
    alphabetically for determinism.
    """
    if len(candidates) == 1:
        return candidates[0]

    # Parse quantization for each and bucket.
    parsed: list[tuple[Path, Optional[str]]] = [
        (p, parse_quantization_from_filename(p.name)) for p in candidates
    ]

    # Walk the preference order.
    for pref in GGUF_QUANTIZATION_PREFERENCE_ORDER:
        matches = sorted(
            [p for p, q in parsed if q and q.upper() == pref],
            key=lambda p: p.name,
        )
        if matches:
            return matches[0]

    # None matched the preference order. Fall back to the first sorted candidate.
    return sorted(candidates, key=lambda p: p.name)[0]


def _select_mmproj_by_precision(candidates: list[Path]) -> Path:
    """Pick the best mmproj file, preferring f16 > bf16 > f32.

    Mirrors the selection logic in `model_utils._select_mmproj_file` but
    operates on Path objects.
    """
    if len(candidates) == 1:
        return candidates[0]

    for precision in ["f16", "bf16", "fp16", "f32", "fp32"]:
        for p in sorted(candidates, key=lambda p: p.name):
            f_lower = p.name.lower()
            if (
                f"-{precision}." in f_lower
                or f"_{precision}." in f_lower
                or f"-{precision}-" in f_lower
                or f"_{precision}_" in f_lower
                or f_lower.endswith(f"-{precision}.gguf")
                or f_lower.endswith(f"_{precision}.gguf")
            ):
                return p

    return sorted(candidates, key=lambda p: p.name)[0]
