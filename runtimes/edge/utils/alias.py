"""Model-id to LLAMAFARM_MODEL_DIR alias derivation for the edge runtime.

The edge runtime receives model IDs in HTTP requests (e.g.
``Qwen/Qwen3-0.6B-GGUF:Q4_K_M``), not project-config aliases. To let
operators who set ``LLAMAFARM_MODEL_DIR`` place files under a predictable
directory without requiring API clients to change, we derive an alias by
stripping the ``org/`` prefix and the ``:quant`` suffix, then validate it.

Kept in its own module (rather than inline in ``server.py``) so unit
tests can import it without triggering the server's heavy runtime
bootstrap (llama.cpp backend initialization, model loader setup, etc.).
"""

from __future__ import annotations

import os

from llamafarm_common import validate_alias


def derive_alias_from_model_id(model_id: str) -> str | None:
    """Derive a ``LLAMAFARM_MODEL_DIR`` alias from an HTTP-received model_id.

    Returns the derived alias on success, or ``None`` if the model_id
    cannot be safely turned into an alias (absolute path, .gguf filename,
    or validation failure). The caller should treat ``None`` as "skip
    the LLAMAFARM_MODEL_DIR tier for this model".

    Examples:
        >>> derive_alias_from_model_id("Qwen/Qwen3-0.6B-GGUF:Q4_K_M")
        'Qwen3-0.6B-GGUF'
        >>> derive_alias_from_model_id("unsloth/Qwen3-1.7B-GGUF")
        'Qwen3-1.7B-GGUF'
        >>> derive_alias_from_model_id("qwen3-small")
        'qwen3-small'
        >>> derive_alias_from_model_id("/path/to/custom.gguf") is None
        True

    Note: this intentionally ignores the ``org/`` namespace, so
    ``foo/my-model`` and ``bar/my-model`` collide on the same alias
    directory. Operators who need to disambiguate can send distinct
    base names in their request model_ids.
    """
    # Absolute paths and bare .gguf filenames aren't aliases.
    if os.path.isabs(model_id) or model_id.endswith(".gguf"):
        return None

    # Drop the :quantization suffix if present.
    base = model_id.split(":", 1)[0]
    # Drop the org/ prefix if present.
    if "/" in base:
        base = base.rsplit("/", 1)[1]

    try:
        validate_alias(base)
    except ValueError:
        return None
    return base
