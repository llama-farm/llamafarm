"""
LlamaFarm Common Utilities

Shared utilities used across all LlamaFarm Python services (server, rag, runtimes).
"""

__version__ = "0.1.0"

# IMPORTANT: offline_mode MUST be imported before any module that transitively
# imports `huggingface_hub` or `transformers`. Those libraries read their
# offline env vars (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE) once at module load
# time. offline_mode's module-level bootstrap sets those vars when
# LLAMAFARM_OFFLINE=1 is present, so importing it first ensures downstream
# libraries see the propagated values.
from . import offline_mode  # noqa: F401 — import for side effects

from .model_utils import (
    GGUF_QUANTIZATION_PREFERENCE_ORDER,
    get_gguf_file_path,
    get_mmproj_file_path,
    list_gguf_files,
    parse_model_with_quantization,
    parse_quantization_from_filename,
    resolve_gguf_path,
    resolve_mmproj_path,
    select_gguf_file,
    select_gguf_file_with_logging,
    validate_alias,
)

# Submodules also importable as llamafarm_common.safe_home, etc.
# Kept as submodule imports to avoid adding their deps to the top-level namespace.
# Usage:
#   from llamafarm_common.safe_home import safe_home, get_data_dir
#   from llamafarm_common.device import get_optimal_device, get_device_info
#   from llamafarm_common.model_cache import ModelCache
#   from llamafarm_common.model_format import detect_model_format

__all__ = [
    "GGUF_QUANTIZATION_PREFERENCE_ORDER",
    "get_gguf_file_path",
    "get_mmproj_file_path",
    "list_gguf_files",
    "offline_mode",
    "parse_model_with_quantization",
    "parse_quantization_from_filename",
    "resolve_gguf_path",
    "resolve_mmproj_path",
    "select_gguf_file",
    "select_gguf_file_with_logging",
    "validate_alias",
]
