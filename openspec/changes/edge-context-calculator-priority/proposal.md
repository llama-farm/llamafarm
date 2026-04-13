## Why

The edge runtime's context calculator uses `n_ctx_train` from GGUF metadata as priority 2, ahead of pattern-matched defaults from `model_context_defaults.yaml` (priority 3). Since nearly every GGUF file includes `n_ctx_train`, the pattern-matching system is effectively dead code — it never wins. On constrained edge devices (Raspberry Pi), this causes small models (e.g., 268MB navlink-v2) to allocate their full 32K training context (2GB+ KV cache + compute buffers), crashing inference via segfault despite having 15GB of available RAM. The memory-based cap doesn't catch it because 15GB is technically enough — the crash is in the native llama.cpp layer, not an OOM.

## What Changes

- **BREAKING**: The context calculator priority order changes. Pattern-matched defaults from `model_context_defaults.yaml` now take priority over `n_ctx_train` from GGUF metadata. Deployments that relied on the implicit behavior of always using `n_ctx_train` may see different (smaller) context sizes if a pattern match exists.
- The four-tier priority becomes: (1) explicit `config_n_ctx` from API/config, (2) pattern match from `model_context_defaults.yaml`, (3) `n_ctx_train` from GGUF metadata, (4) computed max from memory / fallback.
- `PRELOAD_N_CTX` and per-request `n_ctx` continue to work as before (they feed into `config_n_ctx`, priority 1).

## Capabilities

### New Capabilities

- `context-priority`: Context calculator priority order for determining default context size from model metadata, config patterns, and memory constraints

### Modified Capabilities

## Impact

- `runtimes/edge/utils/context_calculator.py` — reorder the priority branches in `get_default_context_size()`
- `runtimes/edge/config/model_context_defaults.yaml` — no structural change, but patterns now actually govern behavior for models that match them
- Universal runtime (`runtimes/universal/`) is unaffected — it has its own context logic
- Existing `PRELOAD_N_CTX` and per-request `n_ctx` override paths are unchanged
