## Context

The edge runtime's `get_default_context_size()` in `runtimes/edge/utils/context_calculator.py` determines context window size using a four-tier priority system. The current order is:

1. `config_n_ctx` (explicit API/config override)
2. `n_ctx_train` (from GGUF metadata)
3. Pattern match from `model_context_defaults.yaml`
4. Computed max from memory / fallback

Since virtually all GGUF files embed `n_ctx_train`, tier 3 (pattern matching) is dead code — it never wins. The `model_context_defaults.yaml` config exists specifically to let operators set sane defaults per model family, but those defaults are ignored.

On a Raspberry Pi 5 (15GB RAM), a 268MB navlink-v2 model with `n_ctx_train=32768` allocates a 2GB+ compute buffer and 576MB KV cache. The memory-based cap (`compute_max_context`) doesn't catch it because 15GB technically fits. But the llama.cpp native layer segfaults during inference with the oversized context on ARM64, crashing the container silently.

## Goals / Non-Goals

**Goals:**
- Pattern-matched defaults from `model_context_defaults.yaml` take priority over `n_ctx_train`
- Existing `config_n_ctx` (from API, `PRELOAD_N_CTX`, per-request `n_ctx`) remains highest priority
- `n_ctx_train` remains a valid fallback when no pattern matches
- No changes to the universal runtime or any code outside the edge runtime

**Non-Goals:**
- Fixing the underlying llama.cpp ARM64 segfault with large contexts (that's upstream)
- Adding new pattern entries to `model_context_defaults.yaml` (operators can do this themselves)
- Changing memory estimation logic in `compute_max_context`
- Making the priority order configurable (complexity not warranted)

## Decisions

### 1. Swap priority of pattern match and n_ctx_train

The `elif` chain in `get_default_context_size()` (lines 396-449) reorders to:

```
1. config_n_ctx          (explicit user choice — unchanged)
2. pattern_n_ctx         (operator's deployment default — promoted)
3. n_ctx_train           (model metadata — demoted)
4. computed max / 2048   (fallback — unchanged)
```

**Why pattern over n_ctx_train:** The pattern config is a deliberate choice by the operator for their deployment. The training context is an artifact of how the model was trained — not a recommendation for how much context to allocate on a constrained device. A 268MB model trained with 32K context doesn't need 32K context to function; it needs enough context for its actual workload (typically 2-4K for tool-calling models).

**Alternative considered:** Adding a separate "edge mode" flag that changes the priority. Rejected because it adds configuration surface area for no benefit — the pattern config already is the edge-specific override mechanism.

**Alternative considered:** Capping `n_ctx_train` based on model size heuristics (e.g., models under 1GB get capped at 4096). Rejected because it's fragile and doesn't generalize across model families.

### 2. Keep the wildcard fallback pattern as the safety net

The `model_context_defaults.yaml` already has a `"*"` catch-all pattern with `n_ctx: 4096`. With the new priority order, this pattern effectively caps all unknown models at 4096 unless a more specific pattern or explicit config overrides it. This is the right behavior for edge — conservative by default, opt-in to larger contexts.

### 3. Log when pattern overrides n_ctx_train

Add an info-level log when a pattern match is used instead of the model's training context, so operators can see the override happening and adjust patterns if needed.

```
Pattern default overrides n_ctx_train (32768 → 4096) for model navlink-v2-Q8_0.gguf
```

## Risks / Trade-offs

- **Existing deployments see smaller context sizes** → Any model matched by a pattern with a smaller `n_ctx` than its `n_ctx_train` will use the pattern value. The `"*"` catch-all means all models default to 4096 instead of their training context. Mitigation: operators can add specific patterns for models that need larger context, or use `PRELOAD_N_CTX`/per-request `n_ctx` to override.
- **Pattern config becomes load-bearing** → If `model_context_defaults.yaml` is missing or empty, behavior falls through to `n_ctx_train` then memory-based fallback, which is the current (broken) behavior. Mitigation: the config file is bundled in the Docker image and the `load_model_context_config()` function already raises `FileNotFoundError` if missing.
- **Docstring/comment drift** → The priority order is documented in the function docstring (line 322-328) and inline comments. These must be updated to match. Minor risk but easy to miss.

## Open Questions

None — the change is small and well-scoped.
