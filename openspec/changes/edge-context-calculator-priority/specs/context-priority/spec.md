## ADDED Requirements

### Requirement: Pattern-matched defaults take priority over training context

The context calculator SHALL use pattern-matched defaults from `model_context_defaults.yaml` before falling back to the model's `n_ctx_train` metadata. The full priority order SHALL be:

1. Explicit `config_n_ctx` (from API request, `PRELOAD_N_CTX`, or per-request `n_ctx`)
2. Pattern match from `model_context_defaults.yaml`
3. `n_ctx_train` from GGUF metadata
4. Computed maximum from available memory
5. Fallback default (2048)

All tiers SHALL be capped by the memory-based maximum computed by `compute_max_context()`.

#### Scenario: Model matches a pattern with smaller context than n_ctx_train
- **WHEN** a GGUF model has `n_ctx_train=32768` and matches a pattern with `n_ctx=4096`
- **THEN** the calculator SHALL use `4096` as the context size

#### Scenario: Model matches no specific pattern but matches wildcard
- **WHEN** a GGUF model does not match any specific pattern but matches the `"*"` catch-all with `n_ctx=4096`
- **THEN** the calculator SHALL use `4096` as the context size, not the model's `n_ctx_train`

#### Scenario: Explicit config_n_ctx overrides pattern match
- **WHEN** a request includes `n_ctx=2048` and the model matches a pattern with `n_ctx=4096`
- **THEN** the calculator SHALL use `2048` (explicit config always wins)

#### Scenario: No pattern match falls through to n_ctx_train
- **WHEN** a GGUF model has `n_ctx_train=8192` and no pattern in `model_context_defaults.yaml` matches (including no wildcard)
- **THEN** the calculator SHALL use `8192` from the model's training context

### Requirement: Log when pattern overrides training context

The context calculator SHALL log an info-level message when a pattern-matched default is used instead of the model's `n_ctx_train`, including both values and the model identifier.

#### Scenario: Pattern override is logged
- **WHEN** a model with `n_ctx_train=32768` is assigned `n_ctx=4096` via pattern match
- **THEN** the calculator SHALL emit an info log indicating the pattern default overrides `n_ctx_train`, showing both values

### Requirement: Docstring reflects updated priority order

The `get_default_context_size()` function docstring SHALL document the updated priority order with pattern match at tier 2 and `n_ctx_train` at tier 3.

#### Scenario: Docstring accuracy
- **WHEN** a developer reads the docstring of `get_default_context_size()`
- **THEN** the documented priority order SHALL list pattern match before `n_ctx_train`
