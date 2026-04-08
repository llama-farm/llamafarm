## Why

Users need to run Gemma 4 models, and our current llama.cpp pin (b7694, Jan 10) predates upstream Gemma 4 support entirely. The Gemma 4 work landed in upstream as a flurry of commits between 2026-04-02 and 2026-04-08 and is still being actively patched, so we need to track close to upstream HEAD until that churn settles.

## What Changes

- Bump the pinned llama.cpp version from `b7694` to `b8708`, the latest upstream tag at the time of implementation. `b8708` includes 7 of the 8 Gemma 4 commits that had landed upstream as of 2026-04-08; the trailing EOG token fix (`d9a12c82`) was cut ~4.5 hours after `b8708` was tagged and is deferred to the planned follow-up bump (see "Coordination notes" below).
- Replace three deprecated llama.cpp APIs with their renamed equivalents in `packages/llamafarm-llama`:
  - `llama_load_model_from_file` → `llama_model_load_from_file`
  - `llama_new_context_with_model` → `llama_init_from_model`
  - `llama_free_model` → `llama_model_free`
  All three are pure renames with identical signatures — no behavioral change.
- Trigger the existing `build-llama.yml` workflow against the new tag so the LlamaFarm-published Linux ARM64 binary is available before merge (upstream does not ship ARM64 binaries).
- Add `.claude/rules/llama_cpp_bindings.md` documenting the rationale for owning our own llama.cpp bindings, where the version is pinned, and the upgrade validation procedure — so this conversation does not have to happen again on the next bump.

## Capabilities

### New Capabilities
- `llama-cpp-binary`: Formalizes the rules governing how the llama.cpp native binary is pinned, propagated, downloaded, and validated across LlamaFarm. No spec exists for this subsystem today; this change is the moment we write it down because we are now committing to a "track upstream" cadence and need testable invariants. Covers: single source of truth for the pinned version, propagation to dependent files, cache key isolation, supported model architectures (including Gemma 4), and the upgrade validation procedure.

### Modified Capabilities
<!-- None. No existing capability spec covers the llama.cpp binary subsystem. -->


## Impact

**Affected code:**
- `llama-cpp-version.txt` (single source of truth)
- `packages/llamafarm-llama/src/llamafarm_llama/_binary.py` (fallback constant)
- `packages/llamafarm-llama/src/llamafarm_llama/_bindings.py` (deprecated cdef declarations)
- `packages/llamafarm-llama/src/llamafarm_llama/llama.py` (deprecated call sites)
- `cli/internal/llamabinary/llamabinary.go` (`Version` constant)
- `cli/internal/llamabinary/llamabinary_test.go` and `download_test.go` (test fixtures)

**Affected APIs:** None at the LlamaFarm public API surface. The deprecated llama.cpp functions we replace are internal to `llamafarm-llama` and the rename is signature-compatible.

**Affected dependencies:** llama.cpp upstream binaries downloaded at runtime by `llamafarm-llama` and bundled via `cli/internal/llamabinary`. The LlamaFarm-published Linux ARM64 artifact must be rebuilt at the new tag.

**Affected systems:**
- `.github/workflows/build-llama.yml` reads `llama-cpp-version.txt` automatically; must be triggered manually via `workflow_dispatch` at the new tag before this change merges, otherwise Linux ARM64 users will fail to download a binary.
- All users running GGUF models will pick up the new binary on next launch (cache key includes the version string, so old cached binaries remain isolated).

**Risk assessment:** Low. The header diff between `b7694` and `b8708` shows zero struct layout changes affecting our cffi bindings (`llama_model_params`, `llama_context_params`, `llama_batch` are byte-identical). Removed APIs (LoRA adapter and cvec helpers) are not used by our code. The risk is concentrated in runtime behavior that only manifests with real model inference, so the validation plan (see design.md) hinges on smoke-testing both a Gemma 4 model and a non-Gemma model end-to-end.

**Follow-up expected:** This is the first of an anticipated 2–3 bumps over the next ~2 weeks while upstream Gemma 4 churn settles. Subsequent bumps should be cheap because the deprecated-API rename and rules documentation land in this change.
