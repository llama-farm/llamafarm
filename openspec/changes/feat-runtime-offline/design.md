## Context

The companion change `feat-cli-models-path` (just landed) gives deployment tooling a way to locate model files and llama.cpp binaries for a target device and emit a source→target transport plan. The canonical on-device layout it documents is:

```
<target-root>/
├── manifest.json                      ← ansible/packer writes this
├── <alias>/
│   ├── model.<QUANT>.gguf             ← GGUF weights
│   └── mmproj.<precision>.gguf        ← optional multimodal projector
```

What's missing is the runtime-side half. Today, `runtimes/edge` loads models via `common/llamafarm_common/model_utils.py`, which:

1. Consults the HuggingFace cache first (`_get_cached_gguf_files`)
2. Calls `HfApi.list_repo_files()` over HTTP if cache is empty
3. Falls back to cache on network failure (but only after trying the network)
4. Calls `snapshot_download()` to fetch missing files

Steps 2 and 4 are blockers for air-gapped devices. Step 3 is "graceful degradation" but still pays the latency of a failed network call on every cold start. Additionally, the llama.cpp binary loader in `packages/llamafarm-llama/src/llamafarm_llama/_binary.py` always tries to download from GitHub if no local copy exists — a fatal flaw when the device has no internet.

The concrete use case driving this change is llamadrone / arc: Raspberry Pi devices running the edge runtime in a Docker container, where models are bind-mounted from the host filesystem and no DNS is reachable. The ops pipeline is `lf models pull` on the build host → `lf models path --format json` → ansible file placement → container start with bind mount. Everything works up to container start; at that point, the runtime still tries to resolve models through the HuggingFace API and fails.

## Goals / Non-Goals

**Goals:**

- Give deployments a single env-var flag (`LLAMAFARM_OFFLINE=1`) that guarantees zero outbound network requests for model + llama.cpp binary resolution.
- Give deployments a flat-directory layout (`LLAMAFARM_MODEL_DIR`) that the runtime can load from, matching the canonical layout emitted by `lf models path`.
- Preserve existing online behavior as the default — this change is strictly additive for current users.
- Make offline failures loud and actionable: name the alias, list the paths tried, reference the `lf` command that would fix it.
- Keep the runtime's responsibility scoped to "load a model, serve requests". All fetching/placement concerns stay in the CLI + ops tooling.

**Non-Goals:**

- CLI changes (handled by `feat-cli-models-path`, already landed).
- Lemonade runtime changes. The edge runtime is the priority; lemonade is out of scope.
- Deep universal-runtime refactoring. Universal already uses `from_pretrained` which honors `HF_HUB_OFFLINE=1` transitively; the only universal work here is propagating the env var. GGUF loader parity with edge is deferred.
- A new config schema field for offline mode. Env-var driven, matching `HF_HUB_OFFLINE` precedent.
- Manifest generation or verification. The `manifest.json` file that downstream tooling places alongside the models is opaque to the runtime.
- Any change to how chat templates are loaded — GGUF metadata embeds the template, so `get_chat_template_from_gguf` continues to work unchanged.

## Decisions

### Decision: Env-var driven, not config-driven

Offline mode and the flat-directory path are set via `LLAMAFARM_OFFLINE` and `LLAMAFARM_MODEL_DIR` environment variables, not via new fields in `llamafarm.yaml`.

**Rationale:**

1. The project config is server-side state that gets pushed around via `lf deploy`. Deployment-time flags like "am I air-gapped" are inherently about the *target device*, not the project. Env vars are the right shape.
2. `HF_HUB_OFFLINE=1` is the existing standard that `huggingface_hub`, `transformers`, and `datasets` already honor. Aligning with that precedent avoids reinventing semantics.
3. Docker compose and ansible idiomatically pass env vars. A config field would force operators to either edit the pushed config per-device (bad) or maintain device-specific overrides (worse).
4. No schema regeneration required (no `nx run generate-types` run, no cross-language type sync), which keeps this PR scoped to Python changes.

**Alternatives considered:**

- `llamafarm.yaml` `runtime.offline: true` field: rejected for the reasons above.
- CLI flag on runtime startup (`--offline`): rejected because it couples the env-var-centric Docker deployment model to a command-line invocation that Docker compose would have to construct.
- Auto-detect from "can I reach huggingface.co": rejected as fragile and slow on cold start.

### Decision: Three-tier resolution order

`resolve_gguf_path` resolves model files in this order, first match wins:

```
  1. $LLAMAFARM_MODEL_DIR/<alias>/   (opt-in, format-sniffed)
          │
          ▼ (not set, or alias dir empty/missing)
  2. HuggingFace cache               (existing code path)
          │
          ▼ (cache miss, LLAMAFARM_OFFLINE unset)
  3. Network download                (snapshot_download)
```

When `LLAMAFARM_OFFLINE=1`, step 3 is removed. When the cache is missing AND offline AND the alias dir is missing, the runtime raises immediately with a full error report.

**Absolute paths are not a tier.** This was initially included in the design, but taint-analysis (CodeQL py/path-injection) flagged the caller-controlled absolute-path passthrough, and the feature duplicated what `LLAMAFARM_MODEL_DIR` already provides. Absolute paths in `runtime.models[].model` are still accepted by the legacy `get_gguf_file_path` entry point — which handles `.gguf`-suffixed inputs via a safe-directory basename lookup under `~/.llamafarm/models/` or `$GGUF_MODELS_DIR/` as implemented in the edge runtime on main. That behavior is preserved unchanged; the tier is only absent from the new alias-aware resolver.

**Rationale:** This preserves all existing behavior by default (HF cache works as before, absolute paths still flow through the legacy entry point), layers in the new alias-directory tier as an opt-in, and adds one failure mode (step 3 removed) that only affects deployments explicitly opted into offline mode.

**Alternatives considered:**

- Keep absolute paths as tier 1 in the new resolver: rejected because it reintroduces the py/path-injection concern and duplicates the legacy entry point's existing safe-directory basename lookup.
- HF cache as highest priority, alias dir as fallback: rejected because the whole point of the alias directory is to *replace* the HF cache on devices where the HF cache structure would be awkward to maintain.
- Single-tier "either env var is set → use only alias dir": rejected because it doesn't let developers test offline mode against a populated HF cache without rebuilding their world.

### Decision: Format sniffing, not exact filenames

The alias-directory resolver discovers files by format (`.gguf` extension + GGUF magic bytes) and the existing mmproj filename heuristic, NOT by requiring specific filenames like `model.Q4_K_M.gguf`.

**Rationale:** The canonical layout emitted by `lf models path` uses normalized filenames (`model.<QUANT>.gguf`, `mmproj.<precision>.gguf`), but downstream tooling may choose to preserve the HF-native filenames instead (`Qwen3-1.7B-Q4_K_M.gguf`, `mmproj-qwen-f16.gguf`). Both should work. Format sniffing decouples the runtime from a rigid filename contract while still identifying the right file for each role.

The sniffing logic reuses the existing heuristic in `common/llamafarm_common/model_utils.py._is_mmproj_file` (already battle-tested), and uses the existing `GGUF_QUANTIZATION_PREFERENCE_ORDER` to disambiguate when multiple candidate weights files are present.

**Alternatives considered:**

- Require the canonical layout names exactly: rejected because it breaks `rsync --preserve-names` workflows that copy HF cache files directly.
- Require a metadata file (`kind`, `role`, etc.) alongside each model: rejected because it adds friction to ops workflows and the file is already self-describing via its extension + magic bytes.

### Decision: Edge runtime first; universal deferred beyond env propagation

The initial implementation updates `runtimes/edge` (specifically `models/gguf_language_model.py`) and the shared `common/llamafarm_common/model_utils.py` and `packages/llamafarm-llama/src/llamafarm_llama/_binary.py`. The universal runtime gets env-var propagation but no GGUF loader refactor in this change.

**Rationale:** The llamadrone use case is edge-only. Universal runtime GGUF loading goes through `runtimes/universal/models/gguf_language_model.py` which is structurally similar to edge's but has additional complexity (cross-runtime model cache, device allocator, multi-model). Getting parity there deserves its own scoping. The env-var propagation (`LLAMAFARM_OFFLINE` → `HF_HUB_OFFLINE`) applies to both runtimes cleanly, so universal's `from_pretrained` calls will still honor offline mode transitively via `huggingface_hub`.

**Alternatives considered:**

- Update both runtimes now: rejected to keep this change reviewable and to avoid conflating two runtimes' quirks.
- Only update `common/llamafarm_common` and let each runtime adopt it later: rejected because edge is the primary target and needs to work end-to-end in this change.

### Decision: Env var propagation happens at import time, not at function-call time

The code that sets `HF_HUB_OFFLINE=1` from `LLAMAFARM_OFFLINE=1` runs during the very first module import that checks offline mode (e.g. at the top of `model_utils.py`), BEFORE any `huggingface_hub` module is imported. This ensures that when `huggingface_hub` first reads its offline state, the env var is already set.

**Rationale:** `huggingface_hub` reads `HF_HUB_OFFLINE` once when its constants module is imported. If we set the env var later, it's too late. The runtime's import order must therefore place the offline-mode bootstrap before any `huggingface_hub` or `transformers` import.

**Mitigation:** Add a dedicated `common/llamafarm_common/offline_mode.py` module that:
1. Reads `LLAMAFARM_OFFLINE` and sets `HF_HUB_OFFLINE` if needed.
2. Exposes `is_offline()` and `strict_offline()` helpers.
3. Logs the resolved mode on first call (idempotent).

This module is imported at the top of `model_utils.py` (which is imported early by both runtimes) and at the top of `runtimes/edge/server.py`. The universal runtime's equivalent startup module imports it too.

**Alternatives considered:**

- Set `HF_HUB_OFFLINE` from the Dockerfile/shell script: works but requires every deployment to know about both env vars. Wrapping into `LLAMAFARM_OFFLINE` is more ergonomic.
- Set `HF_HUB_OFFLINE` in `server.py` after imports: rejected because `huggingface_hub` has already read its state by then.

### Decision: sha256 verification is NOT part of runtime load

The `manifest.json` file that `lf models path --format json` emits (and that downstream tooling places on the device) contains sha256 digests. The runtime does NOT read this file or verify file integrity on load.

**Rationale:** Integrity verification is a deployment-pipeline concern (ansible, rsync `--checksum`, image signing, etc.) and performing it at runtime would add a multi-second startup cost on multi-GB models. If a deployment wants on-device verification, it should do it in the ops layer before starting the container.

**Alternatives considered:**

- Opt-in verification via `LLAMAFARM_VERIFY_MODELS=1`: deferrable to a follow-up if someone actually asks for it. Not in V1.

### Decision: Error messages are structured multi-line strings, not custom exception classes

Offline-mode errors for missing models raise `FileNotFoundError` (or the existing runtime exception type) with a multi-line message that names the alias, lists the tried paths, and references the fix. We do not introduce a new `LlamaFarmOfflineError` exception class.

**Rationale:** The existing exception surface is already understood by callers. Adding a new exception class forces every caller to either catch both or accept that `FileNotFoundError` is a stand-in. A well-formatted message achieves the same operator experience without an API churn.

**Canonical format:**

```
Model 'qwen3-1.7b' not available in offline mode.
  Tried: /opt/llamafarm/models/qwen3-1.7b/
  Tried: /root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-GGUF/
  To fix: run `lf models pull qwen3-1.7b` on a host with internet, then sync the files
          (or use `lf models path --ensure` if the build host has internet).
```

**Alternatives considered:**

- Dedicated `ModelNotFoundOfflineError`: rejected per above.
- Include the full transport plan in the error: rejected as noisy; operators can run `lf models path` themselves.

## Risks / Trade-offs

**Risk:** Setting `HF_HUB_OFFLINE=1` at import time may interact badly with libraries that are already loaded before our bootstrap runs.
**Mitigation:** The bootstrap must run in `common/llamafarm_common/__init__.py` or an explicit early import in every entry point. Integration test covers this by starting the edge runtime with `LLAMAFARM_OFFLINE=1` and asserting that loading a cached model does not touch the network (verified by mock HTTP server that fails any request).

**Risk:** Format sniffing in the alias directory may misclassify a corrupt GGUF file as a valid weights file, then crash at llama-cpp init time with an opaque error.
**Mitigation:** Check magic bytes (`GGUF` at offset 0) in addition to the `.gguf` extension, and surface a clear "file at <path> has corrupt GGUF magic" error before attempting to load. This matches what the CLI's `modelformat` package already does.

**Risk:** Operators who set `LLAMAFARM_MODEL_DIR` but forget to populate it will see a fallthrough to HF cache (in online mode) or a loud error (in offline mode) — surprising in the online case.
**Mitigation:** The startup log line records which mode + model_dir are active. The error in offline mode explicitly lists the alias directory as one of the tried paths. An operator who sees "online mode, model_dir=X, but files actually coming from HF cache" has enough signal to investigate.

**Risk:** The universal runtime's `from_pretrained` cache interactions are complex; relying on `HF_HUB_OFFLINE=1` propagation may not catch every path.
**Mitigation:** This change explicitly defers full universal-runtime parity. The env-var propagation is the minimum viable hook; follow-up work can add universal-runtime-specific offline error messages if needed. The llamadrone use case is edge-only, so this does not block the primary consumer.

**Risk:** The alias-directory resolver may get confused when a single alias directory contains multiple GGUF files and neither matches the mmproj heuristic (e.g. Q4_K_M and Q5_K_M side by side).
**Mitigation:** Reuse the existing `GGUF_QUANTIZATION_PREFERENCE_ORDER` for disambiguation, same logic as `select_gguf_file` in `model_utils.py`. Document the behavior in the offline-operation docs.

**Risk:** Tests that exercise offline mode need to run without any HuggingFace connectivity, but CI runners may have it.
**Mitigation:** Use `unittest.mock.patch` to replace `huggingface_hub.HfApi.list_repo_files` and `huggingface_hub.snapshot_download` with mocks that raise if called. This proves the code path never even attempts to reach the network, regardless of CI connectivity.

## Migration Plan

1. Land the `common/llamafarm_common/offline_mode.py` helper + guards in `model_utils.py` + guards in `_binary.py`. No behavior change for existing deployments because neither env var is set by default.
2. Land the alias-directory resolver in `common/llamafarm_common` (or a new `common/llamafarm_common/model_dir.py`). Plumb into `runtimes/edge/models/gguf_language_model.py`.
3. Update the edge runtime's `server.py` to import `offline_mode` early and emit the startup log line.
4. Add tests covering all three resolution tiers, with mocked network to prove offline mode never makes a request.
5. Update the edge runtime Dockerfile documentation (or the new offline-operation doc) with an example showing `LLAMAFARM_OFFLINE=1`, `LLAMAFARM_MODEL_DIR=/models`, and `HF_HUB_OFFLINE=1` env vars plus a bind mount.
6. Downstream: llamadrone / arc deployment playbook adopts the new env vars and the alias-directory layout. No runtime rollback concerns because the old code paths still work when the vars are unset.

Universal runtime adoption can happen in a separate PR without blocking this one.

## Open Questions

1. **Should `LLAMAFARM_OFFLINE` also force `TRANSFORMERS_OFFLINE=1`?** The `transformers` library honors its own offline var in addition to `HF_HUB_OFFLINE`. Probably yes, for belt-and-suspenders — minimal cost, keeps behavior consistent. Defer decision to implementation time.

2. **Should the startup log line go to stderr or structlog?** The edge runtime uses `structlog`. Stick with that; no reason to special-case this log line.

3. **How should the alias-directory resolver behave if `LLAMAFARM_MODEL_DIR` points at a nonexistent path (not just a missing alias dir)?** Log a warning, then fall through as if unset. An explicit mistake by the operator is different from "I haven't populated it yet." The startup log line should say `model_dir=<path> (not found)` in that case.

4. **Do we need to handle the case where a GGUF file exists in both the alias dir AND the HF cache but they have different quantizations?** Current design: alias dir wins, period. Document this in the offline-operation guide so operators who want a specific quantization know they need to place the right file in the alias dir.

5. **Chat template extraction on non-GGUF models (transformers format) in offline mode.** V1 scope is GGUF-only for the alias-directory resolver, matching `feat-cli-models-path` V1 scope. Transformers models in the universal runtime keep going through `from_pretrained` with `HF_HUB_OFFLINE=1`. This open question is a reminder that future work on universal-runtime GGUF parity will need to re-examine this.
