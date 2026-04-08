## 1. Offline-Mode Bootstrap Module

- [x] 1.1 Create `common/llamafarm_common/offline_mode.py` with `is_offline()`, `model_dir()`, `raise_offline_error()`, `raise_offline_binary_error()` helpers
- [x] 1.2 In that module, read `LLAMAFARM_OFFLINE` (truthy values: `1`, `true`, `yes`, `on`, case-insensitive) at import time and set `os.environ["HF_HUB_OFFLINE"] = "1"` before any `huggingface_hub`/`transformers` import can run
- [x] 1.3 Also set `os.environ["TRANSFORMERS_OFFLINE"] = "1"` when `LLAMAFARM_OFFLINE` is set
- [x] 1.4 Emit a single idempotent log line via `log_startup_mode()`: `mode=online|offline`, plus `model_dir=<path>` if set. Uses structlog when available, stdlib logging otherwise.
- [x] 1.5 Handle the conflict case: when `LLAMAFARM_OFFLINE=1` and `HF_HUB_OFFLINE=0` (or `TRANSFORMERS_OFFLINE=0`), override to `1` and log a warning for each
- [x] 1.6 Import `common/llamafarm_common/offline_mode` from `common/llamafarm_common/__init__.py` BEFORE `model_utils` so propagation runs before `huggingface_hub` is loaded
- [x] 1.7 Unit tests in `common/tests/test_offline_mode.py`: 35 tests covering truthy detection, propagation, overrides, idempotency, log-once, structured error messages

## 2. Alias-Directory Model Resolver

- [x] 2.1 Create `common/llamafarm_common/model_dir.py` with a `resolve_from_model_dir(alias: str) -> ModelDirResult | None` function
- [x] 2.2 Use extension + GGUF magic-byte sniffing (`b"GGUF"` at offset 0) to validate each candidate
- [x] 2.3 Reuse `_is_mmproj_file` heuristic from `model_utils.py` to separate weights from mmproj
- [x] 2.4 Apply `GGUF_QUANTIZATION_PREFERENCE_ORDER` when multiple weights candidates exist
- [x] 2.5 Return frozen dataclass `ModelDirResult(alias, alias_dir, weights_path, mmproj_path)`
- [x] 2.6 Debug log on hits, warning log when `LLAMAFARM_MODEL_DIR` root does not exist
- [x] 2.7 Unit tests in `common/tests/test_model_dir.py`: 19 tests covering all the above cases

## 3. `model_utils.py` Offline Guards and Resolver Integration

- [x] 3.1 Add new public `resolve_gguf_path(model_id, alias, token=None, preferred_quantization=None) -> str` + `resolve_mmproj_path(model_id, alias, token=None) -> str | None` functions implementing the three-tier order (LLAMAFARM_MODEL_DIR → HF cache → network; absolute paths deliberately not a tier, see design.md)
- [x] 3.2 Import `offline_mode` at the top of `model_utils.py` BEFORE `huggingface_hub` is imported so env-var propagation runs in time
- [x] 3.3 Skip the alias-dir tier when `LLAMAFARM_MODEL_DIR` is unset (handled by `resolve_from_model_dir` returning None)
- [x] 3.4 Skip the network-download tier in offline mode; raise structured `FileNotFoundError` with alias name, both tried paths, and `lf models pull` remediation
- [x] 3.5 Keep `get_gguf_file_path` as a standalone public function (already had cache-first logic; added offline guard) — `resolve_gguf_path` is a new layer on top that adds tiers 1 and 2
- [x] 3.6 Add offline guard to `get_mmproj_file_path` — returns None without touching network (mmproj is optional)
- [x] 3.7 Add offline guard to `list_gguf_files` — raises immediately before any `HfApi` construction
- [x] 3.8 Unit tests in `common/tests/test_model_utils_offline.py` (14 tests) covering: list_gguf_files offline + online, offline+cache hit, offline+cache miss (full error format), absolute path, alias dir wins, alias dir miss fallthrough, offline full coverage, mmproj tier, legacy get_mmproj_file_path guard
- [x] 3.9 `unittest.mock.patch` asserts `snapshot_download` and `HfApi` never called in offline tests — passes against all 14 offline tests

## 4. `llamafarm_llama/_binary.py` Offline Guards

- [x] 4.1 Implement `_is_offline()` inline rather than importing from `llamafarm_common` — keeps llamafarm-llama dependency-free (rationale: adding a cross-package dep would pull in huggingface_hub transitively and bloat a lean binary package)
- [x] 4.2 In `get_lib_path`, after the bundled and cached checks, raise `FileNotFoundError` when offline instead of calling `download_binary`
- [x] 4.3 Error message format matches `common/llamafarm_common/offline_mode.raise_offline_binary_error` output: names the platform, lists both tried paths, references `lf runtime binary pull --platform <os>/<arch>`
- [x] 4.4 `get_binary_info` now includes `"offline": bool` field in its output dict
- [x] 4.5 Add `packages/llamafarm-llama/tests/test_binary_offline.py` (19 tests): truthy/falsy env detection, offline + no cache raises w/o download, offline + cached binary succeeds w/o download, online mode still downloads when missing, error lists tried paths and platform, offline flag in binary info. Mocks `download_binary` and asserts-not-called in offline paths.

## 5. Runtime Integration (universal, with edge follow-up)

Context: the edge runtime lives on `feat-runtime-edge-standalone` and is not
yet merged to main. We wired the integration into the **universal** runtime
instead (which is on main), and the edge runtime will get an equivalent
one-line change when that branch rebases on top of this one. The shared
module-level changes (offline_mode, model_utils guards) already benefit edge
transitively because edge imports `llamafarm_common`.

- [x] 5.1 Update `runtimes/universal/models/gguf_language_model.py` to accept an optional `alias: str | None` kwarg and call `resolve_gguf_path(model_id, alias=alias, ...)` when provided, falling back to legacy `get_gguf_file_path` otherwise
- [x] 5.2 Same for mmproj: use `resolve_mmproj_path` when alias is known, fall back to `get_mmproj_file_path` otherwise. Both paths honor offline mode.
- [x] 5.3 Import `llamafarm_common` at the very top of `runtimes/universal/server.py` (before any import that might transitively pull in `huggingface_hub`) so `offline_mode` propagation runs in time
- [x] 5.4 Emit the startup log line via `offline_mode.log_startup_mode()` in the FastAPI lifespan
- [x] 5.5 Chat template extraction is unchanged — `get_chat_template_from_gguf` reads from the GGUF file's embedded metadata regardless of where the file lives
- [x] 5.6 Integration tests in `runtimes/universal/tests/test_gguf_offline_integration.py` (6 tests): alias routes through resolver with model_dir hit, alias+offline+complete miss raises with correct error, no-alias uses legacy path, startup log format, constructor accepts alias kwarg, alias defaults to None
- [x] 5.7 Edge runtime wiring (completed on merge from main after feat-runtime-edge-standalone landed as #799): `runtimes/edge/models/gguf_language_model.py` now accepts an `alias` kwarg and routes through `resolve_gguf_path` / `resolve_mmproj_path` when provided; `runtimes/edge/server.py` imports `llamafarm_common.offline_mode` early for env propagation and emits `log_startup_mode()` in the lifespan. Smoke-verified: constructor accepts good aliases, rejects traversal aliases at construction time.
- [x] 5.8 End-to-end `LLAMAFARM_MODEL_DIR` support on edge runtime: `runtimes/edge/utils/alias.py` auto-derives an alias from the incoming HTTP `model` field by stripping `org/` prefix and `:quant` suffix, validated via `validate_alias`. `runtimes/edge/server.py::load_language` passes the derived alias to `GGUFLanguageModel`, so `LLAMAFARM_MODEL_DIR=/opt/llamafarm/models` + a directory named after the base model ID (e.g. `Qwen3-0.6B-GGUF/`) works for any API client request form without config changes. 23 unit tests in `runtimes/edge/tests/test_alias.py` cover the derivation + validation edge cases.

## 6. Universal Runtime Env-Var Propagation

- [x] 6.1 Located `runtimes/universal/server.py` as the canonical entry point
- [x] 6.2 Added `import llamafarm_common` at the top of `server.py` so `offline_mode` bootstrap runs BEFORE any FastAPI/HF import (handled as part of Section 5)
- [x] 6.3 Verified via smoke test: `uv run python -c "import server"` succeeds and produces no network calls on import — transformers models loaded via `from_pretrained` will honor the propagated `HF_HUB_OFFLINE=1` transitively
- [x] 6.4 Design doc already notes universal-runtime GGUF parity beyond this env-var propagation is deferred

## 7. Startup Log Line

- [x] 7.1 Implemented `log_startup_mode()` in `offline_mode.py` — idempotent via `_startup_logged` flag, uses structlog when available, stdlib logging otherwise. Emits `mode`, `model_dir`, `hf_hub_offline`, `transformers_offline` fields.
- [x] 7.2 Wired into `runtimes/universal/server.py` lifespan (edge will pick it up transitively when the branch rebases; the helper is already available via `llamafarm_common.offline_mode`)
- [x] 7.3 Test coverage: `test_offline_mode.py` has 3 log-related tests (logs-once, online message, offline message), `test_gguf_offline_integration.py` verifies the real universal-runtime structlog path produces the expected tokens

## 8. Documentation

- [x] 8.1 Created `docs/website/docs/models/offline-operation.md` — both env vars, three-tier resolution order, canonical layout, startup verification, end-to-end workflow with `lf models path`
- [x] 8.2 Docker compose snippet with bind-mounted `/models`, all offline env vars, and `LD_LIBRARY_PATH` for the llama.cpp binary
- [x] 8.3 Troubleshooting section covers: missing-model error, missing-binary error, GGUF magic validation warning, nonexistent root path warning, "online mode" mismatch
- [x] 8.4 Cross-references to `lf models path` and `lf runtime binary pull` in both directions
- [ ] 8.5 **Deferred**: Edge runtime README update (the edge runtime itself lives on a different branch; this will happen in the rebase follow-up)

## 9. Validation

- [x] 9.1 `openspec validate feat-runtime-offline --strict` passes
- [x] 9.2 Common test suite: 112 passed (35 offline_mode + 19 model_dir + 14 model_utils_offline + 44 pre-existing model_utils)
- [x] 9.3 llamafarm-llama test suite: 53 passed, 1 pre-existing broken test deselected (TestSourceBuild::test_download_binary_uses_prebuilt_on_linux_arm64 had a stale `bin/libllama.so` expectation from before the b7694 version bump — not caused by this change)
- [x] 9.4 Universal runtime test suite (edge runtime lives on a different branch): 38 passed including 6 new integration tests for offline-mode GGUF loading
- [x] 9.5 Manually verified: resolver picks up a real cached GGUF from `$LLAMAFARM_MODEL_DIR/<alias>/` via format sniffing, no network calls, log line shows `mode=offline model_dir=/tmp/lf-offline-test hf_hub_offline=1 transformers_offline=1`
- [x] 9.6 Manually verified: `LLAMAFARM_OFFLINE=1` without `LLAMAFARM_MODEL_DIR` still resolves via the HF cache tier (cache-first behavior preserved)
- [x] 9.7 Manually verified: missing-everywhere scenario raises `FileNotFoundError` with alias name, both tried paths, `lf models pull` remediation, and the `lf models path --ensure` hint
- [x] 9.8 Manually verified: corrupt `.gguf` file (wrong magic bytes) in alias dir is rejected with a specific warning log, then the resolver cleanly falls through and surfaces the standard offline error
