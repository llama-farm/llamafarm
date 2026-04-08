## Why

Today every `lf models` command boots the Python server to do its work — even read-only commands like `lf models status` and `lf models list`, which only need to walk the local HuggingFace Hub cache directory. `lf models pull` is the worst offender: it boots FastAPI just to make a few HTTPS requests and `mv` files into a well-known directory layout. This makes the CLI unusable in CI and on fresh machines without a full Python environment, and contradicts the offline-deploy story that `feat-cli-models-path` and `feat-runtime-offline` already committed to (CLI owns the model directory; runtime consumes it). We want `lf models *` to be a self-contained Go binary that talks directly to HuggingFace Hub and writes the same on-disk cache layout that `transformers.from_pretrained` and `huggingface_hub.try_to_load_from_cache` already know how to read.

## What Changes

- Add a new internal Go package `cli/internal/hfmodel` — the write-side companion to the existing `cli/internal/hfcache` (read-side). It owns the HF Hub cache write path bit-perfectly so Python loaders find files written by Go without modification.
- Port `parse_model_with_quantization`, `parse_quantization_from_filename`, `is_split_gguf_file`, `select_gguf_file`, and `GGUF_QUANTIZATION_PREFERENCE_ORDER` from `common/llamafarm_common/model_utils.py` into Go. Table-driven tests mirror the Python fixtures one-for-one so behavior stays in lock-step.
- Add a Go HF Hub API client: tree listing (`/api/models/<id>/tree/<rev>?recursive=true`), file metadata (HEAD `resolve/<rev>/<file>` for ETag, X-Linked-Size, X-Linked-ETag, commit hash), Bearer auth, and structured error mapping for 401/403/404/410/451.
- Add HF token discovery in Go matching `huggingface_hub.get_token` order: `$HF_TOKEN` → `$HUGGING_FACE_HUB_TOKEN` → `$HF_HOME/token` → `~/.cache/huggingface/token` → `~/.huggingface/token` (legacy).
- Add a streaming downloader that writes the canonical cache layout: atomic blob writes (`<etag>.tmp` → rename), `snapshots/<commit>/<file>` symlink with Windows copy-fallback matching `huggingface_hub.file_download._create_symlink`, and `refs/main` commit-hash file. Resumable downloads via `Range` header with ETag re-verification before resume.
- Add file locking compatible with `huggingface_hub`'s `filelock`-based `.lock` files so concurrent Go and Python processes coordinate cleanly (`flock` on unix, `LockFileEx` on Windows).
- Switch `lf models status` (Slice 0) and `lf models list` (Slice 1) to use `hfcache` directly. Drop their `EnsureServicesOrExit` calls. No server boot for read-only model commands.
- Switch `lf models pull` (Slices 3 and 4) to use `hfmodel` directly for both single-file (GGUF) and multi-file (transformer) downloads, with bounded concurrency for the multi-file case. No server boot for downloads.
- Preserve the progress event vocabulary today's SSE stream uses (`init`/`start`/`progress`/`cached`/`end`/`done`/`error`/`warning`) so the future server-side adapter (out of scope here) is a thin shim.
- The `POST /v1/models/download` HTTP endpoint and SSE event shape are **unchanged**. Designer and Electron continue to hit it exactly as today; only the CLI's internal codepath bypasses the HTTP layer.

**Out of scope (deliberately, to keep this change shippable):**
- Server-side delegation of `POST /v1/models/download` to the new Go path
- Disk-space precheck port from `server/services/disk_space_service.py`
- Deletion of dead Python download code (`UniversalProvider.download_model`, `stream_download_file`, etc.)
- A standalone `lfmodel` subprocess binary for Python to shell out to
- These are sequenced as future work once Slices 0–4 prove the layout-compat contract.

## Capabilities

### New Capabilities
- `cli-models-native`: The CLI can read, list, status-check, and download HuggingFace Hub models without invoking any Python or booting the LlamaFarm server. Defines the lookup precedence, the HF cache layout the CLI writes, the GGUF quantization selection contract, the progress event vocabulary, the HF token discovery order, the resumable-download semantics, and the structured error mapping for HF API failures.

### Modified Capabilities

*(None — `models-path` from `feat-cli-models-path` already covers the read-side cache layout convention. This change introduces write-side behavior that lives in a separate capability so the two can evolve independently.)*

## Impact

- **CLI code** (Go): new `cli/internal/hfmodel/` package; modifications to `cli/cmd/models.go`, `cli/cmd/models_pull.go`, `cli/cmd/models_shared.go` to drop the orchestrator/server dependency. May share types with `cli/internal/hfcache`.
- **CLI tests** (Go): table-driven unit tests for the GGUF selection port mirroring `common/tests/test_model_utils*.py`; integration test that writes a model via Go and reads it back via a Python script invoking `huggingface_hub.try_to_load_from_cache` (the load-bearing layout-compat contract).
- **No Python changes**: `server/services/runtime_service/providers/universal_provider.py`, `server/api/routers/models/services.py`, and `common/llamafarm_common/model_utils.py` are untouched. The Python download path keeps working for Designer / Electron consumers.
- **No HTTP API changes**: `POST /v1/models/download`, `GET /v1/models`, `POST /v1/models/validate-download`, and `GET /v1/models/{id}/quantizations` keep their current shapes and SSE event vocabulary.
- **No config schema changes**: no new env vars, no `llamafarm.yaml` fields. Honors existing `HF_HUB_CACHE`, `HF_HOME`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, and `LLAMAFARM_OFFLINE` (refuse network calls when set).
- **Downstream consumers**: CI pipelines and fresh-install users no longer need a working Python environment to populate a model cache before running the runtime. Aligns with `feat-runtime-offline`'s "lf models pull on a host with internet, then sync the files" remediation messaging.
- **Risks**: HF cache layout drift (mitigated by the layout-compat integration test), Windows symlink semantics (mitigated by direct port of `_create_symlink`), GGUF selection regex parity (mitigated by mirrored table tests), file-lock interop with `filelock` (mitigated by using the same lock-file naming and `flock`/`LockFileEx` semantics).
