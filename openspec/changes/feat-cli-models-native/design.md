## Context

The LlamaFarm CLI's `lf models` command surface (`status`, `list`, `pull`) currently boots the Python FastAPI server for every invocation, even read-only commands. The server endpoint `POST /v1/models/download` is itself a thin wrapper around `huggingface_hub`-style file fetches plus hand-rolled HF cache layout writes (see `server/services/runtime_service/providers/universal_provider.py::stream_download_file`). The CLI subscribes to the resulting SSE stream and prints progress.

Two recent changes set the stage for moving this work into Go:
- **`feat-cli-models-path`** introduced `cli/internal/hfcache`, a Go package that reads the HuggingFace Hub cache layout natively (snapshots, blobs, refs) without invoking Python. It deliberately mirrors the read-side subset of `common/llamafarm_common/model_utils.py`.
- **`feat-runtime-offline`** committed the runtime to consuming whatever the CLI puts on disk. Its remediation messaging already says "run `lf models pull` on a host with internet, then sync the files" — implying the CLI is the canonical writer of the model cache.

This change extends `hfcache`'s read-side ownership to a write-side companion (`hfmodel`) and migrates the `lf models *` command surface off the server. The HTTP endpoint and SSE event vocabulary are preserved unchanged so Designer and Electron continue working through the existing path.

The key technical constraint is **HF cache layout compatibility**. Python loaders (`transformers.from_pretrained`, `huggingface_hub.try_to_load_from_cache`, `llama_cpp.Llama(model_path=...)`) must be able to find files written by Go without any code changes on the Python side. The layout is stable because changing it would break every existing HF cache on disk.

## Goals / Non-Goals

**Goals:**
- `lf models status`, `lf models list`, and `lf models pull` work in a fresh environment with no Python installed and no LlamaFarm server running.
- The Go-written cache is bit-perfectly compatible with `huggingface_hub`'s reader. Verified by an integration test that writes via Go and reads back via Python.
- GGUF quantization selection in Go produces the same result as Python for every fixture currently covered by `common/tests/test_model_utils*.py`.
- HF token discovery in Go matches `huggingface_hub.get_token`'s precedence order exactly.
- Resumable downloads survive a network drop without restarting from byte 0.
- Concurrent downloads (Go ↔ Python, Go ↔ Go) are coordinated via the same `.lock` files `huggingface_hub.filelock` uses.
- The progress event vocabulary today's SSE stream emits is preserved verbatim so a future server-side adapter (out of scope) is a thin shim.
- `LLAMAFARM_OFFLINE=1` is honored: the Go path refuses network calls and emits the same structured "run `lf models pull` on a host with internet" remediation message that `feat-runtime-offline` defined.

**Non-Goals:**
- Server-side delegation of `POST /v1/models/download` to the new Go path. (Future work; the endpoint stays intact for now.)
- Disk-space precheck port from `server/services/disk_space_service.py`. (The Go path emits a `warning` event vocabulary slot but the actual precheck is left as future work.)
- Deletion of the existing Python download code (`UniversalProvider.download_model`, `stream_download_file`, `get_model_download_info`). It stays in place for Designer and Electron consumers.
- A standalone `lfmodel` subprocess binary for Python to shell out to. The new Go code lives inside `lf` for now; extracting it as a sibling binary is a future packaging decision.
- Xet high-performance transfer protocol (`HF_XET_HIGH_PERFORMANCE=1` in Python). Standard HTTPS resolve URLs only — Xet is a Python-side optimization we can revisit later.
- GGUF metadata extraction (chat templates, tokenizer config). Out of scope; the runtime already handles this from the loaded `.gguf` file.
- Designer/Electron migration to a hypothetical Go binary subprocess. They keep using the Python HTTP endpoint.
- Changes to `common/llamafarm_common/model_utils.py`. The Python side is untouched.

## Decisions

### Decision 1: New `cli/internal/hfmodel` package, not an extension of `hfcache`

`hfcache` is documented as "read-only access to the local HuggingFace Hub cache." Mixing write semantics into it would muddle that contract and force every read consumer to depend on a much larger surface (HTTP client, HF API types, retry/resume state). A separate package keeps the read path lean and lets `hfmodel` import `hfcache` for the read-side primitives it needs (cache root resolution, repo folder name, snapshot path computation).

**Alternatives considered:**
- *One mega-package.* Rejected: bad layering, surprising blast radius for any future change.
- *Inline the writer in `cli/cmd/models_pull.go`.* Rejected: untestable, can't be reused by future commands like `lf init` (which may want to pre-pull a default model).

### Decision 2: Pure Go HTTP client (`net/http`) — no third-party HF SDK

There is no canonical Go HF Hub SDK. The HF API surface we need is small (3 endpoints: tree listing, file metadata, file resolve) and stable. A pure-`net/http` implementation gives us full control over retries, Range headers, ETag re-verification, and error mapping without taking on a dependency we'd need to keep up to date.

**Alternatives considered:**
- *Community-maintained Go HF SDKs.* Rejected: low maintenance signal, would couple us to a third party for a small surface area.
- *cgo bindings to `huggingface_hub`.* Rejected: defeats the purpose (still needs Python at runtime), cross-compilation pain.

### Decision 3: Port `model_utils.py` selection logic, do not call out to Python

`parse_model_with_quantization`, `parse_quantization_from_filename`, `is_split_gguf_file`, `select_gguf_file`, and `GGUF_QUANTIZATION_PREFERENCE_ORDER` are pure logic, ~250 lines total, with existing unit tests. Porting them keeps the CLI self-contained and avoids the "sometimes invoke Python, sometimes don't" UX confusion.

**Mitigation for divergence risk:** the Go test suite uses table-driven tests with the same fixtures as `common/tests/test_model_utils*.py`. Any drift between the two implementations gets caught in CI on either side. If this proves unstable in practice, a follow-up change can extract a shared YAML fixture file that both Python and Go test suites consume.

**Alternatives considered:**
- *Subprocess `python -m llamafarm_common.select_gguf`.* Rejected: defeats the no-Python goal.
- *Re-derive selection logic from scratch.* Rejected: subtle behavioral drift on imatrix quants and split files would surface as user-visible "the wrong quant got picked" bugs.

### Decision 4: Atomic blob writes via `<etag>.tmp` → `os.Rename`

Mirrors what `huggingface_hub.file_download` does today and what the existing Python server-side downloader does (`stream_download_file` writes to `temp_path = blob_path.with_suffix(".tmp")` then `temp_path.rename(blob_path)`). Using the etag (rather than a random suffix) means a partial download is identifiable for resume purposes and won't be confused with a partial of a *different* version of the file.

### Decision 5: Resumable downloads with ETag re-verification

When `<etag>.tmp` exists at startup, send `Range: bytes=<size>-` with `If-Range: <etag>`. If the server responds with `206 Partial Content`, append. If it responds with `200 OK` (etag changed) or `416 Range Not Satisfiable`, delete the tmp file and start over. Refuse to silently merge bytes from a different version of the file — the resulting blob would have a hash mismatch and break Python loaders.

**Alternatives considered:**
- *No resume.* Rejected: model files are multi-GB, single network blip wastes hours. The current Python implementation has this defect; we should not perpetuate it.
- *Resume without ETag check.* Rejected: silent corruption is the worst possible failure mode here.

### Decision 6: Symlink-with-Windows-fallback matches `huggingface_hub._create_symlink` exactly

The reference implementation in `huggingface_hub/file_download.py` lines 596–693:
1. Compute a relative path from `dst_folder` to `abs_src` (the blob).
2. Probe whether the filesystem supports symlinks (`are_symlinks_supported(commonpath)`).
3. If yes: `os.symlink(relative_src, abs_dst)`. Use relative symlinks so the cache survives being moved, and because Windows handles relative symlinks better even in admin mode.
4. If no: `shutil.move(abs_src, abs_dst)` for new blobs (no wasted disk), `shutil.copyfile` for existing blobs (we don't know who else references them).

The Go port replicates step-for-step:
1. `filepath.Rel(dstFolder, absSrc)` for the relative target.
2. Probe symlink support by attempting a tiny test symlink in the target folder once per cache root, cache the result.
3. If supported: `os.Symlink(relativeSrc, absDst)`.
4. If not: `os.Rename(absSrc, absDst)` for new blobs (we just downloaded it, no other referrers), fall back to `io.Copy` for already-cached blobs.

The tiny test-symlink probe avoids the chicken-and-egg of "the cache directory might not exist yet" and matches `huggingface_hub`'s `are_symlinks_supported` cached lookup.

**Alternatives considered:**
- *Always symlink, fail on Windows without dev mode.* Rejected: Windows users would get an opaque "operation not permitted" error and blame us, not their OS.
- *Always copy, never symlink.* Rejected: a 7B model takes 14GB instead of 7GB on disk, halves cache hit rate.

### Decision 7: File locking via `flock` (unix) and `LockFileEx` (Windows), naming compatible with `huggingface_hub.filelock`

`huggingface_hub` uses the `filelock` Python package to coordinate concurrent downloads. Lock files are named `<blob_path>.lock` and held for the duration of the download. The Go port writes to the same `.lock` filenames using `golang.org/x/sys/unix.Flock(fd, LOCK_EX)` on unix and `LockFileEx` (via `golang.org/x/sys/windows`) on Windows. This means a `lf models pull` and a concurrent `huggingface_hub.snapshot_download` of the same model coordinate cleanly — whichever process arrives first holds the lock, the other waits.

**Alternatives considered:**
- *No locking, hope for the best.* Rejected: race between two downloaders writing the same blob ends in a corrupted file, not just one redundant download. Easy to hit when `lf models pull` runs alongside the runtime auto-downloading.
- *A separate Go-only lock file.* Rejected: doesn't coordinate with Python, defeats the purpose.

### Decision 8: HTTP contract is preserved verbatim

`POST /v1/models/download`, `GET /v1/models`, `POST /v1/models/validate-download`, and `GET /v1/models/{id}/quantizations` keep their current shapes. The CLI internally bypasses these endpoints; Designer and Electron continue calling them. No event-vocabulary churn for first-party HTTP consumers.

This means the CLI and the HTTP endpoint may drift in subtle ways (e.g., the CLI gets resumable downloads first). That is acceptable: Designer and Electron run on machines where Python is already a hard dependency, so they benefit less from a Go path. When the future server-delegation slice lands, both consumers will get the same behavior again.

### Decision 9: Progress event vocabulary stays identical to today's SSE events

The Go callback interface emits events with the same field names and `event` discriminator values that `stream_download_file` produces today: `init`, `start`, `progress`, `cached`, `end`, `done`, `error`, `warning`. This costs nothing at the Go layer and means the future server-side delegation slice can re-encode events as SSE without any field renaming.

### Decision 10: Slice ordering optimizes for risk reduction

Slice 0 (`lf models status` → Go) is a 30-line change to drop one function call. It validates that we can drop server-boot from a real command without breaking anything. Slice 1 (`lf models list` → Go) is similarly small. Slices 2–4 are the meat of the work: the HF API client, the GGUF selection port, the writer, and finally the multi-file downloader. Each slice is independently shippable; if Slice 3 reveals a layout-compat surprise, Slices 0 and 1 are already merged and delivering value.

## Risks / Trade-offs

- **HF cache layout drift** → Mitigated by an integration test in `cli/internal/hfmodel/integration_test.go` that downloads a tiny test model via Go, then shells out to a Python script invoking `huggingface_hub.try_to_load_from_cache(repo_id, "config.json")` and asserts the call returns a non-None path. The Python script lives in `cli/internal/hfmodel/testdata/` and is invoked only when a `LF_TEST_PYTHON_HF=1` env var is set, so devs without a Python env can still run unit tests. CI sets the env var.

- **GGUF selection regex divergence** → Mitigated by table-driven Go tests using fixtures copied verbatim from `common/tests/test_model_utils*.py`. Any new fixture added to either side must be added to both; a CI check across both languages catches drift. The Q variants (Q2_K, Q3_K_S/M/L, Q4_0/1, Q4_K_S/M, Q5_0/1, Q5_K_S/M, Q6_K, Q8_0), imatrix variants (IQ2_K, IQ3_K, IQ4_XS, etc.), F16/F32, and split-shard patterns (`-NNNNN-of-NNNNN`) are the high-risk surface.

- **Windows symlink fallback** → Mitigated by porting `_create_symlink` step-for-step (Decision 6). Tested on a Windows CI runner. The probe uses a tiny dummy file in the target folder so it works even when the symlink-permission state is per-directory.

- **File-lock incompatibility with `huggingface_hub.filelock`** → Mitigated by using the same `.lock` filename convention and OS-level advisory locks (`flock` / `LockFileEx`). Python's `filelock` package uses the same primitives under the hood, so the locks are bidirectionally honored. Verified in an integration test that spawns a Go writer and a Python writer pointing at the same blob, asserts they serialize.

- **HF token discovery order divergence** → Mitigated by direct port of `huggingface_hub.constants.HF_TOKEN_PATH` resolution and the `get_token` precedence: `$HF_TOKEN` → `$HUGGING_FACE_HUB_TOKEN` → contents of `$HF_HOME/token` (or `~/.cache/huggingface/token` if HF_HOME unset) → contents of `~/.huggingface/token` (legacy). Unit-tested with each env var set in isolation.

- **Resumable-download corruption from etag mismatch** → Mitigated by `If-Range: <etag>` semantics (Decision 5). The HTTP server enforces the constraint; we trust HTTP. Belt-and-suspenders: after a successful download (resumed or not), the blob's size must equal the metadata's `X-Linked-Size`/`Content-Length`. Mismatch deletes the blob and returns an error.

- **Network errors with poor messages** → Mitigated by structured error mapping: 401 → "run `huggingface-cli login` or set `HF_TOKEN`", 403 → "model `<id>` is gated; visit https://huggingface.co/<id> to accept terms", 404 → "model `<id>` not found; check the spelling" (with optional did-you-mean if a similar repo exists), 410/451 → "model removed or unavailable in your region", network/DNS → "cannot reach huggingface.co — check your connection or set `HF_HUB_OFFLINE=1` to use only the local cache".

- **Disk-space precheck not yet ported** → Accepted trade-off. The Go path emits the `warning` event slot (so the CLI rendering pipeline is ready) but does not actually compute free space. Slice 6 (future work) adds the precheck. In the meantime, a download that fills the disk fails on a write error, which is the same behavior as today's Python path when the precheck is skipped due to failure (`graceful degradation`).

- **Python download path drift over time** → Accepted: Designer and Electron will continue exercising the Python path, which means it can develop subtle behavioral differences from the Go path. Mitigation: we treat the Go path as the canonical implementation going forward, and any bug fixes land in Go first. The Python path is in maintenance mode until the future server-delegation slice retires it.

- **`LLAMAFARM_OFFLINE` semantics** → The Go path checks the same env var the Python `offline_mode` module honors, and refuses network calls when set. It does NOT propagate `HF_HUB_OFFLINE=1` to a child process (there is no Python child process here); it just acts on the var directly. This matches the user's mental model: "if I set `LLAMAFARM_OFFLINE=1`, nothing in `lf` reaches the network."
