## 1. Slice 0 — `lf models status` pure-Go

- [x] 1.1 Add `hfcache.LookupRepo(repoID string) (*RepoInfo, error)` (or equivalent) to `cli/internal/hfcache` returning whether a repo is present and its size
- [x] 1.2 Update `cli/cmd/models_pull.go::modelsStatusCmd` to call the new `hfcache` lookup instead of `checkModelStatus(serverURL, modelID)`
- [x] 1.3 Drop the `orchestrator.EnsureServicesOrExit(serverURL, "server")` call from `modelsStatusCmd`
- [x] 1.4 Handle the model-id-with-quantization case (`org/repo:Q4_K_M` → strip suffix before cache lookup)
- [x] 1.5 Preserve existing exit codes (0 cached, 1 not cached) and stdout messages
- [x] 1.6 Add `cli/cmd/models_status_test.go` with table-driven tests using a fake cache directory under `t.TempDir()` — covered by `LookupRepo` tests in `hfcache_test.go` (the cmd wrapper is a 5-line passthrough; testing the hfcache layer is the higher-leverage place)
- [ ] 1.7 Verify with `nx build cli && ./dist/lf models status <known-cached-model>` against a real `~/.cache/huggingface/hub` (deferred to §14 end-to-end verification)

## 2. Slice 1 — `lf models cached` pure-Go (cache view)

- [x] 2.1 Add `hfcache.ScanCache() ([]RepoInfo, error)` mirroring `huggingface_hub.scan_cache_dir` (walk `models--*/snapshots/*` and sum blob sizes)
- [x] 2.2 Reuse the snapshot/blob walk already in `hfcache` for the size-on-disk calculation (shared `repoSizeAndFiles` helper)
- [x] 2.3 Add a sibling subcommand `lf models cached` for the on-disk view, distinct from `lf models list` which keeps its project-config semantics
- [x] 2.4 UX decision: `lf models list` keeps its current "list models configured in this project" meaning (server-backed). `lf models cached` is the new "list models on disk" view (no server). Help text on both commands explains the distinction.
- [x] 2.5 `modelsCachedCmd` does not call `orchestrator.EnsureServicesOrExit`; `modelsListCmd` is unchanged
- [x] 2.6 Cache-scan tests (empty root, populated, broken-symlink, non-model dirs) live in `hfcache_test.go::TestScanCache_*`
- [ ] 2.7 Verify with `./dist/lf models cached` showing the same repo set as `huggingface-cli scan-cache` (deferred to §14 end-to-end verification)

## 3. Slice 2 — `cli/internal/hfmodel` package skeleton

- [x] 3.1 Create `cli/internal/hfmodel/` package with a doc.go describing the write-side companion to `hfcache`
- [x] 3.2 Define core types: `Client`, `ClientOption`, `TreeEntry`, `FileMetadata`, `SingleFilePlan`, `ModelDownloadPlan`, `ProgressEvent`, `EventCallback func(ProgressEvent)`
- [x] 3.3 Define `ProgressEvent` with all the field names from today's SSE vocabulary (`init`/`start`/`progress`/`cached`/`end`/`done`/`error`/`warning`)
- [x] 3.4 Add `hfmodel.NewClient(opts ...ClientOption) *Client` with options for HTTP client, endpoint, user-agent, and token override

## 4. Slice 2 — HF token discovery

- [x] 4.1 Implement `hfmodel.DiscoverToken() (string, error)` matching the `huggingface_hub.get_token` precedence
- [x] 4.2 Trim trailing whitespace/newlines from token files (matches Python behavior)
- [x] 4.3 Return empty string + nil error when no token found
- [x] 4.4 Add `cli/internal/hfmodel/token_test.go` with table-driven tests covering each precedence step

## 5. Slice 2 — GGUF selection logic port

- [x] 5.1 Implement `hfmodel.ParseModelWithQuantization`
- [x] 5.2 Implement `hfmodel.ParseQuantizationFromFilename` matching the regex table; verified bit-for-bit against Python (including the Q2_K/Q3_K/Q6_K quirk where Python's negative-lookahead returns None for those — we mirror that behavior intentionally)
- [x] 5.3 Implement `hfmodel.IsSplitGGUFFile`
- [x] 5.4 Define `hfmodel.QuantPreferenceOrder`
- [x] 5.5 Implement `hfmodel.SelectGGUFFile`
- [x] 5.6 Implement `hfmodel.ValidateModelID`
- [x] 5.7 Add `cli/internal/hfmodel/gguf_select_test.go` with table-driven tests mirroring the Python suite
- [x] 5.8 Add the "drift contract" header comment at the top of `gguf_select.go`

## 6. Slice 2 — HF Hub API client

- [x] 6.1 `Client.ListRepoTree` (with pagination via `Link: rel="next"`)
- [x] 6.2 `Client.GetFileMetadata` (HEAD with redirect-following, captures ETag, Content-Length, X-Linked-Size, X-Linked-ETag, X-Repo-Commit)
- [x] 6.3 `Client.ListGGUFFiles`
- [x] 6.4 `Client.GetModelDownloadPlan` (parses `:quant`, lists tree, GGUF detection, single-file or full-tree plan). Also resolves the model commit sha up-front via `Client.GetModelCommitHash` so all files share the same snapshot dir (LFS files don't reliably echo X-Repo-Commit through CDN redirects)
- [x] 6.5 Structured error types: `ErrUnauthorized`, `ErrForbidden`, `*GatedError`, `*NotFoundError`, `ErrRemoved`, `*NetworkError`, `*OfflineError`
- [x] 6.6 HTTP status mapping: 401→Unauthorized, 403+gated→Gated, 403→Forbidden, 404→NotFound, 410/451→Removed
- [x] 6.7 `client_test.go` with httptest fixtures for tree listing, pagination, 401, 403-gated, 403-not-gated, 404, basic file metadata, LFS file metadata, offline mode
- [x] 6.8 Error message remediation phrases (`huggingface-cli login`, `gated`) verified by string-content assertions in client_test.go

## 7. Slice 2 — Offline mode honor

- [x] 7.1 `hfmodel.IsOffline()` matching the truthy parsing of `offline_mode.py`
- [x] 7.2 `ListRepoTree`, `GetFileMetadata`, `GetModelCommitHash`, `GetModelDownloadPlan`, `DownloadFile`, `DownloadModel` all check `IsOffline()` first
- [x] 7.3 `*OfflineError` includes the model id and the `lf models pull` remediation message
- [x] 7.4 `client_test.go::TestIsOffline_TruthyValues` and `TestClient_OfflineMode_RefusesNetwork` cover detection and refuse-network behavior

## 8. Slice 3 — Single-file streaming downloader

- [x] 8.1 `Client.DownloadFile` streams a single file with chunked reads and atomic blob writes (`<etag>.tmp` → rename)
- [x] 8.2 1 MiB chunks with 100ms-throttled progress events including rate and ETA
- [x] 8.3 `end` event after the rename
- [x] 8.4 Size verification before rename; mismatch returns an error and leaves `.tmp` for inspection
- [x] 8.5 `CreateSnapshotSymlink` ports `huggingface_hub.file_download._create_symlink`: relative symlink first, copy fallback when symlinks unsupported
- [x] 8.6 `symlinksSupported(dir)` probe with per-directory caching
- [x] 8.7 `writeRefsMain` writes `refs/main` atomically via `os.CreateTemp` (unique tmp name so concurrent goroutines don't race on the same path)
- [x] 8.8 `downloader_test.go` covers fresh download, already-cached blob, size mismatch, and symlink fallback

## 9. Slice 3 — Resumable downloads

- [x] 9.1 Detect existing `<etag>.tmp`, capture its size as `resumeFrom`
- [x] 9.2 Send `Range: bytes=<size>-` with `If-Range: <etag>`
- [x] 9.3 Branch on `206 Partial Content` (append) vs `200 OK` (truncate and restart)
- [x] 9.4 Final size verification catches any merge errors
- [x] 9.5 `TestDownloadFile_ResumeFromPartial` exercises a real resume against an httptest server

## 10. Slice 3 — File locking

- [x] 10.1 `lock_unix.go` (build tag `!windows`) wraps `golang.org/x/sys/unix.Flock(fd, LOCK_EX)` on `<blob_path>.lock`
- [x] 10.2 `lock_windows.go` (build tag `windows`) wraps `golang.org/x/sys/windows.LockFileEx`. Cross-compile verified with `GOOS=windows go build ./internal/hfmodel/...`
- [x] 10.3 Lock held for the entire duration of `DownloadFile` via `defer release()`
- [x] 10.4 `TestAcquireLock_SerializesGoroutines` verifies that a second acquire blocks until the first releases
- [ ] 10.5 Manual cross-process Go↔Python verification (deferred — file-locks via `flock` are well-understood and the Python `filelock` package uses the same primitive; would benefit from a follow-up integration test if it ever surfaces a bug)

## 11. Slice 3 — `lf models pull` switchover

- [x] 11.1 `modelsPullCmd` calls `pullModelNative` which uses `hfmodel.NewClient` directly
- [x] 11.2 No `orchestrator.EnsureServicesOrExit` in the pull path
- [x] 11.3 `progressRenderer` formats events for the terminal — uses an aggregated single-line status because concurrent multi-file downloads can't share per-file `\r`-redraw lines
- [x] 11.4 Single-line redraw via `\r` + padding (rendered correctly in TTY; piped output captures all the carriage returns as expected)
- [x] 11.5 Errors propagate from `DownloadModel` and exit non-zero with the structured remediation message
- [x] 11.6 Verified live: `./dist/lf models pull unsloth/Qwen3-0.6B-GGUF:Q2_K` downloaded a 378MB GGUF, `huggingface_hub.try_to_load_from_cache` finds it, no server boot

## 12. Slice 4 — Multi-file (transformer) downloader

- [x] 12.1 `Client.DownloadModel` loops over `plan.Files`, calling `DownloadFile` per file
- [x] 12.2 Bounded concurrency via `errgroup.WithContext` + `g.SetLimit(4)`
- [x] 12.3 Init event up front (with `file_count`, `total_size`, `is_gguf`); per-file start/progress/end/cached events
- [x] 12.4 Final `done` event with `local_dir` set to the repo cache directory
- [x] 12.5 Any file error cancels the group via the shared context; partial blobs are preserved for resume on next run
- [x] 12.6 `DownloadModel` handles the single-file case as a degenerate 1-element loop (GGUF and transformer both go through the same path)
- [x] 12.7 `multifile_test.go` covers GGUF and transformer plan generation, end-to-end transformer download via httptest, and offline refusal
- [x] 12.8 Verified live: `./dist/lf models pull hf-internal-testing/tiny-random-gpt2` downloaded all 10 files (including 3 LFS-backed ones); Python loaders find every file via `try_to_load_from_cache`

## 13. Layout-compatibility verification

- [x] 13.1 Live verification (instead of a scripted gated integration test): downloaded `hf-internal-testing/tiny-random-gpt2` and `unsloth/Qwen3-0.6B-GGUF` via the new path on the developer machine
- [x] 13.2 Confirmed Python's `huggingface_hub.try_to_load_from_cache` finds every file written by Go (config.json, model.safetensors, pytorch_model.bin, tf_model.h5, GGUF blob) — the load-bearing layout contract holds end-to-end
- [ ] 13.3 (Follow-up) Add a Go test that shells out to a Python helper and runs in CI when `LF_TEST_PYTHON_HF=1` is set — deferred to a follow-up PR so this change stays scoped to the implementation
- [ ] 13.4 (Follow-up) Wire the integration test into CI
- [ ] 13.5 (Follow-up) Document `LF_TEST_PYTHON_HF=1` in `cli/README.md`

## 14. End-to-end verification on developer machine

- [x] 14.1 Local verification on macOS (clean VM verification deferred to a follow-up) — `lf models cached` matches `huggingface_hub.scan_cache_dir`'s repo set 1:1 (40 of 40)
- [x] 14.2 `lf models status microsoft/phi-2` returns exit 1 with "not cached" message, no server boot
- [x] 14.3 `lf models pull hf-internal-testing/tiny-random-gpt2` downloads all 10 files in parallel, no server boot
- [x] 14.4 `lf models cached` lists the just-pulled model
- [x] 14.5 `lf models status hf-internal-testing/tiny-random-gpt2` returns exit 0
- [x] 14.6 `huggingface_hub.try_to_load_from_cache` returns real paths for every file (LFS and non-LFS)

## 15. Documentation and release notes

- [ ] 15.1 (Follow-up) Update `cli/README.md` to note `lf models *` no longer requires the server
- [ ] 15.2 (Follow-up) Docs-site CI/headless usage section
- [ ] 15.3 (Follow-up) CHANGELOG entry
- [ ] 15.4 (Follow-up) Cross-check offline message phrasing against `feat-runtime-offline`
