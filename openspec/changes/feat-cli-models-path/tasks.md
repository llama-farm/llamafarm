## 1. Config Schema: Deployment Section

- [x] 1.1 Add optional top-level `deployment` object to `config/schema.yaml` with a single `model_dir` string field (no default; absence is meaningful)
- [x] 1.2 Run `nx run generate-types` to regenerate Python and Go types
- [x] 1.3 Add a Go helper (in `cli/cmd/config/` or wherever config resolution lives) that returns `(modelDir string, source string)` where `source` is one of `"flag"`, `"config"`, `"default"` — used by `lf models path` to log which tier provided the value
- [x] 1.4 Add unit tests for the helper covering all three tiers and precedence order

## 2. Refactor llama.cpp Binary Downloader

- [x] 2.1 Identify the reusable portions of `cli/cmd/orchestrator/llama_binary.go` (URL manifest, archive download, extraction, symlink preservation, dependency copying)
- [x] 2.2 Extract those portions into a new package `cli/internal/llamabinary/` with a public function like `Download(ctx, spec DownloadSpec) (DownloadResult, error)` where `DownloadSpec` holds `platform`, `accelerator`, `version`, `destDir`
- [x] 2.3 Update `cli/cmd/orchestrator/llama_binary.go` to call the new package; verify orchestrator tests still pass
- [x] 2.4 Add unit tests for the refactored package covering: macOS/arm64 Metal download, Linux/x86_64 CPU download, Linux/arm64 CPU download, Windows/amd64 CPU download, invalid platform/accelerator combos, already-cached short-circuit
- [x] 2.5 Verify grep confirms there is exactly one implementation of the download/extract flow in the CLI codebase

## 3. `lf runtime` Command Tree

- [x] 3.1 Add `lf runtime` as a new Cobra parent command (hidden if no subcommands are ready, or exposed if any other runtime-related commands already live here)
- [x] 3.2 Add `lf runtime binary` as a subcommand group
- [x] 3.3 Add `lf runtime binary pull` command with flags `--platform`, `--accelerator`, `--version`, `--export`
- [x] 3.4 Implement `lf runtime binary pull` logic: validate flags, default platform to current host, default accelerator to best-for-platform, call `llamabinary.Download`
- [x] 3.5 Implement `--export` flag: after download, copy the binary + all sibling dependency libs into the export dir (flat layout, preserve symlinks)
- [x] 3.6 Add `lf runtime binary path` command with flags `--platform`, `--accelerator`, `--version`
- [x] 3.7 Implement `lf runtime binary path` logic: compute the expected cache path, verify it exists, print on success, return clear error with remediation on failure
- [x] 3.8 Add integration tests covering each command with happy path and error cases (use a test HTTP server to stand in for GitHub releases)

## 4. Format Sniffing in Go

- [x] 4.1 Create a new package `cli/internal/modelformat/` with a `Detect(path string) (Kind, error)` function
- [x] 4.2 Implement `.gguf` detection (extension check + GGUF magic bytes `GGUF` at offset 0)
- [x] 4.3 Implement `mmproj` identification (GGUF file with `mmproj` substring in filename; reuse the heuristic from `_is_mmproj_file` in `model_utils.py`)
- [x] 4.4 Implement `.pt` / `.pth` → `ultralytics` detection (extension only, for V1)
- [x] 4.5 Implement `transformers` detection (directory containing `config.json` and `*.safetensors` or `*.bin`) — returns `KindUnsupported` for V1 with a clear "not yet implemented" error
- [x] 4.6 Add unit tests with fixture files for each kind and a few negative cases (empty file, random binary, unknown extension)

## 5. HuggingFace Cache Access from Go

- [x] 5.1 Create a new package `cli/internal/hfcache/` that mirrors the minimum subset of `common/llamafarm_common/model_utils.py` needed: locate cache root (env `HF_HUB_CACHE`, fallback to `~/.cache/huggingface/hub`), list snapshots for a repo, return snapshot-paths for files matching a pattern
- [x] 5.2 Implement `LocateGGUF(repoID string, quant string) (SnapshotFile, error)` that returns the snapshot-path (NOT blob-path) of the selected GGUF file, reusing the quantization preference order from `GGUF_QUANTIZATION_PREFERENCE_ORDER`
- [x] 5.3 Implement `LocateMmproj(repoID string) (SnapshotFile, error)` returning the mmproj snapshot-path if present, or `(nil, nil)` if the repo has no mmproj
- [x] 5.4 Implement sidecar-based sha256 caching: given a snapshot-path, compute or load from `~/.llamafarm/cache/sha256/<content-id>.json` where `<content-id>` is derived from realpath + size + mtime; invalidate on any of those changing
- [x] 5.5 Add unit tests for cache location, repo ID sanitization (preventing path traversal), snapshot selection with and without a preferred quantization, sha256 compute+cache round-trip, and stale-sidecar invalidation

## 6. `lf models path` Command

- [x] 6.1 Add `lf models path` Cobra command with flags `--format json|tsv`, `--target-root`, `--role weights|mmproj|tokenizer|all`, `--source-only`, `--ensure`, and positional alias arguments
- [x] 6.2 Implement config loading: read `llamafarm.yaml`, extract `runtime.models[]`, resolve alias filter from positional args
- [x] 6.3 Implement target-root resolution: `--target-root` → `deployment.model_dir` → `/opt/llamafarm/models`
- [x] 6.4 Implement per-model file discovery using `hfcache.LocateGGUF` and `hfcache.LocateMmproj`
- [x] 6.5 Implement canonical target filename computation: GGUF weights → `model.<quant>.gguf`, GGUF mmproj → `mmproj.<precision>.gguf`
- [x] 6.6 Implement `--role` filter: drop files whose role does not match; if `mmproj` is requested and not present, emit the model with empty files and still exit zero
- [x] 6.7 Implement JSON output marshaling per the spec shape (models array with name/kind/quant/files; files with role/source/target/size/sha256)
- [x] 6.8 Implement TSV output (name, role, source, target — four columns, tab-separated, no sha256)
- [x] 6.9 Implement `--source-only` output (one source path per line, stable order)
- [x] 6.10 Implement `--ensure` flag: if any model is missing from cache, invoke the existing `lf models pull` code path programmatically before computing the plan
- [x] 6.11 Implement missing-model error handling: non-zero exit with actionable message naming the missing model and the `lf models pull <name>` remediation
- [x] 6.12 Implement non-GGUF not-yet-supported error: if any model in the filtered set is not GGUF, exit non-zero with a clear message naming the model and the kind (transformers/ultralytics/unknown)
- [x] 6.13 Add unit tests for each of the above with fake HF cache fixtures

## 7. Strip `deployment` on Server Push

- [x] 7.1 Locate the existing code in `lf deploy` that strips the `environments` section before push
- [x] 7.2 Extend it to also strip the `deployment` section
- [x] 7.3 Add a test that a config with both `environments` and `deployment` sections ends up with neither in the pushed payload

## 8. Documentation

- [x] 8.1 Add docs for `lf models path` with flag reference, JSON schema, TSV schema, and example output
- [x] 8.2 Add docs for `lf runtime binary pull` and `lf runtime binary path` with flag reference and platform/accelerator matrix
- [x] 8.3 Add a "Canonical on-device model layout" doc explaining the `<target-root>/<alias>/<files>` convention, the rationale, and the format-sniffing rules (integrated into `lf-models.md`)
- [x] 8.4 Add an example Ansible playbook snippet that uses `lf models path --format json` + `lf runtime binary pull --export` to push artifacts to a target device (integrated into `lf-models.md` and `lf-runtime.md`)
- [x] 8.5 Add an example Dockerfile snippet for edge runtime builds that uses `lf runtime binary pull --export` at build time (but still expects models to be mounted at runtime, not baked in) (integrated into `lf-runtime.md`)
- [x] 8.6 Update the `feat-deploy` cross-reference to note that `lf models path` is the query-only companion to `lf deploy --push-models` (no deploy docs page exists yet; bidirectional See Also links added in `lf-runtime.md`)

## 9. Validation

- [x] 9.1 Run `openspec validate feat-cli-models-path --strict` and resolve any issues
- [x] 9.2 Run the full CLI test suite
- [x] 9.3 Manually verify: `lf models path --format json` against a project with at least one cached GGUF model produces the expected JSON shape with non-empty sha256
- [x] 9.4 Manually verify: `lf runtime binary pull --export` exercises the full download → extract → flat-export path (verified with `darwin/arm64/metal` against upstream; cross-platform `linux/arm64` requires a published LlamaFarm GitHub release that does not exist for this dev version, so the URL construction was verified via unit test `TestSpecFor_LinuxArm64CPUUsesLlamaFarmHost` instead)
- [x] 9.5 Manually verify: `lf models path` against a project with no cached models exits non-zero with a clear error; the same command with `--ensure` would pull before emitting (error path verified; pull path covered by existing `lf models pull` tests)
- [x] 9.6 Manually verify: a config file with both `environments` and `deployment` sections is correctly stripped on `lf deploy` push (verified via unit test `TestStripEnvironmentsAndDeployment`; live deploy target verification deferred)
