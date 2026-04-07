## Why

Deployment tooling (Ansible, Packer, Dockerfile builds) has no supported way to ask LlamaFarm "which model files do I need to ship, and where should they go on the target?" Today, anyone building an offline or edge image (llamadrone / arc being the concrete case) either hand-traces the HuggingFace cache directory structure or invents brittle glue. We also have no CLI-level way to pre-fetch the llama.cpp binary for a non-host platform, so the edge Docker container silently downloads it on first run — which fails entirely in air-gapped environments.

## What Changes

- Add `lf models path` command — query-only. Reads `llamafarm.yaml`, locates each model's files in the local HuggingFace cache, and emits a transport plan (source → target mapping) as JSON or TSV. Supports filtering by model alias, file role (weights, mmproj, tokenizer), and an `--ensure` flag that pulls missing models first.
- Add `lf runtime binary pull` command — downloads the pinned llama.cpp binary and its dependency libraries for a specified target platform and accelerator, into the local cache. Supports `--export <dir>` to materialize a flat directory of binaries + libs for downstream tooling (copies are acceptable at ~50–200 MB; this is not a model-file concern).
- Add `lf runtime binary path` command — prints the cached binary path for a given platform/accelerator. Query-only.
- Refactor `cli/cmd/orchestrator/llama_binary.go` so its download logic can be reused by the new CLI commands without duplication.
- Extend the LlamaFarm config schema with a new optional top-level `deployment` section containing `model_dir`, which is read by `lf models path` as the default target root. Falls back to the hardcoded `/opt/llamafarm/models` when unset. `--target-root` flag overrides both.
- Document a canonical on-device model layout (`<target-root>/<alias>/<files>`) that the CLI emits as `target` paths but does not create on the host.

## Capabilities

### New Capabilities
- `models-path`: Query-only CLI commands for locating model files in the local HuggingFace cache and emitting transport plans (source → target mappings) that deployment tooling can consume.
- `runtime-binary`: CLI commands for fetching and locating llama.cpp binaries for arbitrary target platforms and accelerators, with optional flat-directory export for downstream transport.

### Modified Capabilities
- `environments`: Extend the config schema with a new top-level `deployment` section (`deployment.model_dir`) that lives alongside the existing `environments` section and provides the default target root for `lf models path`.

## Impact

- **CLI (Go)**: New Cobra commands under `lf models` and `lf runtime binary`. Refactor of `cli/cmd/orchestrator/llama_binary.go` to expose its download logic for reuse. New config helper to resolve `deployment.model_dir`.
- **Config schema (`config/schema.yaml`)**: New optional `deployment` section with a `model_dir` string field. Triggers `nx run generate-types` to regenerate Python and Go types.
- **No runtime changes**: `runtimes/edge` and `runtimes/universal` are untouched. Strict offline mode and flat-directory loading are deliberately deferred to a follow-up change (`feat-runtime-offline`).
- **Downstream consumers**: llamadrone / arc deployment playbooks can adopt these commands immediately to eliminate hand-traced HF cache paths. The edge runtime Dockerfile can use `lf runtime binary pull --export` at build time to pre-fetch binaries without burning model files into the image.
- **Documentation**: New docs for the canonical on-device layout convention, example Ansible playbook snippets, example Dockerfile integration.
