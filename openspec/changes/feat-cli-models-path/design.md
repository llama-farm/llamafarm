## Context

LlamaFarm's deployment story (see in-progress `feat-deploy`) is centered on dev-host-to-server pushes: `lf bundle` ships binaries to an install target, and `lf deploy` triggers remote model downloads or pushes local cache contents over HTTP. That model breaks down for llamadrone / arc, where deployment is driven by ansible and Packer into an rpi-based air-gapped device image. In that pipeline:

- The build host has internet and runs ansible/packer.
- The target device has no internet at runtime.
- Multi-GB model files must be placed into the rpi image or pushed over SSH from the build host — not burned into a Docker image layer.
- The edge runtime Docker container on the device expects files on a mounted host filesystem path.

Today there is no supported way to ask LlamaFarm "for this project, which files do I need to ship and where should they go?" Ops engineers hand-trace the HuggingFace cache layout. There is also no way to pre-fetch the llama.cpp binary for a non-host platform — the edge container silently downloads it on first run, which fails entirely when the device is air-gapped.

Existing machinery we can reuse:

- `common/llamafarm_common/model_utils.py` already has `_get_cached_gguf_files`, `_get_cached_gguf_path`, `get_gguf_file_path`, `get_mmproj_file_path`. These are cache-first and handle quantization selection.
- `cli/cmd/models_pull.go` already wraps `lf models pull` with SSE progress rendering.
- `cli/cmd/orchestrator/llama_binary.go` already downloads llama.cpp binaries for the current host, with per-platform URL manifests, extraction, symlink handling, and dependency copying. It's Go-side, which is where the new CLI commands want to live.

## Goals / Non-Goals

**Goals:**

- Give deployment tooling (ansible, packer, Dockerfile builds) a single command that emits a complete source→target transport plan for a project's model files — no host-side copies.
- Give deployment tooling a single command that fetches the llama.cpp binary for an arbitrary target platform and optionally exports it to a flat directory for transport.
- Keep the build host clean: no duplicated multi-GB model files, no need to pre-materialize a flat layout.
- Reuse existing download logic from `cli/cmd/orchestrator/llama_binary.go` without duplicating it.
- Establish a canonical on-device model layout that runtime tooling (in a follow-up change) can rely on.

**Non-Goals:**

- Runtime-side offline mode, flat-directory loader, or `LLAMAFARM_OFFLINE` env var handling — deferred to `feat-runtime-offline`.
- Shipping the whole HuggingFace cache directory — explicitly rejected by the user. Only per-file paths are emitted.
- Full transformers / ultralytics model support in the initial `lf models path` implementation — GGUF is the priority. Non-GGUF models return an explicit not-yet-supported error.
- A "bundle models into a tar" workflow. The user wants individual file paths that ansible/packer can push; tarballs are downstream tooling's job.
- Changes to `runtimes/edge`, `runtimes/universal`, `runtimes/lemonade`, or any Python runtime code.
- Authentication / gated model handling beyond what `lf models pull` already provides via `HF_TOKEN`.

## Decisions

### Decision: Source field is the snapshot path, not the blob path

The HuggingFace cache stores file content in `models--{org}--{repo}/blobs/<sha>` (content-addressed) and exposes named symlinks in `models--{org}--{repo}/snapshots/<commit-hash>/<filename>`. `lf models path` emits the snapshot path as `source`, not the blob path.

**Rationale:**

1. Filenames matter to downstream tooling. `qwen3-1.7b-Q4_K_M.gguf` tells ansible/packer which file they're copying; `a3f9b21c…` does not.
2. HuggingFace cache garbage collection repacks blobs over time. Snapshot symlinks are stable across GCs; blob paths are not. A transport plan captured to a file should still resolve a week later.
3. When ansible copies a file, following the symlink (`copy:` does this by default) gives it the actual bytes. The symlink issue we talked about for whole-cache sync doesn't apply to per-file copies.

The command MUST still verify the file exists (via `os.path.realpath` / `filepath.EvalSymlinks`) before emitting the path, so stale dangling symlinks fail fast.

**Alternatives considered:**

- Emit blob paths (`…/blobs/<sha>`). Rejected because filenames are lost.
- Emit the `realpath` result. Rejected for the same reason, and because it's less stable across HF cache GCs.
- Emit both `source` and `source_real`. Rejected as unnecessary complexity — one canonical answer is simpler.

### Decision: Target-root resolution precedence

`lf models path` resolves `target_root` in this order, first match wins:

1. `--target-root <path>` CLI flag
2. `deployment.model_dir` from `llamafarm.yaml`
3. Hardcoded fallback: `/opt/llamafarm/models`

**Rationale:** The hardcoded fallback keeps llamadrone's ansible trivial (no flag, no config needed, the convention Just Works). The config field gives teams with nonstandard device layouts a per-project override without touching every playbook. The flag is the escape hatch for ad-hoc cases. This mirrors the three-tier pattern used by `lf deploy` for `server_url`.

**Alternatives considered:**

- Required flag, no default. Rejected — worse UX, forces boilerplate on every playbook.
- Default only, no config. Rejected — doesn't compose with projects that have nonstandard device layouts.
- `environments.<name>.model_dir` per-environment. Rejected for V1 as over-specified; can be added later without breaking the hierarchy.

### Decision: New top-level `deployment` section, not nested under `environments`

The new `model_dir` field lives under a new top-level `deployment` section, not inside the existing `environments` section.

**Rationale:** `environments` is a map of named deploy targets with per-target settings (`server_url`, `deploy_models`, `deploy_data`). `deployment.model_dir` is a single project-wide default for how to lay out files on any target, not a per-environment override. Nesting it under `environments` would either duplicate it or force an environment to be selected, neither of which fits.

This establishes a precedent: the new `deployment` section will be the home for future global deployment settings that are not per-environment (e.g., binary install paths, cache locations).

**Alternatives considered:**

- Nest under `environments.*.model_dir`. Rejected — it's not per-environment.
- Nest under `environments.defaults.model_dir`. Rejected — introduces a "defaults" sub-key that doesn't exist today; bigger schema change for no real benefit.
- Put at top level (`model_dir`). Rejected — pollutes top-level namespace, doesn't scope to deployment.

### Decision: Canonical target layout is `<target-root>/<alias>/<files>`, format-sniffed

The CLI emits `target` paths using this convention:

```
<target-root>/
├── manifest.json                      ← ansible/packer writes this; CLI does not
├── <alias>/
│   ├── model.<quant>.gguf             ← GGUF weights
│   ├── mmproj.<precision>.gguf        ← optional multimodal projector
│   ├── (or) config.json, tokenizer.json, *.safetensors
│   └── (or) *.pt
```

No `kind` metadata file. The `kind` field in JSON output is informational, derived by sniffing file extensions and content (`.gguf` → gguf, `config.json` + `*.safetensors` → transformers, `.pt`/`.pth` → ultralytics). This reuses existing format-detection helpers in `runtimes/edge/utils/model_format.py` (which needs to be accessible from Go — see Risks).

**Rationale:** Fewer files to place. Sniffing is what the runtime already does anyway, so there's no new format knowledge to maintain. The convention is "one directory per model alias, files named for their role".

**Canonical filenames:**

- GGUF weights: `model.<quant>.gguf` (e.g., `model.Q4_K_M.gguf`). Stripping the HF filename and normalizing lets two different upstream repos that both ship a Q4_K_M variant land at the same target path.
- GGUF mmproj: `mmproj.<precision>.gguf` (e.g., `mmproj.f16.gguf`).
- Transformers: files keep their HF names (`config.json`, `tokenizer.json`, etc.) because the transformers loader keys off those exact names.
- Ultralytics: files keep their HF names.

**Alternatives considered:**

- Preserve the original HF filename as the target. Rejected — creates inconsistent target paths that depend on the upstream repo's naming convention and makes ansible playbook authoring harder.
- Include the model's original HF repo ID in the path. Rejected — verbose, forces filesystem paths to mirror HF namespaces.
- Single flat directory, all files together. Rejected — collisions between models, and it obscures which files belong to which alias.

### Decision: Format sniffing lives in Go, mirrored from Python

`runtimes/edge/utils/model_format.py` (Python) has format-sniffing logic today. The new CLI command needs equivalent logic in Go. Instead of shelling out to Python or inventing a new IPC mechanism, we port the essential file-extension + magic-byte checks directly into a small Go package at `cli/internal/modelformat/`.

**Rationale:** Format sniffing is a pure function of a file path. The Python implementation is small enough to reimplement in Go without significant risk. Shelling out would add a Python runtime dependency to the CLI for a trivial operation.

**Alternatives considered:**

- Shell out to Python. Rejected — adds a runtime dependency the CLI otherwise doesn't have.
- Use cgo to call Python. Rejected — overkill.
- Defer sniffing entirely, require a `kind` field in `llamafarm.yaml`. Rejected — the user explicitly said sniffing is fine, and it keeps the config cleaner.

### Decision: sha256 is cached in a sidecar file next to the HF cache entry

The first time `lf models path --format json` is asked for a given file's sha256, it computes the digest and writes it to a sidecar file like `<snapshot-path>.sha256` (or to a CLI-owned cache directory, see Open Questions). Subsequent invocations read the sidecar instead of re-hashing.

**Rationale:** sha256 on a multi-GB GGUF file is a real cost on first run (~10s for a 4 GB file on a fast SSD). Caching makes re-invocation free. Using a sidecar keeps the metadata close to the data and immediately clear in debugging — a user looking at the HF cache can see that a sha256 has been computed.

**Sidecar invalidation:** The sidecar records the source file's `(size, mtime)` alongside the digest. If either changes, the sidecar is discarded and the digest is recomputed.

**Alternatives considered:**

- Always compute sha256. Rejected — expensive and wasteful on repeat invocations.
- Store in a CLI-managed cache directory (e.g., `~/.llamafarm/cache/sha256/`). Acceptable but less locally introspectable. See Open Questions.
- Use HuggingFace's own sha256 metadata (the blob filename IS the git-lfs sha256 for many files). Rejected — not all files are stored as git-lfs blobs, and the sha256 we want is of the file content as ansible will see it, not git-lfs internals. Safer to compute ourselves.

### Decision: `lf runtime binary pull` refactors the orchestrator, does not duplicate it

The existing `cli/cmd/orchestrator/llama_binary.go` has all the download logic: per-platform manifests, archive extraction, symlink preservation, dependency copying. We refactor it to expose a reusable `DownloadBinary(platform, accelerator, version, destDir)` function (or similar), and both the orchestrator and the new `lf runtime binary pull` command call into it.

**Rationale:** Single source of truth. Duplicating the download code would immediately diverge.

**Refactor shape:** Move the reusable bits from `cli/cmd/orchestrator/llama_binary.go` into a new package `cli/internal/llamabinary/` (or keep under `orchestrator` if that's cleaner for package dependencies). The orchestrator keeps its "download to the runtime's expected path" convenience wrapper; the new CLI command calls the lower-level function directly so it can target an arbitrary destination.

### Decision: Export materializes a flat directory, not a tarball

`lf runtime binary pull --export <dir>` writes files directly into `<dir>` in a flat layout. It does NOT produce a tar.gz.

**Rationale:** Ansible's `synchronize` / `copy` modules operate on directories. Packer's file provisioner operates on directories. A tarball would force the consumer to extract it before transport. If someone wants a tarball, they can `tar -czf` the exported dir themselves.

### Decision: `lf models path` is query-only; `--ensure` is the only mutation escape hatch

By default, `lf models path` exits non-zero if any selected model is not cached. It never silently downloads. The `--ensure` flag explicitly enables pull-before-path behavior.

**Rationale:** Surprises are bad. A CI pipeline that runs `lf models path` expects it to complete in milliseconds, not spend ten minutes pulling models over the network. The default is deterministic. `--ensure` is the one documented way to combine pull + path into a single invocation, useful for fresh build environments.

## Risks / Trade-offs

**Risk:** HF cache structure may change in future huggingface_hub versions.
**Mitigation:** We already depend on this structure via `common/llamafarm_common/model_utils.py`. Adding a second consumer in Go amplifies the coupling but doesn't create it. If HF changes the layout, both sites need to update together. Pin the `huggingface_hub` version in `common/pyproject.toml` and document the dependency in `design.md`.

**Risk:** sha256 sidecar files pollute the HF cache and might confuse `huggingface-cli scan` or similar tools.
**Mitigation:** Name sidecars with an unambiguous suffix like `.lf-sha256.json` and place them in a CLI-owned sibling directory if HF tooling turns out to complain. Revisit in an open question below.

**Risk:** Format sniffing in Go diverges from Python. A file the runtime would successfully load might be reported as "not yet supported" by the CLI, or vice versa.
**Mitigation:** The Go sniffer is deliberately minimal (extension + optional magic bytes for GGUF). The Python runtime's sniffer is the authority; if the Go sniffer says "gguf" it must be correct, but if it says "unknown" that's acceptable — the CLI can just skip the file and warn. For the initial GGUF-only scope, the sniffer only needs to identify `.gguf` reliably, which is trivial.

**Risk:** Canonical target filenames (`model.Q4_K_M.gguf`) differ from the upstream HF filenames, so ansible playbooks that hard-code HF filenames will break if they migrate to the new flow.
**Mitigation:** Not really a mitigation — this is intentional (see decision). Document the naming convention in the migration notes and in the command's help text.

**Risk:** `deployment.model_dir` in the config file bleeds into the server-pushed config and confuses the server.
**Mitigation:** Strip `deployment` before any `lf deploy` push, the same way `environments` is stripped today. This is called out in the environments delta spec.

**Risk:** Cross-platform `lf runtime binary pull` may not match what the orchestrator would have done on the target host, causing the edge runtime to fail loading.
**Mitigation:** The `lf runtime binary pull` download matrix is exactly the orchestrator's download matrix (shared code). The only way they can diverge is if the target host's CPU capabilities differ from what the platform flag implies (e.g., an ARM64 host without NEON). Accepted — this is an edge case that can be addressed with a finer-grained accelerator flag later.

**Risk:** `--ensure` is tempting to use in CI and will turn fast queries into slow downloads silently.
**Mitigation:** `--ensure` is opt-in, not the default. Document its cost in help text.

## Migration Plan

1. Land the CLI changes behind the new commands. Existing `lf models pull`, `lf deploy`, `lf bundle`, and the orchestrator's binary download path continue to work unchanged.
2. Add the `deployment` section to `config/schema.yaml` as an optional field. Existing configs without the section are unaffected.
3. Regenerate types with `nx run generate-types`.
4. Update the llamadrone / arc ansible playbooks to use `lf models path` and `lf runtime binary pull --export`. This is downstream work in a different repo, not part of this change.
5. Update edge runtime documentation to describe the `/opt/llamafarm/models` convention and show an example Dockerfile + ansible integration.
6. Follow-up change `feat-runtime-offline` adds runtime-side support for the flat layout and strict offline mode. Until it lands, the runtime continues to use HF cache lookups, which still works in the Docker container as long as the HF cache is mounted at the expected path.

No data migration. No config migration. Additive only.

## Open Questions

1. **sha256 sidecar location**: next to the HF cache file (`<file>.lf-sha256.json`) or in a CLI-owned cache dir (`~/.llamafarm/cache/sha256/`)? Leaning toward a CLI-owned dir so we don't pollute the HF cache namespace, but the trade-off is debuggability.

2. **Should `lf models path` support a `--manifest` mode** that emits only the target-side manifest.json that ansible will ultimately place on the device? Useful for pipelines that want to generate the manifest without the full plan. Can be added later without breaking the V1 shape.

3. **Should `lf runtime binary path` respect the same `deployment` config section** (e.g., `deployment.binary_dir`) for a "where should the binary go on the target" answer? Consistent, but the binary is typically small and its location is less varied than the model dir. Decision: defer to a follow-up unless llamadrone needs it.

4. **Transformers and ultralytics support** in `lf models path` — should we scope in the initial PR, or is the explicit "not yet supported" error good enough? Leaning toward explicit-error-only for V1 to keep scope tight. Confirm with the user before implementation begins.

5. **Alias naming for models** — is "alias" always the user-facing name from `runtime.models[].name` in `llamafarm.yaml`, or do we need a separate `deploy_name` field? Current assumption: always the name field. If naming collides between models in different namespaces, the user can rename one.
