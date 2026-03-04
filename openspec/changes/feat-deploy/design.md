## Context

LlamaFarm is currently localhost-only. The CLI, server, and config all assume a shared filesystem at `~/.llamafarm/`. The `--server-url` flag exists on the root command but nothing uses it for remote operations — no config push, no remote model management, no remote install.

The deployment story spans three phases: bundling/installing LlamaFarm on remote machines (Phase 1), pushing project configs and triggering model downloads (Phase 2), and pushing models/data directly for air-gapped scenarios (Phase 3).

### Existing Infrastructure

| Component | What exists | Reusable? |
|---|---|---|
| CLI binary releases | `release-cli.yml` — 5 platform variants on GitHub Releases | Yes, `lf bundle` downloads these |
| PyApp binaries | `pyapp.yml` — self-contained server/rag/runtime executables | Yes, included in bundles |
| Addon wheel bundles | `build-addon-wheels.yml` — per-platform tar.gz on Releases | Yes, included in bundles |
| Torch install | `hardware_wheels.go` — detects GPU, installs correct torch via `uv pip` | Yes, offline mode needs adaptation |
| `install.sh` | Downloads CLI binary from GitHub Releases | Extended for full install |
| `PUT /{ns}/{proj}` | Full-replacement config update | Yes, for config push |
| `POST /{ns}` | Create project | Yes, for upsert flow |
| `POST /v1/models/download` | SSE-streaming model download from HuggingFace | Yes, for remote model pull |
| `POST /v1/models/validate-download` | Disk space pre-check | Yes, for deploy pre-flight |
| `GET /health` | Health summary | Yes, for deploy pre-flight |
| `lf models pull` SSE renderer | Progress bars with rate + ETA | Refactored for multi-model parallel display |
| `POST /v1/datasets/{id}/files/bulk` | Bulk file upload to dataset | Yes, for data push (Phase 3) |
| Addon registry + installer | YAML registry, wheel download, PYTHONPATH injection | Extended for offline/bundle mode |

## Goals / Non-Goals

**Goals:**
- Install LlamaFarm on a remote machine with a single command (online or offline)
- Deploy a project config from a developer machine to a remote LlamaFarm server
- Trigger model downloads on the remote server after config push
- Support air-gapped environments via bundled archives and model push
- Support dataset upload and ingestion as part of deploy
- Keep the local `lf deploy` (no environment) useful as an explicit "sync config + pull models" for localhost

**Non-Goals:**
- Authentication or API keys for remote server access
- Ongoing file sync / watcher-based remote sync
- Multi-server fleet deployment
- Rollback or versioned deployments
- Remote service lifecycle management (start/stop/restart services)
- Source mode support (binary/PyApp mode only)

## Decisions

### D1: Bundle fetches release artifacts via GitHub API

`lf bundle` downloads pre-built binaries from GitHub Releases rather than building from source. The CLI, PyApp, and addon wheel workflows already produce all needed artifacts.

**Alternatives considered:**
- Build from source locally — too slow, requires full toolchain (Go, Rust/PyApp, Python)
- Require CI to produce bundles — removes user agency; users can build their own CI pipeline if needed

**Implication:** `lf bundle` requires internet access on the machine creating the bundle. The bundle itself is then transferable to air-gapped machines.

### D2: Torch wheels bundled separately from PyApp

The PyApp runtime binary ships with CPU-only torch. The bundle includes accelerator-specific torch wheels in a `torch/` directory. The installer handles the torch upgrade after PyApp extraction.

**Rationale:** This matches the existing two-step install flow (`hardware_wheels.go` already does this). The alternative of building accelerator-specific PyApp variants would multiply CI build time and storage.

**Offline adaptation:** `InstallHardwarePackages()` currently downloads wheels via `uv pip install --index-url`. For offline mode, it needs a local-path mode: `uv pip install torch --find-links /path/to/torch/` using the bundled wheels.

### D3: `lf deploy` upserts via CLI-side create-then-update

Rather than adding a new server upsert endpoint, `lf deploy` tries `PUT /{ns}/{proj}` first. If 404, it `POST /{ns}` to create, then `PUT` to set the full config.

**Alternatives considered:**
- New `PUT-or-create` server endpoint — adds server changes when the CLI can handle it
- Always create first — would fail if project exists

**Rationale:** Minimal server changes. The two-call approach handles both fresh deploys and updates.

### D4: Environments config lives in `llamafarm.yaml`, stripped before push

The `environments` section is a new top-level key in the project config. Before pushing to the server, the CLI strips this section — it's local-only metadata.

**Alternatives considered:**
- Separate `environments.yaml` file — cleaner separation, but adds file management burden
- Environment variables or CLI config — less discoverable, harder to share

**Rationale:** One file to manage. Team members can customize environments and optionally gitignore the section if needed. Stripping before push is trivial.

### D5: Parallel model downloads with pre-flight disk space check

`lf deploy` kicks off all model downloads concurrently and renders per-model + overall progress. Before downloading, it calls `POST /v1/models/validate-download` for each model to check disk space.

**Alternatives considered:**
- Sequential downloads — safer for disk space but much slower
- No pre-flight check — risks partial deploys that fail mid-download

**Rationale:** Models are independent; parallel is faster. Pre-flight catches disk issues before any downloads start. If total space is insufficient, the CLI warns and prompts before proceeding.

### D6: Resumable uploads for model push (Phase 3)

The model upload endpoint uses a resumable protocol (tus-style or chunked upload with server-tracked offset). Clients can resume interrupted uploads without re-sending completed data.

**Alternatives considered:**
- Simple streaming POST — simpler but a 10 GB upload failure means starting over
- rsync/scp — reliable but requires SSH access, not HTTP-native

**Rationale:** Model files are frequently 5-20 GB. Network interruptions on air-gapped network bridges or VPN tunnels are common. Resumable uploads prevent wasted bandwidth and time.

### D7: Dataset deploy uploads docs and triggers ingestion, not ChromaDB

`--with-data` uploads the raw document files to the server and triggers the RAG ingestion pipeline. It does not attempt to transfer ChromaDB vector stores.

**Alternatives considered:**
- Sync ChromaDB directly — fragile, version-sensitive, large
- Skip data entirely — leaves the deploy incomplete for RAG projects

**Rationale:** The server already has ingestion infrastructure. Uploading source docs and re-ingesting is idempotent, version-safe, and uses existing endpoints.

### D8: `install.sh` accepts archive path for single-step offline install

`./install.sh <archive.tar.gz>` detects offline mode, extracts, and installs everything from the archive. No manual untarring needed.

**Alternatives considered:**
- Separate `install-offline.sh` — confusing having two scripts
- Require manual extraction — extra step, error-prone

**Rationale:** One script, two modes. The archive path argument is the mode switch. Simple for users.

## Risks / Trade-offs

**[GitHub API rate limits]** → `lf bundle` downloads from GitHub Releases. Unauthenticated API is rate-limited to 60 requests/hour. Mitigation: support `GITHUB_TOKEN` env var for authenticated requests (5000/hour). Bundle creation is infrequent enough that this is unlikely to be hit in practice.

**[Large bundle sizes]** → A full bundle with CUDA torch + addons could be 3-5 GB. Mitigation: addons are optional (`--addons` flag). Users only bundle what they need. Torch is the largest component and is required for GPU acceleration.

**[PyApp internal venv for torch swap]** → PyApp manages its own internal Python venv. Swapping torch wheels inside it requires understanding PyApp's internal paths (`~/.local/share/pyapp/` or similar). Mitigation: the existing `InstallHardwarePackages()` already handles this — we extend it for offline mode, not replace it.

**[Partial deploy failures]** → If config pushes but model downloads fail, the server has a config referencing missing models. Mitigation: models are lazily loaded on first inference request. A missing model produces a clear error at inference time, not at config push time. The user can re-run `lf deploy` to retry.

**[No auth on remote server]** → Phase 1-3 assume trusted network. Anyone with HTTP access to the server can push configs and models. Mitigation: explicitly out of scope. Document that users should use VPN/firewall for remote servers until auth is added.
