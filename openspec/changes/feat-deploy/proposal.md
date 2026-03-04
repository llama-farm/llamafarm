# Proposal: LlamaFarm Deployment

## Summary

Add deployment capabilities to LlamaFarm: bundling for distribution, deploying projects to remote servers, and supporting air-gapped environments. This enables teams to run LlamaFarm on remote infrastructure beyond the current localhost-only workflow.

## Motivation

Today, LlamaFarm is entirely local. The CLI starts services on localhost, the config watcher syncs files on the local filesystem, and there's no concept of deploying to a remote machine. Users who want to run LlamaFarm on a remote server (GPU box, on-prem, air-gapped) have no supported path — they'd need to manually install, copy configs, and pull models.

## Phases

### Phase 1: Bundle + Install

**Goal:** Get LlamaFarm onto a remote machine with a single command, whether the machine has internet or not.

#### `lf bundle`

Packages all LlamaFarm components into a single distributable archive. Runs on the developer's machine (or CI if the user sets that up themselves).

```bash
lf bundle \
  --platform linux \
  --arch x86_64 \
  --accelerator cuda \
  --addons stt,tts \
  -o llamafarm-bundle-linux-x86_64-cuda.tar.gz
```

**Flags:**
- `--platform` — target OS: `linux`, `darwin`, `windows`
- `--arch` — target architecture: `x86_64`, `arm64`
- `--accelerator` — compute backend: `cuda`, `rocm`, `vulkan`, `cpu`, `metal`
- `--addons` — comma-separated addon names to include
- `-o` / `--output` — output file path
- `--version` — LlamaFarm version to bundle (default: current CLI version)

**Bundle contents:**

```
llamafarm-bundle-linux-x86_64-cuda.tar.gz
├── lf-linux-amd64                          # CLI binary
├── llamafarm-server-linux-x86_64           # PyApp binary (accel-agnostic)
├── llamafarm-rag-linux-x86_64              # PyApp binary (accel-agnostic)
├── llamafarm-runtime-linux-x86_64          # PyApp binary (ships CPU-only torch)
├── torch/                                  # Accelerator-specific torch wheels
│   └── torch-*.whl                         # e.g., CUDA-enabled torch
├── addons/                                 # Optional, if --addons specified
│   ├── stt-wheels-linux-x86_64.tar.gz
│   └── tts-wheels-linux-x86_64.tar.gz
├── addons-registry/                        # Addon YAML definitions
│   ├── stt.yaml
│   └── tts.yaml
├── install.sh                              # Offline-aware installer
└── manifest.json                           # Bundle metadata
```

**`manifest.json` example:**

```json
{
  "version": "0.8.0",
  "platform": "linux",
  "arch": "x86_64",
  "accelerator": "cuda",
  "components": {
    "cli": "lf-linux-amd64",
    "server": "llamafarm-server-linux-x86_64",
    "rag": "llamafarm-rag-linux-x86_64",
    "runtime": "llamafarm-runtime-linux-x86_64-cuda"
  },
  "addons": ["stt", "tts"]
}
```

**Source of binaries:** Downloaded from GitHub Releases for the specified version. The CLI release workflow, PyApp workflow, and addon wheel workflow already produce all needed artifacts.

#### `install.sh` (enhanced)

The existing `install.sh` is extended to support two modes:

**Online mode** (current behavior, extended):
```bash
curl -fsSL https://llamafarm.com/install.sh | bash
```
- Downloads CLI binary (already works)
- Downloads PyApp service binaries (new)
- Runs `lf addons install <names>` for requested addons (new, optional)

**Offline mode** (new):
```bash
./install.sh llamafarm-bundle-linux-x86_64-cuda.tar.gz
```
- Reads `manifest.json` from archive
- Installs CLI binary to `/usr/local/bin/` (or `--install-dir`)
- Installs PyApp binaries to `~/.llamafarm/bin/`
- Installs addon wheels to `~/.llamafarm/addons/`
- Updates `~/.llamafarm/addons.json` state
- Verifies installation (`lf version`, service health)

**Initial target:** Linux. Design for Mac/Windows portability from the start.

---

### Phase 2: Simple Deploy

**Goal:** Push a project config to a remote server and trigger model downloads.

#### Config: `environments` section

A new top-level section in `llamafarm.yaml`:

```yaml
environments:
  staging:
    server_url: http://10.0.1.50:14345
    deploy_models: true       # trigger model pull on deploy (default: true)
    deploy_data: false        # upload + ingest dataset docs (default: false)
  production:
    server_url: http://10.0.1.100:14345
    deploy_models: true
    deploy_data: true
```

The `environments` section is stripped from the config before pushing to the server — it's local-only metadata about deploy targets.

#### `lf deploy`

```bash
lf deploy <environment>            # deploy to named environment
lf deploy --server-url <url>       # ad-hoc deploy to any server
```

**Flow:**

```
lf deploy staging
    │
    ├─ 1. Health check: GET /health on target server
    │     Fail fast with clear error if unreachable
    │
    ├─ 2. Upsert project config
    │     Try PUT /{ns}/{proj} (update)
    │     If 404 → POST /{ns} (create), then PUT
    │     Config is stripped of `environments` before sending
    │
    ├─ 3. Pre-check model disk space
    │     POST /v1/models/validate-download for each model
    │     Warn/abort if insufficient space
    │
    ├─ 4. Trigger parallel model downloads
    │     POST /v1/models/download for each model in runtime.models[]
    │     Concurrent SSE streams with per-model + overall progress
    │     Skip models already cached on server (server returns "cached" status)
    │
    └─ 5. Summary
          Report: config pushed, N models downloaded, any warnings
```

**Override flags:**
- `--with-data` — override `deploy_data` to true for this run
- `--skip-models` — skip model download step
- `--dry-run` — show what would happen without executing

**Server changes needed:**
- Upsert logic: either a new endpoint or CLI-side create-then-update fallback
- No other server changes — existing endpoints cover config push and model download

**CLI reuse:**
- Model download SSE rendering already exists in `lf models pull`
- Server URL flag already exists on root command

---

### Phase 3: Full Deploy (Air-gapped + Data)

**Goal:** Support deploying to servers without internet access, including pushing models and datasets.

#### `lf deploy --push-models`

Instead of triggering the server to pull models from HuggingFace, push model files directly from the developer's local HF cache.

```bash
lf deploy production --push-models
```

**Flow:**
1. For each model in `runtime.models[]`:
   - Locate model files in local `~/.cache/huggingface/hub/`
   - Stream upload to a new server endpoint
   - Server writes blobs into its own HF cache structure

**New server endpoint:**
- `POST /v1/models/upload` — accepts resumable streamed model blob(s), places in HF cache with correct `blobs/`, `snapshots/`, `refs/` structure. Must support resumable uploads for large (10+ GB) model files.

#### `lf deploy --with-data`

Upload dataset documents and trigger ingestion on the remote server.

```bash
lf deploy production --with-data
```

**Flow:**
1. Read dataset configs from `llamafarm.yaml`
2. For each dataset with local document files:
   - Upload files via `POST /v1/datasets/{id}/files/bulk` (endpoint already exists)
   - Trigger ingestion on the remote server
3. Show ingestion progress

**Also configurable per-environment:**
```yaml
environments:
  production:
    server_url: http://10.0.1.100:14345
    deploy_models: true
    deploy_data: true          # auto-upload + ingest datasets on deploy
```

#### Air-gapped Deploy (combining Phase 1 bundle + Phase 3 push)

The full air-gapped workflow:

```
INTERNET MACHINE                    AIR-GAPPED SERVER
┌──────────────────┐                ┌──────────────────┐
│                  │   USB/scp      │                  │
│ lf bundle ...    │──────────────► │ ./install.sh     │
│   → archive.gz   │   (one-time)   │   archive.gz     │
│                  │                │                  │
│ lf deploy prod   │   HTTP         │ Server running   │
│ --push-models    │──────────────► │ Receives config  │
│ --with-data      │   (network)    │ + models + data  │
└──────────────────┘                └──────────────────┘
```

---

## Out of Scope

- Authentication / API keys for remote servers
- Ongoing config sync (watcher-style) to remote servers
- Multi-server orchestration (deploy to fleet)
- Rollback / versioned deployments
- Remote service lifecycle management (start/stop services on remote)

## Resolved Questions

1. **Accelerator-specific PyApp runtime binaries** — PyApp runtime ships with CPU-only torch. The CLI separately installs the accelerator-specific torch (cuda, rocm, vulkan, metal). The bundle must include accelerator-specific torch wheels, and `install.sh` must handle the torch upgrade step after installing the PyApp binary.

2. **Model upload chunking** — Resumable uploads are required for the model push endpoint (Phase 3). Large GGUF files can be 10+ GB.

3. **Environment defaults** — No explicit default environment. `lf deploy` with no arguments deploys to the local server (`localhost:14345`, the existing `--server-url` default). Named environments are only for remote targets.
