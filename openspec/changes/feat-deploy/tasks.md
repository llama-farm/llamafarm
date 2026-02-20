## 1. Config Schema: Environments

- [x] 1.1 Add `environments` schema to `config/schema.yaml` with `server_url` (required), `deploy_models` (default true), `deploy_data` (default false) per named environment
- [x] 1.2 Run `nx run generate-types` to regenerate Python and Go types from updated schema
- [x] 1.3 Add config helper in CLI to resolve environment by name — reads `environments` from loaded config, returns `server_url` and deploy settings

## 2. Phase 1: Bundle Command

- [x] 2.1 Add `lf bundle` Cobra command with `--platform`, `--arch`, `--accelerator`, `--addons`, `--version`, `-o` flags
- [x] 2.2 Implement GitHub Release artifact downloader — fetch CLI binary, PyApp binaries, addon wheel archives for specified platform/version; support `GITHUB_TOKEN` for auth
- [x] 2.3 Implement torch wheel downloader — fetch accelerator-specific torch wheels from PyTorch index URLs (reuse `PyTorchSpec` mappings from `hardware_wheels.go`)
- [x] 2.4 Implement bundle packager — assemble downloaded artifacts into tar.gz with `manifest.json` and embedded `install.sh`
- [x] 2.5 Add manifest.json generation with version, platform, arch, accelerator, components map, and addons list
- [x] 2.6 Validate platform/arch/accelerator combinations against known release matrix; error on invalid combos

## 3. Phase 1: Install Script

- [x] 3.1 Extend `install.sh` to accept an archive path argument and detect offline vs online mode
- [x] 3.2 Implement offline mode: extract archive, read `manifest.json`, install CLI binary to install dir
- [x] 3.3 Add PyApp binary installation to `~/.llamafarm/bin/` in both online and offline modes
- [x] 3.4 Add torch wheel upgrade step in offline mode — use `uv pip install --find-links` with bundled torch wheels
- [x] 3.5 Add addon installation in offline mode — extract addon wheel archives to `~/.llamafarm/addons/`, update `addons.json`
- [x] 3.6 Add installation verification step — run `lf version`, report success or failure

## 4. Phase 2: Deploy Command (Core)

- [x] 4.1 Add `lf deploy` Cobra command accepting optional environment name, with `--server-url`, `--with-data`, `--skip-models`, `--dry-run` flags
- [x] 4.2 Implement environment resolution — resolve server URL from named environment, fall back to `--server-url` flag, fall back to default localhost
- [x] 4.3 Implement health check step — `GET /health` on target server, fail fast with clear error if unreachable
- [x] 4.4 Implement config push with upsert logic — strip `environments` from config, try `PUT /{ns}/{proj}`, on 404 call `POST /{ns}` then retry PUT
- [x] 4.5 Implement model disk space pre-check — call `POST /v1/models/validate-download` for each model, warn and prompt if insufficient space
- [x] 4.6 Implement parallel model download trigger — call `POST /v1/models/download` concurrently for each model in `runtime.models[]`, consume SSE streams in parallel
- [x] 4.7 Implement multi-model progress rendering — display per-model progress bars and overall progress (refactor existing `lf models pull` SSE renderer to support multiple concurrent streams)
- [x] 4.8 Implement deploy summary — print config pushed status, model download results (names, cached vs downloaded), total elapsed time
- [x] 4.9 Implement `--dry-run` mode — print planned actions without executing

## 5. Phase 3: Model Push

- [ ] 5.1 Add `POST /v1/models/upload` server endpoint — accepts resumable chunked upload, writes blobs to HuggingFace cache structure (`blobs/`, `snapshots/`, `refs/`)
- [ ] 5.2 Implement resumable upload protocol on server — track upload offset per upload ID, accept range-based chunk uploads, support resume after interruption
- [ ] 5.3 Add `--push-models` flag to `lf deploy` — locates model files in local HF cache, streams upload to server
- [ ] 5.4 Implement local model file discovery — for each model in config, find cached files in `~/.cache/huggingface/hub/`, warn if not cached locally
- [ ] 5.5 Implement upload progress rendering — per-model progress bars with bytes transferred, rate, and ETA
- [ ] 5.6 Add server-side model existence check — skip upload if model already cached on server

## 6. Phase 3: Data Push

- [ ] 6.1 Add `--with-data` implementation to `lf deploy` — read dataset configs, find local document files, upload via `POST /v1/datasets/{id}/files/bulk`
- [ ] 6.2 Implement local dataset file discovery — for each dataset, resolve document file paths relative to project directory
- [ ] 6.3 Trigger ingestion after upload — call the dataset ingestion endpoint on the remote server after files are uploaded
- [ ] 6.4 Implement data push progress rendering — per-dataset upload progress and ingestion status
