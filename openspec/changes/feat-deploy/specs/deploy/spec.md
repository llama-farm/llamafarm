## ADDED Requirements

### Requirement: CLI deploy command pushes project to server
The `lf deploy` command SHALL push the current project config to a LlamaFarm server and optionally trigger model downloads and data ingestion.

#### Scenario: Deploy to named environment
- **WHEN** user runs `lf deploy staging`
- **THEN** the CLI resolves `staging` from the `environments` section of `llamafarm.yaml` and deploys to the configured `server_url`

#### Scenario: Deploy to ad-hoc server
- **WHEN** user runs `lf deploy --server-url http://10.0.1.50:14345`
- **THEN** the CLI deploys to the specified URL without requiring a named environment

#### Scenario: Deploy with no arguments
- **WHEN** user runs `lf deploy` with no environment name and no `--server-url` flag
- **THEN** the CLI deploys to `localhost:14345` (the default `--server-url`)

#### Scenario: Deploy to unknown environment
- **WHEN** user runs `lf deploy nonexistent`
- **THEN** the CLI exits with an error listing available environment names

### Requirement: Deploy performs health check before proceeding
The deploy command SHALL verify the target server is reachable before pushing config or triggering downloads.

#### Scenario: Server is healthy
- **WHEN** `GET /health` on the target server returns a successful response
- **THEN** deploy proceeds with config push

#### Scenario: Server is unreachable
- **WHEN** `GET /health` on the target server fails or times out
- **THEN** the CLI exits with a clear error message indicating the server is unreachable at the given URL

### Requirement: Deploy upserts project config
The deploy command SHALL create the project on the remote server if it does not exist, or update it if it does.

#### Scenario: Project does not exist on remote
- **WHEN** `PUT /{ns}/{proj}` returns 404
- **THEN** the CLI calls `POST /{ns}` to create the project, then `PUT /{ns}/{proj}` to set the full config

#### Scenario: Project already exists on remote
- **WHEN** `PUT /{ns}/{proj}` returns 200
- **THEN** the config is updated in place

### Requirement: Deploy triggers parallel model downloads
When `deploy_models` is enabled (default), the deploy command SHALL trigger model downloads on the remote server for all models defined in `runtime.models[]`.

#### Scenario: Multiple models downloaded in parallel
- **WHEN** the project config defines 3 models and `deploy_models` is true
- **THEN** the CLI triggers `POST /v1/models/download` for all 3 models concurrently and displays per-model and overall progress

#### Scenario: Model already cached on server
- **WHEN** a model is already present in the server's HuggingFace cache
- **THEN** the server returns a `cached` status and the CLI shows it as already available

#### Scenario: Insufficient disk space for model
- **WHEN** `POST /v1/models/validate-download` indicates insufficient disk space for a model
- **THEN** the CLI warns the user and prompts for confirmation before proceeding

### Requirement: Deploy supports override flags
The deploy command SHALL accept flags that override environment-level settings for a single run.

#### Scenario: Override deploy_data with --with-data
- **WHEN** user runs `lf deploy staging --with-data`
- **THEN** dataset upload and ingestion is performed regardless of the environment's `deploy_data` setting

#### Scenario: Skip models with --skip-models
- **WHEN** user runs `lf deploy staging --skip-models`
- **THEN** model downloads are skipped regardless of the environment's `deploy_models` setting

#### Scenario: Dry run
- **WHEN** user runs `lf deploy staging --dry-run`
- **THEN** the CLI prints what actions would be taken (config push, model downloads, data uploads) without executing them

### Requirement: Deploy prints a summary on completion
After all deploy steps complete, the CLI SHALL print a summary of actions taken.

#### Scenario: Successful deploy with models
- **WHEN** deploy completes successfully with 2 models downloaded
- **THEN** the CLI prints a summary showing: config pushed, 2 models downloaded (with names), total time elapsed
