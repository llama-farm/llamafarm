## ADDED Requirements

### Requirement: Deploy supports pushing model files to server
The `lf deploy --push-models` flag SHALL upload model files from the local HuggingFace cache to the remote server instead of triggering the server to download them.

#### Scenario: Push models from local cache
- **WHEN** user runs `lf deploy production --push-models`
- **THEN** for each model in `runtime.models[]`, the CLI locates the model files in the local `~/.cache/huggingface/hub/` and uploads them to the server

#### Scenario: Model not in local cache
- **WHEN** a model defined in `runtime.models[]` is not present in the local HuggingFace cache
- **THEN** the CLI warns the user and skips that model, suggesting `lf models pull <model>` first

#### Scenario: Model already present on server
- **WHEN** the server already has the model cached
- **THEN** the CLI skips the upload and reports the model as already available

### Requirement: Server accepts model file uploads
The server SHALL provide an endpoint that accepts model file uploads and places them in the HuggingFace cache structure.

#### Scenario: Upload GGUF model blob
- **WHEN** a client uploads a GGUF model file via `POST /v1/models/upload`
- **THEN** the server writes the blob to `~/.cache/huggingface/hub/models--{org}--{repo}/blobs/{etag}` and creates the appropriate `snapshots/` symlink and `refs/` entry

### Requirement: Model uploads are resumable
The model upload endpoint SHALL support resumable uploads to handle large files (10+ GB) and network interruptions.

#### Scenario: Upload interrupted and resumed
- **WHEN** a 15 GB model upload is interrupted at 10 GB
- **THEN** the client can resume from the 10 GB offset without re-uploading the completed portion

#### Scenario: Upload completes in a single request
- **WHEN** a small model file (< 1 GB) is uploaded without interruption
- **THEN** the upload completes normally in a single request

### Requirement: Push models displays upload progress
The CLI SHALL display per-model upload progress with transfer rate and ETA.

#### Scenario: Multi-model upload progress
- **WHEN** user pushes 2 models
- **THEN** the CLI shows per-model progress bars with bytes transferred, transfer rate, and estimated time remaining
