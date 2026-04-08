## ADDED Requirements

### Requirement: Read-only model commands SHALL NOT boot the LlamaFarm server

The CLI commands `lf models status` and `lf models list` SHALL operate entirely against the local HuggingFace Hub cache directory and SHALL NOT invoke `orchestrator.EnsureServicesOrExit`, `orchestrator.EnsureServicesOrExitWithConfig`, or any other code path that starts the Python server, the RAG worker, or the Universal Runtime.

#### Scenario: `lf models status` on a fresh machine with no Python
- **WHEN** a user runs `lf models status unsloth/Qwen3-1.7B-GGUF:Q4_K_M` on a machine that has the LlamaFarm CLI installed but no Python interpreter and no LlamaFarm server processes running
- **THEN** the command exits with code 0 if the model is present in the local HF cache, exits with code 1 if not, and never spawns a Python child process

#### Scenario: `lf models list` does not boot the server
- **WHEN** a user runs `lf models list` and the LlamaFarm server is not running
- **THEN** the command lists every cached HuggingFace model under `$HF_HUB_CACHE` (or the resolved default), printing each repo's id, size on disk, and snapshot path
- **AND** no LlamaFarm server, RAG worker, or runtime process is started

#### Scenario: `lf models status` for an uncached model
- **WHEN** a user runs `lf models status some/model` and the model is not in the local cache
- **THEN** the command prints `✗ Model some/model is not cached` and exits with code 1

### Requirement: `lf models pull` SHALL download HuggingFace models without booting the server

The CLI command `lf models pull <model-id>` SHALL fetch model files from the HuggingFace Hub directly using a Go HTTP client, write them to the local cache in the canonical HuggingFace Hub layout, and emit progress to the terminal — without invoking `orchestrator.EnsureServicesOrExit` or making any HTTP request to the LlamaFarm server.

#### Scenario: pulling a single-file GGUF model with no server running
- **WHEN** a user runs `lf models pull unsloth/Qwen3-0.6B-GGUF:Q4_K_M` on a machine with no LlamaFarm server running
- **THEN** the CLI fetches the selected `.gguf` file directly from `huggingface.co`, writes it to the HF cache, prints progress with rate and ETA, and exits with code 0
- **AND** no LlamaFarm server, RAG worker, or runtime process is started

#### Scenario: pulling a multi-file transformer model
- **WHEN** a user runs `lf models pull nomic-ai/nomic-embed-text-v1.5` on a machine with no LlamaFarm server running
- **THEN** the CLI lists every file in the repository, downloads each one to the HF cache (with bounded concurrency), and emits a progress event per file
- **AND** the resulting cache directory is loadable by `huggingface_hub.snapshot_download(..., local_files_only=True)` without further network access

#### Scenario: pulling a model that is already cached
- **WHEN** a user runs `lf models pull <model-id>` for a model whose blob files all exist with the expected size
- **THEN** the CLI emits a `cached` event for each already-present file and exits with code 0, performing no network downloads

### Requirement: The CLI SHALL write the canonical HuggingFace Hub cache layout

When `lf models pull` writes a downloaded file to disk, it SHALL produce the exact directory structure that `huggingface_hub` produces, so that Python loaders (`transformers.from_pretrained`, `huggingface_hub.try_to_load_from_cache`, `llama_cpp.Llama`) can find the file without any code changes on the Python side.

#### Scenario: blob, snapshot symlink, and refs are all written
- **WHEN** the CLI successfully downloads a file `model.gguf` from repo `org/repo` at commit `abc123...`
- **THEN** the blob is written to `<cache>/models--org--repo/blobs/<etag>`
- **AND** a symlink is created at `<cache>/models--org--repo/snapshots/abc123/model.gguf` pointing at the relative path of the blob
- **AND** the file `<cache>/models--org--repo/refs/main` contains the commit hash `abc123`

#### Scenario: bit-perfect compatibility with `huggingface_hub` reader
- **WHEN** the CLI writes a model via the Go path
- **THEN** invoking `huggingface_hub.try_to_load_from_cache(repo_id="org/repo", filename="model.gguf")` from Python returns the absolute path to the snapshot file (not `None` and not the `_CACHED_NO_EXIST` sentinel)

#### Scenario: Windows fallback when symlinks are unsupported
- **WHEN** the CLI writes a model on a filesystem where symlinks cannot be created (e.g., Windows without developer mode)
- **THEN** the blob file is moved (not symlinked) into `<cache>/models--org--repo/snapshots/<commit>/<filename>` for newly downloaded files
- **AND** for files where the blob already existed prior to this download, the file is copied rather than moved
- **AND** Python loaders can still find the file via `huggingface_hub.try_to_load_from_cache`

### Requirement: GGUF quantization selection SHALL match the Python implementation

The Go GGUF selection logic SHALL produce the same output as `common/llamafarm_common/model_utils.py::select_gguf_file` for every input fixture. Quantization parsing, split-file detection, preference ordering, and the handling of imatrix (`IQ*`) variants SHALL be byte-for-byte identical to the Python behavior.

#### Scenario: explicit quantization request
- **WHEN** the CLI is asked to select from `["model.Q4_K_M.gguf", "model.Q8_0.gguf", "model.F16.gguf"]` with `preferred_quantization="Q8_0"`
- **THEN** it returns `"model.Q8_0.gguf"`

#### Scenario: case-insensitive quantization request
- **WHEN** the CLI is asked to select with `preferred_quantization="q4_k_m"` from a list containing `"model.Q4_K_M.gguf"`
- **THEN** it returns `"model.Q4_K_M.gguf"`

#### Scenario: default preference order
- **WHEN** the CLI is asked to select from `["model.Q8_0.gguf", "model.Q5_K_M.gguf", "model.Q4_K_M.gguf"]` with no preferred quantization
- **THEN** it returns `"model.Q4_K_M.gguf"` (the first preference in the default order)

#### Scenario: split files are deprioritized when a non-split version exists
- **WHEN** the CLI is asked to select from a list containing both `"model.Q4_K_M.gguf"` and `"model-00001-of-00002.Q4_K_M.gguf"`
- **THEN** it returns `"model.Q4_K_M.gguf"` (the non-split version)

#### Scenario: imatrix quantization variants are recognized
- **WHEN** the CLI parses the filename `"model.IQ4_XS.gguf"`
- **THEN** it extracts the quantization `"IQ4_XS"`

#### Scenario: F16 and FP16 normalize to the same value
- **WHEN** the CLI parses both `"model.F16.gguf"` and `"model.FP16.gguf"`
- **THEN** both return `"F16"`

#### Scenario: model id with quantization suffix
- **WHEN** the CLI parses `"unsloth/Qwen3-4B-GGUF:Q8_0"`
- **THEN** it returns `("unsloth/Qwen3-4B-GGUF", "Q8_0")`

#### Scenario: lowercase quantization in suffix is normalized
- **WHEN** the CLI parses `"unsloth/Qwen3-4B-GGUF:q8_0"`
- **THEN** it returns `("unsloth/Qwen3-4B-GGUF", "Q8_0")`

### Requirement: HuggingFace token discovery SHALL match `huggingface_hub.get_token`

The CLI SHALL discover a HuggingFace authentication token using the same precedence order as `huggingface_hub.get_token`. The first non-empty source wins.

Precedence:
1. `$HF_TOKEN` environment variable
2. `$HUGGING_FACE_HUB_TOKEN` environment variable
3. Contents of `$HF_HOME/token` (when `$HF_HOME` is set)
4. Contents of `~/.cache/huggingface/token`
5. Contents of `~/.huggingface/token` (legacy path)

#### Scenario: HF_TOKEN env var takes precedence
- **WHEN** `HF_TOKEN=abc` is set and `~/.cache/huggingface/token` contains `xyz`
- **THEN** the CLI uses `abc` as the auth token

#### Scenario: legacy token file used as last resort
- **WHEN** no env var is set and `~/.cache/huggingface/token` does not exist but `~/.huggingface/token` contains `legacy-token`
- **THEN** the CLI uses `legacy-token` as the auth token

#### Scenario: no token available
- **WHEN** none of the env vars are set and none of the token files exist
- **THEN** the CLI proceeds without a token; HF API requests are unauthenticated

### Requirement: Downloads SHALL be resumable and SHALL refuse to silently merge mismatched bytes

When a partial download exists at startup (the `<etag>.tmp` blob file is present and non-empty), the CLI SHALL attempt to resume using HTTP `Range` and `If-Range` headers. The CLI SHALL refuse to append bytes from a different version of the file even if the resumed bytes would happen to "fit."

#### Scenario: clean resume after network drop
- **WHEN** a previous `lf models pull` was interrupted at 90% and the `.tmp` file exists
- **AND** the file's ETag on the server has not changed
- **THEN** the CLI sends `Range: bytes=<size>-` with `If-Range: <etag>`, receives a `206 Partial Content` response, and appends the remaining bytes

#### Scenario: ETag changed since the partial was written
- **WHEN** the `.tmp` file exists but the file's ETag on the server has changed
- **THEN** the CLI deletes the `.tmp` file and starts the download from byte 0
- **AND** the CLI MUST NOT append the new bytes onto the old partial

#### Scenario: server does not support Range
- **WHEN** the server responds with `200 OK` instead of `206 Partial Content`
- **THEN** the CLI deletes the `.tmp` file and starts the download from byte 0

### Requirement: Concurrent downloads SHALL be coordinated with `huggingface_hub`-compatible file locks

When the CLI is downloading a blob, it SHALL hold an exclusive advisory lock on a `<blob_path>.lock` file using the same naming convention `huggingface_hub.filelock` uses. Concurrent Go and Python processes downloading the same blob SHALL serialize, not corrupt the blob.

#### Scenario: two `lf models pull` processes for the same model
- **WHEN** two `lf models pull <same-model>` invocations run concurrently
- **THEN** the second one waits for the first to release the lock, then either reuses the cached blob (no second download) or downloads the next-needed file

#### Scenario: `lf models pull` racing with a `huggingface_hub.snapshot_download` call
- **WHEN** a Go `lf models pull` and a Python `huggingface_hub.snapshot_download` race for the same blob
- **THEN** they serialize via the shared `.lock` file and the resulting blob is intact (no corruption, no truncation)

### Requirement: HuggingFace API errors SHALL surface actionable, structured messages

The CLI SHALL map HuggingFace HTTP error responses to human-readable error messages with concrete remediation steps. Generic 4xx/5xx errors with no remediation are not acceptable.

#### Scenario: 401 Unauthorized
- **WHEN** an HF API request returns `401 Unauthorized`
- **THEN** the CLI prints an error containing the phrase `huggingface-cli login` or `HF_TOKEN`, instructing the user to authenticate

#### Scenario: 403 Forbidden on a gated model
- **WHEN** an HF API request returns `403 Forbidden` for a gated model
- **THEN** the CLI prints an error containing `gated` and a URL of the form `https://huggingface.co/<model-id>` where the user can accept terms

#### Scenario: 404 Not Found
- **WHEN** an HF API request returns `404 Not Found`
- **THEN** the CLI prints an error stating the model id was not found and suggests checking the spelling

#### Scenario: network error
- **WHEN** the HF API call fails because `huggingface.co` is unreachable (DNS, connection refused, timeout)
- **THEN** the CLI prints an error mentioning that `huggingface.co` could not be reached and references `LLAMAFARM_OFFLINE=1` as the way to operate without network access

### Requirement: The CLI SHALL emit progress events with the existing event vocabulary

The Go downloader SHALL emit progress events using the same event-name discriminators and field names as the existing server-side SSE stream (`init`, `start`, `progress`, `cached`, `end`, `done`, `error`, `warning`). This vocabulary is preserved so a future server-side adapter (out of scope here) can re-emit Go events as SSE without renaming any fields.

#### Scenario: init event before first file download
- **WHEN** `lf models pull` begins downloading a model
- **THEN** the first event emitted has `event="init"` and includes `model_id`, `quantization`, `selected_file`, `total_size`, `is_gguf`, and `file_count` fields

#### Scenario: progress event during a file download
- **WHEN** a file is being downloaded
- **THEN** progress events have `event="progress"` and include `file`, `downloaded`, `total`, `percent`, `bytes_per_sec`, and `eta_seconds` fields

#### Scenario: done event at the end of a successful pull
- **WHEN** all files for a model have been downloaded successfully
- **THEN** the final event has `event="done"` and includes a `local_dir` field naming the repo cache directory

### Requirement: The CLI SHALL honor `LLAMAFARM_OFFLINE=1` and refuse network calls

When the `LLAMAFARM_OFFLINE` environment variable is set to a truthy value (`1`, `true`, `yes`, `on`, case-insensitive), `lf models pull` SHALL refuse to make any HuggingFace Hub API requests and SHALL exit with an error pointing the user at the offline-mode remediation.

#### Scenario: offline mode set, model already cached
- **WHEN** `LLAMAFARM_OFFLINE=1` is set and the requested model's files are all present in the local cache
- **THEN** the CLI emits `cached` events for each file and exits with code 0, making no network requests

#### Scenario: offline mode set, model not cached
- **WHEN** `LLAMAFARM_OFFLINE=1` is set and the requested model is not in the local cache
- **THEN** the CLI exits with a non-zero status and an error message that names the missing model and references running `lf models pull` on a host with internet access

#### Scenario: offline mode does not require Python
- **WHEN** `LLAMAFARM_OFFLINE=1` is set and the CLI is running in an environment with no Python interpreter
- **THEN** the offline check itself does not invoke Python; it reads the env var directly in Go
