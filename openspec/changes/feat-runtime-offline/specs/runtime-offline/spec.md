## ADDED Requirements

### Requirement: `LLAMAFARM_OFFLINE=1` enables strict offline mode

When the environment variable `LLAMAFARM_OFFLINE` is set to a truthy value (`1`, `true`, `yes`, case-insensitive) at process startup, the runtime SHALL operate in strict offline mode. In strict offline mode the runtime MUST NOT initiate any outbound network request on behalf of model loading, chat template resolution, or llama.cpp binary acquisition.

#### Scenario: Offline mode blocks HuggingFace hub listing

- **WHEN** `LLAMAFARM_OFFLINE=1` is set and the runtime needs to locate a GGUF file whose repo is not cached
- **THEN** the runtime SHALL NOT call `huggingface_hub.HfApi.list_repo_files` or equivalent and SHALL raise a clear error naming the missing model

#### Scenario: Offline mode blocks snapshot download

- **WHEN** `LLAMAFARM_OFFLINE=1` is set and the runtime needs to download a model file
- **THEN** the runtime SHALL NOT call `huggingface_hub.snapshot_download` or equivalent and SHALL raise a clear error naming the missing model

#### Scenario: Offline mode blocks llama.cpp binary download

- **WHEN** `LLAMAFARM_OFFLINE=1` is set and the runtime needs the llama.cpp shared library and no cached copy exists
- **THEN** the runtime SHALL NOT fetch from GitHub releases and SHALL raise a clear error pointing at `lf runtime binary pull`

### Requirement: Offline mode propagates `HF_HUB_OFFLINE`

When `LLAMAFARM_OFFLINE=1` is set, the runtime SHALL ensure that `HF_HUB_OFFLINE=1` is also set in the process environment before any `huggingface_hub`, `transformers`, or `datasets` library code runs. This guarantees that transitive calls made by third-party libraries also honor offline mode.

#### Scenario: HF_HUB_OFFLINE is exported automatically

- **WHEN** the runtime process starts with `LLAMAFARM_OFFLINE=1` and no explicit `HF_HUB_OFFLINE` value
- **THEN** `os.environ["HF_HUB_OFFLINE"]` SHALL equal `"1"` before any transformers/huggingface_hub import is consumed

#### Scenario: Explicit HF_HUB_OFFLINE=0 is overridden

- **WHEN** the runtime process starts with `LLAMAFARM_OFFLINE=1` and `HF_HUB_OFFLINE=0`
- **THEN** the runtime SHALL override `HF_HUB_OFFLINE` to `"1"` and log a warning that the conflicting value was replaced

### Requirement: Offline mode errors name the alias, paths tried, and remediation command

In offline mode, any error raised because a model file could not be found SHALL include: the model alias (from `llamafarm.yaml` `runtime.models[].name`, not the HF repo id), the filesystem paths the runtime tried, and the specific `lf` CLI command that would make the missing file available.

#### Scenario: Missing model error format

- **WHEN** `LLAMAFARM_OFFLINE=1` is set and a model alias "qwen3-1.7b" has no files at `$LLAMAFARM_MODEL_DIR/qwen3-1.7b/` nor in the HF cache
- **THEN** the raised error message SHALL name "qwen3-1.7b", list the two tried paths, and reference `lf models pull qwen3-1.7b`

#### Scenario: Missing llama.cpp binary error format

- **WHEN** `LLAMAFARM_OFFLINE=1` is set and no cached llama.cpp shared library exists
- **THEN** the raised error message SHALL reference `lf runtime binary pull` with suggested platform flags

### Requirement: Offline mode does not retry

In offline mode the runtime SHALL raise immediately on missing-resource errors. It SHALL NOT retry, sleep-and-retry, wait for a network to become available, or emit a warning and continue.

#### Scenario: No retry on missing model

- **WHEN** offline mode raises a missing-model error
- **THEN** the error propagates to the caller on the first failure with no retries and no backoff

### Requirement: Runtime logs offline status on startup

On startup, the runtime SHALL emit exactly one log line at `INFO` level that indicates the resolved mode (`offline` or `online`) and, when `LLAMAFARM_MODEL_DIR` is set, its path. This gives operators a single grep-able line to verify deployment state.

#### Scenario: Online startup log

- **WHEN** neither `LLAMAFARM_OFFLINE` nor `LLAMAFARM_MODEL_DIR` is set
- **THEN** the runtime logs a single line containing the tokens "mode=online" and the HF cache path

#### Scenario: Offline startup log with model dir

- **WHEN** `LLAMAFARM_OFFLINE=1` and `LLAMAFARM_MODEL_DIR=/opt/llamafarm/models` are set
- **THEN** the runtime logs a single line containing the tokens "mode=offline", "model_dir=/opt/llamafarm/models"

### Requirement: Existing online behavior is preserved by default

When neither `LLAMAFARM_OFFLINE` nor `LLAMAFARM_MODEL_DIR` is set, the runtime SHALL behave exactly as it did before this change. Cache-first lookup in the HuggingFace cache, on-demand download from HuggingFace or GitHub, and silent warning-then-retry semantics for network failures all remain intact for existing deployments.

#### Scenario: Default deployment is unchanged

- **WHEN** a model is loaded with no offline env vars set
- **THEN** the runtime uses the pre-existing `get_gguf_file_path` code path and fetches missing files over the network as before
