## ADDED Requirements

### Requirement: `lf models path` emits a transport plan

The CLI SHALL provide a `lf models path` command that reads `llamafarm.yaml`, locates each configured model's files in the local HuggingFace cache, and emits a machine-readable transport plan mapping source paths on the current host to target paths on a deployment device. The command SHALL NOT download, copy, or modify any model files, HuggingFace cache contents, or on-device target files. It MAY maintain internal bookkeeping files (e.g. sha256 sidecar files in a CLI-owned cache directory) as an implementation detail; those are not considered "user-visible modifications".

#### Scenario: JSON output contains source, target, size, and sha256 for each file

- **WHEN** the user runs `lf models path --format json` against a project with cached models
- **THEN** the output is a JSON object with a `models` array, where each model has `name`, `kind`, `quant`, and a `files` array, and each file object contains `role`, `source`, `target`, `size`, and `sha256` fields

#### Scenario: TSV output is one file per line

- **WHEN** the user runs `lf models path --format tsv` (or omits `--format`)
- **THEN** the output is tab-separated with columns `name`, `role`, `source`, `target` — one line per file, no `sha256` column

#### Scenario: Empty positional args selects all configured models

- **WHEN** the user runs `lf models path` with no alias arguments
- **THEN** the plan contains every model declared in `runtime.models[]` of the loaded `llamafarm.yaml`

#### Scenario: Positional args filter the plan to specified aliases

- **WHEN** the user runs `lf models path qwen3-1.7b smollm-135m`
- **THEN** the plan contains only those two models, in the order given

### Requirement: `lf models path` source paths are HuggingFace snapshot paths

The `source` field emitted by `lf models path` SHALL be the HuggingFace cache snapshot path (`.../snapshots/<hash>/<filename>`), NOT the resolved blob path (`.../blobs/<sha>`). The command SHALL use `os.path.realpath` or equivalent only for existence verification.

#### Scenario: Source preserves the original filename

- **WHEN** the plan is emitted for a cached GGUF model
- **THEN** the `source` field ends with the original filename (e.g., `qwen3-1.7b-Q4_K_M.gguf`), not a blob hash

#### Scenario: Source remains valid across HuggingFace cache garbage collection

- **WHEN** HuggingFace cache garbage collection repacks blob storage but leaves snapshot symlinks intact
- **THEN** the `source` path from a previously emitted plan SHALL still resolve to a valid file

### Requirement: `lf models path` validates aliases before building target paths

Before constructing any `target` path, the command SHALL validate each model's alias (the `name` field from `runtime.models[]`) to ensure it cannot escape `target_root` via path traversal. Aliases containing `..`, path separators (`/`, `\`), or absolute-path prefixes SHALL be rejected with a clear error. Aliases SHALL match the pattern `^[a-zA-Z0-9._][a-zA-Z0-9._\-]*$`, mirroring the runtime-side `validate_alias` contract in `llamafarm_common.model_utils`.

#### Scenario: Traversal alias rejected

- **WHEN** the user runs `lf models path` against a project whose `runtime.models[0].name` is `../etc/passwd`
- **THEN** the command exits non-zero with an "invalid alias" error before computing any target paths

#### Scenario: Slash-bearing alias rejected

- **WHEN** a model name contains `/` or `\`
- **THEN** the command exits non-zero with an "invalid alias" error

### Requirement: `lf models path` target paths use a resolved target root

The `target` field for each file SHALL be computed as `<target_root>/<alias>/<canonical-filename>`, where `target_root` is resolved in this precedence order: `--target-root` flag, `deployment.model_dir` from `llamafarm.yaml`, hardcoded fallback `/opt/llamafarm/models`. The `alias` component MUST already have passed the validation defined in the previous requirement.

#### Scenario: `--target-root` flag overrides config

- **WHEN** the user runs `lf models path --target-root /custom/models` and the config also specifies `deployment.model_dir: /opt/llamafarm/models`
- **THEN** the `target_root` in output is `/custom/models` and each file's `target` begins with `/custom/models/`

#### Scenario: `deployment.model_dir` is used when flag is absent

- **WHEN** `llamafarm.yaml` specifies `deployment.model_dir: /srv/lf/models` and no `--target-root` flag is given
- **THEN** the `target_root` in output is `/srv/lf/models`

#### Scenario: Hardcoded fallback when neither flag nor config is set

- **WHEN** no `--target-root` flag is given and `llamafarm.yaml` has no `deployment.model_dir`
- **THEN** the `target_root` in output is `/opt/llamafarm/models`

### Requirement: `lf models path` fails loudly when a model is not cached

If any selected model's required files are not present in the HuggingFace cache, the command SHALL exit with a non-zero status and print an actionable error message naming the missing model and the remediation command. The `--ensure` flag SHALL run `lf models pull` before the plan is computed to populate missing files.

#### Scenario: Missing model without `--ensure`

- **WHEN** the user runs `lf models path qwen3-1.7b` and the model is not in the HF cache
- **THEN** the command exits non-zero with a message like `qwen3-1.7b not found in cache. Run 'lf models pull qwen3-1.7b' first.`

#### Scenario: Missing model with `--ensure`

- **WHEN** the user runs `lf models path qwen3-1.7b --ensure` and the model is not in the HF cache
- **THEN** the command first invokes the equivalent of `lf models pull qwen3-1.7b`, then emits the transport plan for the now-cached files

### Requirement: `lf models path` supports role filtering

The command SHALL support a `--role` flag that filters the files included in the plan. Supported values: `weights` (main model weights file), `mmproj` (multimodal projector file, if present), `tokenizer` (tokenizer files for non-GGUF models), and `all` (default, includes every required file).

#### Scenario: `--role weights` returns only the main weights file

- **WHEN** the user runs `lf models path qwen3-1.7b --role weights --format json`
- **THEN** the `files` array for that model contains exactly one entry with `role: "weights"`

#### Scenario: `--role mmproj` returns the mmproj file when present

- **WHEN** the user runs `lf models path <multimodal-model> --role mmproj` and the model has a cached mmproj file
- **THEN** the plan contains only the mmproj file entry

#### Scenario: `--role mmproj` returns an empty file list when absent

- **WHEN** the user runs `lf models path <text-only-model> --role mmproj`
- **THEN** the model appears in the plan with an empty `files` array and the command exits zero

### Requirement: `lf models path` supports `--source-only` shell convenience mode

The `--source-only` flag SHALL cause the command to print one source path per line, with no other output, suitable for use in shell command substitution.

#### Scenario: `--source-only` for a single model and single role

- **WHEN** the user runs `lf models path qwen3-1.7b --role weights --source-only`
- **THEN** the output is exactly one line: the absolute source path of the weights file, with a trailing newline

#### Scenario: `--source-only` with multiple files

- **WHEN** the user runs `lf models path --source-only` and multiple files are selected
- **THEN** the output is one source path per line, in a stable order (by model then by role)

### Requirement: sha256 is cached per file and only emitted for JSON output

The command SHALL compute a sha256 digest for each cached model file the first time it is needed, persist the digest to a sidecar file alongside the HuggingFace cache entry, and reuse it on subsequent invocations. The sha256 field SHALL appear only in JSON output; TSV and `--source-only` output SHALL NOT include it.

#### Scenario: sha256 is computed once and reused

- **WHEN** `lf models path --format json` is run twice on the same unchanged model file
- **THEN** the second run SHALL NOT recompute the sha256 (verified by the sidecar file being reused and the file not being re-hashed)

#### Scenario: sha256 is omitted from TSV output

- **WHEN** `lf models path --format tsv` is run
- **THEN** each output line contains exactly four tab-separated fields (`name`, `role`, `source`, `target`), with no sha256

### Requirement: `lf models path` identifies model kind by format sniffing

The `kind` field in JSON output SHALL be derived by inspecting file extensions and content (reusing existing model-format detection helpers): `.gguf` → `gguf`, `.pt` or `.pth` → `ultralytics`, presence of `config.json` + `*.safetensors` → `transformers`. The command SHALL NOT read a `kind` metadata file.

#### Scenario: GGUF model is detected as kind=gguf

- **WHEN** `lf models path --format json` emits a plan for a model with a `.gguf` weights file
- **THEN** the model's `kind` field is `"gguf"`

#### Scenario: Canonical target filenames for GGUF

- **WHEN** a GGUF model with quantization `Q4_K_M` is emitted
- **THEN** the weights file's `target` field ends with `<alias>/model.Q4_K_M.gguf` and any mmproj file's `target` ends with `<alias>/mmproj.<precision>.gguf`

### Requirement: Initial implementation covers GGUF models

The initial implementation SHALL support GGUF models (single-file and sharded) including associated mmproj files. Support for `transformers` and `ultralytics` model kinds MAY return an explicit "not yet supported" error and SHALL NOT silently emit an incomplete plan.

#### Scenario: Transformers model returns a clear not-yet-supported error

- **WHEN** the user runs `lf models path` against a project containing a non-GGUF transformers model
- **THEN** the command exits non-zero with a message indicating transformers models are not yet supported by `lf models path` and naming the affected model
