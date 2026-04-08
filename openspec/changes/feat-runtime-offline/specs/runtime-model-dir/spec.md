## ADDED Requirements

### Requirement: `LLAMAFARM_MODEL_DIR` enables flat-directory model loading

When the environment variable `LLAMAFARM_MODEL_DIR` is set to a non-empty filesystem path, the runtime SHALL treat it as the root of a canonical on-device model layout (`<LLAMAFARM_MODEL_DIR>/<alias>/<files>`) and consult it before falling back to the HuggingFace cache. The `<alias>` component is the model's `name` field from `runtime.models[]` in `llamafarm.yaml`.

#### Scenario: Unset env var preserves default behavior

- **WHEN** `LLAMAFARM_MODEL_DIR` is unset
- **THEN** the runtime uses the existing HF-cache-first resolution without inspecting any flat directory

#### Scenario: Set env var routes lookup through alias directory

- **WHEN** `LLAMAFARM_MODEL_DIR=/opt/llamafarm/models` and the runtime loads a model with alias "qwen3-1.7b"
- **THEN** the runtime inspects `/opt/llamafarm/models/qwen3-1.7b/` before the HF cache

### Requirement: Alias-directory discovery uses format sniffing, not exact filenames

When resolving a model from the alias directory, the runtime SHALL discover files by format sniffing (extension and GGUF magic bytes) rather than matching specific filenames. This accommodates both the canonical layout emitted by `lf models path` (e.g. `model.Q4_K_M.gguf`, `mmproj.f16.gguf`) and HF-preserved filenames (e.g. `Qwen3-1.7B-Q4_K_M.gguf`, `mmproj-qwen-f16.gguf`).

#### Scenario: Canonical filename is resolved

- **WHEN** `$LLAMAFARM_MODEL_DIR/qwen3-1.7b/model.Q4_K_M.gguf` exists with a valid GGUF magic header
- **THEN** the runtime resolves it as the weights file for alias "qwen3-1.7b"

#### Scenario: Preserved HF filename is resolved

- **WHEN** `$LLAMAFARM_MODEL_DIR/qwen3-1.7b/Qwen3-1.7B-Q4_K_M.gguf` exists with a valid GGUF magic header
- **THEN** the runtime resolves it as the weights file for alias "qwen3-1.7b"

#### Scenario: mmproj is identified by filename heuristic

- **WHEN** an alias directory contains both a main GGUF file and one whose filename matches the mmproj heuristic (`mmproj` substring, or `multimodal` without a quant suffix)
- **THEN** the mmproj file is returned for the mmproj role and NOT as the weights file

#### Scenario: Multiple GGUF files with no mmproj heuristic

- **WHEN** an alias directory contains two GGUF files and neither matches the mmproj heuristic
- **THEN** the runtime selects using the existing quantization preference order (Q4_K_M > Q4_K > Q5_K_M > ... > F16) applied to parsed filenames

#### Scenario: Empty alias directory is treated as not present

- **WHEN** an alias directory exists but contains no `.gguf` files
- **THEN** the runtime treats it as absent and continues to the next resolution tier

### Requirement: Resolution order is deterministic and documented

The new alias-based resolver (`resolve_gguf_path`) SHALL resolve model files in this order, first match wins: (1) `$LLAMAFARM_MODEL_DIR/<alias>/` if set and populated, (2) HuggingFace cache via the existing `get_gguf_file_path` logic, (3) network download via `snapshot_download` (only when NOT in offline mode). This order SHALL be documented alongside the env vars.

Behavior for callers that do NOT pass an alias is unchanged from the legacy entry point. In particular, existing support for `.gguf`-suffixed model IDs (including those with directory components, which are resolved via basename lookup under `~/.llamafarm/models/` or `$GGUF_MODELS_DIR/` as implemented in the edge runtime on main) SHALL continue to work for all callers that route through `get_gguf_file_path` without an alias.

Absolute filesystem paths are NOT a resolution tier within `resolve_gguf_path` itself — callers that want alias-directory support must use an alias or `$LLAMAFARM_MODEL_DIR`. This narrower restriction avoids the taint-analysis concern with caller-controlled absolute paths flowing into new filesystem operations while preserving all pre-existing behavior of the legacy `get_gguf_file_path` entry point (which continues to accept `.gguf`-suffixed inputs via its safe-directory basename lookup).

#### Scenario: Legacy `.gguf` basename lookup is preserved

- **WHEN** a caller invokes `get_gguf_file_path("/some/dir/model.gguf")` directly (no alias), and `~/.llamafarm/models/model.gguf` or `$GGUF_MODELS_DIR/model.gguf` exists
- **THEN** the resolver returns the matching safe-directory path, matching the behavior added to main prior to this change

#### Scenario: Edge runtime preserves legacy path for absolute model IDs

- **WHEN** an HTTP request sends an absolute path as the model ID to the edge runtime
- **THEN** the auto-derived alias is `None` (absolute paths are skipped by the derivation helper), the `GGUFLanguageModel` constructor takes the legacy branch, and `get_gguf_file_path` handles the request using its pre-existing safe-directory lookup

#### Scenario: Model dir shadows HF cache when both populated

- **WHEN** both the alias directory and the HF cache contain the same model and `LLAMAFARM_MODEL_DIR` is set
- **THEN** the alias directory file is loaded

#### Scenario: Falls through to HF cache when alias dir is absent

- **WHEN** `LLAMAFARM_MODEL_DIR` is set but the specific alias directory is missing, and `LLAMAFARM_OFFLINE` is unset
- **THEN** the runtime falls through to the HF cache for that model

#### Scenario: Offline mode prevents fallthrough to network

- **WHEN** `LLAMAFARM_MODEL_DIR` is set, the alias dir is absent, the HF cache lacks the model, and `LLAMAFARM_OFFLINE=1` is set
- **THEN** the runtime raises the offline error without attempting a network download

### Requirement: Alias-directory loading produces the same runtime result as HF-cache loading

From the perspective of subsequent chat completion, embedding, or inference calls, loading a model via alias directory SHALL be indistinguishable from loading the same model via HF cache. The resolver's responsibility is limited to returning a valid filesystem path; downstream code (llama-cpp initialization, chat template extraction, mmproj loading) SHALL NOT need changes.

#### Scenario: Chat completion works identically

- **WHEN** a model is loaded from an alias directory vs. the HF cache
- **THEN** a subsequent chat completion call produces the same structural response (content, tool calls, finish reason) in both setups

#### Scenario: Chat template is extracted from GGUF metadata

- **WHEN** a model is loaded from an alias directory
- **THEN** the chat template is extracted from the GGUF file's embedded metadata via the existing `get_chat_template_from_gguf` path, not from an external file
