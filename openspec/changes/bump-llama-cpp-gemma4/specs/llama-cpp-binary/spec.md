## ADDED Requirements

### Requirement: llama.cpp version is pinned in a single source of truth

The pinned upstream llama.cpp version SHALL be defined in exactly one file (`llama-cpp-version.txt` at the repository root) and propagated to dependent code via reads from that file or as fallback constants. The fallback constants SHALL match the value in `llama-cpp-version.txt` whenever the file is updated.

#### Scenario: Reading the canonical version

- **WHEN** any LlamaFarm component needs to know the pinned llama.cpp version
- **THEN** it reads from `llama-cpp-version.txt` (preferred) or uses a hardcoded fallback that matches the current value of that file

#### Scenario: Fallback constants stay in sync

- **WHEN** `llama-cpp-version.txt` is updated to a new tag
- **THEN** every fallback constant referencing the previous tag is updated in the same commit, including:
  - `packages/llamafarm-llama/src/llamafarm_llama/_binary.py` (the hardcoded fallback in `_read_llama_cpp_version`)
  - `cli/internal/llamabinary/llamabinary.go` (the `Version` package variable)
  - `cli/internal/llamabinary/llamabinary_test.go` and `download_test.go` (test fixture strings)

#### Scenario: Drift detection

- **WHEN** any of the four propagation locations references a llama.cpp version that does not match `llama-cpp-version.txt`
- **THEN** the change SHALL be rejected at review or by automation, because the single-source-of-truth invariant has been broken

### Requirement: Cached binaries are isolated by version

The on-disk cache for downloaded llama.cpp binaries SHALL include the version string in the cache key, so that bumping the pinned version does not require manual cache cleanup and so that multiple versions can coexist on a single host without collision.

#### Scenario: User upgrades to a new pinned version

- **WHEN** a user who has previously downloaded llama.cpp binary version `bX` updates LlamaFarm to a build pinned at version `bY`
- **THEN** the next launch downloads the `bY` binary into a separate cache subdirectory (`<cache>/bY/`), leaves the `bX` binary undisturbed, and uses `bY` for inference

#### Scenario: Cross-platform binaries are isolated

- **WHEN** a host downloads binaries for two different platform/accelerator combinations (e.g. via `lf runtime binary pull` for cross-platform deployment)
- **THEN** each platform/accelerator combination is cached in its own subdirectory and does not overwrite the host's own binary

### Requirement: Supported model architectures track upstream

The set of model architectures LlamaFarm supports for inference SHALL be defined as "whatever the pinned llama.cpp version supports." Adding support for a new architecture is accomplished by bumping the pinned version, not by modifying LlamaFarm's binding code.

#### Scenario: Loading a Gemma 4 model

- **WHEN** the pinned llama.cpp version includes Gemma 4 support (`b8708` or later) AND a user attempts to load a Gemma 4 GGUF model through the universal runtime
- **THEN** the model loads successfully and produces coherent inference output

#### Scenario: Loading a previously-supported model

- **WHEN** the pinned llama.cpp version is bumped AND a user attempts to load a model architecture that was supported by both the old and new versions (e.g. Llama 3, Qwen, Mistral)
- **THEN** the model loads and runs inference with no behavioral regression compared to the previous pinned version

### Requirement: Linux ARM64 binary is built and validated before merge

Upstream llama.cpp does not ship pre-built Linux ARM64 binaries. LlamaFarm SHALL build its own Linux ARM64 binary at every pinned version via the `build-llama.yml` GitHub Actions workflow. Before merging any version-bump PR, the workflow SHALL be triggered as a build smoke test to confirm the new pin compiles successfully. The actual release-asset publication is decoupled and happens automatically when the next LlamaFarm release tag is pushed.

#### Scenario: Bumping the pinned version (smoke test)

- **WHEN** a PR updates `llama-cpp-version.txt` to a new tag `bX`
- **THEN** the implementer triggers `build-llama.yml` via `workflow_dispatch` against `bX` and confirms the build succeeds and uploads a `llama-bX-bin-linux-arm64.zip` artifact to the workflow run, before merging the PR

#### Scenario: Cutting a LlamaFarm release with a new pin

- **WHEN** a LlamaFarm release tag (`v*`) is pushed at a commit that contains a bumped `llama-cpp-version.txt`
- **THEN** `build-llama.yml` re-runs automatically against that tag, builds llama.cpp at the new pin, and the `Release` step (gated on `if: startsWith(github.ref, 'refs/tags/')`) attaches `llama-bX-bin-linux-arm64.zip` to the LlamaFarm release as a downloadable asset

#### Scenario: Linux ARM64 user downloads binary

- **WHEN** a Linux ARM64 user runs LlamaFarm and `llamafarm-llama` needs to download the binary for the pinned version
- **THEN** `_binary.py` resolves the artifact URL via `_get_llamafarm_release_version`, downloads the LlamaFarm-published ARM64 zip from the LlamaFarm release that ships with their LlamaFarm version, and extracts `libllama.so` into the version-keyed cache directory

### Requirement: Upgrade procedure validates header diff and runtime behavior

Before merging any change that bumps the pinned llama.cpp version, the implementer SHALL verify that no struct layout changes affect the cffi declarations in `_bindings.py` AND SHALL run end-to-end inference smoke tests against at least one model.

#### Scenario: Reviewing the header diff

- **WHEN** preparing a llama.cpp version bump
- **THEN** the implementer runs `git diff <old>..<new> -- include/llama.h ggml/include/ggml.h` against an upstream checkout, identifies any struct layout changes affecting `llama_model_params`, `llama_context_params`, `llama_batch`, or other types declared in `_bindings.py`, and updates the cffi declarations to match (or documents in the change's design.md that no updates are needed)

#### Scenario: Smoke-testing inference

- **WHEN** preparing a llama.cpp version bump
- **THEN** the implementer loads at least one GGUF model end-to-end through the universal runtime, generates output, and verifies the output is coherent — NOT relying solely on unit tests passing, because binding-layer correctness can pass tests while runtime inference is broken

#### Scenario: Smoke-testing the model that motivated the bump

- **WHEN** the bump is motivated by support for a specific model architecture
- **THEN** the implementer additionally smoke-tests that specific architecture (e.g. Gemma 4 for this bump), because passing tests on previously-supported architectures does not prove the new architecture works

### Requirement: Deprecated llama.cpp APIs are replaced when bumping past their deprecation

When a llama.cpp version bump introduces deprecation markers on APIs that LlamaFarm currently calls, the bump SHALL replace those calls with the upstream-recommended replacements in the same change, provided the replacements have signature-compatible behavior.

#### Scenario: Renamed API with identical signature

- **WHEN** an upstream API LlamaFarm calls (e.g. `llama_load_model_from_file`) is marked deprecated AND the upstream deprecation message names a replacement (e.g. `llama_model_load_from_file`) with an identical signature
- **THEN** the version bump replaces the call site in `_bindings.py` and `llama.py` with the new name in the same commit

#### Scenario: Replacement requires non-trivial migration

- **WHEN** an upstream API LlamaFarm calls is marked deprecated AND the replacement requires a non-trivial signature change or refactor (e.g. the array-based LoRA API replacing per-adapter calls)
- **THEN** the version bump leaves the deprecated call in place AND opens a follow-up issue documenting the migration work, rather than blocking the bump on a larger refactor

#### Scenario: Deprecated APIs LlamaFarm does not call

- **WHEN** upstream marks APIs deprecated that LlamaFarm does not call (e.g. token/vocab/session helpers we never wrapped)
- **THEN** no action is required; LlamaFarm tracks only the deprecated APIs it actually depends on
