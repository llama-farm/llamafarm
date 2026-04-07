## ADDED Requirements

### Requirement: `lf runtime binary pull` downloads llama.cpp for a target platform

The CLI SHALL provide a `lf runtime binary pull` command that downloads the pinned llama.cpp library and its required dependency files (e.g., `libggml*`, Metal shaders, CUDA runtime libs on Windows) for a specified target platform and accelerator into the local llamafarm-llama cache directory. The command SHALL reuse the download logic already present in `cli/cmd/orchestrator/llama_binary.go`, refactored so the download path is shared and not duplicated.

#### Scenario: Pull for current host with defaults

- **WHEN** the user runs `lf runtime binary pull` with no flags on a macOS ARM64 host
- **THEN** the command downloads the `darwin/arm64` Metal build of the pinned llama.cpp version into the local cache and exits zero

#### Scenario: Pull for a cross-platform target

- **WHEN** the user runs `lf runtime binary pull --platform linux/arm64 --accelerator cpu` on a macOS host
- **THEN** the command downloads the Linux ARM64 CPU build into the local cache (at a platform-scoped path) and exits zero

#### Scenario: Already-cached binary is not re-downloaded

- **WHEN** the user runs `lf runtime binary pull --platform linux/arm64` and the binary for that platform/version is already cached
- **THEN** the command exits zero without network activity and logs that the binary was already cached

### Requirement: `lf runtime binary pull --export` materializes a flat directory

When `--export <dir>` is specified, the command SHALL copy the downloaded binary and all dependency files into the given directory in a flat layout (no nested platform subdirectories) suitable for ansible/packer pickup. The copy SHALL include the main shared library and all sibling dependency libraries needed at runtime.

#### Scenario: Export produces a self-contained directory

- **WHEN** the user runs `lf runtime binary pull --platform linux/arm64 --export /tmp/lf-bin`
- **THEN** `/tmp/lf-bin/` contains `libllama.so` and all required dependency files (e.g., `libggml.so*`, `libggml-cpu.so`, `libmtmd.so*`) as regular files or preserved symlinks

#### Scenario: Export to an existing directory overwrites prior contents

- **WHEN** the user runs `lf runtime binary pull --export <dir>` and `<dir>` already exists from a previous run
- **THEN** the command replaces the prior contents with the current version's files and exits zero

### Requirement: `lf runtime binary pull` supports platform and accelerator flags

The command SHALL accept `--platform <os>/<arch>` (e.g., `linux/arm64`, `darwin/arm64`, `windows/amd64`) and `--accelerator <backend>` (one of `cpu`, `cuda`, `vulkan`, `metal`, `rocm`). When `--accelerator` is omitted, the command SHALL select the best supported backend for the given platform using the same selection logic used by the runtime orchestrator today.

#### Scenario: Explicit platform + accelerator

- **WHEN** the user runs `lf runtime binary pull --platform linux/x86_64 --accelerator vulkan`
- **THEN** the Vulkan Linux x86_64 binary is downloaded

#### Scenario: Accelerator inferred from platform

- **WHEN** the user runs `lf runtime binary pull --platform darwin/arm64` without `--accelerator`
- **THEN** the command selects `metal` automatically and downloads the Metal build

#### Scenario: Invalid platform/accelerator combo fails fast

- **WHEN** the user runs `lf runtime binary pull --platform darwin/arm64 --accelerator cuda`
- **THEN** the command exits non-zero with a message indicating CUDA is not supported on macOS and listing valid accelerators for that platform

### Requirement: `lf runtime binary pull` supports version override

The command SHALL accept a `--version <tag>` flag that overrides the pinned llama.cpp version. When omitted, the version SHALL be read from `llama-cpp-version.txt` at the repository root or from the embedded build-time constant for installed CLI binaries.

#### Scenario: Explicit version is downloaded

- **WHEN** the user runs `lf runtime binary pull --version b7800`
- **THEN** the command downloads llama.cpp release `b7800` instead of the pinned version

### Requirement: `lf runtime binary path` prints the cached binary path

The CLI SHALL provide a `lf runtime binary path` query command that prints the absolute path to the cached llama.cpp binary for the specified platform and accelerator. The command SHALL NOT download or modify anything. If no binary is cached for the requested platform/accelerator, the command SHALL exit non-zero with a clear error message.

#### Scenario: Path for cached binary

- **WHEN** the user runs `lf runtime binary path --platform linux/arm64 --accelerator cpu` and the binary is cached
- **THEN** the command prints the absolute path to `libllama.so` on one line and exits zero

#### Scenario: Path for uncached binary

- **WHEN** the user runs `lf runtime binary path --platform linux/arm64` and nothing is cached for that platform
- **THEN** the command exits non-zero with a message like `llama.cpp binary not found for linux/arm64. Run 'lf runtime binary pull --platform linux/arm64' first.`

#### Scenario: Default platform is the current host

- **WHEN** the user runs `lf runtime binary path` with no flags on a macOS ARM64 host
- **THEN** the command resolves to the darwin/arm64/metal cached path if present

### Requirement: `lf runtime binary` download logic is shared with the orchestrator

The binary download, extraction, symlink preservation, and dependency-copying logic used by `lf runtime binary pull` SHALL be the same code path used by the runtime orchestrator (`cli/cmd/orchestrator/llama_binary.go`). The refactor SHALL extract a reusable function/package rather than duplicating the logic into `cmd/runtime_binary.go`.

#### Scenario: Orchestrator continues to use the shared downloader

- **WHEN** the CLI orchestrator launches the edge or universal runtime and needs the llama.cpp binary
- **THEN** it invokes the same shared download function that `lf runtime binary pull` uses

#### Scenario: No duplicated download code paths

- **WHEN** the refactor is complete
- **THEN** grep-level inspection confirms there is exactly one implementation of the llama.cpp download/extract flow in the CLI codebase
