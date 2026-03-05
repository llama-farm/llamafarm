## ADDED Requirements

### Requirement: CLI bundle command packages LlamaFarm for distribution
The `lf bundle` command SHALL download pre-built LlamaFarm binaries from GitHub Releases and package them into a single distributable archive.

#### Scenario: Create bundle for Linux CUDA target
- **WHEN** user runs `lf bundle --platform linux --arch x86_64 --accelerator cuda -o bundle.tar.gz`
- **THEN** the CLI downloads the CLI binary, PyApp service binaries (server, rag, runtime), and accelerator-specific torch wheels from GitHub Releases for the current CLI version
- **AND** packages them into `bundle.tar.gz` with a `manifest.json` and embedded `install.sh`

#### Scenario: Create bundle with addons
- **WHEN** user runs `lf bundle --platform linux --arch x86_64 --accelerator cuda --addons stt,tts -o bundle.tar.gz`
- **THEN** the bundle additionally includes addon wheel archives and addon registry YAML files for the specified addons

#### Scenario: Create bundle for specific version
- **WHEN** user runs `lf bundle --version v0.8.0 --platform linux --arch x86_64 --accelerator cpu -o bundle.tar.gz`
- **THEN** the CLI downloads release artifacts for version `v0.8.0` instead of the current CLI version

#### Scenario: Invalid platform/arch/accelerator combination
- **WHEN** user specifies a platform/arch/accelerator combination that has no published release artifacts
- **THEN** the CLI exits with a clear error message listing valid combinations

### Requirement: Bundle manifest describes archive contents
The bundle SHALL include a `manifest.json` file that describes its contents, versions, and target platform.

#### Scenario: Manifest contains all metadata
- **WHEN** a bundle is created
- **THEN** `manifest.json` SHALL contain: `version` (LlamaFarm version), `platform`, `arch`, `accelerator`, `components` (map of component name to binary filename), and `addons` (list of included addon names)

### Requirement: Bundle fetches artifacts from GitHub Releases
The `lf bundle` command SHALL download binary artifacts from the GitHub Releases for the `llama-farm/llamafarm` repository.

#### Scenario: Authenticated download with GITHUB_TOKEN
- **WHEN** `GITHUB_TOKEN` environment variable is set
- **THEN** the CLI uses it for authenticated GitHub API requests to avoid rate limits

#### Scenario: Download fails due to rate limit
- **WHEN** GitHub API returns a 403 rate limit response
- **THEN** the CLI exits with an error suggesting the user set `GITHUB_TOKEN`

### Requirement: Bundle includes accelerator-specific torch wheels
The bundle SHALL include torch wheels matching the specified `--accelerator` flag, separate from the PyApp runtime binary.

#### Scenario: CUDA torch wheels included
- **WHEN** `--accelerator cuda` is specified
- **THEN** the bundle includes CUDA-enabled torch wheels in a `torch/` directory

#### Scenario: CPU-only bundle
- **WHEN** `--accelerator cpu` is specified
- **THEN** no additional torch wheels are included (PyApp ships with CPU torch)
