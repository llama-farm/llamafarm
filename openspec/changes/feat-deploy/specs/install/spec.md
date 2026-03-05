## ADDED Requirements

### Requirement: install.sh supports offline mode via archive argument
The `install.sh` script SHALL accept a bundle archive path as its first argument to perform a fully offline installation.

#### Scenario: Offline install from bundle archive
- **WHEN** user runs `./install.sh llamafarm-bundle-linux-x86_64-cuda.tar.gz`
- **THEN** the script extracts the archive, reads `manifest.json`, installs the CLI binary to the install directory, installs PyApp service binaries to `~/.llamafarm/bin/`, and verifies the installation

#### Scenario: Offline install with custom install directory
- **WHEN** user runs `INSTALL_DIR=/opt/bin ./install.sh bundle.tar.gz`
- **THEN** the CLI binary is installed to `/opt/bin/lf` instead of the default location

#### Scenario: Online install without arguments (existing behavior preserved)
- **WHEN** user runs `./install.sh` with no arguments
- **THEN** the script downloads the CLI binary from GitHub Releases as it does today

### Requirement: install.sh installs PyApp service binaries
In both online and offline mode, the installer SHALL place PyApp service binaries in `~/.llamafarm/bin/`.

#### Scenario: PyApp binaries installed from bundle
- **WHEN** offline install completes
- **THEN** `~/.llamafarm/bin/` contains the server, rag, and runtime PyApp binaries from the bundle

#### Scenario: PyApp binaries installed online
- **WHEN** online install runs
- **THEN** PyApp service binaries are downloaded from GitHub Releases and placed in `~/.llamafarm/bin/`

### Requirement: install.sh installs accelerator-specific torch from bundle
When installing from a bundle that includes torch wheels, the installer SHALL upgrade the runtime's torch to the accelerator-specific version.

#### Scenario: CUDA torch installed from bundle
- **WHEN** offline install runs with a CUDA bundle
- **THEN** the installer replaces the runtime's CPU-only torch with the CUDA torch wheels from the `torch/` directory in the bundle

#### Scenario: CPU bundle skips torch upgrade
- **WHEN** offline install runs with a CPU bundle (no `torch/` directory)
- **THEN** no torch upgrade is performed; the PyApp's built-in CPU torch is used

### Requirement: install.sh installs addons from bundle
When the bundle includes addon wheel archives, the installer SHALL install them.

#### Scenario: Addons installed from bundle
- **WHEN** offline install runs with a bundle containing `addons/stt-wheels-*.tar.gz`
- **THEN** the installer extracts and installs the addon wheels to `~/.llamafarm/addons/stt/` and updates `~/.llamafarm/addons.json`

#### Scenario: Bundle without addons
- **WHEN** offline install runs with a bundle that has no `addons/` directory
- **THEN** no addon installation is performed

### Requirement: install.sh verifies installation
After installation, the script SHALL verify that core components are functional.

#### Scenario: Verification passes
- **WHEN** installation completes
- **THEN** the script runs `lf version` and reports success with the installed version

#### Scenario: Verification fails
- **WHEN** `lf version` fails after installation
- **THEN** the script exits with a non-zero code and a clear error message
