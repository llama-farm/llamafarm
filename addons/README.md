# LlamaFarm Addon Registry

This directory contains the centralized addon registry for LlamaFarm.

## Structure

```
addons/
  ├── README.md
  └── registry/
      ├── stt.yaml      # Speech-to-Text addon
      ├── tts.yaml      # Text-to-Speech addon
      └── speech.yaml   # Composite addon (depends on stt + tts)
```

Each addon has its own YAML file in the `registry/` directory. This prevents a giant monolithic registry file and makes it easier to add/modify addons.

## Architecture

The addon registry is consumed by multiple components:

```
addons/registry/*.yaml
    ├── CLI (Go) - Loads at runtime for addon commands
    ├── Server (Python) - Serves addon metadata to clients via API
    └── Build Script (Python) - Generates wheel bundles for distribution
```

### Benefits

1. **Modular**: Each addon is independently versioned and maintained
2. **Scalable**: Adding new addons doesn't create merge conflicts
3. **Cross-Language**: Both Go (CLI) and Python (Server) read the same files
4. **Version Control**: Changes are tracked in git
5. **Easy to Edit**: YAML is human-readable and easy to maintain
6. **API-Friendly**: Server can serve addon metadata directly to Designer/clients

## Addon Schema

### Basic Addon

```yaml
name: stt                    # Unique identifier
display_name: Speech-to-Text # Human-readable name
description: Transcribe audio using Whisper models
component: universal-runtime # Target service
version: "1.0.0"            # Addon version

packages:                    # Python packages to install
  - faster-whisper>=1.0.0
  - av>=12.0.0

hardware_notes:              # Optional hardware guidance
  cuda: CUDA provides 10-50x faster transcription
  metal: MPS not supported, falls back to CPU
  cpu: CPU works but is slower
```

### Composite Addon (with Dependencies)

Composite addons depend on other addons and don't install packages themselves:

```yaml
name: speech
display_name: Speech Processing
description: Full speech processing with STT and TTS
component: universal-runtime
version: "1.0.0"

# Dependencies on other addons
dependencies:
  - stt
  - tts

# No direct packages (meta-addon)
packages: []

hardware_notes:
  cuda: CUDA acceleration benefits both STT and TTS
```

When installing a composite addon, all dependencies are automatically installed first.

## Adding a New Addon

### 1. Create Addon YAML File

Create a new file in `addons/registry/`:

```bash
# Example: addons/registry/embeddings.yaml
cat > addons/registry/embeddings.yaml <<EOF
name: embeddings
display_name: Text Embeddings
description: Generate embeddings with sentence-transformers
component: universal-runtime
version: "1.0.0"

packages:
  - sentence-transformers>=2.0.0
  - faiss-cpu>=1.7.0

hardware_notes:
  cuda: CUDA significantly speeds up embedding generation
  cpu: CPU works but is slower for large batches
EOF
```

### 2. Test Locally

```bash
# Verify the registry loads correctly
python -c "from server.api.routers.addons.registry import get_addon_registry; import json; print(json.dumps(get_addon_registry()['embeddings'], indent=2))"

# Build wheels for testing
python tools/build_addon_wheels.py --addon embeddings --platform macos-arm64
```

### 3. Test Installation

```bash
# Create branch and push
git checkout -b feat-add-embeddings-addon
git commit -am "feat(addons): add embeddings addon"
git push

# Trigger branch build
gh workflow run test-addon-branch.yml \
  -f branch=feat-add-embeddings-addon \
  -f addon=embeddings \
  -f platform=macos-arm64 \
  -f create_release=true
```

## Addon Dependencies

Addons can depend on other addons using the `dependencies` field. This enables:

1. **Composite Addons**: Convenience addons that install multiple related addons
2. **Shared Components**: Multiple addons can share common dependencies
3. **Simplified Installation**: Users install one addon, get everything they need

### Dependency Resolution

When installing an addon with dependencies:

```bash
$ lf addons install speech

Installing Speech Processing and its dependencies:
  - Speech-to-Text
  - Text-to-Speech
  - Speech Processing

Detected hardware: Metal
Installing Speech-to-Text...
# ... installs stt ...

Installing Text-to-Speech...
# ... installs tts ...

Installing Speech Processing...
Meta-addon (no packages to install)

All addons installed successfully!
```

**Features**:
- Automatic dependency resolution
- Correct install order (dependencies first)
- Circular dependency detection
- Skips already-installed dependencies
- Meta-addons (no packages) are tracked but don't download wheels

### Creating a Composite Addon

To create an addon that combines other addons:

```yaml
name: my-composite
display_name: My Composite Feature
description: Combines multiple addons for a complete feature
component: universal-runtime
version: "1.0.0"

dependencies:
  - stt
  - tts
  - embeddings

packages: []  # Empty for meta-addons
```

## Component Integration

### CLI (Go)

**File**: [cli/cmd/addons_registry.go](../cli/cmd/addons_registry.go)

The CLI loads all YAML files from `addons/registry/` at runtime:

```go
// Load registry (cached via sync.Once)
if err := LoadAddonRegistry(); err != nil {
    // handle error
}

// Access addon
addon := AddonRegistry["stt"]

// Check dependencies
if len(addon.Dependencies) > 0 {
    // Resolve and install dependencies first
}
```

**Fallback Locations**:
1. `~/.llamafarm/src/addons/registry/` (source mode)
2. Relative to CLI binary (binary distribution)
3. `../addons/registry/` (development)

### Server (Python)

**File**: [server/api/routers/addons/registry.py](../server/api/routers/addons/registry.py)

The server loads all YAML files and caches the result:

```python
from server.api.routers.addons.registry import get_addon_registry

# Get registry (cached)
registry = get_addon_registry()
addon = registry["stt"]

# Access dependencies
dependencies = addon.get("dependencies", [])
```

### Build Script (Python)

**File**: [tools/build_addon_wheels.py](../tools/build_addon_wheels.py)

The build script loads all addons and skips meta-addons (no packages):

```python
# Loads from addons/registry/*.yaml
specs = load_addon_specs()

# Meta-addons are skipped during build
if not spec["packages"]:
    print(f"Skipping {addon_name} (meta-addon)")
    return
```

## API Endpoints

The server exposes addon metadata via REST API.

### GET /v1/addons

Returns list of all addons with installation status:

```json
[
  {
    "name": "stt",
    "display_name": "Speech-to-Text",
    "description": "Transcribe audio using Whisper models",
    "component": "universal-runtime",
    "version": "1.0.0",
    "dependencies": [],
    "installed": true,
    "installed_at": "2026-02-02T12:34:56Z"
  },
  {
    "name": "speech",
    "display_name": "Speech Processing",
    "description": "Full speech processing with STT and TTS",
    "component": "universal-runtime",
    "version": "1.0.0",
    "dependencies": ["stt", "tts"],
    "installed": false,
    "installed_at": null
  }
]
```

This endpoint is used by:
- Designer UI to show available addons
- CLI (future) for remote addon discovery
- API clients for automation

## Package Requirements

Packages listed in `packages` must meet these criteria:

1. **Binary Wheels Available**: Must have pre-built wheels for target platforms
   - Check with: `pip download --only-binary=:all: package-name`
2. **Version Specs**: Use semantic versioning constraints
   - `package>=1.0.0` - Minimum version
   - `package>=1.0.0,<2.0.0` - Version range
3. **Platform Compatibility**: Must work on all target platforms:
   - macOS (ARM64, x86_64)
   - Linux (x86_64, arm64)
   - Windows (x86_64)

If a package doesn't have binary wheels, either:
- Exclude it from the addon (document as manual install)
- Build wheels manually and host them
- Use an alternative package

## Hardware Notes

Hardware notes provide guidance for users based on their system:

```yaml
hardware_notes:
  cuda: CUDA provides 10-50x faster transcription
  metal: MPS not supported by faster-whisper, falls back to CPU
  rocm: ROCm supported with similar performance to CUDA
  cpu: CPU works but is slower
```

These are displayed when installing an addon, helping users understand performance implications.

## Versioning

- **Addon Version**: Tracks the addon definition version (bumped when packages change)
- **Package Versions**: Constrained via version specs in `packages` list

When updating packages:
1. Update `packages` list in the addon's YAML file
2. Increment addon `version`
3. Test with snapshot release before merging

## Validation

Before committing changes:

```bash
# Validate YAML syntax
yamllint addons/registry/*.yaml

# Test Python loading
python -c "from server.api.routers.addons.registry import get_addon_registry; print(len(get_addon_registry()))"

# Test Go loading
./dist/lf addons list

# Test build script
python tools/build_addon_wheels.py --addon all --platform macos-arm64 --output /tmp/test
```

## Examples

### View Available Addons

```bash
$ lf addons list

Available Addons:

NAME    DESCRIPTION                              COMPONENT          STATUS
speech  Full speech processing with STT and TTS  universal-runtime  Not installed
stt     Transcribe audio using Whisper models    universal-runtime  Not installed
tts     High-quality neural TTS with Kokoro      universal-runtime  Not installed
```

### Install an Addon

```bash
# Install single addon
$ lf addons install stt

# Install composite addon (installs all dependencies)
$ lf addons install speech
```

### Uninstall an Addon

```bash
$ lf addons uninstall stt
```

**Note**: Uninstalling an addon that other addons depend on does NOT uninstall the dependents.

## Future Enhancements

Potential improvements to the registry system:

- [ ] JSON Schema validation for addon YAML files
- [ ] Automated tests for registry changes
- [ ] Conflict detection (two addons requiring incompatible packages)
- [ ] Platform-specific package variations
- [ ] License information in registry
- [ ] Download size estimates
- [ ] Automatic dependency cleanup on uninstall
