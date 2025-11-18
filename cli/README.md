# LlamaFarm CLI (lf)

Go-based CLI for creating projects, managing datasets, and chatting with your runtime. The CLI also launches the Designer web UI for users who prefer a visual interface.

## Install
The recommended install flow is the top-level script:

```bash
curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
```

Windows users can download `lf.exe` from the [latest release](https://github.com/llama-farm/llamafarm/releases/latest).

### Build from Source
```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm/cli
go build -o lf
```

Optional:
```bash
sudo mv lf /usr/local/bin/
```

## Upgrading

The CLI includes an auto-upgrade feature that can automatically download and install new versions:

```bash
# Upgrade to latest version
lf version upgrade

# Upgrade to specific version
lf version upgrade v1.2.3

# Preview upgrade without executing
lf version upgrade --dry-run

# Force upgrade even if same version
lf version upgrade --force

# Install to custom directory
lf version upgrade --install-dir ~/.local/bin
```

The upgrade command handles:
- Cross-platform binary downloads from GitHub releases
- SHA256 checksum verification for security
- Automatic sudo/elevation when needed for system directories
- Atomic binary replacement to prevent corruption
- Fallback to user directories if system installation fails

## Commands
See the [CLI reference](../docs/website/docs/cli/index.md) for an exhaustive list. Some quick examples:

```bash
lf init my-project
lf start                                              # Starts server, RAG worker, and Designer web UI
lf datasets create -s pdf_ingest -b main_db research
lf datasets upload research ./docs/*.pdf
lf datasets process research
lf rag query --database main_db "summarize"
lf chat "hello"
```

**Designer Web UI**: When you run `lf start`, the Designer is automatically launched at `http://localhost:7724`. This provides a visual interface for managing projects, uploading datasets via drag-and-drop, and testing your AI—all without additional commands. See the [Designer documentation](../docs/website/docs/designer/index.md) for details.

### Running Backend Services Manually
When developing locally without Docker orchestration, start the server and RAG worker via Nx from the repository root:

```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

npm install -g nx
nx init --useDotNxInstallation --interactive=false

# Option A: single command
nx dev

# Option B: separate terminals
nx start rag    # Terminal 1
nx start server # Terminal 2
```

Then run the CLI (installed binary or `go run main.go ...`) in another terminal.

## Tests
```bash
go test ./...
```

## Environment Variables

### `LF_VERSION_REF`
Override the git ref (branch, tag, or commit SHA) used to download Python source code from the repository. Useful for CI/CD testing or development against specific branches.

```bash
# Test with a feature branch
LF_VERSION_REF=feat-new-feature lf start

# Test with a specific version tag
LF_VERSION_REF=v1.2.3 lf start

# Test with a specific commit
LF_VERSION_REF=abc123def456... lf start
```

**Default behavior:**
- Release builds (e.g., `v1.2.3`) download matching source code tags
- Dev builds automatically use the `main` branch
- Source code is cached in `~/.llamafarm/src` and only re-downloaded when versions change

### Other Environment Variables
- `LLAMAFARM_SESSION_ID` – reuse a session for `lf chat`
- `OLLAMA_HOST` – point to a different Ollama endpoint (default: `http://localhost:11434`)
- `LF_DATA_DIR` – override the data directory (default: `~/.llamafarm`)

## Development Notes
- Commands live under `cmd/` as Cobra subcommands.
- Shared helpers (HTTP clients, config resolution) live in `cmd/*` modules.
- Regenerate Go config types after schema updates via `config/generate_types.py`.
- The CLI's `CurrentVersion` variable is set via `-ldflags` during build: `-X 'github.com/llamafarm/cli/cmd/version.CurrentVersion=v1.2.3'`
