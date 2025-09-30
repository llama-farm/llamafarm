# LlamaFarm CLI (lf)

Go-based CLI for creating projects, managing datasets, and chatting with your runtime.

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

## Commands
See the [CLI reference](../docs/website/docs/cli/index.md) for an exhaustive list. Some quick examples:

```bash
lf init my-project
lf start
lf datasets create -s pdf_ingest -b main_db research
lf datasets upload research ./docs/*.pdf
lf datasets process research
lf rag query --database main_db "summarize"
lf chat "hello"
```

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

## Development Notes
- Commands live under `cmd/` as Cobra subcommands.
- Shared helpers (HTTP clients, config resolution) live in `cmd/*` modules.
- Regenerate Go config types after schema updates via `config/generate-types.sh`.
