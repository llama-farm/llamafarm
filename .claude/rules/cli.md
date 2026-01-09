# LlamaFarm CLI

## Overview
- The CLI is the primary user interface for LlamaFarm
- Installed globally; the `lf` binary should be in the user's `PATH`
- Orchestrates local service lifecycle (server, RAG worker, runtime processes)

## Architecture

### Entry Points
- `cli/main.go` - Application entry point
- `cli/cmd/root.go` - Root command and global flags
- `cli/cmd/*.go` - Individual command implementations

### Key Subsystems
- **Orchestrator** (`cli/cmd/orchestrator/`) - Manages process spawning, service health checks, and hardware detection
  - `orchestrator.go` - Main orchestration logic
  - `process_manager.go` - Process lifecycle management
  - `hardware_detect.go` - GPU/CPU detection for optimal runtime configuration
  - `python_env.go` - Python virtual environment management via `uv`
  - `services.go` - Service definitions and startup sequences
- **Config** (`cli/cmd/config/`) - YAML configuration parsing and validation
- **Utils** (`cli/cmd/utils/`) - Shared utilities (HTTP client, formatting, logging)
- **Version** (`cli/cmd/version/`) - Version checking and self-upgrade functionality

### Common Commands
- `lf init <project>` - Initialize a new project
- `lf start` - Start all services (server, RAG, runtime)
- `lf chat` - Interactive chat session
- `lf models pull` - Download models
- `lf services status` - Check service health
- `lf rag ingest` - Ingest documents into RAG

## Development

### Running During Development
- Build: `nx build cli` produces binary at `./dist/lf`
- Unless actively modifying CLI code, prefer the system-installed `lf` binary over the dev build

### Testing
- Unit tests are colocated with source files (`*_test.go`)
- Run tests: `cd cli && go test ./...`
