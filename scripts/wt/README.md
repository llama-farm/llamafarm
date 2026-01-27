# wt - LlamaFarm Worktree Manager

Manages multiple concurrent LlamaFarm development environments using git worktrees. Enables parallel agent coding sessions without port conflicts or service collisions.

## Why Worktrees?

When running multiple AI coding agents (Claude, Cursor, etc.) on the same codebase, each needs:
- Isolated file changes (no merge conflicts mid-session)
- Separate service instances (server, runtime, designer)
- Non-conflicting ports
- Independent data directories

Git worktrees provide the file isolation. `wt` handles everything else.

## Quick Start

```bash
# Install wt globally
./wt self-install

# Create a new worktree (creates branch, installs deps, builds, starts services)
wt create feat/my-feature

# Create and switch to the worktree directory
wt create --go feat/my-feature

# List all worktrees
wt list

# Switch to a worktree (changes directory)
wt switch feat/my-feature

# Open Designer in browser
wt open
```

## Installation

### Option 1: Self-install (Recommended)

```bash
# From the llamafarm repo
./scripts/wt/wt self-install
source ~/.zshrc  # or ~/.bashrc
```

This creates a symlink at `/usr/local/bin/wt` and adds shell initialization to your RC file for directory switching and tab completion.

### Option 2: Manual Setup

Add to your shell config:

**Bash** (`~/.bashrc`):
```bash
eval "$(wt init)"
```

**Zsh** (`~/.zshrc`):
```bash
eval "$(wt init)"
```

**Fish** (`~/.config/fish/config.fish`):
```fish
wt init fish | source
```

**Explicit shell** (when $SHELL is incorrect):
```bash
eval "$(wt init zsh)"
```

## Commands

### Worktree Management

| Command | Description |
|---------|-------------|
| `wt create <name>` | Create worktree, install deps, build, start services |
| `wt create --go <name>` | Same as above, then cd into the worktree |
| `wt list` | List all worktrees with status and ports |
| `wt delete [name]` | Stop services and remove worktree |
| `wt switch <name>` | Change directory to worktree |

### Build & Install

| Command | Description |
|---------|-------------|
| `wt install [name]` | Install all dependencies (parallel) |
| `wt build [name]` | Build CLI with correct server URL |

### Service Lifecycle

| Command | Description |
|---------|-------------|
| `wt start [name]` | Start all services (server, runtime, RAG, designer) |
| `wt stop [name]` | Stop all services |
| `wt restart [name]` | Restart all services |
| `wt logs [service]` | Tail logs (server\|rag\|runtime\|designer\|all) |
| `wt status [name]` | Show detailed service status |

### Reverse Proxy (Port-Free URLs)

| Command | Description |
|---------|-------------|
| `wt proxy start` | Start Caddy reverse proxy (requires sudo) |
| `wt proxy stop` | Stop Caddy reverse proxy |
| `wt proxy status` | Show proxy status and routes |
| `wt proxy install` | Install as macOS LaunchDaemon (auto-start on boot) |
| `wt proxy uninstall` | Remove LaunchDaemon |
| `wt proxy reload` | Regenerate Caddyfile and reload |

### Utilities

| Command | Description |
|---------|-------------|
| `wt lf [name] <args>` | Run lf CLI targeting worktree's server |
| `wt url [name]` | Print service URLs |
| `wt open [name]` | Open Designer in browser |
| `wt health [name]` | Check health of all services |
| `wt vscode [name]` | Open in VS Code with proper settings |
| `wt cursor [name]` | Open in Cursor with proper settings |
| `wt prune` | Remove worktrees for merged branches |
| `wt gc` | Clean up orphaned data directories |
| `wt doctor` | Diagnose common issues |

## Architecture

### Directory Structure

```
~/worktrees/llamafarm/           # WT_ROOT - git worktrees
├── .wt/                         # Metadata directory
│   ├── ports.json               # Port assignments registry
│   └── Caddyfile                # Generated Caddy config
├── feat-my-feature/             # Worktree directory (sanitized branch name)
│   ├── .env.wt                  # Auto-generated environment file
│   ├── cli/
│   ├── server/
│   ├── designer/
│   └── ...
└── feat-other-feature/

~/.llamafarm/worktrees/          # WT_DATA_ROOT - isolated data
├── feat-my-feature/
│   ├── logs/                    # Service logs
│   ├── pids/                    # PID files for service management
│   ├── broker/                  # Celery filesystem broker
│   │   ├── in/
│   │   ├── processed/
│   │   └── results/
│   └── projects/                # LlamaFarm projects
└── feat-other-feature/
```

### Port Allocation

Each worktree gets a unique port offset (100-999) computed from a hash of its name:

| Service | Port Formula | Example (offset=150) |
|---------|--------------|----------------------|
| Server | 8000 + offset | 8150 |
| Designer | 5000 + offset | 5150 |
| Runtime | 11000 + offset | 11150 |

Port assignments are stored in `$WT_ROOT/.wt/ports.json` to ensure consistency and detect collisions.

### Caddy Reverse Proxy

When the proxy is running, access worktrees via port-free URLs:

```
http://server.feat-my-feature.localhost
http://designer.feat-my-feature.localhost
http://runtime.feat-my-feature.localhost
```

This requires Caddy (`brew install caddy`) and uses `.localhost` domains which resolve to 127.0.0.1 without `/etc/hosts` modifications.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WT_ROOT` | `~/worktrees/llamafarm` | Directory for git worktrees |
| `WT_DATA_ROOT` | `~/.llamafarm/worktrees` | Directory for isolated data |

## Generated Files

### `.env.wt`

Auto-generated in each worktree with isolated configuration:

```bash
# Worktree Identity
WT_NAME=feat-my-feature
WT_PATH=/Users/you/worktrees/llamafarm/feat-my-feature

# Port Assignments
LF_SERVER_PORT=8150
LF_DESIGNER_PORT=5150
LF_RUNTIME_PORT=11150

# Data Isolation
LF_DATA_DIR=/Users/you/.llamafarm/worktrees/feat-my-feature

# Celery Configuration
CELERY_BROKER_URL=filesystem://
CELERY_BROKER_FOLDER=/Users/you/.llamafarm/worktrees/feat-my-feature/broker
```

## Workflow Examples

### Agent Development Session

```bash
# Create isolated environment for an AI agent
wt create --go feat/agent-task

# Agent works in this worktree...
# Services are already running

# Check status
wt status

# View logs if something goes wrong
wt logs server

# When done, clean up
wt delete feat-agent-task
```

### Multiple Parallel Agents

```bash
# Terminal 1: Agent working on feature A
wt create --go feat/feature-a
# Services on ports 8150, 5150, 11150

# Terminal 2: Agent working on feature B
wt create --go feat/feature-b
# Services on ports 8267, 5267, 11267

# Both agents work independently with no conflicts
```

### Using Port-Free URLs

```bash
# One-time setup
wt proxy install
wt proxy start

# Access any worktree without remembering ports
open http://designer.feat-feature-a.localhost
open http://designer.feat-feature-b.localhost
```

## Troubleshooting

### Diagnose Issues

```bash
wt doctor
```

Checks:
- Required tools (git, go, uv, npm, jq, curl)
- Port conflicts
- Stale PID files
- Lock files

### Clean Up Orphaned Data

```bash
wt gc
```

Removes data directories that no longer have corresponding worktrees.

### Reset a Stuck Worktree

```bash
wt stop my-worktree
wt start my-worktree
```

### Force Delete

```bash
wt delete my-worktree --force
```

## Files

| File | Description |
|------|-------------|
| `wt` | Main executable script (includes shell init via `wt init`) |
| `wt-test.sh` | Test suite for core functionality |

## Requirements

- bash 4.0+
- git
- go 1.24+
- uv (Python package manager)
- npm
- jq
- curl
- caddy (optional, for reverse proxy)
- fzf (optional, for interactive worktree selection)
