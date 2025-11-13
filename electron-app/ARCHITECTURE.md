# LlamaFarm Desktop - Architecture

This document describes the technical architecture of the LlamaFarm Desktop application.

## Overview

LlamaFarm Desktop is an Electron application that provides a native desktop experience for the LlamaFarm AI platform. It handles automatic installation, backend orchestration, health monitoring, and provides a seamless user experience without requiring terminal interaction.

## Technology Stack

- **Electron 28**: Cross-platform desktop framework
- **TypeScript 5**: Type-safe development
- **electron-vite**: Fast Vite-based build system
- **electron-builder**: Application packaging and distribution
- **Axios**: HTTP client for health checks and downloads
- **Node.js 18+**: Runtime environment

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                      (Renderer Process)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐        ┌──────────────────────────────┐  │
│  │    Splash    │        │       Main Window            │  │
│  │   Screen     │   →    │  ┌────────────────────────┐  │  │
│  │              │        │  │    Status Bar          │  │  │
│  │  Progress    │        │  ├────────────────────────┤  │  │
│  │  Updates     │        │  │    Designer UI         │  │  │
│  └──────────────┘        │  │  (iframe/embedded)     │  │  │
│                          │  └────────────────────────┘  │  │
│                          └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ▲ IPC ▼
┌─────────────────────────────────────────────────────────────┐
│                      Preload Script                          │
│                   (Secure IPC Bridge)                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Exposes safe APIs via contextBridge:                │    │
│  │  - backend.getStatus()                              │    │
│  │  - backend.restart()                                │    │
│  │  - backend.onStatusChange()                         │    │
│  │  - cli.getInfo()                                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ▲ IPC ▼
┌─────────────────────────────────────────────────────────────┐
│                      Main Process                            │
│                    (Node.js Runtime)                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │ Window Manager  │  │  CLI Installer   │  │  Backend   │ │
│  │                 │  │                  │  │  Manager   │ │
│  │ - Splash        │  │ - Platform       │  │            │ │
│  │ - Main Window   │  │   Detection      │  │ - Start    │ │
│  │ - Updates       │  │ - Download       │  │ - Stop     │ │
│  │                 │  │ - Verify         │  │ - Monitor  │ │
│  └─────────────────┘  └──────────────────┘  └────────────┘ │
│                                                    │         │
│                                      ┌─────────────▼──────┐  │
│                                      │  Health Checker    │  │
│                                      │                    │  │
│                                      │ - Poll /health     │  │
│                                      │ - Track status     │  │
│                                      │ - Auto-recovery    │  │
│                                      └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   LlamaFarm Backend                          │
│                   (Child Process)                            │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────┐    ┌────────────┐    ┌──────────────┐      │
│  │   CLI      │ → │   Server   │    │  RAG Worker  │      │
│  │  (lf)      │    │  (FastAPI) │    │  (Celery)    │      │
│  └────────────┘    └────────────┘    └──────────────┘      │
│                           │                   │              │
│                           ▼                   ▼              │
│                    ┌──────────────────────────────┐          │
│                    │   Docker Containers          │          │
│                    │   - Chroma DB                │          │
│                    │   - Redis                    │          │
│                    └──────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Main Process

The main process is the application's core, running in Node.js. It has full access to system resources and manages application lifecycle.

#### Window Manager (`src/main/window-manager.ts`)

**Responsibilities**:
- Create and manage Electron windows
- Control window lifecycle (show, hide, close)
- Update window contents via IPC

**Windows**:
1. **Splash Window** (500x400, frameless)
   - Shows during initialization
   - Displays progress updates
   - Auto-closes when ready

2. **Main Window** (1400x900)
   - Embeds Designer UI
   - Shows status bar
   - Provides restart controls

#### CLI Installer (`src/main/backend/cli-installer.ts`)

**Responsibilities**:
- Detect if CLI is installed
- Download appropriate binary for platform
- Install to app's userData directory
- Verify installation

**Platform Detection**:
```typescript
Platform: darwin  → OS: macOS
Arch:     arm64   → Binary: lf-darwin-arm64

Platform: darwin  → OS: macOS
Arch:     x64     → Binary: lf-darwin-amd64

Platform: win32   → OS: Windows
Arch:     x64     → Binary: lf-windows-amd64.exe

Platform: linux   → OS: Linux
Arch:     x64     → Binary: lf-linux-amd64
```

**Download Process**:
1. Fetch latest release from GitHub API
2. Construct download URL
3. Download binary with progress tracking
4. Save to `{userData}/bin/lf`
5. Make executable (Unix)
6. Verify with `lf version`

#### Backend Manager (`src/main/backend/backend-manager.ts`)

**Responsibilities**:
- Start backend process (`lf start --no-tui`)
- Monitor process health
- Handle crashes and auto-restart
- Graceful shutdown

**Lifecycle**:
```
stopped → starting → running → (monitor) → running
                                   ↓
                              degraded/error
                                   ↓
                              auto-restart (3x max)
```

**Process Management**:
- Spawns `lf start --no-tui` as child process
- Captures stdout/stderr
- Monitors exit events
- Sets environment variables
- Handles graceful termination

**Auto-Recovery**:
- Exponential backoff: 2s, 4s, 8s, 16s, 32s
- Max 3 restart attempts
- Resets on successful startup
- User notification on failure

#### Health Checker (`src/main/backend/health-checker.ts`)

**Responsibilities**:
- Poll `/health` endpoint
- Parse component status
- Determine overall health
- Trigger UI updates

**Health States**:
- **healthy**: All components ready
- **degraded**: Some components not ready
- **unhealthy**: No components ready or unreachable

**Polling Strategy**:
- During startup: Every 3 seconds
- During runtime: Every 10 seconds
- Timeout: 5 seconds per request

### Preload Script

The preload script acts as a secure bridge between main and renderer processes.

**Security Features**:
- Context isolation enabled
- No nodeIntegration
- Explicit API surface via contextBridge
- No direct access to Node.js APIs

**Exposed API**:
```typescript
window.llamafarm = {
  backend: {
    getStatus: () => Promise<BackendStatus>
    restart: () => Promise<{ success: boolean; error?: string }>
    stop: () => Promise<{ success: boolean; error?: string }>
    onStatusChange: (callback: (status: BackendStatus) => void) => void
  },
  cli: {
    getInfo: () => Promise<{ isInstalled: boolean; path: string | null }>
  },
  splash: {
    onStatus: (callback: (status: SplashStatus) => void) => void
  },
  platform: string,
  version: string
}
```

### Renderer Process

The renderer process runs the UI in a Chromium browser context.

#### Splash Screen (`src/renderer/splash.html`)

**Features**:
- Beautiful gradient background
- Animated logo
- Progress bar
- Status messages
- Error display

**Updates**:
Receives status updates via IPC and updates UI elements.

#### Main Window (`src/renderer/index.html`)

**Features**:
- Persistent status bar
- Embedded Designer (iframe)
- Restart/stop controls
- Real-time status updates

**Designer Integration**:
- Development: Loads from `http://localhost:3000`
- Production: Serves from bundled files

## Data Flow

### Startup Sequence

```
1. App Launch
   ↓
2. Main Process Initialization
   ↓
3. Create Splash Window
   ↓
4. CLI Installation Check
   ├─ Found → Continue
   └─ Not Found → Download & Install
   ↓
5. Start Backend Manager
   ↓
6. Spawn `lf start --no-tui`
   ↓
7. Wait for Health Check (max 3 min)
   ├─ Success → Continue
   └─ Failure → Retry or Error
   ↓
8. Create Main Window
   ↓
9. Close Splash
   ↓
10. Start Health Monitoring (every 10s)
```

### Status Update Flow

```
Health Checker (Main)
   ↓ Poll /health
Backend Status Change
   ↓ IPC Event
Preload Script
   ↓ Callback
Renderer Process
   ↓ Update UI
Status Bar / Window Content
```

### User Action Flow

```
User Clicks "Restart"
   ↓
Renderer: llamafarm.backend.restart()
   ↓ IPC invoke
Main Process: backend:restart handler
   ↓
Backend Manager: stop() then start()
   ↓
Status Updates via IPC
   ↓
UI Updates
```

## Security Model

### Process Isolation

- **Main Process**: Full system access (trusted)
- **Renderer Process**: Sandboxed (untrusted)
- **Preload Script**: Bridge with explicit API

### IPC Security

- Context isolation: ✅ Enabled
- Node integration: ❌ Disabled
- Web security: ✅ Enabled
- Remote module: ❌ Disabled

### API Exposure

Only safe, controlled APIs are exposed to renderer:
- No filesystem access
- No shell execution
- No arbitrary IPC
- Validated inputs

## Performance Considerations

### Resource Usage

- **Memory**: ~200MB base + backend (~2-4GB total)
- **CPU**: Low when idle, spikes during processing
- **Disk**: ~50MB app + 5GB for backend/models

### Optimization Strategies

1. **Lazy Loading**: Load Designer only when needed
2. **Process Pooling**: Reuse child processes
3. **Efficient Polling**: Adaptive health check intervals
4. **Smart Caching**: Cache CLI binary, config files

### Startup Optimization

- CLI already installed: ~30s to ready
- CLI needs download: ~2-3 min to ready
- Subsequent launches: ~20-30s

## Error Handling

### Error Categories

1. **Installation Errors**
   - Network failures
   - Permission issues
   - Unsupported platform

2. **Runtime Errors**
   - Backend crashes
   - Health check failures
   - Port conflicts

3. **User Errors**
   - Docker not running
   - Insufficient resources
   - Invalid configuration

### Recovery Strategies

1. **Automatic**:
   - Retry downloads (3x)
   - Restart backend (3x)
   - Reconnect health checks

2. **User-Assisted**:
   - Show error dialog
   - Provide solutions
   - Link to docs

3. **Graceful Degradation**:
   - Show partial UI
   - Enable manual controls
   - Preserve user data

## Future Architecture

### Planned Enhancements

1. **Multi-Process Model**
   - Separate processes for heavy tasks
   - Worker threads for background operations

2. **State Management**
   - Redux/Zustand for app state
   - Persistent storage

3. **Plugin System**
   - Dynamic loading
   - Sandboxed execution

4. **Distributed Backend**
   - Remote backend support
   - Multi-project management

## References

- [Electron Security](https://www.electronjs.org/docs/latest/tutorial/security)
- [IPC Best Practices](https://www.electronjs.org/docs/latest/tutorial/ipc)
- [Context Isolation](https://www.electronjs.org/docs/latest/tutorial/context-isolation)
- [Process Model](https://www.electronjs.org/docs/latest/tutorial/process-model)
