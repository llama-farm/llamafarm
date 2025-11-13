# LlamaFarm Desktop - Development Guide

This guide covers development setup, architecture decisions, and implementation details for the LlamaFarm Desktop application.

## Architecture Overview

### Design Principles

1. **Security First**: Sandboxed renderer, context isolation, no nodeIntegration
2. **Automatic Everything**: CLI installation, backend startup, health monitoring, recovery
3. **Native Feel**: Platform-specific behaviors, system tray, native notifications
4. **Resilient**: Auto-restart on crash, graceful degradation, comprehensive error handling

### Process Model

```
Main Process (Node.js)
├── CLI Installer (downloads & installs lf CLI)
├── Backend Manager (manages lf start process)
├── Health Checker (polls /health endpoint)
└── Window Manager (creates & manages windows)
    ├── Splash Window (startup screen)
    └── Main Window (embeds Designer)

Preload Script (secure bridge)
└── Exposes safe IPC APIs to renderer

Renderer Process (Chromium)
├── Splash Screen (HTML/CSS/JS)
└── Main Window (embeds Designer via iframe)
```

## Development Workflow

### Initial Setup

```bash
# From the electron-app directory
npm install

# Build the Designer first (required for production builds)
cd ../designer
npm install
npm run build

# Return to electron-app
cd ../electron-app
```

### Running in Development

```bash
# Start the backend manually (in another terminal)
cd ..
lf start  # or nx dev

# Start the Electron app
cd electron-app
npm run dev
```

**Development Mode Behavior:**
- Loads Designer from `http://localhost:3000` (or the dev server port)
- Opens DevTools automatically
- Hot reload for renderer changes
- Main process requires restart for changes

### Building

```bash
# Build TypeScript
npm run build

# Package for current platform
npm run pack:mac    # macOS
npm run pack:win    # Windows
npm run pack:linux  # Linux

# Build distributable
npm run dist:mac    # Creates DMG
npm run dist:win    # Creates installer + portable
npm run dist:linux  # Creates AppImage + deb
```

## Implementation Details

### CLI Installation

**File**: `src/main/backend/cli-installer.ts`

The CLI installer:
1. Checks if `lf` is already installed (system PATH or app's bin directory)
2. If not found, downloads the appropriate binary from GitHub releases
3. Stores it in the app's userData directory for portability
4. Verifies installation by running `lf version`

**Platform Detection:**
- macOS: `darwin-amd64` or `darwin-arm64`
- Windows: `windows-amd64.exe`
- Linux: `linux-amd64`, `linux-arm64`

**Storage Location:**
- macOS: `~/Library/Application Support/llamafarm-desktop/bin/lf`
- Windows: `%APPDATA%/llamafarm-desktop/bin/lf.exe`
- Linux: `~/.config/llamafarm-desktop/bin/lf`

### Backend Management

**File**: `src/main/backend/backend-manager.ts`

The backend manager:
1. Spawns `lf start --no-tui` as a child process
2. Captures stdout/stderr for logging
3. Monitors process exit events
4. Implements auto-restart with exponential backoff
5. Provides graceful shutdown on app quit

**Health Monitoring:**
- Polls `/health` endpoint every 10 seconds
- Tracks component status (server, rag, etc.)
- Updates UI via IPC when status changes
- Triggers restart on consecutive failures

**Auto Recovery:**
- Max 3 restart attempts
- Exponential backoff: 2s, 4s, 8s, 16s, 32s (capped at 30s)
- Resets attempt counter on successful startup
- Shows user-friendly error messages

### Window Management

**File**: `src/main/window-manager.ts`

**Splash Window:**
- 500x400, frameless, transparent
- Shows during CLI installation and backend startup
- Updates progress bar based on initialization steps
- Automatically closes when main window is ready

**Main Window:**
- 1400x900 (or smaller based on screen size)
- Embeds Designer UI via iframe (production) or loads dev URL (development)
- Hidden until ready to prevent flickering
- Status bar shows backend health

### IPC Communication

**Files**: `src/main/index.ts`, `src/preload/index.ts`

**Security:**
- Context isolation enabled
- Node integration disabled
- Only safe APIs exposed via contextBridge
- All IPC handlers use `handle/invoke` pattern (not `on/send`)

**Available APIs:**
```typescript
window.llamafarm = {
  backend: {
    getStatus: () => Promise<BackendStatus>
    restart: () => Promise<void>
    stop: () => Promise<void>
    onStatusChange: (callback) => void
  },
  cli: {
    getInfo: () => Promise<CLIInfo>
  },
  splash: {
    onStatus: (callback) => void
  }
}
```

## Testing Strategy

### Manual Testing

1. **Fresh Install Test**
   - Delete CLI: `rm ~/Library/Application\ Support/llamafarm-desktop/bin/lf`
   - Delete app data: `rm -rf ~/Library/Application\ Support/llamafarm-desktop`
   - Launch app and verify CLI downloads
   - Verify backend starts successfully

2. **Backend Crash Recovery**
   - Start app
   - Kill backend: `pkill -f "lf start"`
   - Verify app detects crash
   - Verify auto-restart works

3. **Graceful Shutdown**
   - Start app with backend running
   - Quit app (Cmd+Q)
   - Verify backend stops cleanly
   - No orphaned processes

4. **Health Degradation**
   - Start app
   - Stop a service manually (e.g., kill RAG worker)
   - Verify status bar shows degraded state
   - Verify recovery

### Automated Testing

TODO: Add automated tests

```bash
# Unit tests for business logic
npm test

# E2E tests with Spectron/Playwright
npm run test:e2e
```

## Common Issues & Solutions

### "CLI installation failed"

**Cause**: Network error, GitHub API rate limit, or unsupported platform

**Solution**:
- Check internet connection
- Verify platform is supported
- Install CLI manually and app will detect it

### "Backend won't start"

**Cause**: Port 8000 in use, Docker not running, or corrupted LlamaFarm installation

**Solution**:
- Check `lf start` works manually
- Free up port 8000
- Reinstall LlamaFarm: delete `~/.llamafarm`

### "White screen"

**Cause**: Designer not loading, backend not ready, or network issue

**Solution**:
- Check DevTools console for errors
- Verify backend is healthy (status bar)
- Check Designer URL is correct

## Future Enhancements

### Short Term
- [ ] System tray integration
- [ ] Native notifications
- [ ] Log viewer in UI
- [ ] Settings panel (ports, auto-start, etc.)
- [ ] Auto-updater integration

### Medium Term
- [ ] Windows support
- [ ] Linux support
- [ ] Multiple project support
- [ ] Resource monitoring (CPU, memory)
- [ ] Custom themes

### Long Term
- [ ] Embedded Ollama (no separate install)
- [ ] Embedded Docker (via Docker Desktop API)
- [ ] Plugin system
- [ ] Cloud sync

## Release Process

1. **Update Version**
   ```bash
   npm version patch  # or minor/major
   ```

2. **Build for All Platforms**
   ```bash
   npm run dist:mac
   npm run dist:win
   npm run dist:linux
   ```

3. **Test Built Apps**
   - Test on clean VMs/machines
   - Verify CLI installation
   - Verify backend startup
   - Test core workflows

4. **Create GitHub Release**
   - Tag with version number
   - Upload DMG, exe, AppImage
   - Update release notes

5. **Update Documentation**
   - Update download links in README
   - Update screenshots if needed
   - Update changelog

## Resources

- [Electron Docs](https://www.electronjs.org/docs/latest)
- [electron-builder Docs](https://www.electron.build/)
- [electron-vite Docs](https://electron-vite.org/)
- [LlamaFarm Main Docs](../docs/website/docs/intro.md)

## Getting Help

- Issues: https://github.com/llama-farm/llamafarm/issues
- Discord: https://discord.gg/RrAUXTCVNF
- Discussions: https://github.com/llama-farm/llamafarm/discussions
