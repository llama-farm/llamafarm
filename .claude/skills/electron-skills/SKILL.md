---
name: electron-skills
description: Electron patterns for LlamaFarm Desktop. Covers main/renderer processes, IPC, security, and packaging.
allowed-tools: Read, Grep, Glob
user-invocable: false
---

# Electron Skills for LlamaFarm Desktop

Electron 28 + Electron Vite patterns for the LlamaFarm Desktop application.

## Overview

This skill extends [typescript-skills](../typescript-skills/SKILL.md) with Electron-specific patterns for main/renderer process architecture, IPC communication, security, and performance.

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | Electron 28 | Desktop application framework |
| Build | electron-vite 2 | Vite-based build for main/preload/renderer |
| Updates | electron-updater | Auto-update via GitHub releases |
| Packaging | electron-builder | Cross-platform packaging (macOS/Win/Linux) |

## Architecture

```
electron-app/
  src/
    main/           # Main process (Node.js context)
      index.ts      # App entry, lifecycle, IPC handlers
      backend/      # CLI installer, model downloader
      window-manager.ts
      menu-manager.ts
      logger.ts
    preload/        # Preload scripts (bridge context)
      index.ts      # contextBridge API exposure
    renderer/       # Renderer process (browser context)
      index.html    # Main window
      splash.html   # Splash screen
```

## Core Principles

1. **Process isolation** - Main, preload, and renderer are separate contexts
2. **Context isolation** - Always use `contextBridge.exposeInMainWorld`
3. **No Node in renderer** - `nodeIntegration: false` always
4. **Type-safe IPC** - Define channel types and payload schemas
5. **Secure by default** - Minimize exposed APIs in preload

## Related Documents

- [electron.md](./electron.md) - IPC patterns, main/renderer communication
- [security.md](./security.md) - Context isolation, preload security, CSP
- [performance.md](./performance.md) - Window management, memory, startup

## Shared TypeScript Patterns

This skill inherits from [typescript-skills](../typescript-skills/SKILL.md):
- [patterns.md](../typescript-skills/patterns.md) - Idiomatic TypeScript
- [typing.md](../typescript-skills/typing.md) - Strict typing, generics
- [security.md](../typescript-skills/security.md) - Input validation, XSS prevention

## Quick Reference

### IPC Handler Pattern (Main Process)
```typescript
ipcMain.handle('cli:info', async () => {
  const isInstalled = await this.cliInstaller.isInstalled()
  return {
    isInstalled,
    path: isInstalled ? this.cliInstaller.getCLIPath() : null
  }
})
```

### Preload Bridge Pattern
```typescript
const api = {
  cli: {
    getInfo: () => ipcRenderer.invoke('cli:info')
  },
  platform: process.platform,
  version: process.versions.electron
}

contextBridge.exposeInMainWorld('llamafarm', api)
```

### BrowserWindow Configuration
```typescript
new BrowserWindow({
  webPreferences: {
    preload: path.join(__dirname, '../preload/index.js'),
    nodeIntegration: false,
    contextIsolation: true
  }
})
```

## Checklist Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| IPC | 2 | 3 | 2 | 1 |
| Security | 4 | 3 | 2 | 1 |
| Performance | 1 | 3 | 3 | 2 |
