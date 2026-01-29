# Electron IPC and Process Patterns

Main/renderer process communication and Electron architecture patterns.

## Process Model

Electron has three process types with different capabilities:

| Process | Context | Node.js | DOM | Purpose |
|---------|---------|---------|-----|---------|
| Main | Node.js | Yes | No | App lifecycle, native APIs, windows |
| Preload | Bridge | Limited | Yes | Secure IPC bridge |
| Renderer | Browser | No | Yes | UI rendering |

## IPC Communication

### Handle/Invoke Pattern (Request/Response)

Main process handler:
```typescript
import { ipcMain } from 'electron'

// Define typed response
interface CLIInfo {
  isInstalled: boolean
  path: string | null
}

// Register handler
ipcMain.handle('cli:info', async (): Promise<CLIInfo> => {
  const isInstalled = await this.cliInstaller.isInstalled()
  return {
    isInstalled,
    path: isInstalled ? this.cliInstaller.getCLIPath() : null
  }
})
```

Preload bridge:
```typescript
import { contextBridge, ipcRenderer } from 'electron'

const api = {
  cli: {
    getInfo: (): Promise<CLIInfo> => ipcRenderer.invoke('cli:info')
  }
}

contextBridge.exposeInMainWorld('llamafarm', api)
```

Renderer usage:
```typescript
// Type declaration for exposed API
declare global {
  interface Window {
    llamafarm: {
      cli: {
        getInfo: () => Promise<CLIInfo>
      }
    }
  }
}

// Use the API
const info = await window.llamafarm.getInfo()
```

### Send/On Pattern (One-way Events)

Main to renderer:
```typescript
// Main process sends
mainWindow.webContents.send('splash-status', {
  message: 'Loading...',
  progress: 50
})

// Preload exposes listener
const api = {
  splash: {
    onStatus: (callback: (status: SplashStatus) => void) => {
      const listener = (_event: IpcRendererEvent, status: SplashStatus) => {
        callback(status)
      }
      ipcRenderer.on('splash-status', listener)
      // Return cleanup function
      return () => ipcRenderer.removeListener('splash-status', listener)
    }
  }
}
```

### Channel Naming Convention

Use namespaced channel names with colons for request-response (handle/invoke) and hyphens for one-way events (send/on):

```typescript
// Pattern: domain:action (for handle/invoke - request-response)
'cli:info'           // Get CLI information
'cli:install'        // Install CLI
'backend:status'     // Backend status check (invoke)
'backend:restart'    // Restart backend (invoke)

// Pattern: domain-event (for send/on - one-way from main to renderer)
'splash-status'      // Splash screen updates
'backend-status'     // Backend status broadcasts
```

**Note**: The colon vs hyphen distinction helps developers quickly identify communication patterns in the codebase.

## Window Management

### Creating Windows
```typescript
import { BrowserWindow, screen } from 'electron'

createMainWindow(): BrowserWindow {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize

  const mainWindow = new BrowserWindow({
    width: Math.min(1400, width),
    height: Math.min(900, height),
    show: false, // Don't show until ready
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      nodeIntegration: false,
      contextIsolation: true
    },
    titleBarStyle: 'hidden', // Custom title bar
    trafficLightPosition: { x: 20, y: 18 }, // macOS traffic lights
    backgroundColor: '#1a1f2e'
  })

  mainWindow.once('ready-to-show', () => {
    mainWindow.show()
  })

  return mainWindow
}
```

### Splash Screen Pattern
```typescript
createSplashWindow(): BrowserWindow {
  const splash = new BrowserWindow({
    width: 500,
    height: 400,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: false,
    movable: true,
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  })

  // Load embedded HTML or file
  splash.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(splashHTML)}`)
  splash.center()
  splash.show()

  return splash
}
```

## App Lifecycle

### Startup Sequence
```typescript
class ElectronApp {
  constructor() {
    this.setupEventHandlers()
    this.setupIPCHandlers()
  }

  private setupEventHandlers(): void {
    app.on('ready', () => this.onReady())
    app.on('window-all-closed', () => this.onWindowsClosed())
    app.on('activate', () => this.onActivate()) // macOS dock click
    app.on('before-quit', () => this.onBeforeQuit())
    app.on('will-quit', (event) => this.onWillQuit(event))
  }

  private async onReady(): Promise<void> {
    // 1. Create splash screen
    const splash = this.createSplashWindow()

    try {
      // 2. Initialize services
      await this.initializeServices()

      // 3. Create main window
      this.createMainWindow()
    } catch (error) {
      this.handleStartupError(error)
    }
  }
}
```

### Graceful Shutdown
```typescript
private async onWillQuit(event: Electron.Event): Promise<void> {
  event.preventDefault()

  try {
    // Stop background services
    await execAsync(`"${this.cliPath}" services stop`, { timeout: 30000 })

    // Cleanup windows
    this.windowManager.cleanup()

    // Close logger
    await logger.close()

    app.exit(0)
  } catch (error) {
    console.error('Shutdown error:', error)
    app.exit(1)
  }
}
```

## External Process Management

### Spawning CLI Commands
```typescript
import { exec } from 'child_process'
import { promisify } from 'util'

const execAsync = promisify(exec)

async startServices(): Promise<void> {
  const timeout = 600000 // 10 minutes for first-time setup

  const { stdout, stderr } = await execAsync(
    `"${this.cliPath}" services start`,
    { timeout }
  )

  console.log('Services output:', stdout)
  if (stderr) console.warn('Services stderr:', stderr)
}
```

### Retry Logic for External Services
```typescript
async waitForServer(): Promise<void> {
  const maxAttempts = 120
  let attempts = 0
  let delay = 500
  const maxDelay = 2000

  while (attempts < maxAttempts) {
    try {
      const response = await axios.get('http://127.0.0.1:14345/health', {
        timeout: 3000
      })
      if (response.status === 200) return
    } catch {
      // Server not ready yet
    }

    await new Promise(resolve => setTimeout(resolve, delay))
    delay = Math.min(delay * 1.1, maxDelay) // Exponential backoff
    attempts++
  }

  throw new Error(`Server failed to respond after ${maxAttempts} attempts`)
}
```

---

## Checklist

### IPC-001: Use handle/invoke for request-response
- **Description**: Use ipcMain.handle with ipcRenderer.invoke for async operations
- **Search**: `grep -r "ipcMain.on\|ipcRenderer.send" electron-app/src/`
- **Pass**: Request-response uses handle/invoke pattern
- **Fail**: Using on/send for request-response
- **Severity**: High
- **Fix**: Replace with ipcMain.handle and ipcRenderer.invoke

### IPC-002: Type IPC channel payloads
- **Description**: Define TypeScript interfaces for all IPC payloads
- **Search**: `grep -r "ipcMain.handle\|ipcRenderer.invoke" electron-app/src/`
- **Pass**: All handlers have typed parameters and return types
- **Fail**: Using `any` or untyped payloads
- **Severity**: High
- **Fix**: Define interfaces for each channel's request/response

### IPC-003: Return cleanup functions from listeners
- **Description**: Preload event listeners must return cleanup functions
- **Search**: `grep -r "ipcRenderer.on" electron-app/src/preload/`
- **Pass**: All listeners return removeListener function
- **Fail**: No cleanup mechanism for event subscriptions
- **Severity**: Medium
- **Fix**: Return `() => ipcRenderer.removeListener(channel, listener)`

### IPC-004: Namespace channel names
- **Description**: IPC channels should use domain:action naming
- **Search**: `grep -rE "ipc(Main|Renderer)\.(handle|invoke|on|send)\(" electron-app/src/`
- **Pass**: All channels follow namespace:action pattern
- **Fail**: Generic or unclear channel names
- **Severity**: Low
- **Fix**: Rename to pattern like 'cli:info', 'backend:status'

### IPC-005: Hide window until ready
- **Description**: Windows should not show until content is loaded
- **Search**: `grep -r "new BrowserWindow" electron-app/src/`
- **Pass**: show: false with ready-to-show listener
- **Fail**: Window visible during load
- **Severity**: Medium
- **Fix**: Set show: false, add `once('ready-to-show', () => show())`

### IPC-006: Quote executable paths
- **Description**: External command paths must be quoted for spaces
- **Search**: `grep -r "execAsync\|exec(" electron-app/src/`
- **Pass**: All paths wrapped in quotes: `"${path}"`
- **Fail**: Unquoted paths that may contain spaces
- **Severity**: Critical
- **Fix**: Use template literals with quotes: `"${this.cliPath}" command`

### IPC-007: Set command timeouts
- **Description**: External commands must have timeouts
- **Search**: `grep -r "execAsync" electron-app/src/`
- **Pass**: All execAsync calls include timeout option
- **Fail**: Commands without timeout can hang indefinitely
- **Severity**: High
- **Fix**: Add `{ timeout: milliseconds }` option

### IPC-008: Handle macOS activation
- **Description**: App should recreate window on dock click
- **Search**: `grep -r "app.on.*activate" electron-app/src/`
- **Pass**: activate handler creates window if none exist
- **Fail**: Missing or empty activate handler
- **Severity**: Medium
- **Fix**: Add `if (BrowserWindow.getAllWindows().length === 0) createMainWindow()`
