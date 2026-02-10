# Electron Performance Patterns

Window management, memory optimization, and startup performance.

## Startup Performance

### Deferred Window Display
```typescript
createMainWindow(): BrowserWindow {
  const mainWindow = new BrowserWindow({
    show: false,  // Don't show until ready
    backgroundColor: '#1a1f2e',  // Prevent white flash
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  })

  // Show when content is ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show()
  })

  // Fallback timeout to prevent infinite loading
  setTimeout(() => {
    if (!mainWindow.isVisible()) {
      console.log('Main window timeout - showing anyway')
      mainWindow.show()
    }
  }, 5000)

  return mainWindow
}
```

### Splash Screen Pattern
```typescript
async onReady(): Promise<void> {
  // 1. Show splash immediately
  const splash = this.windowManager.createSplashWindow()

  try {
    // 2. Initialize in background
    await this.ensureCLI()
    await this.waitForServer()
    await this.ensureModels()

    // 3. Create main window (hidden)
    this.windowManager.createMainWindow()

    // 4. Main window shows via ready-to-show, closes splash
  } catch (error) {
    this.handleStartupError(error)
  }
}
```

### Progress Updates During Startup
```typescript
updateSplash(status: {
  message: string
  progress?: number
  models?: ModelStatus[]
}): void {
  if (this.splashWindow && !this.splashWindow.isDestroyed()) {
    this.splashWindow.webContents.send('splash-status', status)
  }
}

// Usage during initialization
this.windowManager.updateSplash({
  message: 'Checking for LlamaFarm CLI...',
  progress: 10
})
```

## Window Management

### Responsive Window Sizing
```typescript
import { screen } from 'electron'

createMainWindow(): BrowserWindow {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize

  return new BrowserWindow({
    width: Math.min(1400, width),   // Cap at max size
    height: Math.min(900, height),
    minWidth: 800,   // Set minimum sizes
    minHeight: 600
  })
}
```

### Window Reference Management
```typescript
class WindowManager {
  private splashWindow: BrowserWindow | null = null
  private mainWindow: BrowserWindow | null = null

  // Always check before using
  closeSplash(): void {
    if (this.splashWindow && !this.splashWindow.isDestroyed()) {
      this.splashWindow.close()
      this.splashWindow = null
    }
  }

  // Clear reference on close
  createMainWindow(): BrowserWindow {
    this.mainWindow = new BrowserWindow({ ... })

    this.mainWindow.on('closed', () => {
      this.mainWindow = null  // Prevent memory leak
    })

    return this.mainWindow
  }

  // Cleanup all windows
  cleanup(): void {
    if (this.splashWindow && !this.splashWindow.isDestroyed()) {
      this.splashWindow.close()
    }
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.close()
    }
  }
}
```

## External Process Management

### Exponential Backoff for Health Checks
```typescript
async waitForServer(): Promise<void> {
  const maxAttempts = 120
  let attempts = 0
  let delay = 500        // Start with 500ms
  const maxDelay = 2000  // Cap at 2 seconds

  while (attempts < maxAttempts) {
    try {
      // Use 127.0.0.1 instead of localhost to avoid IPv6 delays
      const response = await axios.get('http://127.0.0.1:14345/health', {
        timeout: 3000
      })

      if (response.status === 200) {
        console.log('Server is ready!')
        return
      }
    } catch {
      // Server not ready yet
    }

    await new Promise(resolve => setTimeout(resolve, delay))
    delay = Math.min(delay * 1.1, maxDelay)  // Gradual backoff
    attempts++
  }

  throw new Error(`Server failed to respond after ${maxAttempts} attempts`)
}
```

### Retry Logic for Flaky Operations
```typescript
async startServices(): Promise<void> {
  const timeout = 600000  // 10 minutes for first-time setup
  const maxRetries = 2

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      if (attempt > 1) {
        console.log(`Retry attempt ${attempt}/${maxRetries}...`)
        await new Promise(resolve => setTimeout(resolve, 3000))
      }

      await execAsync(
        `"${this.cliPath}" services start`,
        { timeout }
      )

      // Verify services started
      await new Promise(resolve => setTimeout(resolve, 2000))
      const verifyResult = await execAsync(
        `"${this.cliPath}" services status`,
        { timeout: 60000 }
      )

      if (verifyResult.stdout.includes('State: running')) {
        return  // Success
      }
    } catch (error) {
      if (attempt >= maxRetries) {
        throw error
      }
    }
  }
}
```

## Memory Management

### Logger with File Rotation
```typescript
class Logger {
  private writeStream: fs.WriteStream | null = null

  async initialize(): Promise<void> {
    // Rotate log if too large (> 10MB)
    try {
      const stats = await fs.promises.stat(this.logFile)
      if (stats.size > 10 * 1024 * 1024) {
        const rotatedPath = this.logFile.replace('.log', `.${Date.now()}.log`)
        await fs.promises.rename(this.logFile, rotatedPath)
      }
    } catch {
      // File doesn't exist yet
    }

    this.writeStream = fs.createWriteStream(this.logFile, { flags: 'a' })
  }

  async close(): Promise<void> {
    if (this.writeStream) {
      await new Promise<void>((resolve) => {
        this.writeStream!.end(() => resolve())
      })
      this.writeStream = null
    }
  }
}
```

### Cleanup Event Listeners
```typescript
// Preload: Return cleanup functions
const api = {
  backend: {
    onStatusChange: (callback: (status: BackendStatus) => void) => {
      const listener = (_event: IpcRendererEvent, status: BackendStatus) => {
        callback(status)
      }
      ipcRenderer.on('backend-status', listener)

      // Return cleanup function
      return () => ipcRenderer.removeListener('backend-status', listener)
    }
  }
}

// Renderer: Use cleanup on unmount
useEffect(() => {
  const cleanup = window.llamafarm.backend.onStatusChange((status) => {
    setStatus(status)
  })

  return cleanup  // Remove listener on unmount
}, [])
```

## CSS Performance

### Inject CSS After Load
```typescript
mainWindow.webContents.on('did-finish-load', () => {
  mainWindow?.webContents.insertCSS(`
    /* Make the header draggable */
    header {
      -webkit-app-region: drag;
      padding-left: 80px !important;
    }

    /* Keep buttons clickable */
    header button, header a {
      -webkit-app-region: no-drag;
    }
  `)
})
```

### Efficient Splash Screen Updates
```typescript
// Use efficient DOM updates in splash screen
function renderModels(models: ModelStatus[]): void {
  const container = document.getElementById('models-container')

  // Clear and rebuild (simple for small lists)
  container.innerHTML = ''

  models.forEach(model => {
    const item = document.createElement('div')
    item.className = 'model-item'

    // Use textContent for safety and performance
    const name = document.createElement('div')
    name.textContent = model.display_name

    item.appendChild(name)
    container.appendChild(item)
  })
}
```

## Build Performance

### Electron Vite Configuration
```typescript
// electron.vite.config.ts
import { defineConfig, externalizeDepsPlugin } from 'electron-vite'

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()],  // Externalize Node dependencies
    build: {
      outDir: 'dist/main'
    }
  },
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      outDir: 'dist/preload'
    }
  },
  renderer: {
    root: 'src/renderer',
    build: {
      outDir: '../../dist/renderer'
    }
  }
})
```

---

## Checklist

### PERF-001: Hide window until ready
- **Description**: Windows should not show until content is loaded
- **Search**: `grep -r "new BrowserWindow" electron-app/src/`
- **Pass**: show: false with ready-to-show event handler
- **Fail**: Window shows immediately or during load
- **Severity**: High
- **Fix**: Set show: false, add once('ready-to-show') handler

### PERF-002: Set background color
- **Description**: Windows should have background color to prevent white flash
- **Search**: `grep -r "backgroundColor" electron-app/src/`
- **Pass**: backgroundColor set in BrowserWindow options
- **Fail**: No background color causes white flash
- **Severity**: Medium
- **Fix**: Add backgroundColor matching app theme

### PERF-003: Use splash screen for long startups
- **Description**: Show splash screen during initialization
- **Search**: `grep -r "createSplashWindow\|splash" electron-app/src/`
- **Pass**: Splash shown during async initialization
- **Fail**: User sees blank/frozen window during startup
- **Severity**: High
- **Fix**: Create splash window first, close after main ready

### PERF-004: Null window references on close
- **Description**: Clear window references to prevent memory leaks
- **Search**: `grep -r "on.*closed" electron-app/src/`
- **Pass**: Window variables set to null in closed handler
- **Fail**: References kept after window closed
- **Severity**: Medium
- **Fix**: Add `window.on('closed', () => { this.window = null })`

### PERF-005: Use 127.0.0.1 not localhost
- **Description**: Use IP address to avoid DNS/IPv6 resolution delays
- **Search**: `grep -r "localhost" electron-app/src/main/`
- **Pass**: Health checks use 127.0.0.1
- **Fail**: Using localhost for local connections
- **Severity**: Low
- **Fix**: Replace localhost with 127.0.0.1 for performance

### PERF-006: Implement exponential backoff
- **Description**: Retries should use increasing delays
- **Search**: `grep -r "while.*attempts\|retry" electron-app/src/`
- **Pass**: Delay increases between retry attempts
- **Fail**: Fixed delay between retries
- **Severity**: Medium
- **Fix**: Use `delay = Math.min(delay * multiplier, maxDelay)`

### PERF-007: Set command timeouts
- **Description**: External commands must have timeouts
- **Search**: `grep -r "execAsync" electron-app/src/`
- **Pass**: All execAsync calls include timeout option
- **Fail**: Commands can hang indefinitely
- **Severity**: High
- **Fix**: Add { timeout: milliseconds } to all execAsync calls

### PERF-008: Return cleanup functions
- **Description**: Event listener subscriptions must return cleanup
- **Search**: `grep -r "ipcRenderer.on" electron-app/src/preload/`
- **Pass**: All listeners return removeListener function
- **Fail**: Event subscriptions cannot be cleaned up
- **Severity**: Medium
- **Fix**: Return `() => ipcRenderer.removeListener(channel, listener)`

### PERF-009: Rotate log files
- **Description**: Log files should be rotated to prevent growth
- **Search**: `grep -r "WriteStream\|createWriteStream" electron-app/src/`
- **Pass**: Log rotation implemented for large files
- **Fail**: Unbounded log file growth
- **Severity**: Low
- **Fix**: Check file size and rename before creating new stream
