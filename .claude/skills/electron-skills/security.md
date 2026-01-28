# Electron Security Patterns

Context isolation, preload security, and secure Electron configuration.

## Security Model

Electron security is built on process isolation:

```
Main Process (Node.js)     Preload (Bridge)        Renderer (Browser)
-------------------        ----------------        ------------------
Full Node.js access        Limited Node.js         No Node.js access
Native OS APIs             contextBridge only      Web APIs only
File system access         Exposes safe APIs       Uses exposed APIs
Child processes            No direct IPC access    Sandboxed
```

## Context Isolation

### Always Enable Context Isolation
```typescript
new BrowserWindow({
  webPreferences: {
    nodeIntegration: false,    // REQUIRED: Disable Node in renderer
    contextIsolation: true,    // REQUIRED: Isolate preload context
    preload: path.join(__dirname, '../preload/index.js')
  }
})
```

### Why Context Isolation Matters
- Prevents renderer from accessing Node.js globals
- Isolates preload script's execution context
- Protects against prototype pollution attacks
- Limits blast radius of XSS vulnerabilities

## Preload Script Security

### Expose Minimal API Surface
```typescript
// GOOD: Expose specific, typed functions
const api = {
  cli: {
    getInfo: () => ipcRenderer.invoke('cli:info')
  },
  platform: process.platform  // Safe read-only value
}

contextBridge.exposeInMainWorld('llamafarm', api)

// BAD: Never expose raw IPC
const badApi = {
  ipcRenderer: ipcRenderer,  // NEVER DO THIS
  invoke: ipcRenderer.invoke // NEVER DO THIS
}
```

### Type-Safe API Exposure
```typescript
// Define the exposed API type
interface LlamaFarmAPI {
  cli: {
    getInfo: () => Promise<{ isInstalled: boolean; path: string | null }>
  }
  backend: {
    getStatus: () => Promise<BackendStatus>
    onStatusChange: (callback: (status: BackendStatus) => void) => () => void
  }
  platform: NodeJS.Platform
  version: string
}

const api: LlamaFarmAPI = {
  cli: {
    getInfo: () => ipcRenderer.invoke('cli:info')
  },
  backend: {
    getStatus: () => ipcRenderer.invoke('backend:status'),
    onStatusChange: (callback) => {
      const listener = (_event: IpcRendererEvent, status: BackendStatus) => {
        callback(status)
      }
      ipcRenderer.on('backend-status', listener)
      return () => ipcRenderer.removeListener('backend-status', listener)
    }
  },
  platform: process.platform,
  version: process.versions.electron
}

contextBridge.exposeInMainWorld('llamafarm', api)
export type { LlamaFarmAPI }
```

## DOM Security in Embedded HTML

### Safe DOM Building
```typescript
// When building DOM in splash/embedded HTML, use safe methods

// GOOD: textContent escapes HTML
const name = document.createElement('div')
name.className = 'model-name'
name.textContent = model.display_name  // Safe: escapes HTML

// BAD: innerHTML with untrusted data
element.innerHTML = model.display_name  // XSS vulnerability

// GOOD: Sanitize status to allowlist
const VALID_STATUSES = ['present', 'downloading', 'checking', 'error']
const safeStatus = VALID_STATUSES.includes(model.status)
  ? model.status
  : 'checking'

icon.className = 'model-icon ' + safeStatus  // Safe: validated
```

### Sanitize Numeric Values
```typescript
// Prevent CSS injection through numeric values
const safeProgress = Math.max(0, Math.min(100, Number(model.progress) || 0))
progressBar.style.width = safeProgress + '%'  // Safe: bounded number
```

## Window Security Configuration

### Secure BrowserWindow Options
```typescript
new BrowserWindow({
  webPreferences: {
    nodeIntegration: false,
    contextIsolation: true,
    preload: path.join(__dirname, '../preload/index.js'),

    // Additional security options
    sandbox: true,                    // Enable renderer sandbox
    webSecurity: true,                // Enable same-origin policy
    allowRunningInsecureContent: false,
    experimentalFeatures: false
  }
})
```

### When webSecurity Can Be Disabled
```typescript
// Only disable for localhost development with trusted content
new BrowserWindow({
  webPreferences: {
    webSecurity: false  // Only for localhost CORS during development
  }
})

// Document the reason and scope
// webSecurity: false - Required for localhost:14345 API access
// The Designer UI is served from localhost and calls localhost APIs
```

## External Content Loading

### Validate URLs Before Loading
```typescript
// Only load from trusted origins
const ALLOWED_ORIGINS = [
  'http://localhost:14345',
  'http://127.0.0.1:14345'
]

loadDesigner(url: string): void {
  const parsed = new URL(url)
  const origin = parsed.origin

  if (!ALLOWED_ORIGINS.includes(origin)) {
    throw new Error(`Untrusted origin: ${origin}`)
  }

  this.mainWindow.loadURL(url)
}
```

### Safe External Link Handling
```typescript
import { shell } from 'electron'

// Open external links in system browser
{
  label: 'Documentation',
  click: async () => {
    await shell.openExternal('https://docs.llamafarm.dev')
  }
}

// Validate URL before opening
async openExternalSafe(url: string): Promise<void> {
  const parsed = new URL(url)

  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error(`Invalid protocol: ${parsed.protocol}`)
  }

  await shell.openExternal(url)
}
```

## Auto-Updater Security

### Secure Update Configuration
```typescript
import { autoUpdater } from 'electron-updater'

// Only auto-update in production
if (app.isPackaged) {
  autoUpdater.autoDownload = true
  autoUpdater.autoInstallOnAppQuit = true

  // Verify update signatures (electron-builder handles this)
  // Updates are signed and verified automatically

  autoUpdater.checkForUpdatesAndNotify()
}
```

### Handle Update Errors Gracefully
```typescript
autoUpdater.on('error', (err) => {
  // Log but don't expose error details to user
  console.error('Auto-updater error:', err)
  // Don't show raw error messages that might leak system info
})
```

## File System Security

### Validate File Paths
```typescript
import * as path from 'path'
import * as os from 'os'

// Ensure paths stay within expected directories
function getSafePath(relativePath: string): string {
  const homeDir = os.homedir()
  const basePath = path.join(homeDir, '.llamafarm')
  const fullPath = path.resolve(basePath, relativePath)

  // Prevent directory traversal
  if (!fullPath.startsWith(basePath)) {
    throw new Error('Path traversal detected')
  }

  return fullPath
}
```

### Secure Executable Permissions
```typescript
// Set proper permissions on downloaded executables
await fsPromises.chmod(this.cliPath, 0o755)  // rwxr-xr-x
```

---

## Checklist

### SEC-001: Context isolation enabled
- **Description**: All BrowserWindows must have contextIsolation: true
- **Search**: `grep -r "contextIsolation" electron-app/src/`
- **Pass**: contextIsolation: true in all webPreferences
- **Fail**: contextIsolation: false or missing
- **Severity**: Critical
- **Fix**: Set contextIsolation: true in webPreferences

### SEC-002: Node integration disabled
- **Description**: Renderer processes must not have Node.js access
- **Search**: `grep -r "nodeIntegration" electron-app/src/`
- **Pass**: nodeIntegration: false in all webPreferences
- **Fail**: nodeIntegration: true
- **Severity**: Critical
- **Fix**: Set nodeIntegration: false, use preload for IPC

### SEC-003: No raw IPC exposure
- **Description**: Preload must not expose ipcRenderer directly
- **Search**: `grep -r "ipcRenderer" electron-app/src/preload/`
- **Pass**: Only specific functions exposed via contextBridge
- **Fail**: Raw ipcRenderer or invoke exposed
- **Severity**: Critical
- **Fix**: Wrap each IPC call in a specific function

### SEC-004: Validate exposed API inputs
- **Description**: Functions exposed to renderer must validate inputs
- **Search**: `grep -r "contextBridge.exposeInMainWorld" electron-app/src/`
- **Pass**: Exposed functions validate parameters before IPC
- **Fail**: Parameters passed directly to ipcRenderer
- **Severity**: High
- **Fix**: Add input validation in exposed functions

### SEC-005: Use textContent not innerHTML
- **Description**: DOM manipulation must use textContent for untrusted data
- **Search**: `grep -r "innerHTML" electron-app/src/`
- **Pass**: innerHTML only used with static/sanitized content
- **Fail**: innerHTML with dynamic/untrusted data
- **Severity**: High
- **Fix**: Use textContent or createElement with textContent

### SEC-006: Allowlist dynamic class names
- **Description**: Dynamic CSS classes must be validated against allowlist
- **Search**: `grep -r "className.*+" electron-app/src/`
- **Pass**: Dynamic values checked against VALID_* arrays
- **Fail**: Unvalidated strings concatenated to className
- **Severity**: Medium
- **Fix**: Validate against allowlist before concatenation

### SEC-007: Sanitize numeric style values
- **Description**: Numeric values for styles must be bounded
- **Search**: `grep -r "style\.(width|height|left|top)" electron-app/src/`
- **Pass**: Values bounded with Math.max/min
- **Fail**: Unbounded numbers used in styles
- **Severity**: Medium
- **Fix**: Use `Math.max(0, Math.min(100, Number(value) || 0))`

### SEC-008: Validate URLs before loading
- **Description**: Only load content from trusted origins
- **Search**: `grep -r "loadURL" electron-app/src/`
- **Pass**: URLs validated against allowlist
- **Fail**: Arbitrary URLs loaded without validation
- **Severity**: High
- **Fix**: Check URL origin against ALLOWED_ORIGINS array

### SEC-009: Auto-update only in production
- **Description**: Auto-updater should only run in packaged builds
- **Search**: `grep -r "autoUpdater" electron-app/src/`
- **Pass**: Update checks guarded by app.isPackaged
- **Fail**: Updates checked in development mode
- **Severity**: Medium
- **Fix**: Wrap with `if (app.isPackaged) { ... }`

### SEC-010: Prevent directory traversal
- **Description**: File paths must be validated to prevent traversal
- **Search**: `grep -r "path.join\|path.resolve" electron-app/src/`
- **Pass**: Paths validated to stay within base directory
- **Fail**: User input used directly in path operations
- **Severity**: High
- **Fix**: Use path.resolve and check startsWith(basePath)
