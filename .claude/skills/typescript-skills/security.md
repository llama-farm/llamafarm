# Security Patterns for LlamaFarm TypeScript

XSS prevention, input validation, and secure coding practices.

## XSS Prevention

### Sanitize User Input for Display
```typescript
const MAX_CONFIG_VALUE_LENGTH = 100

export const sanitizeConfigValue = (value: unknown): string => {
  if (!value) return 'Not set'

  const str = String(value)
    .replace(/[<>'"]/g, '') // Remove HTML/script injection characters
    .trim()

  return str.length > MAX_CONFIG_VALUE_LENGTH
    ? str.substring(0, MAX_CONFIG_VALUE_LENGTH) + '...'
    : str
}
```

### React's Built-in Protection
React escapes values in JSX by default:
```tsx
// SAFE: React escapes the content
<div>{userInput}</div>

// DANGEROUS: Bypasses React's protection
<div dangerouslySetInnerHTML={{ __html: userInput }} />
```

### Sanitize HTML Content
When HTML must be rendered, use a sanitizer:
```tsx
import rehypeSanitize from 'rehype-sanitize'
import ReactMarkdown from 'react-markdown'

// Safe markdown rendering
<ReactMarkdown rehypePlugins={[rehypeSanitize]}>
  {userContent}
</ReactMarkdown>
```

## URL Validation

### Validate URLs Before Use
```typescript
export const isValidAndSafeURL = (urlString: string): boolean => {
  try {
    const url = new URL(urlString)

    // Only allow http and https protocols
    if (!['http:', 'https:'].includes(url.protocol)) {
      return false
    }

    // Warn about localhost/private IPs in production
    const hostname = url.hostname.toLowerCase()
    const isLocalhost =
      hostname === 'localhost' ||
      hostname === '127.0.0.1' ||
      hostname.startsWith('192.168.') ||
      hostname.startsWith('10.') ||
      hostname.startsWith('172.')

    if (import.meta.env.PROD && isLocalhost) {
      console.warn('Localhost/private IP in production:', hostname)
    }

    return true
  } catch {
    return false
  }
}
```

### Extract Safe Hostname
```typescript
export const extractSafeHostname = (urlValue: unknown): string => {
  if (!urlValue) return 'Not set'

  const urlString = String(urlValue)

  if (!isValidAndSafeURL(urlString)) {
    return 'Invalid URL'
  }

  try {
    const url = new URL(urlString)
    const hostname = sanitizeConfigValue(url.hostname)
    const port = url.port ? sanitizeConfigValue(url.port) : ''
    return port ? `${hostname}:${port}` : hostname
  } catch {
    return 'Invalid URL'
  }
}
```

## Navigation State Validation

### Validate Router State
```typescript
export const validateNavigationState = (state: unknown): {
  database: string
  strategyName: string
  strategyType: string
  currentConfig: Record<string, unknown>
  isDefault: boolean
} => {
  const s = state as Record<string, unknown>

  // Validate database name (alphanumeric and underscores only)
  const database =
    typeof s?.database === 'string' && /^[a-zA-Z0-9_]+$/.test(s.database)
      ? s.database
      : 'main_database'

  // Validate strategy name
  const strategyName =
    typeof s?.strategyName === 'string' &&
    /^[a-zA-Z0-9\s_-]+$/.test(s.strategyName)
      ? s.strategyName
      : ''

  // Validate strategy type against allowed list
  const allowedTypes = ['BasicSimilarityStrategy', 'MultiQueryStrategy']
  const strategyType =
    typeof s?.strategyType === 'string' && allowedTypes.includes(s.strategyType)
      ? s.strategyType
      : 'BasicSimilarityStrategy'

  // Validate config is object
  const currentConfig =
    s?.currentConfig &&
    typeof s.currentConfig === 'object' &&
    !Array.isArray(s.currentConfig)
      ? (s.currentConfig as Record<string, unknown>)
      : {}

  const isDefault = typeof s?.isDefault === 'boolean' ? s.isDefault : false

  return { database, strategyName, strategyType, currentConfig, isDefault }
}
```

## Input Sanitization

### Sanitize Filter Keys
```typescript
const MAX_FILTER_KEY_LENGTH = 50

export const sanitizeFilterKey = (key: string): string => {
  return key.replace(/[^a-zA-Z0-9_-]/g, '').substring(0, MAX_FILTER_KEY_LENGTH)
}
```

### Sanitize Filter Values
```typescript
const MAX_FILTER_VALUE_LENGTH = 200

export const sanitizeFilterValue = (value: string): string => {
  return value
    .replace(/[<>'"\\]/g, '')
    .trim()
    .substring(0, MAX_FILTER_VALUE_LENGTH)
}
```

### Parse Numeric Values Safely
```typescript
const parseNumericValue = (raw: string): number | null => {
  const num = Number(raw)

  if (Number.isNaN(num)) return null
  if (!Number.isFinite(num)) return null
  if (Math.abs(num) > Number.MAX_SAFE_INTEGER) return null

  return num
}
```

## Reserved Names Protection

```typescript
export const RESERVED_STRATEGY_NAMES = [
  'default',
  'null',
  'undefined',
  'none',
  'system',
  'admin',
  'root',
  'all',
  'any',
]

export const validateStrategyName = (name: string): string | null => {
  const trimmedName = name.trim()

  if (!trimmedName) {
    return 'Strategy name is required'
  }

  if (RESERVED_STRATEGY_NAMES.includes(trimmedName.toLowerCase())) {
    return `"${trimmedName}" is a reserved name.`
  }

  if (!/^[a-zA-Z0-9_-]+$/.test(trimmedName)) {
    return 'Cannot contain spaces or special characters'
  }

  if (trimmedName.length > 100) {
    return 'Strategy name must be 100 characters or less'
  }

  return null
}
```

## Electron Security

### Context Isolation
```typescript
const mainWindow = new BrowserWindow({
  webPreferences: {
    nodeIntegration: false,      // Disable Node.js in renderer
    contextIsolation: true,      // Isolate preload scripts
    preload: path.join(__dirname, '../preload/index.js'),
  },
})
```

### Safe DOM Building in Preload
```typescript
// Build DOM elements safely to prevent XSS
const item = document.createElement('div')
item.className = 'model-item'

const name = document.createElement('div')
name.className = 'model-name'
name.textContent = model.display_name // textContent escapes HTML

// Sanitize status to only allow known values
const VALID_STATUSES = ['present', 'downloading', 'checking', 'error']
const safeStatus = VALID_STATUSES.includes(model.status) ? model.status : 'checking'
```

## API Security

### URL Encoding for Path Parameters
```typescript
const { data } = await apiClient.get(
  `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(projectId)}`
)
```

### Validate Response Data
```typescript
function isValidProject(data: unknown): data is Project {
  return (
    typeof data === 'object' &&
    data !== null &&
    'namespace' in data &&
    typeof (data as Record<string, unknown>).namespace === 'string' &&
    'name' in data &&
    typeof (data as Record<string, unknown>).name === 'string'
  )
}
```

---

## Checklist

### SECURITY-001: No dangerouslySetInnerHTML with user input
- **Description**: Never use dangerouslySetInnerHTML with unsanitized content
- **Search**: `grep -r "dangerouslySetInnerHTML" designer/src/`
- **Pass**: No usage, or only with sanitized/static content
- **Fail**: Using with user-provided data
- **Severity**: Critical
- **Fix**: Use React's default escaping or rehype-sanitize

### SECURITY-002: URL encoding for path parameters
- **Description**: Path parameters must be URL encoded
- **Search**: `grep -rE "apiClient\.(get|post|put|delete)\(" designer/src/api/`
- **Pass**: All dynamic path segments use encodeURIComponent
- **Fail**: Unencoded user input in URLs
- **Severity**: Critical
- **Fix**: Wrap with `encodeURIComponent()`

### SECURITY-003: Validate URL protocols
- **Description**: Only allow http/https protocols
- **Search**: `grep -r "new URL" designer/src/`
- **Pass**: Protocol validated before use
- **Fail**: Accepting file://, javascript:, data: URLs
- **Severity**: Critical
- **Fix**: Check url.protocol against allowed list

### SECURITY-004: Sanitize navigation state
- **Description**: Router location.state must be validated
- **Search**: `grep -r "location.state" designer/src/`
- **Pass**: State validated with type guards
- **Fail**: Direct use of unvalidated state
- **Severity**: Critical
- **Fix**: Use validateNavigationState() or similar

### SECURITY-005: Electron context isolation enabled
- **Description**: Electron windows must use context isolation
- **Search**: `grep -r "contextIsolation" electron-app/src/`
- **Pass**: contextIsolation: true in all windows
- **Fail**: contextIsolation: false or missing
- **Severity**: High
- **Fix**: Set contextIsolation: true in webPreferences

### SECURITY-006: Node integration disabled
- **Description**: Electron renderer must not have Node access
- **Search**: `grep -r "nodeIntegration" electron-app/src/`
- **Pass**: nodeIntegration: false in all windows
- **Fail**: nodeIntegration: true
- **Severity**: High
- **Fix**: Set nodeIntegration: false, use preload for IPC

### SECURITY-007: Length limits on user input
- **Description**: User input must have maximum length limits
- **Search**: `grep -r "substring\|slice\|MAX_" designer/src/utils/security.ts`
- **Pass**: All sanitization functions limit length
- **Fail**: Unbounded string processing
- **Severity**: Medium
- **Fix**: Add substring(0, MAX_LENGTH) to sanitizers
