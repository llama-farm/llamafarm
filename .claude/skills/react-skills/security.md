# React Security Best Practices

Guidelines for preventing XSS, injection attacks, and other security vulnerabilities.

## Core Principles

1. **Never trust user input** - Sanitize all data from users, URLs, and external sources
2. **Escape by default** - React escapes JSX by default, but edge cases exist
3. **Validate navigation state** - URL params and location.state can be manipulated
4. **Use allowlists** - Prefer allowlists over blocklists for validation

## XSS Prevention

### React's Built-in Protection

React automatically escapes values in JSX:

```typescript
// Safe - React escapes the content
const userInput = '<script>alert("xss")</script>'
return <div>{userInput}</div> // Renders as text, not HTML
```

### dangerouslySetInnerHTML

Avoid unless absolutely necessary:

```typescript
// DANGEROUS - only use with sanitized content
<div dangerouslySetInnerHTML={{ __html: sanitizedHtml }} />
```

If you must use it, sanitize with a library like DOMPurify or rehype-sanitize.

## Checklist Items

### 1. No Unsanitized dangerouslySetInnerHTML

| Attribute | Value |
|-----------|-------|
| Description | dangerouslySetInnerHTML must use sanitized content |
| Search Pattern | `grep -n "dangerouslySetInnerHTML" --include="*.tsx"` |
| Pass Criteria | All uses wrapped with sanitization (DOMPurify, rehype-sanitize) |
| Severity | Critical |
| Fix | Sanitize with `DOMPurify.sanitize(html)` or use rehype-sanitize |

### 2. URL Parameters Are Validated

| Attribute | Value |
|-----------|-------|
| Description | useParams values must be validated before use |
| Search Pattern | `grep -A5 "useParams" --include="*.tsx"` |
| Pass Criteria | Params validated with regex or allowlist before use |
| Severity | High |
| Fix | Validate with regex: `/^[a-zA-Z0-9_-]+$/` |

### 3. Navigation State Is Validated

| Attribute | Value |
|-----------|-------|
| Description | location.state can be manipulated and must be validated |
| Search Pattern | `grep -A10 "useLocation\|location\.state" --include="*.tsx"` |
| Pass Criteria | State validated with type guards and default values |
| Severity | High |
| Fix | Use validation function like `validateNavigationState()` |

### 4. URLs Are Protocol-Validated

| Attribute | Value |
|-----------|-------|
| Description | User-provided URLs must validate protocol (http/https only) |
| Search Pattern | `grep -E "new URL\(|href=\{" --include="*.tsx"` |
| Pass Criteria | URLs validated with isValidAndSafeURL or similar |
| Severity | Critical |
| Fix | Check `['http:', 'https:'].includes(url.protocol)` |

### 5. User Content Has Length Limits

| Attribute | Value |
|-----------|-------|
| Description | User input should have max length to prevent DoS |
| Search Pattern | `grep -E "maxLength\|substring\|slice" --include="*.ts"` |
| Pass Criteria | Text inputs have maxLength, displayed values truncated |
| Severity | Medium |
| Fix | Add maxLength to inputs, truncate display values |

### 6. Sensitive Data Not in URL

| Attribute | Value |
|-----------|-------|
| Description | Tokens, passwords, API keys must not appear in URLs |
| Search Pattern | `grep -E "token=\|password=\|api_key=" --include="*.ts" --include="*.tsx"` |
| Pass Criteria | No sensitive data in query params or path |
| Severity | Critical |
| Fix | Use headers or POST body for sensitive data |

### 7. External Links Use rel="noopener"

| Attribute | Value |
|-----------|-------|
| Description | Links to external sites need noopener to prevent tabnabbing |
| Search Pattern | `grep -E "target=[\"']_blank" --include="*.tsx"` |
| Pass Criteria | All target="_blank" links have rel="noopener noreferrer" |
| Severity | Medium |
| Fix | Add `rel="noopener noreferrer"` to external links |

### 8. Form Actions Are Protected

| Attribute | Value |
|-----------|-------|
| Description | Forms should prevent double submission and validate input |
| Search Pattern | `grep -E "<form|onSubmit" --include="*.tsx"` |
| Pass Criteria | Forms disable submit during loading, validate before submit |
| Severity | Medium |
| Fix | Add disabled state during submission, validate inputs |

## Sanitization Utilities

### Config Value Sanitization

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

### URL Validation

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
    const isLocalhost = hostname === 'localhost' ||
                       hostname === '127.0.0.1' ||
                       hostname.startsWith('192.168.') ||
                       hostname.startsWith('10.') ||
                       hostname.startsWith('172.')

    if (import.meta.env.PROD && isLocalhost) {
      console.warn('Localhost/private IP detected in production:', hostname)
    }

    return true
  } catch {
    return false
  }
}
```

### Navigation State Validation

```typescript
export const validateNavigationState = (state: unknown): {
  database: string
  strategyName: string
  strategyType: string
  currentConfig: Record<string, any>
  isDefault: boolean
} => {
  const s = state as any

  // Validate database name (alphanumeric and underscores only)
  const database = typeof s?.database === 'string' &&
                   /^[a-zA-Z0-9_]+$/.test(s.database)
    ? s.database
    : 'main_database'

  // Validate strategy name (alphanumeric, hyphens, underscores)
  const strategyName = typeof s?.strategyName === 'string' &&
                       /^[a-zA-Z0-9_-]+$/.test(s.strategyName)
    ? s.strategyName
    : ''

  // Validate against allowlist
  const allowedTypes = ['BasicStrategy', 'AdvancedStrategy']
  const strategyType = typeof s?.strategyType === 'string' &&
                       allowedTypes.includes(s.strategyType)
    ? s.strategyType
    : 'BasicStrategy'

  // Validate config is an object (contents still untrusted)
  const currentConfig = s?.currentConfig &&
                        typeof s.currentConfig === 'object' &&
                        !Array.isArray(s.currentConfig)
    ? s.currentConfig
    : {}

  const isDefault = typeof s?.isDefault === 'boolean' ? s.isDefault : false

  return { database, strategyName, strategyType, currentConfig, isDefault }
}
```

### Filter Key/Value Sanitization

```typescript
const MAX_FILTER_KEY_LENGTH = 50
const MAX_FILTER_VALUE_LENGTH = 200

export const sanitizeFilterKey = (key: string): string => {
  return key.replace(/[^a-zA-Z0-9_-]/g, '').substring(0, MAX_FILTER_KEY_LENGTH)
}

export const sanitizeFilterValue = (value: string): string => {
  return value
    .replace(/[<>'"\\]/g, '')
    .trim()
    .substring(0, MAX_FILTER_VALUE_LENGTH)
}
```

## Markdown Rendering

Use rehype-sanitize when rendering user-provided markdown:

```typescript
import ReactMarkdown from 'react-markdown'
import rehypeSanitize from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'

function SafeMarkdown({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeSanitize]}
    >
      {content}
    </ReactMarkdown>
  )
}
```

## Reserved Names

Prevent system name collisions:

```typescript
export const RESERVED_NAMES = [
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

export const validateName = (name: string): string | null => {
  const trimmed = name.trim()

  if (!trimmed) return 'Name is required'

  if (RESERVED_NAMES.includes(trimmed.toLowerCase())) {
    return `"${trimmed}" is a reserved name`
  }

  if (!/^[a-zA-Z0-9_-]+$/.test(trimmed)) {
    return 'Cannot contain special characters'
  }

  if (trimmed.length > 100) {
    return 'Name must be 100 characters or less'
  }

  return null
}
```

## Clipboard API Security

```typescript
async function copyToClipboard(text: string) {
  try {
    // Sanitize before copying
    const sanitized = text.replace(/[<>]/g, '')
    await navigator.clipboard.writeText(sanitized)
    return true
  } catch (err) {
    console.error('Failed to copy:', err)
    return false
  }
}
```
