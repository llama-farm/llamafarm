import type { CapturedRequest } from '../contexts/DevToolsContext'

/**
 * Escape single quotes for shell strings (used in cURL)
 */
function escapeShellSingleQuote(str: string): string {
  return str.replace(/'/g, "'\\''")
}

/**
 * Case-insensitive header lookup
 */
function getHeader(headers: Record<string, string>, key: string): string | undefined {
  const lowerKey = key.toLowerCase()
  for (const [k, v] of Object.entries(headers)) {
    if (k.toLowerCase() === lowerKey) return v
  }
  return undefined
}

/**
 * Generate a cURL command from a captured request
 */
export function generateCurl(request: CapturedRequest): string {
  const lines: string[] = []

  lines.push(`curl -X ${request.method} '${escapeShellSingleQuote(request.fullUrl)}'`)

  // Add headers (skip content-type if we're adding -d with JSON)
  const headers = { ...request.headers }
  for (const [key, value] of Object.entries(headers)) {
    // Skip internal headers
    if (key.toLowerCase() === 'content-length') continue
    lines.push(`  -H '${escapeShellSingleQuote(key)}: ${escapeShellSingleQuote(value)}'`)
  }

  // Add body if present
  if (request.body && request.method !== 'GET') {
    const bodyStr =
      typeof request.body === 'string'
        ? request.body
        : JSON.stringify(request.body, null, 2)

    // For multipart, show a comment instead (case-insensitive check)
    const contentType = getHeader(headers, 'content-type')
    if (contentType?.includes('multipart/form-data')) {
      lines.push(`  # Note: multipart/form-data body not shown`)
      lines.push(`  # Use appropriate -F flags for file uploads`)
    } else {
      lines.push(`  -d '${escapeShellSingleQuote(bodyStr)}'`)
    }
  }

  return lines.join(' \\\n')
}

/**
 * Escape a string for Python double-quoted strings
 */
function escapePythonString(str: string): string {
  return str.replace(/\\/g, '\\\\').replace(/"/g, '\\"')
}

/**
 * Convert a JavaScript value to Python literal syntax
 * Properly handles null -> None, true -> True, false -> False
 * without corrupting string values that contain these words
 */
function toPythonLiteral(value: unknown, indent = 0): string {
  const spaces = '    '.repeat(indent)
  const nextSpaces = '    '.repeat(indent + 1)

  if (value === null) return 'None'
  if (value === true) return 'True'
  if (value === false) return 'False'
  if (typeof value === 'number') return String(value)
  if (typeof value === 'string') return `"${escapePythonString(value)}"`

  if (Array.isArray(value)) {
    if (value.length === 0) return '[]'
    const items = value.map(v => `${nextSpaces}${toPythonLiteral(v, indent + 1)}`).join(',\n')
    return `[\n${items}\n${spaces}]`
  }

  if (typeof value === 'object' && value !== null) {
    const entries = Object.entries(value)
    if (entries.length === 0) return '{}'
    const items = entries
      .map(([k, v]) => `${nextSpaces}"${escapePythonString(k)}": ${toPythonLiteral(v, indent + 1)}`)
      .join(',\n')
    return `{\n${items}\n${spaces}}`
  }

  return String(value)
}

/**
 * Generate Python requests code from a captured request
 */
export function generatePython(request: CapturedRequest): string {
  const lines: string[] = []

  lines.push('import requests')
  lines.push('')

  // Headers - filter out content-length (case-insensitive)
  const headers: Record<string, string> = {}
  for (const [key, value] of Object.entries(request.headers)) {
    if (key.toLowerCase() !== 'content-length') {
      headers[key] = value
    }
  }

  const hasHeaders = Object.keys(headers).length > 0
  if (hasHeaders) {
    lines.push('headers = {')
    for (const [key, value] of Object.entries(headers)) {
      lines.push(`    "${escapePythonString(key)}": "${escapePythonString(value)}",`)
    }
    lines.push('}')
    lines.push('')
  }

  // Body
  const contentType = getHeader(headers, 'content-type')
  const hasBody = request.body && request.method !== 'GET'
  if (hasBody) {
    if (contentType?.includes('multipart/form-data')) {
      lines.push('# Note: For file uploads, use the files parameter')
      lines.push('# files = {"file": open("path/to/file", "rb")}')
      lines.push('')
    } else {
      // Convert to Python literal syntax (properly handles null/true/false without affecting strings)
      const bodyStr = toPythonLiteral(request.body)
      lines.push(`payload = ${bodyStr}`)
      lines.push('')
    }
  }

  // Request call
  const method = request.method.toLowerCase()
  const args: string[] = [`"${escapePythonString(request.fullUrl)}"`]

  if (hasHeaders) {
    args.push('headers=headers')
  }

  if (hasBody && !contentType?.includes('multipart/form-data')) {
    args.push('json=payload')
  }

  lines.push(`response = requests.${method}(`)
  lines.push(`    ${args.join(',\n    ')}`)
  lines.push(')')
  lines.push('')
  lines.push('print(response.status_code)')
  lines.push('print(response.json())')

  return lines.join('\n')
}

/**
 * Escape a string for JavaScript double-quoted strings
 */
function escapeJavaScriptString(str: string): string {
  return str.replace(/\\/g, '\\\\').replace(/"/g, '\\"')
}

/**
 * Generate JavaScript fetch code from a captured request
 */
export function generateJavaScript(request: CapturedRequest): string {
  const lines: string[] = []

  // Build fetch options
  const options: Record<string, unknown> = {
    method: request.method,
  }

  // Headers - filter out content-length (case-insensitive)
  const headers: Record<string, string> = {}
  for (const [key, value] of Object.entries(request.headers)) {
    if (key.toLowerCase() !== 'content-length') {
      headers[key] = value
    }
  }

  if (Object.keys(headers).length > 0) {
    options.headers = headers
  }

  // Body
  const contentType = getHeader(headers, 'content-type')
  if (request.body && request.method !== 'GET') {
    if (contentType?.includes('multipart/form-data')) {
      lines.push('// Note: For file uploads, use FormData')
      lines.push('// const formData = new FormData();')
      lines.push('// formData.append("file", fileInput.files[0]);')
      lines.push('')
    } else {
      options.body = 'JSON.stringify(payload)'
    }
  }

  // Generate the code
  const hasJsonBody = request.body && request.method !== 'GET' && !contentType?.includes('multipart/form-data')
  if (hasJsonBody) {
    // Always use JSON.stringify to ensure valid JS syntax
    const bodyStr = JSON.stringify(request.body, null, 2)
    lines.push(`const payload = ${bodyStr};`)
    lines.push('')
  }

  // Use JSON.stringify for the URL to handle any special characters
  lines.push(`const response = await fetch("${escapeJavaScriptString(request.fullUrl)}", {`)
  lines.push(`  method: "${request.method}",`)

  if (options.headers) {
    lines.push(`  headers: ${JSON.stringify(options.headers, null, 2).split('\n').map((l, i) => i === 0 ? l : '  ' + l).join('\n')},`)
  }

  if (hasJsonBody) {
    lines.push(`  body: JSON.stringify(payload),`)
  }

  lines.push('});')
  lines.push('')
  lines.push('const data = await response.json();')
  lines.push('console.log(data);')

  return lines.join('\n')
}

export type CodeFormat = 'curl' | 'python' | 'javascript'

export function generateCode(request: CapturedRequest, format: CodeFormat): string {
  switch (format) {
    case 'curl':
      return generateCurl(request)
    case 'python':
      return generatePython(request)
    case 'javascript':
      return generateJavaScript(request)
    default:
      return ''
  }
}
