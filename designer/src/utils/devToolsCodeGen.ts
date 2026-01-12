import type { CapturedRequest } from '../contexts/DevToolsContext'

/**
 * Generate a cURL command from a captured request
 */
export function generateCurl(request: CapturedRequest): string {
  const lines: string[] = []

  lines.push(`curl -X ${request.method} '${request.fullUrl}'`)

  // Add headers (skip content-type if we're adding -d with JSON)
  const headers = { ...request.headers }
  for (const [key, value] of Object.entries(headers)) {
    // Skip internal headers
    if (key.toLowerCase() === 'content-length') continue
    lines.push(`  -H '${key}: ${value}'`)
  }

  // Add body if present
  if (request.body && request.method !== 'GET') {
    const bodyStr =
      typeof request.body === 'string'
        ? request.body
        : JSON.stringify(request.body, null, 2)

    // For multipart, show a comment instead
    if (headers['Content-Type']?.includes('multipart/form-data')) {
      lines.push(`  # Note: multipart/form-data body not shown`)
      lines.push(`  # Use appropriate -F flags for file uploads`)
    } else {
      lines.push(`  -d '${bodyStr.replace(/'/g, "'\\''")}'`)
    }
  }

  return lines.join(' \\\n')
}

/**
 * Generate Python requests code from a captured request
 */
export function generatePython(request: CapturedRequest): string {
  const lines: string[] = []

  lines.push('import requests')
  lines.push('')

  // Headers
  const headers = { ...request.headers }
  // Remove content-length as requests handles it
  delete headers['content-length']
  delete headers['Content-Length']

  const hasHeaders = Object.keys(headers).length > 0
  if (hasHeaders) {
    lines.push('headers = {')
    for (const [key, value] of Object.entries(headers)) {
      lines.push(`    "${key}": "${value}",`)
    }
    lines.push('}')
    lines.push('')
  }

  // Body
  const hasBody = request.body && request.method !== 'GET'
  if (hasBody) {
    if (headers['Content-Type']?.includes('multipart/form-data')) {
      lines.push('# Note: For file uploads, use the files parameter')
      lines.push('# files = {"file": open("path/to/file", "rb")}')
      lines.push('')
    } else {
      // Use word boundaries to avoid replacing "null"/"true"/"false" inside string values
      const bodyStr = JSON.stringify(request.body, null, 4)
        .replace(/\bnull\b/g, 'None')
        .replace(/\btrue\b/g, 'True')
        .replace(/\bfalse\b/g, 'False')
      lines.push(`payload = ${bodyStr}`)
      lines.push('')
    }
  }

  // Request call
  const method = request.method.toLowerCase()
  const args: string[] = [`"${request.fullUrl}"`]

  if (hasHeaders) {
    args.push('headers=headers')
  }

  if (hasBody && !headers['Content-Type']?.includes('multipart/form-data')) {
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
 * Generate JavaScript fetch code from a captured request
 */
export function generateJavaScript(request: CapturedRequest): string {
  const lines: string[] = []

  // Build fetch options
  const options: Record<string, any> = {
    method: request.method,
  }

  // Headers
  const headers = { ...request.headers }
  delete headers['content-length']
  delete headers['Content-Length']

  if (Object.keys(headers).length > 0) {
    options.headers = headers
  }

  // Body
  if (request.body && request.method !== 'GET') {
    if (headers['Content-Type']?.includes('multipart/form-data')) {
      lines.push('// Note: For file uploads, use FormData')
      lines.push('// const formData = new FormData();')
      lines.push('// formData.append("file", fileInput.files[0]);')
      lines.push('')
    } else {
      options.body = 'JSON.stringify(payload)'
    }
  }

  // Generate the code
  const hasJsonBody = request.body && request.method !== 'GET' && !headers['Content-Type']?.includes('multipart/form-data')
  if (hasJsonBody) {
    // If body is already a string, show it directly; otherwise stringify the object
    if (typeof request.body === 'string') {
      // Body is already a string - show as-is (it's already JSON)
      lines.push(`const payload = ${request.body};`)
    } else {
      const bodyStr = JSON.stringify(request.body, null, 2)
      lines.push(`const payload = ${bodyStr};`)
    }
    lines.push('')
  }

  lines.push(`const response = await fetch("${request.fullUrl}", {`)
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
