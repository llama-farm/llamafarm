import { apiClient } from './client'

export interface BundleRequest {
  platform: string
  arch: string
  accelerator: string
  addons: string[]
  version: string
}

export interface BundleSummary {
  id: string
  version: string
  platform: string
  arch: string
  accelerator: string
  addons: string[]
  size: number
  filename: string
  created_at: string
}

export interface BundleEstimate {
  estimated_bytes: number
  components: Record<string, number>
}

export async function getBundleVersion(): Promise<string> {
  const { data } = await apiClient.get<{ version: string }>('/bundle/version')
  return data.version
}

export async function listBundles(): Promise<BundleSummary[]> {
  const { data } = await apiClient.get<BundleSummary[]>('/bundles')
  return data
}

export async function deleteBundle(id: string): Promise<void> {
  await apiClient.delete(`/bundles/${id}`)
}

export async function estimateBundleSize(
  req: BundleRequest
): Promise<BundleEstimate> {
  const { data } = await apiClient.post<BundleEstimate>('/bundle/estimate', req)
  return data
}

export function getBundleDownloadUrl(id: string): string {
  const base = apiClient.defaults.baseURL || '/api/v1'
  return `${base}/bundles/${id}/download`
}

/**
 * Start a bundle creation via SSE. Returns the EventSource URL
 * and a function to initiate the POST request returning an SSE stream.
 */
export async function createBundleStream(
  req: BundleRequest,
  onProgress: (data: any) => void,
  onComplete: (data: any) => void,
  onError: (msg: string) => void,
): Promise<AbortController> {
  const controller = new AbortController()
  const base = apiClient.defaults.baseURL || '/api/v1'
  const url = `${base}/bundle`

  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const text = await response.text()
        onError(text || `HTTP ${response.status}`)
        return
      }
      const reader = response.body?.getReader()
      if (!reader) {
        onError('No response body')
        return
      }
      const decoder = new TextDecoder()
      let buffer = ''
      let currentEvent = ''

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        console.debug('[bundle-sse] chunk:', chunk)
        buffer += chunk

        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              console.debug('[bundle-sse] event:', currentEvent, data)
              if (currentEvent === 'progress') {
                onProgress(data)
              } else if (currentEvent === 'complete') {
                onComplete(data)
              } else if (currentEvent === 'error') {
                onError(data.message || 'Unknown error')
              }
            } catch {
              // ignore parse errors
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err.message || 'Network error')
      }
    })

  return controller
}
