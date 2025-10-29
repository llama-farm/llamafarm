import { apiClient } from './client'
import {
  ListModelsResponse,
  DownloadModelRequest,
  DownloadEvent,
} from '../types/model'

/**
 * List all cached models available on disk
 * @param provider - The provider to list models for (default: universal)
 * @returns Promise<ListModelsResponse> - List of cached models
 */
export async function listCachedModels(
  provider = 'universal'
): Promise<ListModelsResponse> {
  const response = await apiClient.get<ListModelsResponse>(
    `/models?provider=${provider}`
  )
  return response.data
}

/**
 * Download a model with streaming progress
 * @param request - The download request containing model name and provider
 * @returns AsyncIterableIterator<DownloadEvent> - Stream of download events
 */
export async function* downloadModel(
  request: DownloadModelRequest
): AsyncIterableIterator<DownloadEvent> {
  // Get base URL from apiClient config
  const baseURL = apiClient.defaults.baseURL || '/api/v1'
  const response = await fetch(`${baseURL}/models/download`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`Failed to download model: ${response.statusText}`)
  }

  const reader = response.body?.getReader()
  const decoder = new TextDecoder()

  if (!reader) {
    throw new Error('No response body')
  }

  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || '' // Keep incomplete line in buffer

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6)
        if (data.trim()) {
          try {
            const event = JSON.parse(data) as DownloadEvent
            yield event
          } catch (e) {
            console.error('Failed to parse SSE data:', data, e)
          }
        }
      }
    }
  }
}

/**
 * Default export with all model service functions
 */
export default {
  listCachedModels,
  downloadModel,
}

