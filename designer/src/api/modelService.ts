import { apiClient } from './client'
import {
  ListModelsResponse,
  ListCachedModelsResponse,
  DownloadModelRequest,
  DownloadEvent,
  DeleteModelResponse,
} from '../types/model'

/**
 * List all models for a project
 */
export async function listModels(
  namespace: string,
  projectId: string
): Promise<ListModelsResponse> {
  const response = await apiClient.get<ListModelsResponse>(
    `/projects/${namespace}/${encodeURIComponent(projectId)}/models`
  )
  return response.data
}

/**
 * List all cached models available on disk
 * @param provider - The provider to list models for (default: universal)
 * @returns Promise<ListCachedModelsResponse> - List of cached models
 */
export async function listCachedModels(
  provider = 'universal'
): Promise<ListCachedModelsResponse> {
  const response = await apiClient.get<ListCachedModelsResponse>(
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
 * Delete a cached model from disk
 * @param modelName - The model identifier to delete (e.g., "meta-llama/Llama-2-7b-hf")
 * @param provider - The provider (default: universal)
 * @returns Promise<DeleteModelResponse> - Info about deleted model including freed space
 */
export async function deleteModel(
  modelName: string,
  provider = 'universal'
): Promise<DeleteModelResponse> {
  const response = await apiClient.delete<DeleteModelResponse>(
    `/models/${encodeURIComponent(modelName)}?provider=${provider}`
  )
  return response.data
}

/**
 * Default export with all model service functions
 */
export default {
  listModels,
  listCachedModels,
  downloadModel,
  deleteModel,
}
