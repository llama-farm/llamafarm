import { apiClient } from './client'
import { ListModelsResponse } from '../types/model'

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

export default { listModels }
