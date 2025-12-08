// Helper to format bytes
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
}

// Map model IDs to HuggingFace identifiers
export const modelIdToHuggingFace: Record<string, string> = {
  'bge-small-en-v1.5': 'BAAI/bge-small-en-v1.5',
  'bge-base-en-v1.5': 'BAAI/bge-base-en-v1.5',
  'bge-large-en-v1.5': 'BAAI/bge-large-en-v1.5',
  'bge-m3': 'BAAI/bge-m3',
  'e5-base-v2': 'intfloat/e5-base-v2',
  'e5-large-v2': 'intfloat/e5-large-v2',
  'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
}

// Check if a model is on disk
export function isModelOnDisk(
  modelId: string,
  cachedModelsResponse?: { data?: Array<{ name: string }> }
): boolean {
  if (!cachedModelsResponse?.data) return false
  const hfId = modelIdToHuggingFace[modelId] || modelId
  return cachedModelsResponse.data.some(m => {
    const modelName = m.name.toLowerCase()
    const searchId = hfId.toLowerCase()
    return (
      modelName.includes(searchId.split('/').pop() || '') ||
      modelName === searchId
    )
  })
}

// Get disk size for a model
export function getModelDiskSize(
  modelId: string,
  cachedModelsResponse?: { data?: Array<{ name: string; size?: number }> }
): number | null {
  if (!cachedModelsResponse?.data) return null
  const hfId = modelIdToHuggingFace[modelId] || modelId
  const found = cachedModelsResponse.data.find(m => {
    const modelName = m.name.toLowerCase()
    const searchId = hfId.toLowerCase()
    return (
      modelName.includes(searchId.split('/').pop() || '') ||
      modelName === searchId
    )
  })
  return found?.size || null
}
