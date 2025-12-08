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

// Sanitize model name by removing spaces and special characters
export function sanitizeModelName(name: string): string {
  return name
    .trim()
    .replace(/[^a-zA-Z0-9_-]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
}

// Format ETA in seconds to human-readable string
export function formatETA(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${Math.round(seconds / 3600)}h`
}

// Validate model name
export function validateModelName(
  name: string,
  existingNames: string[],
  currentName?: string
): { isValid: boolean; error?: string } {
  if (!name || name.trim().length === 0) {
    return { isValid: false, error: 'Model name is required' }
  }

  if (name.trim().length > 100) {
    return { isValid: false, error: 'Model name must be 100 characters or less' }
  }

  // Check for duplicate names (excluding current name if renaming)
  const normalizedName = name.trim().toLowerCase()
  const isDuplicate = existingNames.some(
    existing =>
      existing.toLowerCase() === normalizedName &&
      existing !== currentName
  )

  if (isDuplicate) {
    return { isValid: false, error: 'A model with this name already exists' }
  }

  // Check for invalid characters
  if (!/^[a-zA-Z0-9_-]+$/.test(name.trim())) {
    return {
      isValid: false,
      error: 'Model name can only contain letters, numbers, hyphens, and underscores',
    }
  }

  return { isValid: true }
}
