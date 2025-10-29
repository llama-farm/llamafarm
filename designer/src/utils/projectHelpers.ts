/**
 * Shared utility functions for working with project data
 */

/**
 * Extract all model names from project config
 * @param config - Project configuration object
 * @returns Array of model names
 */
export function getModelNames(config: Record<string, any>): string[] {
  // Try multi-model format first (runtime.models)
  if (config?.runtime?.models) {
    return Object.values(config.runtime.models)
      .map((modelConfig: any) => modelConfig?.model)
      .filter((modelName): modelName is string => typeof modelName === 'string' && modelName.length > 0)
  }

  // Try legacy single-model format (runtime.model)
  if (config?.runtime?.model && typeof config.runtime.model === 'string') {
    return [config.runtime.model]
  }

  return []
}

/**
 * Get a display string for model names
 * @param config - Project configuration object
 * @returns Formatted model name string (e.g., "model1" or "model1 +2 more")
 */
export function getModelDisplayName(config: Record<string, any>): string {
  const models = getModelNames(config)

  if (models.length === 0) {
    return 'No model'
  }

  if (models.length === 1) {
    return models[0]
  }

  return `${models[0]} +${models.length - 1} more`
}

/**
 * Format a timestamp for display
 * @param timestamp - ISO timestamp string, null, or undefined
 * @returns Formatted date string or 'Never'
 */
export function formatLastModified(timestamp: string | null | undefined): string {
  if (!timestamp) {
    return 'Never'
  }

  try {
    const date = new Date(timestamp)
    if (isNaN(date.getTime())) {
      return 'Invalid date'
    }

    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`

    return date.toLocaleDateString()
  } catch (e) {
    return 'Invalid date'
  }
}

/**
 * Parse a timestamp to Unix time for sorting
 * @param timestamp - ISO timestamp string, null, or undefined
 * @returns Unix timestamp in milliseconds, or 0 if invalid
 */
export function parseTimestamp(timestamp: string | null | undefined): number {
  if (!timestamp) {
    return 0
  }

  try {
    const date = new Date(timestamp)
    return isNaN(date.getTime()) ? 0 : date.getTime()
  } catch (e) {
    return 0
  }
}
