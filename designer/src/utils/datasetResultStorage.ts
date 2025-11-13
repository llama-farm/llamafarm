/**
 * Utilities for persisting dataset processing results in localStorage
 * This allows previous processing results to be displayed until the next run
 */

const RESULT_PREFIX = 'dataset-processing-result'

/**
 * Generate storage key for a dataset result
 */
function getStorageKey(namespace: string, project: string, dataset: string): string {
  return `${RESULT_PREFIX}:${namespace}:${project}:${dataset}`
}

/**
 * Save processing result for a dataset
 */
export function saveDatasetResult(
  namespace: string,
  project: string,
  dataset: string,
  result: any
): void {
  try {
    const key = getStorageKey(namespace, project, dataset)
    localStorage.setItem(key, JSON.stringify(result))
  } catch (error) {
    console.warn('Failed to save dataset result to localStorage:', error)
  }
}

/**
 * Load processing result for a dataset
 * Returns null if no result is stored
 */
export function loadDatasetResult(
  namespace: string,
  project: string,
  dataset: string
): any | null {
  try {
    const key = getStorageKey(namespace, project, dataset)
    const stored = localStorage.getItem(key)
    return stored ? JSON.parse(stored) : null
  } catch (error) {
    console.warn('Failed to load dataset result from localStorage:', error)
    return null
  }
}

/**
 * Clear processing result for a dataset
 */
export function clearDatasetResult(
  namespace: string,
  project: string,
  dataset: string
): void {
  try {
    const key = getStorageKey(namespace, project, dataset)
    localStorage.removeItem(key)
  } catch (error) {
    console.warn('Failed to clear dataset result from localStorage:', error)
  }
}

/**
 * Clear all dataset results (useful for cleanup)
 */
export function clearAllDatasetResults(): void {
  try {
    const keys = Object.keys(localStorage)
    keys.forEach(key => {
      if (key.startsWith(RESULT_PREFIX)) {
        localStorage.removeItem(key)
      }
    })
  } catch (error) {
    console.warn('Failed to clear all dataset results from localStorage:', error)
  }
}

