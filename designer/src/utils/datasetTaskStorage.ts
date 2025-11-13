/**
 * Utilities for persisting dataset processing task IDs in sessionStorage
 * This allows processing status to survive page navigation and refreshes
 */

const TASK_ID_PREFIX = 'dataset-processing-task'

/**
 * Generate storage key for a dataset
 */
function getStorageKey(namespace: string, project: string, dataset: string): string {
  return `${TASK_ID_PREFIX}:${namespace}:${project}:${dataset}`
}

/**
 * Save a processing task ID for a dataset
 */
export function saveDatasetTaskId(
  namespace: string,
  project: string,
  dataset: string,
  taskId: string
): void {
  try {
    const key = getStorageKey(namespace, project, dataset)
    sessionStorage.setItem(key, taskId)
  } catch (error) {
    console.warn('Failed to save dataset task ID to sessionStorage:', error)
  }
}

/**
 * Load a processing task ID for a dataset
 * Returns null if no task ID is stored
 */
export function loadDatasetTaskId(
  namespace: string,
  project: string,
  dataset: string
): string | null {
  try {
    const key = getStorageKey(namespace, project, dataset)
    return sessionStorage.getItem(key)
  } catch (error) {
    console.warn('Failed to load dataset task ID from sessionStorage:', error)
    return null
  }
}

/**
 * Clear a processing task ID for a dataset
 */
export function clearDatasetTaskId(
  namespace: string,
  project: string,
  dataset: string
): void {
  try {
    const key = getStorageKey(namespace, project, dataset)
    sessionStorage.removeItem(key)
  } catch (error) {
    console.warn('Failed to clear dataset task ID from sessionStorage:', error)
  }
}

/**
 * Clear all dataset task IDs (useful for cleanup)
 */
export function clearAllDatasetTaskIds(): void {
  try {
    const keys = Object.keys(sessionStorage)
    keys.forEach(key => {
      if (key.startsWith(TASK_ID_PREFIX)) {
        sessionStorage.removeItem(key)
      }
    })
  } catch (error) {
    console.warn('Failed to clear all dataset task IDs from sessionStorage:', error)
  }
}

