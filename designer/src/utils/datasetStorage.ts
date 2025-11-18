/**
 * Generic utilities for persisting dataset data in browser storage
 * Supports both sessionStorage (for temporary task IDs) and localStorage (for persistent results)
 */

type StorageType = 'session' | 'local'

/**
 * Get the appropriate storage object
 */
function getStorage(type: StorageType): Storage {
  return type === 'session' ? sessionStorage : localStorage
}

/**
 * Generate storage key for a dataset
 */
function getStorageKey(
  prefix: string,
  namespace: string,
  project: string,
  dataset: string
): string {
  return `${prefix}:${namespace}:${project}:${dataset}`
}

/**
 * Save a value to storage
 */
function saveToStorage<T>(
  type: StorageType,
  prefix: string,
  namespace: string,
  project: string,
  dataset: string,
  value: T,
  shouldStringify: boolean = false
): void {
  try {
    const key = getStorageKey(prefix, namespace, project, dataset)
    const storage = getStorage(type)
    const valueToStore = shouldStringify ? JSON.stringify(value) : String(value)
    storage.setItem(key, valueToStore)
  } catch (error) {
    console.warn(`Failed to save to ${type}Storage:`, error)
  }
}

/**
 * Load a value from storage
 */
function loadFromStorage<T>(
  type: StorageType,
  prefix: string,
  namespace: string,
  project: string,
  dataset: string,
  shouldParse: boolean = false
): T | null {
  try {
    const key = getStorageKey(prefix, namespace, project, dataset)
    const storage = getStorage(type)
    const stored = storage.getItem(key)
    
    if (!stored) return null
    
    return shouldParse ? JSON.parse(stored) : (stored as T)
  } catch (error) {
    console.warn(`Failed to load from ${type}Storage:`, error)
    return null
  }
}

/**
 * Clear a specific item from storage
 */
function clearFromStorage(
  type: StorageType,
  prefix: string,
  namespace: string,
  project: string,
  dataset: string
): void {
  try {
    const key = getStorageKey(prefix, namespace, project, dataset)
    const storage = getStorage(type)
    storage.removeItem(key)
  } catch (error) {
    console.warn(`Failed to clear from ${type}Storage:`, error)
  }
}

/**
 * Clear all items with a specific prefix from storage
 */
function clearAllFromStorage(type: StorageType, prefix: string): void {
  try {
    const storage = getStorage(type)
    const keys = Object.keys(storage)
    keys.forEach(key => {
      if (key.startsWith(prefix)) {
        storage.removeItem(key)
      }
    })
  } catch (error) {
    console.warn(`Failed to clear all from ${type}Storage:`, error)
  }
}

// Task ID storage (sessionStorage)
const TASK_ID_PREFIX = 'dataset-processing-task'

export function saveDatasetTaskId(
  namespace: string,
  project: string,
  dataset: string,
  taskId: string
): void {
  saveToStorage('session', TASK_ID_PREFIX, namespace, project, dataset, taskId)
}

export function loadDatasetTaskId(
  namespace: string,
  project: string,
  dataset: string
): string | null {
  return loadFromStorage<string>('session', TASK_ID_PREFIX, namespace, project, dataset)
}

export function clearDatasetTaskId(
  namespace: string,
  project: string,
  dataset: string
): void {
  clearFromStorage('session', TASK_ID_PREFIX, namespace, project, dataset)
}

export function clearAllDatasetTaskIds(): void {
  clearAllFromStorage('session', TASK_ID_PREFIX)
}

// Result storage (localStorage)
const RESULT_PREFIX = 'dataset-processing-result'

export function saveDatasetResult(
  namespace: string,
  project: string,
  dataset: string,
  result: any
): void {
  saveToStorage('local', RESULT_PREFIX, namespace, project, dataset, result, true)
}

export function loadDatasetResult(
  namespace: string,
  project: string,
  dataset: string
): any | null {
  return loadFromStorage<any>('local', RESULT_PREFIX, namespace, project, dataset, true)
}

export function clearDatasetResult(
  namespace: string,
  project: string,
  dataset: string
): void {
  clearFromStorage('local', RESULT_PREFIX, namespace, project, dataset)
}

export function clearAllDatasetResults(): void {
  clearAllFromStorage('local', RESULT_PREFIX)
}

