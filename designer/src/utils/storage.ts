// Generic localStorage helpers for arrays and sets with safe parsing

export function getStoredArray<T = unknown>(
  key: string,
  validator?: (item: unknown) => item is T
): T[] {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return []
    const arr = JSON.parse(raw)
    if (!Array.isArray(arr)) return []
    if (validator) return arr.filter(validator)
    return arr as T[]
  } catch {
    return []
  }
}

export function setStoredArray<T = unknown>(key: string, list: T[]): void {
  try {
    localStorage.setItem(key, JSON.stringify(list))
  } catch {}
}

export function getStoredSet(key: string): Set<string> {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return new Set()
    const arr = JSON.parse(raw)
    return Array.isArray(arr) ? new Set(arr) : new Set()
  } catch {
    return new Set()
  }
}

export function setStoredSet(key: string, set: Set<string>): void {
  try {
    localStorage.setItem(key, JSON.stringify(Array.from(set)))
  } catch {}
}

// Model description storage helpers
const MODEL_DESCRIPTIONS_KEY = 'llamafarm:model-descriptions'

interface ModelDescriptions {
  [modelName: string]: string
}

export function getModelDescription(modelName: string): string {
  try {
    const raw = localStorage.getItem(MODEL_DESCRIPTIONS_KEY)
    if (!raw) return ''
    const descriptions: ModelDescriptions = JSON.parse(raw)
    return descriptions[modelName] || ''
  } catch {
    return ''
  }
}

export function setModelDescription(modelName: string, description: string): void {
  try {
    const raw = localStorage.getItem(MODEL_DESCRIPTIONS_KEY)
    const descriptions: ModelDescriptions = raw ? JSON.parse(raw) : {}
    if (description.trim()) {
      descriptions[modelName] = description
    } else {
      delete descriptions[modelName]
    }
    localStorage.setItem(MODEL_DESCRIPTIONS_KEY, JSON.stringify(descriptions))
  } catch {}
}
