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
