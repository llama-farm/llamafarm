/**
 * Namespace utilities for Project API
 * 
 * This module provides functions to determine the current user namespace
 * for API calls. Currently uses a default namespace, but can be extended
 * to support user authentication and multi-tenancy.
 */

/**
 * Default namespace for projects when no user authentication is present
 */
const DEFAULT_NAMESPACE = 'default'

/**
 * Get the current user's namespace for API calls
 * 
 * Checks localStorage for a stored namespace, falling back to default.
 * Ready for authentication integration via setCurrentNamespace().
 * 
 * @returns The current namespace string
 */
export function getCurrentNamespace(): string {
  try {
    const storedNamespace = localStorage.getItem('userNamespace')
    if (storedNamespace) {
      return storedNamespace
    }
  } catch {
    // Fall back to default if localStorage is not available
  }
  
  return DEFAULT_NAMESPACE
}

/**
 * Set the current user namespace
 * 
 * @param namespace - The namespace to set
 */
export function setCurrentNamespace(namespace: string): void {
  try {
    localStorage.setItem('userNamespace', namespace)
  } catch {
    // Silently fail if localStorage is not available
    console.warn('Could not save namespace to localStorage')
  }
}

/**
 * Clear the current namespace (useful for logout)
 */
export function clearCurrentNamespace(): void {
  try {
    localStorage.removeItem('userNamespace')
  } catch {
    // Silently fail if localStorage is not available
  }
}
