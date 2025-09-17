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
 * TODO: Update this when user authentication is implemented
 * This could check localStorage, JWT tokens, or other auth mechanisms
 * 
 * @returns The current namespace string
 */
export function getCurrentNamespace(): string {
  // For now, return a default namespace
  // In the future, this could:
  // - Check localStorage for user info
  // - Parse JWT tokens
  // - Call user info API
  // - Use React context for user state
  
  try {
    // Check if there's a stored user namespace (for future use)
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
