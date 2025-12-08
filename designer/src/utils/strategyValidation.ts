/**
 * Strategy name validation utilities - Matching project/database naming rules
 * Frontend validation focuses on user experience, backend handles business logic
 */

export interface ValidationResult {
  isValid: boolean
  error?: string
}

/**
 * Simple strategy name validation for UX feedback
 * Matches the same rules as project and database naming
 * @param name - The strategy name to validate
 * @returns Validation result with user-friendly error message
 */
export const validateStrategyName = (name: string): ValidationResult => {
  const trimmedName = name?.trim()

  if (!trimmedName) {
    return { isValid: false, error: 'Strategy name is required' }
  }

  if (trimmedName.length > 100) {
    return { isValid: false, error: 'Strategy name is too long (max 100 characters)' }
  }

  // Check for valid characters: only alphanumeric, underscores, and hyphens (no spaces or special characters)
  const validNamePattern = /^[a-zA-Z0-9_-]+$/
  if (!validNamePattern.test(trimmedName)) {
    return { isValid: false, error: 'Strategy name can only contain letters, numbers, underscores (_), and hyphens (-)' }
  }

  return { isValid: true }
}

