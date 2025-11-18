/**
 * Dataset name validation utilities
 * Mirrors project validation patterns for consistency
 */

export interface ValidationResult {
  isValid: boolean
  error?: string
}

/**
 * Simple dataset name validation for UX feedback
 * @param name - The dataset name to validate
 * @returns Validation result with user-friendly error message
 */
export const validateDatasetName = (name: string): ValidationResult => {
  const trimmedName = name?.trim()

  if (!trimmedName) {
    return { isValid: false, error: 'Dataset name is required' }
  }

  if (trimmedName.length > 100) {
    return { isValid: false, error: 'Dataset name is too long (max 100 characters)' }
  }

  // Check for valid characters: only alphanumeric, underscores, and hyphens (no spaces or special characters)
  const validNamePattern = /^[a-zA-Z0-9_-]+$/
  if (!validNamePattern.test(trimmedName)) {
    return { isValid: false, error: 'Dataset name can only contain letters, numbers, underscores (_), and hyphens (-)' }
  }

  return { isValid: true }
}

/**
 * Optimistic duplicate name check for better UX (not for security)
 * @param name - The dataset name to check
 * @param existingNames - Array of existing dataset names
 * @param currentName - Current name (for updates, to allow keeping same name)
 * @returns true if duplicate found
 */
export const checkForDuplicateDatasetName = (
  name: string,
  existingNames: string[],
  currentName: string | null = null
): boolean => {
  if (!name || !existingNames) return false
  
  const normalizedName = name.trim().toLowerCase()
  const normalizedCurrent = currentName?.trim().toLowerCase()
  
  return existingNames.some(existing => 
    existing.toLowerCase() === normalizedName && 
    existing.toLowerCase() !== normalizedCurrent
  )
}

/**
 * Validates dataset name and checks for duplicates with user-friendly messages
 * @param name - The dataset name to validate
 * @param existingNames - Array of existing dataset names
 * @param currentName - Current name (for updates)
 * @returns Validation result with appropriate error message
 */
export const validateDatasetNameWithDuplicateCheck = (
  name: string,
  existingNames: string[],
  currentName: string | null = null
): ValidationResult => {
  // Basic validation first
  const basicValidation = validateDatasetName(name)
  if (!basicValidation.isValid) {
    return basicValidation
  }
  
  // Check for duplicates (optimistic UX check)
  if (checkForDuplicateDatasetName(name, existingNames, currentName)) {
    return { isValid: false, error: 'A dataset with this name already exists' }
  }
  
  return { isValid: true }
}

