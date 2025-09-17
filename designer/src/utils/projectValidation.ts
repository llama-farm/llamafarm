/**
 * Project validation utilities - Simplified UX-focused approach
 * Frontend validation focuses on user experience, backend handles business logic
 */

export interface ValidationResult {
  isValid: boolean
  error?: string
}

/**
 * Simple project name validation for UX feedback
 * @param name - The project name to validate
 * @returns Validation result with user-friendly error message
 */
export const validateProjectName = (name: string): ValidationResult => {
  const trimmedName = name?.trim()
  
  if (!trimmedName) {
    return { isValid: false, error: 'Project name is required' }
  }
  
  if (trimmedName.length > 100) {
    return { isValid: false, error: 'Project name is too long (max 100 characters)' }
  }
  
  return { isValid: true }
}

/**
 * Optimistic duplicate name check for better UX (not for security)
 * @param name - The project name to check
 * @param existingNames - Array of existing project names
 * @param currentName - Current name (for updates, to allow keeping same name)
 * @returns true if duplicate found
 */
export const checkForDuplicateName = (
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
 * Sanitizes a project name by trimming and normalizing spaces
 * @param name - The project name to sanitize
 * @returns Sanitized project name
 */
export const sanitizeProjectName = (name: string): string => {
  return name.trim().replace(/\s+/g, ' ') // Replace multiple spaces with single space
}

/**
 * Validates project name and checks for duplicates with user-friendly messages
 * @param name - The project name to validate
 * @param existingNames - Array of existing project names
 * @param currentName - Current name (for updates)
 * @returns Validation result with appropriate error message
 */
export const validateProjectNameWithDuplicateCheck = (
  name: string,
  existingNames: string[],
  currentName: string | null = null
): ValidationResult => {
  // Basic validation first
  const basicValidation = validateProjectName(name)
  if (!basicValidation.isValid) {
    return basicValidation
  }
  
  // Check for duplicates (optimistic UX check)
  if (checkForDuplicateName(name, existingNames, currentName)) {
    return { isValid: false, error: 'A project with this name already exists' }
  }
  
  return { isValid: true }
}
