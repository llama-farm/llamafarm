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

export interface DatasetValidationError {
  index: number
  name: string
  error: string
}

export interface AllDatasetsValidationResult {
  isValid: boolean
  errors: DatasetValidationError[]
}

/**
 * Validates an array of datasets for name format and duplicates
 * Consolidates validation logic for bulk dataset validation
 * @param datasets - Array of dataset objects with name property
 * @returns Validation result with array of errors (if any)
 */
export const validateAllDatasetNames = (
  datasets: Array<{ name?: string }>
): AllDatasetsValidationResult => {
  const errors: DatasetValidationError[] = []
  const seenNames: string[] = []
  
  for (let i = 0; i < datasets.length; i++) {
    const dataset = datasets[i]
    const datasetName = dataset?.name
    
    // Validate individual dataset name format (including empty/missing names)
    const validation = validateDatasetName(datasetName || '')
    if (!validation.isValid) {
      errors.push({
        index: i,
        name: datasetName || '(empty)',
        error: validation.error || 'Invalid dataset name'
      })
      continue // Skip duplicate check if name is invalid
    }
    
    // Check for duplicates (case-insensitive)
    if (checkForDuplicateDatasetName(datasetName!, seenNames)) {
      errors.push({
        index: i,
        name: datasetName!,
        error: 'Duplicate dataset name'
      })
    }
    
    seenNames.push(datasetName!.trim())
  }
  
  return {
    isValid: errors.length === 0,
    errors
  }
}

