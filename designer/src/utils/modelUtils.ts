/**
 * Shared utility functions for model management
 */

import { validateProjectName, checkForDuplicateName, type ValidationResult } from './projectValidation'

/**
 * Sanitizes a model identifier to a valid model name
 * Converts to lowercase, replaces spaces and special chars with hyphens
 */
export function sanitizeModelName(modelIdentifier: string): string {
  return modelIdentifier
    .toLowerCase()
    .replace(/\//g, '-')  // Replace / with -
    .replace(/:/g, '-')   // Replace : with -
    .replace(/\s+/g, '-') // Replace spaces with -
    .replace(/[^a-zA-Z0-9_-]/g, '') // Remove any other special characters except - and _
    .replace(/-+/g, '-')  // Replace multiple dashes with single dash
    .replace(/^-|-$/g, '') // Remove leading/trailing dashes
}

/**
 * Format bytes to human-readable string
 */
export function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
  let i = Math.floor(Math.log(bytes) / Math.log(1024))
  if (i >= units.length) i = units.length - 1
  const val = bytes / Math.pow(1024, i)
  return `${val.toFixed(i >= 2 ? 1 : 0)} ${units[i]}`
}

/**
 * Format estimated time remaining in seconds to human-readable string
 */
export function formatETA(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return ''
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  if (m >= 60) {
    const h = Math.floor(m / 60)
    const rm = m % 60
    return `~${h}h ${rm}m`
  }
  if (m > 0) return `~${m}m ${s}s`
  return `~${s}s`
}

/**
 * Validates model name with duplicate check
 * Reuses project validation logic since model names follow same rules
 */
export function validateModelName(
  name: string,
  existingModelNames: string[],
  currentName: string | null = null
): ValidationResult {
  const trimmedName = name?.trim()

  if (!trimmedName) {
    return { isValid: false, error: 'Model name is required' }
  }

  // Use same validation as project names
  const validation = validateProjectName(trimmedName)
  if (!validation.isValid) {
    return validation
  }

  // Check for duplicates
  if (checkForDuplicateName(trimmedName, existingModelNames, currentName)) {
    return { isValid: false, error: 'A model with this name already exists' }
  }

  return { isValid: true }
}

