/**
 * Home Page Utilities for Project Creation from Chat
 * 
 * This module provides utilities to create projects from home page chat messages
 * while preserving the existing session management system.
 */

import { getCurrentNamespace } from './namespaceUtils'
import projectService from '../api/projectService'
import { CreateProjectRequest } from '../types/project'

/**
 * Result of creating a project from chat
 */
export interface CreateProjectFromChatResult {
  namespace: string
  projectName: string
}

/**
 * Generate a project name from a chat message
 * Takes the first few meaningful words and sanitizes them for project naming
 */
function generateProjectNameFromMessage(message: string): string {
  // Take first 3-4 meaningful words, sanitize for project name
  const words = message
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ') // Replace non-alphanumeric with spaces
    .split(/\s+/)
    .filter(word => word.length > 2) // Filter out short words like "an", "is", etc.
    .slice(0, 3) // Take first 3 words

  const baseName = words.join('-') || 'project'
  
  // Add timestamp to ensure uniqueness
  const timestamp = Date.now().toString().slice(-4) // Last 4 digits of timestamp
  
  return `${baseName}-${timestamp}`
}

/**
 * Create a new project from a chat message
 * 
 * @param message - The chat message that triggered project creation
 * @param userNamespace - Optional namespace override (defaults to current namespace)
 * @returns Promise with the created project's namespace and name
 */
export async function createProjectFromChat(
  message: string,
  userNamespace?: string
): Promise<CreateProjectFromChatResult> {
  if (!message || !message.trim()) {
    throw new Error('Message cannot be empty')
  }

  const namespace = userNamespace || getCurrentNamespace()
  const projectName = generateProjectNameFromMessage(message)
  
  try {
    console.log('🏠 Creating project from chat:', { namespace, projectName, message: message.substring(0, 100) })
    
    // Use the existing project creation API
    const request: CreateProjectRequest = {
      name: projectName,
      config_template: 'default' // Use default template
    }
    
    const response = await projectService.createProject(namespace, request)
    
    console.log('✅ Project created successfully:', response.project)
    
    return {
      namespace,
      projectName: response.project.name // Use the actual name returned by the API
    }
  } catch (error) {
    console.error('❌ Failed to create project from chat:', error)
    throw new Error(`Failed to create project: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

/**
 * Validate that a message is suitable for project creation
 * 
 * @param message - The message to validate
 * @returns true if valid, false otherwise
 */
export function validateMessageForProjectCreation(message: string): boolean {
  console.log('🔍 Validating message for project creation:', { message, type: typeof message })
  
  if (!message || typeof message !== 'string') {
    console.log('❌ Validation failed: message is not a string or is falsy')
    return false
  }
  
  const trimmed = message.trim()
  console.log('🔍 Trimmed message:', { trimmed, length: trimmed.length })
  
  // Must have content
  if (trimmed.length === 0) {
    console.log('❌ Validation failed: message is empty after trimming')
    return false
  }
  
  // Must be reasonable length (not too short or too long)
  if (trimmed.length < 3 || trimmed.length > 1000) {
    console.log('❌ Validation failed: message length is invalid', { length: trimmed.length, min: 3, max: 1000 })
    return false
  }
  
  console.log('✅ Validation passed')
  return true
}

/**
 * Generate a URL-safe encoded message for navigation
 * 
 * @param message - The message to encode
 * @returns URL-encoded message
 */
export function encodeMessageForUrl(message: string): string {
  return encodeURIComponent(message.trim())
}

/**
 * Decode a message from URL parameters
 * 
 * @param encodedMessage - The URL-encoded message
 * @returns Decoded message
 */
export function decodeMessageFromUrl(encodedMessage: string): string {
  try {
    return decodeURIComponent(encodedMessage)
  } catch (error) {
    console.warn('Failed to decode message from URL:', error)
    return ''
  }
}
