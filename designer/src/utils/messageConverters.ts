/**
 * Type conversion utilities for transforming between API and UI message types
 * 
 * These functions provide safe, explicit conversion between the API layer (ChatMessage)
 * and UI layer (ChatboxMessage) types while maintaining type safety and consistency.
 */

import { ChatMessage } from '../types/chat'
import { ChatboxMessage } from '../types/chatbox'
import { generateMessageId } from './idGenerator'

/**
 * Convert an API message to a UI message structure
 * Maps role field to type field and adds required UI fields
 * 
 * @param apiMessage - Message from API response
 * @param id - Optional unique ID (will generate if not provided)
 * @param timestamp - Optional timestamp (defaults to current time)
 * @returns ChatboxMessage ready for UI display
 * 
 * @example
 * const apiMsg: ChatMessage = { role: 'assistant', content: 'Hello!' }
 * const uiMsg = apiMessageToUiMessage(apiMsg)
 * // Result: { id: 'msg-...', type: 'assistant', content: 'Hello!', timestamp: Date }
 */
export function apiMessageToUiMessage(
  apiMessage: ChatMessage,
  id?: string,
  timestamp: Date = new Date()
): ChatboxMessage {
  return {
    id: id || generateMessageId(),
    content: apiMessage.content,
    type: apiMessage.role, // Direct mapping: role -> type
    timestamp,
    isLoading: false
  }
}

/**
 * Convert a UI message to an API message structure
 * Maps type field to role field and strips UI-specific fields
 * Note: 'error' type messages cannot be converted (not valid for API)
 * 
 * @param uiMessage - Message from UI state
 * @returns ChatMessage ready for API request, or null if conversion not possible
 * 
 * @example
 * const uiMsg: ChatboxMessage = { 
 *   id: 'msg-123', 
 *   type: 'user', 
 *   content: 'Hello!', 
 *   timestamp: new Date() 
 * }
 * const apiMsg = uiMessageToApiMessage(uiMsg)
 * // Result: { role: 'user', content: 'Hello!' }
 */
export function uiMessageToApiMessage(
  uiMessage: ChatboxMessage
): ChatMessage | null {
  // Error type is UI-specific and cannot be converted to API format
  if (uiMessage.type === 'error') {
    console.warn('Cannot convert error message to API format:', uiMessage.content)
    return null
  }

  return {
    role: uiMessage.type, // Direct mapping: type -> role
    content: uiMessage.content
  }
}

/**
 * Convert an array of UI messages to API messages
 * Filters out error messages and other non-convertible messages
 * 
 * @param uiMessages - Array of UI messages
 * @returns Array of API messages (may be shorter due to filtering)
 * 
 * @example
 * const uiMessages = [
 *   { id: '1', type: 'user', content: 'Hi', timestamp: new Date() },
 *   { id: '2', type: 'error', content: 'Error occurred', timestamp: new Date() },
 *   { id: '3', type: 'assistant', content: 'Hello!', timestamp: new Date() }
 * ]
 * const apiMessages = uiMessagesToApiMessages(uiMessages)
 * // Result: [{ role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello!' }]
 */
export function uiMessagesToApiMessages(
  uiMessages: ChatboxMessage[]
): ChatMessage[] {
  return uiMessages
    .map(uiMessageToApiMessage)
    .filter((msg): msg is ChatMessage => msg !== null)
}

/**
 * Create a UI message from scratch with proper defaults
 * Useful for creating new messages in the UI layer
 * 
 * @param content - Message content
 * @param type - Message type
 * @param options - Optional additional fields
 * @returns Complete ChatboxMessage
 */
export function createUiMessage(
  content: string,
  type: ChatboxMessage['type'],
  options: Partial<Pick<ChatboxMessage, 'sources' | 'metadata' | 'isLoading'>> = {}
): ChatboxMessage {
  return {
    id: generateMessageId(),
    content,
    type,
    timestamp: new Date(),
    isLoading: false,
    ...options
  }
}

/**
 * Type guard to check if a message type is valid for API conversion
 */
export function isApiCompatibleType(type: ChatboxMessage['type']): type is ChatMessage['role'] {
  return type === 'user' || type === 'assistant' || type === 'system'
}

export default {
  apiMessageToUiMessage,
  uiMessageToApiMessage,
  uiMessagesToApiMessages,
  createUiMessage,
  isApiCompatibleType
}
