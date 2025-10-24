/**
 * Chat UI Types for frontend chat interface and session management
 *
 * This file contains types for UI state management, session persistence, and frontend-specific
 * chat functionality. These types include UI-specific fields and behaviors not present in the API layer.
 *
 * Related to: ../types/chat.ts (API layer types)
 */

import { BaseMessage } from './chat'

/**
 * Enhanced message structure for UI layer with frontend-specific fields
 * Extends BaseMessage to maintain consistency with API layer
 *
 * Key differences from ChatMessage (API layer):
 * - Uses 'type' instead of 'role' (includes 'error' for UI feedback)
 * - Includes UI state: id, timestamp, isLoading
 * - Includes presentation data: sources, metadata
 *
 * @example
 * const message: ChatboxMessage = {
 *   id: 'msg-123',
 *   type: 'user',
 *   content: 'Hello!',
 *   timestamp: new Date(),
 *   isLoading: false
 * }
 */
export interface ChatboxMessage extends BaseMessage {
  /** Unique identifier for UI tracking and updates */
  id: string
  /** Message type - includes 'error' for UI-specific error display */
  type: 'user' | 'assistant' | 'system' | 'error'
  /** When the message was created/received */
  timestamp: Date
  /** Whether the message is currently being processed (UI state) */
  isLoading?: boolean
  /** Whether the message is currently being streamed */
  isStreaming?: boolean
  /** Whether the message stream was cancelled mid-response */
  cancelled?: boolean
  /** Additional context or citation sources for display */
  sources?: any[]
  /** Extended metadata for UI purposes */
  metadata?: any
}

/**
 * Session metadata for chat persistence and management
 * Used by the session management layer for storing chat history and user session data
 *
 * @example
 * const session: ChatSession = {
 *   id: 'session-abc123',
 *   createdAt: new Date('2024-01-01'),
 *   lastActivity: new Date(),
 *   messageCount: 5,
 *   title: 'Chat about React hooks...'
 * }
 */
export interface ChatSession {
  /** Unique session identifier */
  id: string
  /** When the session was first created */
  createdAt: Date
  /** Last time the session had activity */
  lastActivity: Date
  /** Number of messages in the session */
  messageCount: number
  /** Display title for the session (optional) */
  title?: string
}
