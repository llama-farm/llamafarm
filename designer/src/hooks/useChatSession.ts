import { useState, useEffect, useCallback } from 'react'
import { useActiveProject } from './useActiveProject'
import {
  getExistingSession,
  createOptimisticSession,
  reconcileSessionWithServer,
  loadChatHistory,
  addMessageToHistory,
  initializeChatHistory,
  cleanupPendingSessions,
} from '../utils/projectSessionManager'
import { ChatboxMessage } from '../types/chatbox'

/**
 * Hook that provides session management for chatbox with the expected API
 * Bridges between the existing session management utilities and useChatbox expectations
 */
export function useChatSession(initialSessionId?: string) {
  const activeProject = useActiveProject()
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(initialSessionId || null)
  const [persistedMessages, setPersistedMessages] = useState<ChatboxMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)

  // Cleanup on mount
  useEffect(() => {
    cleanupPendingSessions(60) // 60 minutes
  }, [])

  // Load session when activeProject or initialSessionId changes
  useEffect(() => {
    if (!activeProject) {
      setCurrentSessionId(null)
      setPersistedMessages([])
      return
    }

    setIsLoading(true)

    try {
      // Look for existing session for this project
      const existingSessionId = getExistingSession(
        activeProject.namespace,
        activeProject.project,
        'designer' // Using 'designer' for chat service type
      )

      if (existingSessionId) {
        setCurrentSessionId(existingSessionId)
        const history = loadChatHistory(existingSessionId)
        // Convert from ChatMessage to ChatboxMessage format
        const chatboxMessages = history.map(msg => ({
          id: msg.id,
          type: msg.role === 'user' ? 'user' as const : 'assistant' as const,
          content: msg.content,
          timestamp: new Date(msg.timestamp),
        }))
        setPersistedMessages(chatboxMessages)
      } else if (initialSessionId) {
        // Use provided initial session ID if no existing session
        setCurrentSessionId(initialSessionId)
        const history = loadChatHistory(initialSessionId)
        const chatboxMessages = history.map(msg => ({
          id: msg.id,
          type: msg.role === 'user' ? 'user' as const : 'assistant' as const,
          content: msg.content,
          timestamp: new Date(msg.timestamp),
        }))
        setPersistedMessages(chatboxMessages)
      } else {
        setCurrentSessionId(null)
        setPersistedMessages([])
      }
    } catch (error) {
      console.error('Failed to load session:', error)
      setCurrentSessionId(null)
      setPersistedMessages([])
    } finally {
      setIsLoading(false)
    }
  }, [activeProject?.namespace, activeProject?.project, initialSessionId])

  // Save messages to the current session
  const saveSessionMessages = useCallback((sessionId: string, messages: ChatboxMessage[]) => {
    if (!activeProject) return

    try {
      // Clear existing history and save new messages
      const history = loadChatHistory(sessionId)
      
      // Only save if messages have changed to avoid unnecessary writes
      if (history.length !== messages.length || 
          messages.some((msg, i) => !history[i] || history[i].content !== msg.content)) {
        
        // Initialize empty history
        initializeChatHistory(sessionId)
        
        // Add all messages
        messages.forEach(msg => {
          addMessageToHistory(sessionId, {
            role: msg.type === 'user' ? 'user' : 'assistant',
            content: msg.content,
            timestamp: msg.timestamp.toISOString(),
          })
        })
      }
    } catch (error) {
      console.error('Failed to save session messages:', error)
    }
  }, [activeProject])

  // Create a new session
  const createNewSession = useCallback(() => {
    if (!activeProject) {
      console.warn('Cannot create session without active project')
      return null
    }

    // Create optimistic session
    const sessionId = createOptimisticSession(
      activeProject.namespace,
      activeProject.project,
      'designer'
    )
    
    setCurrentSessionId(sessionId)
    setPersistedMessages([])
    
    return sessionId
  }, [activeProject])

  // Set the current session ID (for when server provides session ID)
  const setSessionId = useCallback((sessionId: string) => {
    if (!activeProject) return

    // If we have an existing session, reconcile it with the server ID
    if (currentSessionId) {
      try {
        const finalSessionId = reconcileSessionWithServer(currentSessionId, sessionId)
        setCurrentSessionId(finalSessionId)
        
        // Reload messages in case session ID changed
        const history = loadChatHistory(finalSessionId)
        const chatboxMessages = history.map(msg => ({
          id: msg.id,
          type: msg.role === 'user' ? 'user' as const : 'assistant' as const,
          content: msg.content,
          timestamp: new Date(msg.timestamp),
        }))
        setPersistedMessages(chatboxMessages)
        
        console.log('Reconciled session:', currentSessionId, '->', finalSessionId)
      } catch (error) {
        console.error('Failed to reconcile session:', error)
        setCurrentSessionId(sessionId)
      }
    } else {
      setCurrentSessionId(sessionId)
    }
  }, [activeProject, currentSessionId])

  return {
    currentSessionId,
    messages: persistedMessages,
    saveSessionMessages,
    createNewSession,
    setSessionId,
    isLoading,
  }
}
