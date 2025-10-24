/**
 * Enhanced Chatbox Hook with Project Session Management
 *
 * Integrates with the project session manager for Designer Chat service
 * Maintains backward compatibility while adding project context session management
 */

import { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { useDeleteProjectChatSession, useProjectChat } from './useChat'
import type { ProjectChatMutation } from './useChat'
import { createChatRequest, chatProjectStreaming } from '../api/chatService'
import { useProjectSession } from './useProjectSession'
import { ChatboxMessage } from '../types/chatbox'
import { ChatStreamChunk, NetworkError } from '../types/chat'
import { generateMessageId } from '../utils/idGenerator'
import { useActiveProject } from './useActiveProject'

/**
 * Convert project session message to chatbox message format
 */
function projectSessionToChatboxMessage(msg: {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}): ChatboxMessage {
  return {
    id: msg.id,
    type: msg.role === 'user' ? 'user' : 'assistant',
    content: msg.content,
    timestamp: new Date(msg.timestamp),
  }
}

/**
 * Enhanced chatbox hook with project session management for Designer Chat
 */
export function useChatboxWithProjectSession(enableStreaming: boolean = true) {
  const streamingEnabled =
    enableStreaming && !import.meta.env.VITE_DISABLE_STREAMING

  // Project session management for Designer Chat
  const projectSession = useProjectSession({
    chatService: 'designer',
    autoCreate: false, // Sessions created on first message
  })

  // Active project context
  const activeProject = useActiveProject()
  const ns = activeProject?.namespace || ''
  const proj = activeProject?.project || ''

  // Use sessionId from projectSession; do not create local fixed IDs here

  // UI state
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  // Refs for streaming
  const streamingAbortControllerRef = useRef<AbortController | null>(null)
  // Fallback removed; no timeout tracking required
  const isMountedRef = useRef(true)

  // API hooks (project-scoped); must be called unconditionally to satisfy React rules.
  // We will guard usage when ns/proj are missing.
  const projectChat = useProjectChat(ns, proj) as unknown as ProjectChatMutation
  const deleteProjectSessionMutation = useDeleteProjectChatSession(ns, proj)

  // Get current state from project session system (always used)
  const projectSessionMessages = useMemo(() => {
    return projectSession.messages.map(projectSessionToChatboxMessage)
  }, [projectSession.messages])
  const isLoadingSession = false // Session loading is now synchronous

  // Cleanup timeout and abort streaming on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false
      if (streamingAbortControllerRef.current) {
        streamingAbortControllerRef.current.abort()
      }
    }
  }, [])

  // Non-streaming fallback removed

  // Add message to both streaming state and project session
  // Legacy addMessage helper removed; we now add directly via projectSession

  // Simplified streaming state: single temporary assistant message
  const [tempAssistant, setTempAssistant] = useState<ChatboxMessage | null>(
    null
  )

  // Combine project session messages with temporary streaming messages
  const currentMessages = useMemo(() => {
    return tempAssistant
      ? [...projectSessionMessages, tempAssistant]
      : [...projectSessionMessages]
  }, [projectSessionMessages, tempAssistant])

  // Handle sending message with streaming or non-streaming API integration
  const sendMessage = useCallback(
    async (messageContent: string) => {
      // Validate input
      if (!messageContent || messageContent.trim() === '') {
        return false
      }

      if ((projectChat?.isPending ?? false) || isStreaming) {
        return false
      }

      messageContent = messageContent.trim()

      // Sessions will be created when API responds with session ID

      // Cancel any existing streaming request before starting a new one
      if (streamingAbortControllerRef.current) {
        streamingAbortControllerRef.current.abort()
        streamingAbortControllerRef.current = null
      }

      // Clear any previous errors
      setError(null)

      // Add user message to project session
      projectSession.addMessage(messageContent, 'user')

      // Create a temporary assistant message for streaming display
      const assistantMessageId = generateMessageId()
      setTempAssistant({
        id: assistantMessageId,
        type: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
        isLoading: false,
      })

      let timeoutId: NodeJS.Timeout | undefined

      try {
        // Create chat request
        const chatRequest = createChatRequest(messageContent)

        // Streaming path only
        setIsStreaming(true)

        // Create abort controller for this request
        const abortController = new AbortController()
        streamingAbortControllerRef.current = abortController

        // Set a timeout for streaming requests
        timeoutId = setTimeout(() => {
          console.log('Streaming request timed out after 1 minute')
          abortController.abort()
        }, 60000)

        let accumulatedContent = ''
        await chatProjectStreaming(
          ns,
          proj,
          chatRequest,
          projectSession.sessionId || undefined,
          {
            onChunk: (chunk: ChatStreamChunk) => {
              // Handle role assignment (first chunk)
              if (chunk.choices?.[0]?.delta?.role && !accumulatedContent) {
                return
              }

              // Handle content chunks
              if (chunk.choices?.[0]?.delta?.content) {
                accumulatedContent += chunk.choices[0].delta.content
                setTempAssistant(prev =>
                  prev && prev.id === assistantMessageId
                    ? {
                        ...prev,
                        content: accumulatedContent,
                        isStreaming: true,
                      }
                    : prev
                )
              }
            },
            onError: (error: Error) => {
              console.error('Streaming error:', error)
              clearTimeout(timeoutId)
              setIsStreaming(false)

              // Determine cancellation vs other errors
              const isAbortError =
                error instanceof Error && error.name === 'AbortError'
              const isWrappedCancel =
                error instanceof NetworkError &&
                (error.message.toLowerCase().includes('cancelled') ||
                  error.message.toLowerCase().includes('canceled') ||
                  error.message.toLowerCase().includes('aborted'))
              const isUserCancelled =
                isAbortError ||
                (error as any)?.code === 'USER_CANCELLED' ||
                (error as any)?.name === 'UserCancelledError' ||
                isWrappedCancel

              if (!isUserCancelled && error instanceof NetworkError) {
                // Show a single error banner; do not add error lines to chat
                setError(error.message)
              } else {
                // For user cancellations: keep partial text, mark cancelled; suppress toast
                if (isUserCancelled) {
                  setError(null)
                  setTempAssistant(prev =>
                    prev && prev.id === assistantMessageId
                      ? { ...prev, isStreaming: false, cancelled: true }
                      : prev
                  )
                } else {
                  // Other errors: show a single error line, no toast
                  const errorMessage =
                    error instanceof NetworkError
                      ? error.message
                      : 'Streaming connection failed'
                  setError(null)
                  setError(errorMessage)
                }
              }
            },
            onComplete: () => {
              clearTimeout(timeoutId)
              setIsStreaming(false)

              // If we got content, finalize the message
              if (accumulatedContent && accumulatedContent.trim()) {
                // Save final message to project session and clear temp assistant
                try {
                  projectSession.addMessage(accumulatedContent, 'assistant')
                } catch (err) {
                  console.warn('Failed to save to project session:', err)
                }
              }
              setTempAssistant(null)
            },
            signal: abortController.signal,
          }
        )
        // For streaming, we return true immediately as the request is initiated
        // The actual success/failure will be handled by the streaming callbacks
        return true
      } catch (error) {
        console.error('Chat error:', error)
        setIsStreaming(false)

        // If user cancelled, suppress toast and extra messages but keep partials marked cancelled
        const isAbortError =
          error instanceof Error && error.name === 'AbortError'
        const isWrappedCancel =
          error instanceof NetworkError &&
          (error.message.toLowerCase().includes('cancelled') ||
            error.message.toLowerCase().includes('canceled') ||
            error.message.toLowerCase().includes('aborted'))
        const isUserCancelled =
          isAbortError ||
          (error as any)?.code === 'USER_CANCELLED' ||
          (error as any)?.name === 'UserCancelledError' ||
          isWrappedCancel
        if (isUserCancelled) {
          setError(null)
          setTempAssistant(prev =>
            prev && prev.id === assistantMessageId
              ? { ...prev, isStreaming: false, cancelled: true }
              : prev
          )
          return false
        }

        // For other errors, clear the temporary streaming message
        setTempAssistant(null)

        // Set error message
        const errorMessage =
          error instanceof Error
            ? error.message
            : 'An unexpected error occurred'
        setError(errorMessage)

        return false
      } finally {
        // Clear the abort controller reference and timeout
        streamingAbortControllerRef.current = null
        if (timeoutId) {
          clearTimeout(timeoutId)
        }
      }
    },
    [projectChat, streamingEnabled, isStreaming, projectSession, tempAssistant]
  )

  // Handle clear chat
  const clearChat = useCallback(async () => {
    if (deleteProjectSessionMutation?.isPending) return false

    try {
      // Delete on server if we have a current session
      if (projectSession.sessionId) {
        try {
          await deleteProjectSessionMutation?.mutateAsync(
            projectSession.sessionId
          )
        } catch (e) {
          // Non-fatal; continue clearing local state
          // console.warn('Server session delete failed:', e)
        }
      }
      // Use project session system
      projectSession.clearHistory()
      // Also clear any temporary streaming messages
      setTempAssistant(null)
      setError(null)
      return true
    } catch (error) {
      console.error('Clear chat error:', error)
      const errorMessage =
        error instanceof Error ? error.message : 'Failed to clear chat'
      setError(errorMessage)
      return false
    }
  }, [projectSession, deleteProjectSessionMutation])

  // Handle input change
  const updateInput = useCallback((value: string) => {
    setInputValue(value)
  }, [])

  // Clear error
  const clearError = useCallback(() => {
    setError(null)
  }, [])

  // Cancel streaming
  const cancelStreaming = useCallback(() => {
    if (streamingAbortControllerRef.current && isStreaming) {
      streamingAbortControllerRef.current.abort()
      setIsStreaming(false)

      // Update any streaming messages to show they were cancelled
      setTempAssistant(prev =>
        prev ? { ...prev, isStreaming: false, cancelled: true } : prev
      )
      // onError will append the single system line; suppress toast
      setError(null)
    }
  }, [isStreaming])

  // Reset to new session (clear current session - new one will be created on next message)
  const resetSession = useCallback(() => {
    // Cancel any active streaming first
    if (isStreaming) {
      cancelStreaming()
    }

    // Clear current session - new one will be created on first message
    setTempAssistant(null)
    setError(null)
    setInputValue('')

    // Return empty string since we don't create sessions proactively
    return ''
  }, [isStreaming, cancelStreaming])

  const result = {
    // State
    sessionId: projectSession.sessionId,
    messages: currentMessages,
    inputValue,
    error: error || projectSession.error,

    // Loading states
    isSending: (projectChat?.isPending ?? false) || isStreaming,
    isStreaming,
    isClearing: deleteProjectSessionMutation?.isPending ?? false,
    isLoadingSession,

    // Actions
    sendMessage,
    clearChat,
    updateInput,
    clearError,
    resetSession,
    cancelStreaming,
    // Session utilities for UI
    listSessions: projectSession.listSessions,
    selectSession: projectSession.selectSession,

    // Computed values
    hasMessages: currentMessages.length > 0,
    canSend:
      !(projectChat?.isPending ?? false) &&
      !isStreaming &&
      inputValue.trim().length > 0,
  }

  return result
}

export default useChatboxWithProjectSession
