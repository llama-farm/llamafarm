import { useState, useCallback, useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useChatInference, useDeleteChatSession, chatKeys } from './useChat'
import { createChatRequest, chatInferenceStreaming } from '../api/chatService'
import { generateMessageId } from '../utils/idGenerator'
import { ChatboxMessage } from '../types/chatbox'
import { ChatStreamChunk } from '../types/chat'
import { useChatSession } from './useChatSession'

/**
 * Custom hook for managing chatbox state and API interactions
 * Extracts chat logic from the Chatbox component for better reusability and testability
 * Now includes session persistence and restoration with streaming support
 */
export function useChatbox(initialSessionId?: string, enableStreaming: boolean = true) {
  const streamingEnabled = enableStreaming && !import.meta.env.VITE_DISABLE_STREAMING
  
  // Session management with persistence
  const {
    currentSessionId: sessionId,
    messages: persistedMessages,
    saveSessionMessages,
    createNewSession,
    setSessionId,
    isLoading: isLoadingSession,
  } = useChatSession(initialSessionId)

  // Local state
  const [messages, setMessages] = useState<ChatboxMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [hasInitialSync, setHasInitialSync] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)

  // Refs for streaming and debounced save
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const streamingAbortControllerRef = useRef<AbortController | null>(null)
  const fallbackTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)

  // API hooks
  const queryClient = useQueryClient()
  const chatMutation = useChatInference()
  const deleteSessionMutation = useDeleteChatSession()

  // Debounced save function to avoid blocking on every message change
  const debouncedSave = useCallback(
    (sessionId: string, messages: ChatboxMessage[]) => {
      // Clear existing timeout
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current)
      }

      // Set new timeout for debounced save
      saveTimeoutRef.current = setTimeout(() => {
        saveSessionMessages(sessionId, messages)
      }, 500) // 500ms delay
    },
    [saveSessionMessages]
  )

  // Sync persisted messages with local state - reset when session changes
  useEffect(() => {
    // Reset state when session changes (project switching)
    setHasInitialSync(false)
    
    // Load messages from new session immediately
    if (persistedMessages.length > 0) {
      setMessages(persistedMessages)
    } else {
      setMessages([])
    }
    setHasInitialSync(true)
  }, [sessionId, persistedMessages])

  // Save messages to persistence when they change (with debouncing)
  useEffect(() => {
    // Save if we have a valid session ID and either:
    // 1. We've done initial sync (loaded from persistence), OR
    // 2. We have messages to save (new session with messages)
    if (sessionId && (hasInitialSync || messages.length > 0)) {
      debouncedSave(sessionId, messages)

      // IMMEDIATELY update React Query cache for cross-component access
      queryClient.setQueryData(chatKeys.session(sessionId), messages)
    }
  }, [messages, sessionId, debouncedSave, hasInitialSync, queryClient])

  // Cleanup timeout and abort streaming on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current)
      }
      if (fallbackTimeoutRef.current) {
        clearTimeout(fallbackTimeoutRef.current)
      }
      if (streamingAbortControllerRef.current) {
        streamingAbortControllerRef.current.abort()
      }
    }
  }, [])

  // Add message to state
  const addMessage = useCallback((message: Omit<ChatboxMessage, 'id'>) => {
    const newMessage: ChatboxMessage = {
      ...message,
      id: generateMessageId(),
    }

    setMessages(prev => [...prev, newMessage])
    return newMessage.id
  }, [])

  // Update message by ID
  const updateMessage = useCallback(
    (id: string, updates: Partial<ChatboxMessage>) => {
      setMessages(prev => {
        const updated = prev.map(msg =>
          msg.id === id ? { ...msg, ...updates } : msg
        )
        return updated
      })
    },
    []
  )

  // Helper function to execute fallback non-streaming request
  const executeFallbackRequest = useCallback(
    async (chatRequest: any, assistantMessageId: string) => {
      console.log('Executing fallback non-streaming request')
      const response = await chatMutation.mutateAsync({
        chatRequest,
        sessionId: sessionId || undefined,
      })

      // Set session ID if received from server (for new sessions)
      if (response.sessionId && response.sessionId !== sessionId) {
        setSessionId(response.sessionId)
        if (!hasInitialSync) {
          setHasInitialSync(true)
        }
      }

      // Update assistant message with response
      if (response.data.choices && response.data.choices.length > 0) {
        const assistantResponse = response.data.choices[0].message.content
        updateMessage(assistantMessageId, {
          content: assistantResponse,
          isLoading: false,
        })
      } else {
        updateMessage(assistantMessageId, {
          content: "Sorry, I didn't receive a proper response.",
          isLoading: false,
        })
      }

      return true
    },
    [chatMutation, sessionId, setSessionId, updateMessage, hasInitialSync]
  )

  // Handle sending message with streaming or fallback
  const sendMessage = useCallback(
    async (messageContent: string) => {
      if (!messageContent.trim() || chatMutation.isPending || isStreaming) return false

      // Clear any previous errors
      setError(null)

      // Add user message immediately (optimistic update)
      addMessage({
        type: 'user',
        content: messageContent,
        timestamp: new Date(),
      })

      // Add loading assistant message
      const assistantMessageId = addMessage({
        type: 'assistant',
        content: 'Thinking...',
        timestamp: new Date(),
        isLoading: true,
      })

      let timeoutId: NodeJS.Timeout | undefined
      let accumulatedContent = ''

      try {
        // Create chat request
        const chatRequest = createChatRequest(messageContent)

        if (streamingEnabled) {
          // Streaming path
          setIsStreaming(true)

          // Create abort controller for this request
          const abortController = new AbortController()
          streamingAbortControllerRef.current = abortController

          // Set a timeout for streaming requests
          timeoutId = setTimeout(() => {
            console.log('Streaming request timed out after 1 minute')
            abortController.abort()
          }, 60000)

          const responseSessionId = await chatInferenceStreaming(
            chatRequest,
            sessionId || undefined,
            {
              onChunk: (chunk: ChatStreamChunk) => {
                if (!isMountedRef.current) return

                // Handle streaming chunk content
                if (chunk.choices && chunk.choices.length > 0) {
                  const choice = chunk.choices[0]
                  if (choice.delta && choice.delta.content) {
                    accumulatedContent += choice.delta.content
                    updateMessage(assistantMessageId, {
                      content: accumulatedContent,
                      isLoading: false,
                      isStreaming: true,
                    })
                  }

                  // Check if streaming is complete
                  if (choice.finish_reason) {
                    updateMessage(assistantMessageId, {
                      content: accumulatedContent || 'Response completed',
                      isLoading: false,
                      isStreaming: false,
                    })
                  }
                }
              },
              onError: (error) => {
                console.error('Streaming error:', error)
                if (!isMountedRef.current) return
                // Will be handled in catch block
              },
              onComplete: () => {
                if (!isMountedRef.current) return
                console.log('Streaming completed')
                setIsStreaming(false)
                if (timeoutId) clearTimeout(timeoutId)
              },
              signal: abortController.signal,
            }
          )

          // Set session ID if received from server
          if (responseSessionId && responseSessionId !== sessionId) {
            setSessionId(responseSessionId)
            if (!hasInitialSync) {
              setHasInitialSync(true)
            }
          }

          setIsStreaming(false)
          if (timeoutId) clearTimeout(timeoutId)
          return true

        } else {
          // Non-streaming fallback
          return await executeFallbackRequest(chatRequest, assistantMessageId)
        }

      } catch (error) {
        console.error('Chat error:', error)
        setIsStreaming(false)
        if (timeoutId) clearTimeout(timeoutId)

        // Check if this was an abort error (user cancelled)
        if (error instanceof Error && error.name === 'AbortError') {
          updateMessage(assistantMessageId, {
            content: accumulatedContent || 'Request was cancelled',
            isLoading: false,
            isStreaming: false,
          })
          return false
        }

        // Remove loading message
        setMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))

        // Set error message
        const errorMessage =
          error instanceof Error
            ? error.message
            : 'An unexpected error occurred'
        setError(errorMessage)

        // Add error message to chat
        addMessage({
          type: 'error',
          content: `Error: ${errorMessage}`,
          timestamp: new Date(),
        })

        return false
      }
    },
    [
      chatMutation,
      sessionId,
      setSessionId,
      addMessage,
      updateMessage,
      isStreaming,
      hasInitialSync,
      streamingEnabled,
      executeFallbackRequest
    ]
  )

  // Handle clear chat
  const clearChat = useCallback(async () => {
    if (deleteSessionMutation.isPending) return false

    try {
      // If we don't have a valid sessionId (e.g., mock/test mode), skip server deletion
      if (sessionId && !sessionId.startsWith('local_')) {
        await deleteSessionMutation.mutateAsync(sessionId)
      }

      // Clear local messages and errors
      setMessages([])
      setError(null)

      // Reset initial sync flag to allow fresh sync with new session
      setHasInitialSync(false)

      // Create new session (this will update sessionId and trigger persistence)
      createNewSession()

      return true
    } catch (error) {
      console.error('Delete session error:', error)
      const errorMessage =
        error instanceof Error ? error.message : 'Failed to clear chat'
      setError(errorMessage)
      return false
    }
  }, [deleteSessionMutation, sessionId, createNewSession])

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
      setMessages(prev => prev.map(msg => 
        msg.isStreaming 
          ? { ...msg, isStreaming: false, content: msg.content + ' [Cancelled]' }
          : msg
      ))
    }
  }, [isStreaming])

  // Reset to new session
  const resetSession = useCallback(() => {
    // Cancel any active streaming first
    if (isStreaming) {
      cancelStreaming()
    }
    
    const newSessionId = createNewSession()
    setMessages([])
    setError(null)
    setInputValue('')

    // Reset initial sync flag to allow fresh sync with new session
    setHasInitialSync(false)

    return newSessionId
  }, [createNewSession, isStreaming, cancelStreaming])

  return {
    // State
    sessionId,
    messages,
    inputValue,
    error,

    // Loading states
    isSending: chatMutation.isPending || isStreaming,
    isStreaming,
    isClearing: deleteSessionMutation.isPending,
    isLoadingSession,

    // Actions
    sendMessage,
    clearChat,
    updateInput,
    clearError,
    resetSession,
    cancelStreaming,
    addMessage,
    updateMessage,

    // Computed values
    hasMessages: messages.length > 0,
    canSend: !chatMutation.isPending && !isStreaming && inputValue.trim().length > 0,
  }
}

export default useChatbox