import { useState, useCallback, useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useChatInference, useDeleteChatSession, chatKeys } from './useChat'
import { createChatRequest, chatInferenceStreaming } from '../api/chatService'
import { generateMessageId } from '../utils/idGenerator'
import useChatSession from './useChatSession'
import { ChatboxMessage } from '../types/chatbox'
import { ChatStreamChunk, NetworkError } from '../types/chat'

/**
 * Custom hook for managing chatbox state and API interactions
 * Extracts chat logic from the Chatbox component for better reusability and testability
 * Now includes session persistence and restoration with streaming support
 */
export function useChatbox(initialSessionId?: string, enableStreaming: boolean = true) {
  const streamingEnabled = enableStreaming
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

  // Sync persisted messages with local state ONLY on true initial load
  useEffect(() => {
    // CRITICAL: Only load from persistence on TRUE component mount (not navigation)
    // We determine this by checking if this is genuinely the first load for this session
    if (!hasInitialSync && messages.length === 0) {
      if (persistedMessages.length > 0) {
        setMessages(persistedMessages)
      }
      setHasInitialSync(true)
    }
  }, [persistedMessages, hasInitialSync, messages.length, sessionId])

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
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current)
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

  // Handle sending message with streaming or non-streaming API integration
  const sendMessage = useCallback(
    async (messageContent: string) => {
      if (!messageContent.trim() || chatMutation.isPending || isStreaming) return false

      // Cancel any existing streaming request before starting a new one
      if (streamingAbortControllerRef.current) {
        streamingAbortControllerRef.current.abort()
        streamingAbortControllerRef.current = null
      }

      // Clear any previous errors
      setError(null)

      // Add user message immediately (optimistic update)
      addMessage({
        type: 'user',
        content: messageContent,
        timestamp: new Date(),
      })

      // Add loading/streaming assistant message
      const assistantMessageId = addMessage({
        type: 'assistant',
        content: streamingEnabled ? '' : 'Thinking...',
        timestamp: new Date(),
        isLoading: !streamingEnabled,
        isStreaming: streamingEnabled,
      })

      let timeoutId: NodeJS.Timeout | undefined

      try {
        // Create chat request
        const chatRequest = createChatRequest(messageContent)

        if (streamingEnabled) {
          // Streaming path
          setIsStreaming(true)
          
          // Create abort controller for this request
          const abortController = new AbortController()
          streamingAbortControllerRef.current = abortController
          
          timeoutId = setTimeout(() => {
            console.log('Streaming request timed out after 1 minute')
            abortController.abort()
          }, 60000)

          let accumulatedContent = ''
          let finalSessionId = sessionId

          const responseSessionId = await chatInferenceStreaming(
            chatRequest,
            sessionId,
            {
              onChunk: (chunk: ChatStreamChunk) => {
                // Handle role assignment (first chunk)
                if (chunk.choices?.[0]?.delta?.role && !accumulatedContent) {
                  // Role chunk - no content to add yet
                  return
                }

                // Handle content chunks
                if (chunk.choices?.[0]?.delta?.content) {
                  accumulatedContent += chunk.choices[0].delta.content
                  updateMessage(assistantMessageId, {
                    content: accumulatedContent,
                    isStreaming: true,
                  })
                }
              },
              onError: (error: Error) => {
                console.error('Streaming error:', error)
                clearTimeout(timeoutId) // Clear the timeout
                setIsStreaming(false)
                
                // Remove streaming message
                setMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))

                // For abort errors or network issues, try fallback to non-streaming
                if (error instanceof NetworkError && 
                    (error.message.includes('cancelled') || error.message.includes('aborted'))) {
                  // Silently fall back to non-streaming mode
                  setTimeout(async () => {
                    try {
                      const chatRequest = createChatRequest(messageContent)
                      const response = await chatMutation.mutateAsync({
                        chatRequest,
                        sessionId,
                      })
                      
                      // Add the response as a new message
                      if (response.data.choices && response.data.choices.length > 0) {
                        const assistantResponse = response.data.choices[0].message.content
                        addMessage({
                          type: 'assistant',
                          content: assistantResponse,
                          timestamp: new Date(),
                        })
                      } else {
                        addMessage({
                          type: 'assistant',
                          content: "Sorry, I didn't receive a proper response.",
                          timestamp: new Date(),
                        })
                      }
                    } catch (fallbackError) {
                      console.error('Fallback request also failed:', fallbackError)
                      const errorMessage = fallbackError instanceof Error 
                        ? fallbackError.message 
                        : 'Failed to get response'
                      setError(errorMessage)
                      addMessage({
                        type: 'error',
                        content: `Error: ${errorMessage}`,
                        timestamp: new Date(),
                      })
                    }
                  }, 100)
                } else {
                  // For other errors, show error message
                  const errorMessage = error instanceof NetworkError 
                    ? error.message 
                    : 'Streaming connection failed'
                  setError(errorMessage)

                  addMessage({
                    type: 'error',
                    content: `Error: ${errorMessage}`,
                    timestamp: new Date(),
                  })
                }
              },
              onComplete: () => {
                clearTimeout(timeoutId) // Clear the timeout
                setIsStreaming(false)
                
                // If we didn't get any content through streaming, fall back to non-streaming
                if (!accumulatedContent || accumulatedContent.trim() === '') {
                  // Remove the streaming message and try non-streaming
                  setMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))
                  
                  // Retry with non-streaming
                  setTimeout(async () => {
                    try {
                      const chatRequest = createChatRequest(messageContent)
                      const response = await chatMutation.mutateAsync({
                        chatRequest,
                        sessionId,
                      })
                      
                      // Add the response as a new message
                      if (response.data.choices && response.data.choices.length > 0) {
                        const assistantResponse = response.data.choices[0].message.content
                        addMessage({
                          type: 'assistant',
                          content: assistantResponse,
                          timestamp: new Date(),
                        })
                      } else {
                        addMessage({
                          type: 'assistant',
                          content: "Sorry, I didn't receive a proper response.",
                          timestamp: new Date(),
                        })
                      }
                    } catch (fallbackError) {
                      console.error('Fallback request also failed:', fallbackError)
                      addMessage({
                        type: 'error',
                        content: 'Error: Failed to get response',
                        timestamp: new Date(),
                      })
                    }
                  }, 100)
                  
                  return
                }
                
                updateMessage(assistantMessageId, {
                  content: accumulatedContent,
                  isStreaming: false,
                  isLoading: false,
                })
              },
              signal: abortController.signal,
            }
          )

          finalSessionId = responseSessionId

          // Set session ID if received from server (for new sessions)
          if (finalSessionId && finalSessionId !== sessionId) {
            setSessionId(finalSessionId)
            // Mark as having initial sync since this is a new session with messages
            if (!hasInitialSync) {
              setHasInitialSync(true)
            }
          }

          return true

        } else {
          // Non-streaming path (fallback)
          const response = await chatMutation.mutateAsync({
            chatRequest,
            sessionId,
          })

          // Set session ID if received from server (for new sessions)
          if (response.sessionId && response.sessionId !== sessionId) {
            setSessionId(response.sessionId)
            // Mark as having initial sync since this is a new session with messages
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
        }

      } catch (error) {
        console.error('Chat error:', error)
        setIsStreaming(false)

        // Remove loading/streaming message
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
      } finally {
        // Clear the abort controller reference and timeout
        streamingAbortControllerRef.current = null
        if (timeoutId) {
          clearTimeout(timeoutId)
        }
      }
    },
    [
      chatMutation, 
      sessionId, 
      setSessionId, 
      addMessage, 
      updateMessage, 
      streamingEnabled, 
      isStreaming,
      hasInitialSync
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
