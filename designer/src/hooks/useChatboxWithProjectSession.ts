/**
 * Enhanced Chatbox Hook with Project Session Management
 * 
 * Integrates with the project session manager for Designer Chat service
 * Maintains backward compatibility while adding project context session management
 */

import { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { useChatInference, useDeleteChatSession } from './useChat'
import { createChatRequest, chatInferenceStreaming } from '../api/chatService'
import { useProjectSession } from './useProjectSession'
import { ChatboxMessage } from '../types/chatbox'
import { ChatStreamChunk, NetworkError } from '../types/chat'
import { generateMessageId } from '../utils/idGenerator'

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
export function useChatboxWithProjectSession(
  enableStreaming: boolean = true
) {
  const streamingEnabled = enableStreaming && !import.meta.env.VITE_DISABLE_STREAMING
  
  // Project session management for Designer Chat
  const projectSession = useProjectSession({
    chatService: 'designer',
    autoCreate: false, // Sessions created on first message
  })
  
  // UI state
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  
  // Refs for streaming
  const streamingAbortControllerRef = useRef<AbortController | null>(null)
  const fallbackTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)

  // API hooks
  const chatMutation = useChatInference()
  const deleteSessionMutation = useDeleteChatSession()

  // Get current state from project session system (always used)
  const currentSessionId = projectSession.sessionId
  const projectSessionMessages = useMemo(() => {
    return projectSession.messages.map(projectSessionToChatboxMessage)
  }, [projectSession.messages, currentSessionId])
  const isLoadingSession = false // Session loading is now synchronous

  // Cleanup timeout and abort streaming on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false
      if (fallbackTimeoutRef.current) {
        clearTimeout(fallbackTimeoutRef.current)
      }
      if (streamingAbortControllerRef.current) {
        streamingAbortControllerRef.current.abort()
      }
    }
  }, [])

  // Helper function to execute fallback non-streaming request
  const executeFallbackRequest = useCallback(async (
    messageContent: string,
    currentSessionId: string,
    onSuccess: (response: any) => void,
    onError: (error: Error) => void
  ) => {
    // Check if component is still mounted before proceeding
    if (!isMountedRef.current) {
      return
    }

    try {
      const chatRequest = createChatRequest(messageContent)
      const response = await chatMutation.mutateAsync({
        chatRequest,
        sessionId: currentSessionId,
      })
      
      // Check if component is still mounted before updating state
      if (!isMountedRef.current) {
        return
      }
      
      onSuccess(response)
    } catch (fallbackError) {
      // Check if component is still mounted before updating state
      if (!isMountedRef.current) {
        return
      }
      
      console.error('Fallback request also failed:', fallbackError)
      onError(fallbackError instanceof Error ? fallbackError : new Error('Unknown fallback error'))
    }
  }, [chatMutation])

  // Add message to both streaming state and project session
  const addMessage = useCallback((message: Omit<ChatboxMessage, 'id'>) => {
    const newMessage: ChatboxMessage = {
      ...message,
      id: generateMessageId(),
    }
    
    // Check if this is a placeholder message that should skip project session in persistent mode
    const isThinkingPlaceholder = message.content === 'Thinking...' && message.type === 'assistant';
    const shouldSkipProjectSession = isThinkingPlaceholder && !projectSession.isTemporaryMode;
    
    if (!shouldSkipProjectSession) {
      // Add to project session system - it will create temporary session if needed
      try {
        projectSession.addMessage(message.content, message.type === 'user' ? 'user' : 'assistant')
      } catch (err) {
        console.error('Failed to add message to project session:', err)
        // Don't fail silently - this is a critical error
        throw err
      }
    }
    
    return newMessage.id
  }, [projectSession])
  
  // Update message helper (for streaming updates before final save to project session)
  const [streamingMessages, setStreamingMessages] = useState<ChatboxMessage[]>([])
  const updateMessage = useCallback(
    (id: string, updates: Partial<ChatboxMessage>) => {
      // For project session system, we maintain temporary streaming messages
      // These are later replaced when final message is saved to project session
      setStreamingMessages(prev => {
        const existing = prev.find(msg => msg.id === id)
        if (existing) {
          return prev.map(msg => msg.id === id ? { ...msg, ...updates } : msg)
        } else {
          // Add new streaming message
          return [...prev, { id, type: 'assistant', content: '', timestamp: new Date(), ...updates } as ChatboxMessage]
        }
      })
    },
    []
  )
  
  // Combine project session messages with temporary streaming messages
  const currentMessages = useMemo(() => {
    const combined = [...projectSessionMessages, ...streamingMessages]
    
    // Filter out "Thinking..." placeholder messages for UI display
    const filteredForUI = combined.filter(msg => {
      const isThinkingPlaceholder = msg.type === 'assistant' && msg.content === 'Thinking...' && !msg.isStreaming && !msg.isLoading;
      return !isThinkingPlaceholder;
    });
    
    return filteredForUI
  }, [projectSessionMessages, streamingMessages])
  
  
  // Handle sending message with streaming or non-streaming API integration
  const sendMessage = useCallback(
    async (messageContent: string) => {
      // Validate input
      if (!messageContent || messageContent.trim() === '') {
        return false;
      }
      
      if (chatMutation.isPending || isStreaming) {
        return false;
      }
      
      messageContent = messageContent.trim();
      
      // Sessions will be created when API responds with session ID
      
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
        content: 'Thinking...',
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
          
          // Set a timeout for streaming requests
          timeoutId = setTimeout(() => {
            console.log('Streaming request timed out after 1 minute')
            abortController.abort()
          }, 60000)
          
          let accumulatedContent = ''
          let deferredSessionId: string | null = null
          
          const responseSessionId = await chatInferenceStreaming(
            chatRequest,
            currentSessionId || undefined,
            {
              onChunk: (chunk: ChatStreamChunk) => {
                // Handle role assignment (first chunk)
                if (chunk.choices?.[0]?.delta?.role && !accumulatedContent) {
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
                clearTimeout(timeoutId)
                setIsStreaming(false)
                
                // Remove streaming message
                setStreamingMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))

                // Only attempt fallback for network errors that are NOT user-initiated cancellations
                // User cancellations (AbortError) should not trigger fallback
                const isUserCancellation = error instanceof Error && error.name === 'AbortError'
                const isNetworkError = error instanceof NetworkError && 
                    (error.message.includes('cancelled') || error.message.includes('aborted'))
                
                if (isNetworkError && !isUserCancellation) {
                  // Clear any existing fallback timeout
                  if (fallbackTimeoutRef.current) {
                    clearTimeout(fallbackTimeoutRef.current)
                  }
                  
                  // Set up tracked fallback timeout
                  fallbackTimeoutRef.current = setTimeout(() => {
                    fallbackTimeoutRef.current = null
                    
                    executeFallbackRequest(
                      messageContent,
                      currentSessionId || '',
                      (response) => {
                        // Add the response as a new message
                        if (response.data.choices && response.data.choices.length > 0) {
                          const assistantResponse = response.data.choices[0].message.content
                          
                          // Skip empty responses
                          if (!assistantResponse || assistantResponse.trim() === '') {
                            addMessage({
                              type: 'assistant',
                              content: "Sorry, I didn't receive a proper response.",
                              timestamp: new Date(),
                            })
                          } else {
                            addMessage({
                              type: 'assistant',
                              content: assistantResponse,
                              timestamp: new Date(),
                            })
                          }
                        } else {
                          addMessage({
                            type: 'assistant',
                            content: "Sorry, I didn't receive a proper response.",
                            timestamp: new Date(),
                          })
                        }
                      },
                      (fallbackError) => {
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
                    )
                  }, 100)
                } else {
                  // For user cancellations or other errors, show error message
                  const errorMessage = isUserCancellation 
                    ? 'Request was cancelled'
                    : (error instanceof NetworkError 
                        ? error.message 
                        : 'Streaming connection failed')
                  
                  if (!isUserCancellation) {
                    setError(errorMessage)
                  }

                  addMessage({
                    type: 'error',
                    content: `Error: ${errorMessage}`,
                    timestamp: new Date(),
                  })
                }
              },
              onComplete: () => {
                clearTimeout(timeoutId)
                setIsStreaming(false)
                
                // If we got content, finalize the message
                if (accumulatedContent && accumulatedContent.trim()) {
                  // Save final message to project session and remove temporary streaming message
                  try {
                    // Add final response to project session (will go to temp messages since streaming happens before session transfer)
                    projectSession.addMessage(accumulatedContent, 'assistant')
                    
                    // Remove the temporary streaming message
                    setStreamingMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))
                    
                    // NOW handle session creation/reconciliation after all messages are added
                    // Use a small delay to ensure the addMessage state update has completed
                    setTimeout(() => {
                      if (deferredSessionId) {
                        try {
                          // Check if we have any existing session
                          const existingSessionId = currentSessionId || projectSession.sessionId
                          
                          if (existingSessionId) {
                            // Check if reconciliation is actually needed
                            if (existingSessionId !== deferredSessionId) {
                              // Session IDs differ, reconciliation needed
                              projectSession.reconcileWithServer(existingSessionId, deferredSessionId)
                            }
                          } else {
                            // Truly no existing session, create new one with all temp messages
                            projectSession.createSessionFromServer(deferredSessionId)
                          }
                        } catch (sessionError) {
                          console.error('Failed to handle deferred session creation:', sessionError)
                          // Don't fail the whole request for session management errors
                        }
                      }
                    }, 10); // Small delay to ensure state updates have completed
                  } catch (err) {
                    console.warn('Failed to save to project session:', err)
                    // Keep the message in streaming state with final content
                    updateMessage(assistantMessageId, {
                      content: accumulatedContent,
                      isStreaming: false,
                      isLoading: false,
                    })
                  }
                } else {
                  // No content received, try non-streaming fallback
                  setStreamingMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))
                  
                  // Handle deferred session even without content
                  if (deferredSessionId) {
                    try {
                      const existingSessionId = currentSessionId || projectSession.sessionId
                      if (!existingSessionId) {
                        projectSession.createSessionFromServer(deferredSessionId)
                      }
                    } catch (sessionError) {
                      console.error('Failed to handle deferred session after streaming failure:', sessionError)
                    }
                  }
                  
                  // Clear any existing fallback timeout
                  if (fallbackTimeoutRef.current) {
                    clearTimeout(fallbackTimeoutRef.current)
                  }
                  
                  // Set up tracked fallback timeout
                  fallbackTimeoutRef.current = setTimeout(() => {
                    fallbackTimeoutRef.current = null
                    
                    executeFallbackRequest(
                      messageContent,
                      currentSessionId || '',
                      (response) => {
                        // Add the response as a new message
                        if (response.data.choices && response.data.choices.length > 0) {
                          const assistantResponse = response.data.choices[0].message.content
                          
                          // Skip empty responses
                          if (!assistantResponse || assistantResponse.trim() === '') {
                            addMessage({
                              type: 'assistant',
                              content: "Sorry, I didn't receive a proper response.",
                              timestamp: new Date(),
                            })
                          } else {
                            addMessage({
                              type: 'assistant',
                              content: assistantResponse,
                              timestamp: new Date(),
                            })
                          }
                        } else {
                          addMessage({
                            type: 'assistant',
                            content: "Sorry, I didn't receive a proper response.",
                            timestamp: new Date(),
                          })
                        }
                      },
                      (fallbackError) => {
                        console.error('Fallback request also failed:', fallbackError)
                        addMessage({
                          type: 'error',
                          content: 'Error: Failed to get response',
                          timestamp: new Date(),
                        })
                      }
                    )
                  }, 100)
                }
              },
              signal: abortController.signal,
            }
          )
          
          // Store session ID for deferred processing after all messages are added
          if (responseSessionId) {
            deferredSessionId = responseSessionId
          }

          // For streaming, we return true immediately as the request is initiated
          // The actual success/failure will be handled by the streaming callbacks
          return true
          
        } else {
          // Non-streaming path
          const response = await chatMutation.mutateAsync({
            chatRequest,
            sessionId: currentSessionId || undefined,
          })
          
          // Handle session reconciliation if we got a session ID from server
          if (response.sessionId) {
            try {
              // Check if we have any existing session (even if currentSessionId is null)
              const existingSessionId = currentSessionId || projectSession.sessionId
              
              if (existingSessionId) {
                // Check if reconciliation is actually needed
                if (existingSessionId !== response.sessionId) {
                  // Session IDs differ, reconciliation needed
                  projectSession.reconcileWithServer(existingSessionId, response.sessionId)
                }
              } else {
                // Truly no existing session, create new one
                projectSession.createSessionFromServer(response.sessionId)
              }
            } catch (sessionError) {
              console.error('Failed to handle session from server response:', sessionError)
              // Don't fail the whole request for session management errors
            }
          }
          
          // Update assistant message with response
          if (response.data.choices && response.data.choices.length > 0) {
            const assistantResponse = response.data.choices[0].message.content
            
            // Skip empty responses
            if (!assistantResponse || assistantResponse.trim() === '') {
              updateMessage(assistantMessageId, {
                content: "Sorry, I didn't receive a proper response.",
                isLoading: false,
              })
            } else {
              // Save final message to project session and remove temporary one
              try {
                projectSession.addMessage(assistantResponse, 'assistant')
                setStreamingMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))
              } catch (err) {
                console.warn('Failed to save to project session:', err)
                updateMessage(assistantMessageId, {
                  content: assistantResponse,
                  isLoading: false,
                })
              }
            }
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
        setStreamingMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))
        
        // Set error message
        const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred'
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
      currentSessionId,
      addMessage,
      updateMessage,
      streamingEnabled,
      isStreaming,
      projectSession,
      executeFallbackRequest
    ]
  )
  
  // Handle clear chat
  const clearChat = useCallback(async () => {
    if (deleteSessionMutation.isPending) return false
    
    try {
      // Use project session system
      projectSession.clearHistory()
      // Also clear any temporary streaming messages
      setStreamingMessages([])
      setError(null)
      return true
    } catch (error) {
      console.error('Clear chat error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to clear chat'
      setError(errorMessage)
      return false
    }
  }, [projectSession])
  
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
      setStreamingMessages(prev => prev.map(msg => 
        msg.isStreaming 
          ? { ...msg, isStreaming: false, content: msg.content + ' [Cancelled]' }
          : msg
      ))
    }
  }, [isStreaming])
  
  // Reset to new session (clear current session - new one will be created on next message)
  const resetSession = useCallback(() => {
    // Cancel any active streaming first
    if (isStreaming) {
      cancelStreaming()
    }
    
    // Clear current session - new one will be created on first message
    setStreamingMessages([])
    setError(null)
    setInputValue('')
    
    // Return empty string since we don't create sessions proactively
    return ''
  }, [isStreaming, cancelStreaming])
  
  const result = {
    // State
    sessionId: currentSessionId,
    messages: currentMessages,
    inputValue,
    error: error || projectSession.error,
    
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
    hasMessages: currentMessages.length > 0,
    canSend: !chatMutation.isPending && !isStreaming && inputValue.trim().length > 0,
  }
  
  return result;
}

export default useChatboxWithProjectSession