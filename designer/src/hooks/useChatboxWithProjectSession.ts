/**
 * Enhanced Chatbox Hook with Project Session Management
 * 
 * Integrates with the project session manager for Designer Chat service
 * Maintains backward compatibility while adding project context session management
 */

import { useState, useCallback, useEffect, useRef } from 'react'
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
  
  // API hooks
  const chatMutation = useChatInference()
  const deleteSessionMutation = useDeleteChatSession()
  
  // Get current state from project session system (always used)
  const currentSessionId = projectSession.sessionId
  const projectSessionMessages = projectSession.messages.map(projectSessionToChatboxMessage)
  const isLoadingSession = projectSession.isLoading
  
  // Add message helper
  const addMessage = useCallback((message: Omit<ChatboxMessage, 'id'>) => {
    const newMessage: ChatboxMessage = {
      ...message,
      id: generateMessageId(),
    }
    
    // Always use project session system
    try {
      projectSession.addMessage(message.content, message.type === 'user' ? 'user' : 'assistant')
    } catch (err) {
      console.warn('Failed to add message to project session:', err)
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
  const currentMessages = [...projectSessionMessages, ...streamingMessages]
  
  // Cleanup timeout and abort streaming on unmount
  useEffect(() => {
    return () => {
      if (streamingAbortControllerRef.current) {
        streamingAbortControllerRef.current.abort()
      }
    }
  }, [])
  
  // Handle sending message with streaming or non-streaming API integration
  const sendMessage = useCallback(
    async (messageContent: string) => {
      if (!messageContent.trim() || chatMutation.isPending || isStreaming) return false
      
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
          
          // Set a timeout for streaming requests
          timeoutId = setTimeout(() => {
            console.log('Streaming request timed out after 1 minute')
            abortController.abort()
          }, 60000)
          
          let accumulatedContent = ''
          
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
                
                // For abort errors, try fallback to non-streaming
                if (error instanceof NetworkError && 
                    (error.message.includes('cancelled') || error.message.includes('aborted'))) {
                  // Silently fall back to non-streaming mode
                  setTimeout(async () => {
                    try {
                      const response = await chatMutation.mutateAsync({
                        chatRequest,
                        sessionId: currentSessionId || undefined,
                      })
                      
                      // Add the response
                      if (response.data.choices && response.data.choices.length > 0) {
                        const assistantResponse = response.data.choices[0].message.content
                        addMessage({
                          type: 'assistant',
                          content: assistantResponse,
                          timestamp: new Date(),
                        })
                      }
                    } catch (fallbackError) {
                      console.error('Fallback request failed:', fallbackError)
                      setError('Failed to get response')
                      addMessage({
                        type: 'error',
                        content: 'Error: Failed to get response',
                        timestamp: new Date(),
                      })
                    }
                  }, 100)
                } else {
                  // Show error message
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
                clearTimeout(timeoutId)
                setIsStreaming(false)
                
                // If we got content, finalize the message
                if (accumulatedContent && accumulatedContent.trim()) {
                  // Save final message to project session and remove temporary streaming message
                  try {
                    projectSession.addMessage(accumulatedContent, 'assistant')
                    // Remove the temporary streaming message
                    setStreamingMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))
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
                  
                  setTimeout(async () => {
                    try {
                      const response = await chatMutation.mutateAsync({
                        chatRequest,
                        sessionId: currentSessionId || undefined,
                      })
                      
                      if (response.data.choices && response.data.choices.length > 0) {
                        const assistantResponse = response.data.choices[0].message.content
                        addMessage({
                          type: 'assistant',
                          content: assistantResponse,
                          timestamp: new Date(),
                        })
                      }
                    } catch (fallbackError) {
                      console.error('Fallback request failed:', fallbackError)
                      addMessage({
                        type: 'error',
                        content: 'Error: Failed to get response',
                        timestamp: new Date(),
                      })
                    }
                  }, 100)
                }
              },
              signal: abortController.signal,
            }
          )
          
          // Handle session creation if we got a new session ID from server
          if (responseSessionId && !currentSessionId) {
            try {
              projectSession.createSessionFromServer(responseSessionId)
            } catch (sessionError) {
              console.error('Failed to create session from server response:', sessionError)
              // Don't fail the whole request for session management errors
            }
          }
          
          return true
          
        } else {
          // Non-streaming path
          const response = await chatMutation.mutateAsync({
            chatRequest,
            sessionId: currentSessionId || undefined,
          })
          
          // Handle session creation if we got a new session ID from server
          if (response.sessionId && !currentSessionId) {
            try {
              projectSession.createSessionFromServer(response.sessionId)
            } catch (sessionError) {
              console.error('Failed to create session from server response:', sessionError)
              // Don't fail the whole request for session management errors
            }
          }
          
          // Update assistant message with response
          if (response.data.choices && response.data.choices.length > 0) {
            const assistantResponse = response.data.choices[0].message.content
            
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
  
  return {
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
}

export default useChatboxWithProjectSession