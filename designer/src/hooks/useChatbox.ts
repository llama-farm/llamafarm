import { useState, useCallback, useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useChatInference, useDeleteChatSession, chatKeys } from './useChat'
import { createChatRequest } from '../api/chatService'
import { generateMessageId } from '../utils/idGenerator'
import useChatSession from './useChatSession'
import { ChatboxMessage } from '../types/chatbox'

/**
 * Custom hook for managing chatbox state and API interactions
 * Extracts chat logic from the Chatbox component for better reusability and testability
 * Now includes session persistence and restoration
 */
export function useChatbox(initialSessionId?: string) {
  // Session management with persistence
  const {
    currentSessionId: sessionId,
    messages: persistedMessages,
    saveSessionMessages,
    createNewSession,
    setSessionId,
    isLoading: isLoadingSession
  } = useChatSession(initialSessionId)
  
  // Local state
  const [messages, setMessages] = useState<ChatboxMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [hasInitialSync, setHasInitialSync] = useState(false)
  
  // Ref for debounced save timeout
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  
  // API hooks
  const queryClient = useQueryClient()
  const chatMutation = useChatInference()
  const deleteSessionMutation = useDeleteChatSession()
  
  // Debounced save function to avoid blocking on every message change
  const debouncedSave = useCallback((sessionId: string, messages: ChatboxMessage[]) => {
    // Clear existing timeout
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current)
    }
    
    // Set new timeout for debounced save
    saveTimeoutRef.current = setTimeout(() => {
      saveSessionMessages(sessionId, messages)
    }, 500) // 500ms delay
  }, [saveSessionMessages])

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

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current)
      }
    }
  }, [])
  
  // Add message to state
  const addMessage = useCallback((message: Omit<ChatboxMessage, 'id'>) => {
    const newMessage: ChatboxMessage = {
      ...message,
      id: generateMessageId()
    }
    
    setMessages(prev => [...prev, newMessage])
    return newMessage.id
  }, [])

  // Update message by ID
  const updateMessage = useCallback((id: string, updates: Partial<ChatboxMessage>) => {
    setMessages(prev => {
      const updated = prev.map(msg => 
        msg.id === id ? { ...msg, ...updates } : msg
      )
      return updated
    })
  }, [])

  // Handle sending message with API integration
  const sendMessage = useCallback(async (messageContent: string) => {
    if (!messageContent.trim() || chatMutation.isPending) return false

    // Clear any previous errors
    setError(null)

    // Add user message immediately (optimistic update)
    addMessage({
      type: 'user',
      content: messageContent,
      timestamp: new Date()
    })

    // Add loading assistant message
    const assistantMessageId = addMessage({
      type: 'assistant',
      content: 'Thinking...',
      timestamp: new Date(),
      isLoading: true
    })

    try {
      // Create chat request
      const chatRequest = createChatRequest(messageContent)

      // Send to API
      const response = await chatMutation.mutateAsync({
        chatRequest,
        sessionId
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
          isLoading: false
        })
      } else {
        updateMessage(assistantMessageId, {
          content: 'Sorry, I didn\'t receive a proper response.',
          isLoading: false
        })
      }

      return true
    } catch (error) {
      console.error('Chat error:', error)

      // Remove loading message
      setMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))

      // Set error message
      const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred'
      setError(errorMessage)

      // Add error message to chat
      addMessage({
        type: 'error',
        content: `Error: ${errorMessage}`,
        timestamp: new Date()
      })

      return false
    }
  }, [chatMutation, sessionId, setSessionId, addMessage, updateMessage])

  // Handle clear chat
  const clearChat = useCallback(async () => {
    if (deleteSessionMutation.isPending) return false

    try {
      await deleteSessionMutation.mutateAsync(sessionId)

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
      const errorMessage = error instanceof Error ? error.message : 'Failed to clear chat'
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

  // Reset to new session
  const resetSession = useCallback(() => {
    const newSessionId = createNewSession()
    setMessages([])
    setError(null)
    setInputValue('')
    
    // Reset initial sync flag to allow fresh sync with new session
    setHasInitialSync(false)
    
    return newSessionId
  }, [createNewSession])

  return {
    // State
    sessionId,
    messages,
    inputValue,
    error,
    
    // Loading states
    isSending: chatMutation.isPending,
    isClearing: deleteSessionMutation.isPending,
    isLoadingSession,
    
    // Actions
    sendMessage,
    clearChat,
    updateInput,
    clearError,
    resetSession,
    addMessage,
    updateMessage,
    
    // Computed values
    hasMessages: messages.length > 0,
    canSend: !chatMutation.isPending && inputValue.trim().length > 0,
  }
}

export default useChatbox
