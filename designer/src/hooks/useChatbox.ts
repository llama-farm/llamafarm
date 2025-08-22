import { useState, useCallback, useEffect } from 'react'
import { useChatInference, useDeleteChatSession } from './useChat'
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
    isLoading: isLoadingSession
  } = useChatSession(initialSessionId)
  
  // Local state
  const [messages, setMessages] = useState<ChatboxMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  
  // API hooks
  const chatMutation = useChatInference()
  const deleteSessionMutation = useDeleteChatSession()
  
  // Sync persisted messages with local state ONLY on initial load
  useEffect(() => {
    // Only sync from persistence if we don't have any current messages
    // This prevents overwriting live updates
    if (messages.length === 0) {
      if (persistedMessages && persistedMessages.length > 0) {
        setMessages(persistedMessages)
      }
    }
  }, [persistedMessages, isLoadingSession])
  
  // Save messages to persistence when they change
  useEffect(() => {
    if (messages.length > 0) {
      saveSessionMessages(sessionId, messages)
    }
  }, [messages, sessionId, saveSessionMessages])
  
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
    setMessages(prev => prev.map(msg => 
      msg.id === id ? { ...msg, ...updates } : msg
    ))
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

      // Update assistant message with response
      if (response.choices && response.choices.length > 0) {
        const assistantResponse = response.choices[0].message.content
        
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
  }, [chatMutation, sessionId, addMessage, updateMessage])

  // Handle clear chat
  const clearChat = useCallback(async () => {
    if (deleteSessionMutation.isPending) return false

    try {
      await deleteSessionMutation.mutateAsync(sessionId)

      // Clear local messages and errors
      setMessages([])
      setError(null)

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
