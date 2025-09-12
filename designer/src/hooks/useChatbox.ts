import { useState, useCallback, useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useDeleteChatSession, chatKeys } from './useChat'
import { generateMessageId } from '../utils/idGenerator'
import useChatSession from './useChatSession'
import useActiveProject from './useActiveProject'
import useStreamingChat from './useStreamingChat'
import { ChatboxMessage } from '../types/chatbox'
import { ChatMessage } from '../types/chat'
import { createUserFriendlyErrorMessage } from '../utils/streamingApi'

/**
 * Custom hook for managing chatbox state and API interactions
 * Now includes streaming support, project integration, and enhanced session management
 */
export function useChatbox(initialSessionId?: string) {
  // Active project integration
  const activeProject = useActiveProject()
  
  // Session management with persistence (project-aware)
  const {
    currentSessionId: sessionId,
    messages: persistedMessages,
    saveSessionMessages,
    createNewSession,
    setSessionId,
    isLoading: isLoadingSession
  } = useChatSession(initialSessionId, activeProject?.namespace, activeProject?.project)
  
  // Local state
  const [messages, setMessages] = useState<ChatboxMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [hasInitialSync, setHasInitialSync] = useState(false)
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  
  // Ref for debounced save timeout
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  
  // API hooks
  const queryClient = useQueryClient()
  const deleteSessionMutation = useDeleteChatSession()
  
  // Streaming chat integration
  const streamingChat = useStreamingChat({
    activeProject,
    sessionId,
    onChunk: useCallback((chunk: string) => {
      // Update streaming message content in real-time
      if (streamingMessageId) {
        updateMessage(streamingMessageId, (prev) => ({
          ...prev,
          content: (prev.content === 'Thinking...' ? '' : prev.content) + chunk,
        }))
      }
    }, [streamingMessageId]),
    onComplete: useCallback((fullResponse: string, responseSessionId?: string) => {
      // Finalize streaming message
      if (streamingMessageId) {
        updateMessage(streamingMessageId, {
          content: fullResponse,
          isLoading: false,
        })
        setStreamingMessageId(null)
      }
      
      // Update session ID if received from server
      if (responseSessionId && responseSessionId !== sessionId) {
        setSessionId(responseSessionId)
        if (!hasInitialSync) {
          setHasInitialSync(true)
        }
      }
    }, [streamingMessageId, sessionId, setSessionId, hasInitialSync]),
    onError: useCallback((streamError: Error) => {
      console.error('Streaming error:', streamError)
      
      // Remove streaming message on error
      if (streamingMessageId) {
        setMessages(prev => prev.filter(msg => msg.id !== streamingMessageId))
        setStreamingMessageId(null)
      }
      
      // Set error state with friendly message
      const friendlyMessage = createUserFriendlyErrorMessage(streamError)
      setError(friendlyMessage)
      
      // Add error message to chat
      addMessage({
        type: 'error',
        content: `Error: ${friendlyMessage}`,
        timestamp: new Date()
      })
    }, [streamingMessageId])
  })
  
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

  // Reset chat state when active project changes
  const activeProjectKey = activeProject ? `${activeProject.namespace}/${activeProject.project}` : null
  const prevActiveProjectRef = useRef<string | null>(null)
  
  useEffect(() => {

    
    // Check if project has changed (not initial load)
    if (prevActiveProjectRef.current && prevActiveProjectRef.current !== activeProjectKey) {
      
      
      // Project has changed - NUCLEAR RESET of all chat state
      setMessages([])
      setError(null)
      setInputValue('')
      setHasInitialSync(false)
      setStreamingMessageId(null)
      
      // Abort any ongoing streaming
      if (streamingChat.isStreaming) {
        streamingChat.abortStream()
      }
      
    }
    
    // Update the previous project reference
    prevActiveProjectRef.current = activeProjectKey
  }, [activeProjectKey, streamingChat, sessionId, messages.length, hasInitialSync])

  // Sync persisted messages with local state on initial load and after project changes
  useEffect(() => {
    
    // Load messages when:
    // 1. This is the first time we're syncing for this session (!hasInitialSync)
    // 2. OR when we have no local messages but have persisted messages (after project switch)
    if (!hasInitialSync || (messages.length === 0 && persistedMessages.length > 0)) {
      
      setMessages(persistedMessages)
      setHasInitialSync(true)
    }
  }, [persistedMessages, hasInitialSync, messages.length, sessionId, activeProjectKey])
  
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
  const updateMessage = useCallback((id: string, updates: Partial<ChatboxMessage> | ((prev: ChatboxMessage) => Partial<ChatboxMessage>)) => {
    setMessages(prev => {
      const updated = prev.map(msg => {
        if (msg.id === id) {
          const newUpdates = typeof updates === 'function' ? updates(msg) : updates
          return { ...msg, ...newUpdates }
        }
        return msg
      })
      return updated
    })
  }, [])

  // Handle sending message with streaming API integration
  const sendMessage = useCallback(async (messageContent: string) => {
    if (!messageContent.trim()) return false

    // Check if we have an active project
    if (!activeProject) {
      const errorMessage = 'Please select a project to start chatting'
      setError(errorMessage)
      
      addMessage({
        type: 'error',
        content: errorMessage,
        timestamp: new Date()
      })
      return false
    }

    // Check if already streaming
    if (streamingChat.isStreaming || streamingMessageId) {
      return false
    }

    // Clear any previous errors
    setError(null)

    // Add user message immediately (optimistic update)
    addMessage({
      type: 'user',
      content: messageContent,
      timestamp: new Date()
    })

    // Add loading assistant message for streaming
    const assistantMessageId = addMessage({
      type: 'assistant',
      content: 'Thinking...',
      timestamp: new Date(),
      isLoading: true
    })
    
    setStreamingMessageId(assistantMessageId)

    try {
      // Convert message to chat format
      const chatMessages: ChatMessage[] = [
        { role: 'user', content: messageContent }
      ]
      
      // Create streaming request
      const request = {
        messages: chatMessages,
        stream: true
      }

      // Send streaming request
      await streamingChat.sendStreamingMessage(request)

      return true
    } catch (error) {
      console.error('Chat error:', error)

      // Remove loading message
      setMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))
      setStreamingMessageId(null)

      // Set error message with friendly formatting
      const friendlyMessage = createUserFriendlyErrorMessage(error)
      setError(friendlyMessage)

      // Add error message to chat
      addMessage({
        type: 'error',
        content: `Error: ${friendlyMessage}`,
        timestamp: new Date()
      })

      return false
    }
  }, [activeProject, streamingChat, streamingMessageId, addMessage])

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
      const friendlyMessage = createUserFriendlyErrorMessage(error)
      setError(friendlyMessage)
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

  // Debug utility - expose to window for debugging
  useEffect(() => {
    if (typeof window !== 'undefined') {
      (window as any).debugChatState = () => {
        const state = {
          activeProject,
          sessionId,
          messagesCount: messages.length,
          hasInitialSync,
          error,
          localStorage: {} as Record<string, string | null>
        }
        
        // Collect all localStorage keys that look like session keys
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i)
          if (key && (key.startsWith('session_') || key.startsWith('chatbox_'))) {
            state.localStorage[key] = localStorage.getItem(key)
          }
        }
        return state
      }
    }
  }, [activeProject, sessionId, messages.length, hasInitialSync, error])

  // Abort current streaming
  const abortStreaming = useCallback(() => {
    if (streamingChat.isStreaming) {
      streamingChat.abortStream()
      
      // Clean up streaming state
      if (streamingMessageId) {
        setMessages(prev => prev.filter(msg => msg.id !== streamingMessageId))
        setStreamingMessageId(null)
      }
    }
  }, [streamingChat, streamingMessageId])

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
    activeProject,
    
    // Loading/streaming states
    isSending: streamingChat.isStreaming || streamingChat.isLoading,
    isClearing: deleteSessionMutation.isPending,
    isLoadingSession,
    isStreaming: streamingChat.isStreaming,
    
    // Actions
    sendMessage,
    clearChat,
    updateInput,
    clearError,
    resetSession,
    addMessage,
    updateMessage,
    abortStreaming,
    
    // Computed values
    hasMessages: messages.length > 0,
    canSend: streamingChat.canSend && inputValue.trim().length > 0,
    hasActiveProject: !!activeProject,
    projectInfo: activeProject ? `${activeProject.namespace}/${activeProject.project}` : null,
  }
}

export default useChatbox
