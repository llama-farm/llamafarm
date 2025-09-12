/**
 * Streaming chat hook for real-time chat with project integration
 * Handles streaming responses, project validation, and session management
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  ChatRequest,
  ChatSessionContext,
  StreamControl,
} from '../types/chat'
import { ActiveProject } from './useActiveProject'
import { 
  startChatStream, 
  sendChatRequest, 
  buildChatAPIURL,
  createUserFriendlyErrorMessage 
} from '../utils/streamingApi'

interface StreamingChatState {
  isStreaming: boolean
  streamingContent: string
  error: string | null
  abortController: AbortController | null
}

interface StreamingChatOptions {
  activeProject: ActiveProject | null
  sessionId?: string
  serverURL?: string
  temperature?: number
  maxTokens?: number
  onChunk?: (chunk: string) => void
  onComplete?: (fullResponse: string, sessionId?: string) => void
  onError?: (error: Error) => void
}

/**
 * Hook for managing streaming chat with project integration
 */
export function useStreamingChat(options: StreamingChatOptions) {
  const {
    activeProject,
    sessionId,
    serverURL = '/api', // Default to relative API path
    temperature,
    maxTokens,
    onChunk,
    onComplete,
    onError,
  } = options

  const [state, setState] = useState<StreamingChatState>({
    isStreaming: false,
    streamingContent: '',
    error: null,
    abortController: null,
  })

  const streamControlRef = useRef<StreamControl | null>(null)
  const accumulatedContentRef = useRef<string>('')

  // Reset state when project changes and cleanup AbortController
  useEffect(() => {
    setState(prev => {
      // Cleanup existing abortController if it exists
      if (prev.abortController && !prev.abortController.signal.aborted) {
        prev.abortController.abort()
      }
      
      return {
        ...prev,
        error: null,
        streamingContent: '',
        isStreaming: false,
        abortController: null, // Clear the reference
      }
    })
    accumulatedContentRef.current = ''
    streamControlRef.current = null // Clear stream control reference
  }, [activeProject?.namespace, activeProject?.project])

  // Build session context from current state
  const buildSessionContext = useCallback((): ChatSessionContext | null => {
    if (!activeProject) {
      return null
    }

    return {
      serverURL,
      namespace: activeProject.namespace,
      projectID: activeProject.project,
      sessionID: sessionId,
      temperature,
      maxTokens,
      streaming: true,
    }
  }, [activeProject, sessionId, serverURL, temperature, maxTokens])

  // Streaming mutation
  const streamingMutation = useMutation({
    mutationFn: async ({ request, context }: { request: ChatRequest; context: ChatSessionContext }) => {
      // Reset state
      setState(prev => ({
        ...prev,
        isStreaming: true,
        streamingContent: '',
        error: null,
      }))
      accumulatedContentRef.current = ''

      try {
        const { chunks, control, getSessionId } = await startChatStream(request, context)
        streamControlRef.current = control

        const reader = chunks.getReader()
        
        while (true) {
          const { done, value } = await reader.read()
          
          if (done) break
          
          // Accumulate content
          accumulatedContentRef.current += value
          
          // Update streaming state
          setState(prev => ({
            ...prev,
            streamingContent: accumulatedContentRef.current,
          }))
          
          // Call chunk callback
          onChunk?.(value)
        }
        
        // Stream complete - cleanup and update state
        const finalResponse = accumulatedContentRef.current
        const finalSessionId = getSessionId()
        
        setState(prev => ({
          ...prev,
          isStreaming: false,
          abortController: null, // Clear abortController reference
        }))
        
        onComplete?.(finalResponse, finalSessionId)
        
        return {
          response: finalResponse,
          sessionId: finalSessionId,
        }
        
      } catch (error) {
        const friendlyMessage = createUserFriendlyErrorMessage(error)
        
        setState(prev => ({
          ...prev,
          isStreaming: false,
          error: friendlyMessage,
          abortController: null, // Clear abortController reference on error
        }))
        
        const errorObj = error instanceof Error ? error : new Error(friendlyMessage)
        onError?.(errorObj)
        throw errorObj
      } finally {
        // Always clear references in finally block
        streamControlRef.current = null
      }
    },
    retry: false, // Don't retry streaming requests
  })

  // Non-streaming fallback mutation
  const fallbackMutation = useMutation({
    mutationFn: async ({ request, context }: { request: ChatRequest; context: ChatSessionContext }) => {
      const result = await sendChatRequest(request, context)
      onComplete?.(result.response, result.sessionId)
      return result
    },
    retry: 2, // Allow retries for fallback
  })

  // Send streaming message
  const sendStreamingMessage = useCallback(
    async (request: ChatRequest): Promise<{ response: string; sessionId?: string } | null> => {
      const context = buildSessionContext()
      
      if (!context) {
        const error = new Error('No active project selected. Please select a project to start chatting.')
        setState(prev => ({
          ...prev,
          error: error.message,
        }))
        onError?.(error)
        return null
      }

      try {
        return await streamingMutation.mutateAsync({ request, context })
      } catch (streamingError) {
        console.warn('Streaming failed, attempting fallback:', streamingError)
        
        // Try fallback to non-streaming
        try {
          return await fallbackMutation.mutateAsync({ request, context })
        } catch (fallbackError) {
          throw fallbackError
        }
      }
    },
    [buildSessionContext, streamingMutation, fallbackMutation, onError]
  )

  // Abort current stream with proper cleanup
  const abortStream = useCallback(() => {
    if (streamControlRef.current && streamControlRef.current.isActive) {
      streamControlRef.current.abort()
    }
    
    setState(prev => {
      // Also abort the abortController if it exists
      if (prev.abortController && !prev.abortController.signal.aborted) {
        prev.abortController.abort()
      }
      
      return {
        ...prev,
        isStreaming: false,
        error: 'Stream was cancelled',
        abortController: null, // Clear abortController reference
      }
    })
    
    // Clear stream control reference
    streamControlRef.current = null
  }, [])

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({
      ...prev,
      error: null,
    }))
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Cleanup any ongoing streams and abort controllers on unmount
      if (streamControlRef.current && streamControlRef.current.isActive) {
        streamControlRef.current.abort()
      }
      
      setState(prev => {
        if (prev.abortController && !prev.abortController.signal.aborted) {
          prev.abortController.abort()
        }
        return {
          ...prev,
          isStreaming: false,
          abortController: null,
        }
      })
    }
  }, [])

  // Check if project is available
  const hasActiveProject = !!activeProject

  // Get current API URL for debugging
  const getCurrentAPIURL = useCallback(() => {
    const context = buildSessionContext()
    return context ? buildChatAPIURL(context) : null
  }, [buildSessionContext])

  return {
    // State
    isStreaming: state.isStreaming,
    streamingContent: state.streamingContent,
    error: state.error,
    hasActiveProject,
    
    // Actions
    sendStreamingMessage,
    abortStream,
    clearError,
    
    // Computed values
    canSend: hasActiveProject && !state.isStreaming,
    isLoading: streamingMutation.isPending || fallbackMutation.isPending,
    
    // Debug utilities
    getCurrentAPIURL,
    currentProject: activeProject,
  }
}

export default useStreamingChat
