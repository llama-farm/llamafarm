import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  sendProjectChatMessage,
  streamProjectChatMessage,
  createProjectChatRequest,
  ProjectChatRequest,
  ProjectChatStreamingOptions,
  ProjectChatMessage,
  ProjectChatResult,
  buildConversationHistory,
} from '../api/projectChatService'

/**
 * Query keys for project chat-related queries
 * Follows the hierarchical pattern used in existing hooks
 */
export const projectChatKeys = {
  all: ['projectChat'] as const,
  sessions: () => [...projectChatKeys.all, 'sessions'] as const,
  session: (sessionId: string) => [...projectChatKeys.sessions(), sessionId] as const,
  conversations: () => [...projectChatKeys.all, 'conversations'] as const,
  conversation: (namespace: string, projectId: string, sessionId?: string) => 
    [...projectChatKeys.conversations(), namespace, projectId, sessionId] as const,
  completions: () => [...projectChatKeys.all, 'completions'] as const,
  completion: (namespace: string, projectId: string) => 
    [...projectChatKeys.completions(), namespace, projectId] as const,
}

/**
 * Parameters for project chat completion
 */
export interface ProjectChatParams {
  namespace: string
  projectId: string
  request: ProjectChatRequest
  sessionId?: string
}

/**
 * Parameters for streaming project chat
 */
export interface ProjectChatStreamParams extends ProjectChatParams {
  options?: ProjectChatStreamingOptions
}

// ProjectChatResult is now imported from the API service

/**
 * Hook to send non-streaming project chat messages
 * @returns Mutation function for sending project chat messages
 */
export const useProjectChatCompletion = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async ({ namespace, projectId, request, sessionId }: ProjectChatParams): Promise<ProjectChatResult> => {
      return await sendProjectChatMessage(namespace, projectId, request, sessionId)
    },
    onSuccess: (data, variables) => {
      // Invalidate conversation queries for this project
      queryClient.invalidateQueries({ 
        queryKey: projectChatKeys.conversation(variables.namespace, variables.projectId) 
      })
      
      // Update completion cache
      queryClient.setQueryData(
        projectChatKeys.completion(variables.namespace, variables.projectId),
        data.completion
      )
    },
    onError: (error) => {
      console.error('Failed to send project chat message:', error)
    }
  })
}

/**
 * Hook to send streaming project chat messages
 * @returns Mutation function for streaming project chat messages
 */
export const useProjectChatStreaming = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async ({ 
      namespace, 
      projectId, 
      request, 
      sessionId, 
      options 
    }: ProjectChatStreamParams): Promise<string> => {
      return await streamProjectChatMessage(namespace, projectId, request, sessionId, options)
    },
    onSuccess: (sessionId, variables) => {
      // Invalidate conversation queries for this project
      queryClient.invalidateQueries({ 
        queryKey: projectChatKeys.conversation(variables.namespace, variables.projectId) 
      })
      
      // Update session cache if we have a session ID
      if (sessionId) {
        queryClient.invalidateQueries({ 
          queryKey: projectChatKeys.session(sessionId) 
        })
      }
    },
    onError: (error) => {
      console.error('Failed to stream project chat message:', error)
    }
  })
}

/**
 * Hook to send a simple text message to project chat
 * Convenience wrapper around useProjectChatCompletion
 * @returns Mutation function for sending simple text messages
 */
export const useProjectChatMessage = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async ({ 
      namespace, 
      projectId, 
      message, 
      sessionId,
      options = {}
    }: {
      namespace: string
      projectId: string
      message: string
      sessionId?: string
      options?: Partial<ProjectChatRequest>
    }): Promise<ProjectChatResult> => {
      const request = createProjectChatRequest(message, options)
      return await sendProjectChatMessage(namespace, projectId, request, sessionId)
    },
    retry: false, // Disable retries to prevent multiple requests
    onSuccess: (data, variables) => {
      // Invalidate conversation queries for this project
      queryClient.invalidateQueries({ 
        queryKey: projectChatKeys.conversation(variables.namespace, variables.projectId) 
      })
      
      // Update completion cache
      queryClient.setQueryData(
        projectChatKeys.completion(variables.namespace, variables.projectId),
        data.completion
      )
    },
    onError: (error) => {
      console.error('Failed to send project chat message:', error)
    }
  })
}

/**
 * Hook to send a streaming text message to project chat
 * Convenience wrapper around useProjectChatStreaming
 * @returns Mutation function for sending streaming text messages
 */
export const useProjectChatStreamingMessage = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async ({ 
      namespace, 
      projectId, 
      message, 
      sessionId,
      requestOptions = {},
      streamingOptions = {}
    }: {
      namespace: string
      projectId: string
      message: string
      sessionId?: string
      requestOptions?: Partial<ProjectChatRequest>
      streamingOptions?: ProjectChatStreamingOptions
    }): Promise<string> => {
      const request = createProjectChatRequest(message, requestOptions)
      return await streamProjectChatMessage(namespace, projectId, request, sessionId, streamingOptions)
    },
    onSuccess: (sessionId, variables) => {
      // Invalidate conversation queries for this project
      queryClient.invalidateQueries({ 
        queryKey: projectChatKeys.conversation(variables.namespace, variables.projectId) 
      })
      
      // Update session cache if we have a session ID
      if (sessionId) {
        queryClient.invalidateQueries({ 
          queryKey: projectChatKeys.session(sessionId) 
        })
      }
    },
    onError: (error) => {
      console.error('Failed to stream project chat message:', error)
    }
  })
}

/**
 * Hook to send a conversation with history to project chat
 * @returns Mutation function for sending conversation with context
 */
export const useProjectChatConversation = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async ({ 
      namespace, 
      projectId, 
      messages, 
      sessionId,
      options = {},
      maxHistoryMessages = 50
    }: {
      namespace: string
      projectId: string
      messages: ProjectChatMessage[]
      sessionId?: string
      options?: Partial<ProjectChatRequest>
      maxHistoryMessages?: number
    }): Promise<ProjectChatResult> => {
      // Build conversation history with message limit
      const conversationHistory = buildConversationHistory(messages, maxHistoryMessages)
      
      const request: ProjectChatRequest = {
        messages: conversationHistory,
        ...options
      }
      
      return await sendProjectChatMessage(namespace, projectId, request, sessionId)
    },
    onSuccess: (data, variables) => {
      // Invalidate conversation queries for this project
      queryClient.invalidateQueries({ 
        queryKey: projectChatKeys.conversation(variables.namespace, variables.projectId) 
      })
      
      // Update completion cache
      queryClient.setQueryData(
        projectChatKeys.completion(variables.namespace, variables.projectId),
        data.completion
      )
    },
    onError: (error) => {
      console.error('Failed to send project chat conversation:', error)
    }
  })
}

/**
 * Hook to send a streaming conversation with history to project chat
 * @returns Mutation function for streaming conversation with context
 */
export const useProjectChatStreamingConversation = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async ({ 
      namespace, 
      projectId, 
      messages, 
      sessionId,
      requestOptions = {},
      streamingOptions = {},
      maxHistoryMessages = 50
    }: {
      namespace: string
      projectId: string
      messages: ProjectChatMessage[]
      sessionId?: string
      requestOptions?: Partial<ProjectChatRequest>
      streamingOptions?: ProjectChatStreamingOptions
      maxHistoryMessages?: number
    }): Promise<string> => {
      // Build conversation history with message limit
      const conversationHistory = buildConversationHistory(messages, maxHistoryMessages)
      
      const request: ProjectChatRequest = {
        messages: conversationHistory,
        ...requestOptions
      }
      
      return await streamProjectChatMessage(namespace, projectId, request, sessionId, streamingOptions)
    },
    onSuccess: (sessionId, variables) => {
      // Invalidate conversation queries for this project
      queryClient.invalidateQueries({ 
        queryKey: projectChatKeys.conversation(variables.namespace, variables.projectId) 
      })
      
      // Update session cache if we have a session ID
      if (sessionId) {
        queryClient.invalidateQueries({ 
          queryKey: projectChatKeys.session(sessionId) 
        })
      }
    },
    onError: (error) => {
      console.error('Failed to stream project chat conversation:', error)
    }
  })
}

/**
 * Hook to get all project chat mutation states
 * Useful for components that need to show loading states across different chat operations
 */
export const useProjectChatMutations = () => {
  const completion = useProjectChatCompletion()
  const streaming = useProjectChatStreaming()
  const message = useProjectChatMessage()
  const streamingMessage = useProjectChatStreamingMessage()
  const conversation = useProjectChatConversation()
  const streamingConversation = useProjectChatStreamingConversation()
  
  return {
    completion,
    streaming,
    message,
    streamingMessage,
    conversation,
    streamingConversation,
    isLoading: completion.isPending || 
               streaming.isPending || 
               message.isPending || 
               streamingMessage.isPending ||
               conversation.isPending ||
               streamingConversation.isPending,
    error: completion.error || 
           streaming.error || 
           message.error || 
           streamingMessage.error ||
           conversation.error ||
           streamingConversation.error,
  }
}

/**
 * Utility hook to extract and validate chat parameters from active project
 * @param activeProject - The active project information
 * @returns Validated namespace and projectId, or null if invalid
 */
export const useProjectChatParams = (activeProject: { namespace: string; project: string } | null) => {
  if (!activeProject?.namespace || !activeProject?.project) {
    return null
  }
  
  return {
    namespace: activeProject.namespace,
    projectId: activeProject.project
  }
}

/**
 * Helper hook to create optimized chat request builders
 * @param namespace - Project namespace
 * @param projectId - Project identifier
 * @returns Object with pre-configured request builders
 */
export const useProjectChatRequestBuilders = (namespace: string, projectId: string) => {
  const chatMessage = useProjectChatMessage()
  const streamingMessage = useProjectChatStreamingMessage()
  const conversation = useProjectChatConversation()
  const streamingConversation = useProjectChatStreamingConversation()
  
  return {
    /**
     * Send a simple text message
     */
    sendMessage: (message: string, sessionId?: string, options?: Partial<ProjectChatRequest>) => 
      chatMessage.mutateAsync({ namespace, projectId, message, sessionId, options }),
    
    /**
     * Send a streaming text message
     */
    sendStreamingMessage: (
      message: string, 
      sessionId?: string, 
      requestOptions?: Partial<ProjectChatRequest>,
      streamingOptions?: ProjectChatStreamingOptions
    ) => streamingMessage.mutateAsync({ 
      namespace, 
      projectId, 
      message, 
      sessionId, 
      requestOptions, 
      streamingOptions 
    }),
    
    /**
     * Send a conversation with history
     */
    sendConversation: (
      messages: ProjectChatMessage[], 
      sessionId?: string, 
      options?: Partial<ProjectChatRequest>,
      maxHistoryMessages?: number
    ) => conversation.mutateAsync({ 
      namespace, 
      projectId, 
      messages, 
      sessionId, 
      options, 
      maxHistoryMessages 
    }),
    
    /**
     * Send a streaming conversation with history
     */
    sendStreamingConversation: (
      messages: ProjectChatMessage[], 
      sessionId?: string, 
      requestOptions?: Partial<ProjectChatRequest>,
      streamingOptions?: ProjectChatStreamingOptions,
      maxHistoryMessages?: number
    ) => streamingConversation.mutateAsync({ 
      namespace, 
      projectId, 
      messages, 
      sessionId, 
      requestOptions, 
      streamingOptions, 
      maxHistoryMessages 
    }),
    
    // Expose loading and error states
    isLoading: chatMessage.isPending || 
               streamingMessage.isPending || 
               conversation.isPending || 
               streamingConversation.isPending,
    error: chatMessage.error || 
           streamingMessage.error || 
           conversation.error || 
           streamingConversation.error,
  }
}

/**
 * Export all hooks and utilities
 */
export default {
  useProjectChatCompletion,
  useProjectChatStreaming,
  useProjectChatMessage,
  useProjectChatStreamingMessage,
  useProjectChatConversation,
  useProjectChatStreamingConversation,
  useProjectChatMutations,
  useProjectChatParams,
  useProjectChatRequestBuilders,
  projectChatKeys,
}
