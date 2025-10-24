import { useMutation, useQueryClient } from '@tanstack/react-query'
import type { UseMutationResult } from '@tanstack/react-query'
import {
  chatInference,
  deleteChatSession,
  chatProject,
  deleteProjectChatSession,
} from '../api/chatService'
import { ChatRequest, ChatResponse, DeleteSessionResponse } from '../types/chat'

// Query key factory for chat operations
export const chatKeys = {
  all: ['chat'] as const,
  sessions: () => [...chatKeys.all, 'sessions'] as const,
  session: (sessionId: string) => [...chatKeys.sessions(), sessionId] as const,
  messages: () => [...chatKeys.all, 'messages'] as const,
  messageHistory: (sessionId: string) =>
    [...chatKeys.messages(), sessionId] as const,
}

/**
 * Mutation hook for sending chat messages
 * Handles loading, error, and success states with proper cache management
 */
export function useChatInference() {
  const queryClient = useQueryClient()

  return useMutation<
    { data: ChatResponse; sessionId: string },
    Error,
    { chatRequest: ChatRequest; sessionId?: string },
    { sessionId?: string }
  >({
    mutationFn: ({ chatRequest, sessionId }) =>
      chatInference(chatRequest, sessionId),

    // Store context for potential rollback
    onMutate: async ({ sessionId }) => {
      if (sessionId) {
        // Cancel any outgoing queries for this session
        await queryClient.cancelQueries({
          queryKey: chatKeys.session(sessionId),
        })
      }
      return { sessionId }
    },

    onSuccess: (response, variables) => {
      const { data, sessionId: responseSessionId } = response
      const { sessionId: requestSessionId } = variables

      // Use the session ID from the response (server-provided) or the request
      const finalSessionId = responseSessionId || requestSessionId

      if (finalSessionId) {
        // Update session cache with new message
        queryClient.setQueryData(
          chatKeys.session(finalSessionId),
          (oldData: ChatResponse[] | undefined) => {
            return oldData ? [...oldData, data] : [data]
          }
        )

        // Invalidate related queries
        queryClient.invalidateQueries({
          queryKey: chatKeys.messageHistory(finalSessionId),
        })
      }
    },

    onError: (_error, _variables, context) => {
      if (context?.sessionId) {
        // Invalidate potentially stale cache
        queryClient.invalidateQueries({
          queryKey: chatKeys.session(context.sessionId),
        })
      }
    },

    // Configure retry logic for chat requests
    retry: (failureCount, error) => {
      // Don't retry validation errors
      if (error.name === 'ValidationError') {
        return false
      }

      // Don't retry client errors (4xx)
      if (
        error.name === 'ChatApiError' &&
        (error as any).status >= 400 &&
        (error as any).status < 500
      ) {
        return false
      }

      // Retry network errors and server errors up to 2 times
      return failureCount < 2
    },

    retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 10000),
  })
}

/**
 * Mutation hook for deleting chat sessions
 * Handles success/error callbacks and cache invalidation
 */
export function useDeleteChatSession() {
  const queryClient = useQueryClient()

  return useMutation<
    DeleteSessionResponse,
    Error,
    string, // sessionId
    { sessionId: string }
  >({
    mutationFn: (sessionId: string) => deleteChatSession(sessionId),

    onMutate: async sessionId => {
      // Cancel any outgoing queries for this session
      await queryClient.cancelQueries({
        queryKey: chatKeys.session(sessionId),
      })

      return { sessionId }
    },

    onSuccess: (_data, sessionId) => {
      // Remove session data from cache
      queryClient.removeQueries({
        queryKey: chatKeys.session(sessionId),
      })

      queryClient.removeQueries({
        queryKey: chatKeys.messageHistory(sessionId),
      })

      // Invalidate sessions list if we're tracking it
      queryClient.invalidateQueries({
        queryKey: chatKeys.sessions(),
      })
    },

    onError: (error, sessionId) => {
      console.error(`Failed to delete session ${sessionId}:`, error)

      // Refresh session data in case it still exists
      queryClient.invalidateQueries({
        queryKey: chatKeys.session(sessionId),
      })
    },

    // Don't retry deletion operations by default
    retry: false,
  })
}

/**
 * Project-scoped chat mutation
 */
export function useProjectChat(namespace: string, project: string) {
  const queryClient = useQueryClient()
  return useMutation<
    { data: ChatResponse; sessionId: string },
    Error,
    { chatRequest: ChatRequest; sessionId?: string },
    { sessionId?: string }
  >({
    mutationFn: ({ chatRequest, sessionId }) =>
      chatProject(namespace, project, chatRequest, sessionId),
    onMutate: async ({ sessionId }) => {
      if (sessionId) {
        await queryClient.cancelQueries({
          queryKey: chatKeys.session(sessionId),
        })
      }
      return { sessionId }
    },
    onSuccess: (response, variables) => {
      const { data, sessionId: responseSessionId } = response
      const { sessionId: requestSessionId } = variables
      const finalSessionId = responseSessionId || requestSessionId
      if (finalSessionId) {
        queryClient.setQueryData(
          chatKeys.session(finalSessionId),
          (oldData: ChatResponse[] | undefined) => {
            return oldData ? [...oldData, data] : [data]
          }
        )
        queryClient.invalidateQueries({
          queryKey: chatKeys.messageHistory(finalSessionId),
        })
      }
    },
    onError: (_error, _variables, context) => {
      if (context?.sessionId) {
        queryClient.invalidateQueries({
          queryKey: chatKeys.session(context.sessionId),
        })
      }
    },
    retry: (failureCount, error) => {
      if (error.name === 'ValidationError') return false
      if (
        error.name === 'ChatApiError' &&
        (error as any).status >= 400 &&
        (error as any).status < 500
      )
        return false
      return failureCount < 2
    },
    retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 10000),
  })
}

// Export a stable alias for the mutation type used by designer chat hook
export type ProjectChatMutation = UseMutationResult<
  { data: ChatResponse; sessionId: string },
  Error,
  { chatRequest: ChatRequest; sessionId?: string },
  { sessionId?: string }
>

/**
 * Project-scoped delete session mutation
 */
export function useDeleteProjectChatSession(
  namespace: string,
  project: string
) {
  const queryClient = useQueryClient()
  return useMutation<
    DeleteSessionResponse,
    Error,
    string,
    { sessionId: string }
  >({
    mutationFn: (sessionId: string) =>
      deleteProjectChatSession(namespace, project, sessionId),
    onMutate: async sessionId => {
      await queryClient.cancelQueries({ queryKey: chatKeys.session(sessionId) })
      return { sessionId }
    },
    onSuccess: (_data, sessionId) => {
      queryClient.removeQueries({ queryKey: chatKeys.session(sessionId) })
      queryClient.removeQueries({
        queryKey: chatKeys.messageHistory(sessionId),
      })
      queryClient.invalidateQueries({ queryKey: chatKeys.sessions() })
    },
    onError: (_error, sessionId) => {
      queryClient.invalidateQueries({ queryKey: chatKeys.session(sessionId) })
    },
    retry: false,
  })
}

export default {
  useChatInference,
  useDeleteChatSession,
  chatKeys,
}
