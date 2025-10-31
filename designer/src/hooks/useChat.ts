/**
 * Legacy chat hooks - minimal functions kept for compatibility
 * All new code should use useChatCompletions.ts instead
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { deleteChatSession } from '../api/chatService'
import { DeleteSessionResponse } from '../types/chat'

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
 * Mutation hook for deleting chat sessions
 * Handles success/error callbacks and cache invalidation
 * @deprecated Consider using project-scoped session management instead
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

export default {
  useDeleteChatSession,
  chatKeys,
}
