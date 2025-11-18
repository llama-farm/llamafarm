/**
 * Unified Chat Completions Hook
 *
 * Provides a single interface for all chat completion operations.
 * Supports both streaming and non-streaming modes with session management.
 *
 * Usage:
 * - Project Chat: Pass current project's namespace and project ID
 * - Dev Chat: Pass "llamafarm" namespace and "project_seed" project ID
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  sendChatCompletion,
  streamChatCompletion,
  createChatCompletionRequest,
  ChatCompletionResult,
} from '../api/chatCompletionsService'
import { ChatRequest, StreamingChatOptions } from '../types/chat'

/**
 * Query keys for chat completions
 */
export const chatCompletionsKeys = {
  all: ['chatCompletions'] as const,
  completions: (namespace: string, projectId: string) =>
    [...chatCompletionsKeys.all, namespace, projectId] as const,
  session: (namespace: string, projectId: string, sessionId: string) =>
    [
      ...chatCompletionsKeys.completions(namespace, projectId),
      'session',
      sessionId,
    ] as const,
}

/**
 * Parameters for non-streaming chat completion
 */
export interface ChatCompletionParams {
  namespace: string
  projectId: string
  request: ChatRequest
  sessionId?: string
}

/**
 * Parameters for streaming chat completion
 */
export interface StreamingChatCompletionParams extends ChatCompletionParams {
  options?: StreamingChatOptions
}

/**
 * Hook for non-streaming chat completions
 *
 * @example
 * ```tsx
 * const completion = useChatCompletion()
 *
 * const result = await completion.mutateAsync({
 *   namespace: 'llamafarm',
 *   projectId: 'project_seed',
 *   request: { messages: [{ role: 'user', content: 'Hello' }] },
 *   sessionId: 'existing-session-id'
 * })
 * ```
 */
export function useChatCompletion() {
  const queryClient = useQueryClient()

  return useMutation<ChatCompletionResult, Error, ChatCompletionParams>({
    mutationFn: async ({ namespace, projectId, request, sessionId }) => {
      return await sendChatCompletion(namespace, projectId, request, sessionId)
    },
    onSuccess: (data, variables) => {
      // Invalidate completion queries for this project
      queryClient.invalidateQueries({
        queryKey: chatCompletionsKeys.completions(
          variables.namespace,
          variables.projectId
        ),
      })

      // Update session cache if we have a session ID
      if (data.sessionId) {
        queryClient.invalidateQueries({
          queryKey: chatCompletionsKeys.session(
            variables.namespace,
            variables.projectId,
            data.sessionId
          ),
        })
      }
    },
    onError: error => {
      console.error('Failed to send chat completion:', error)
    },
  })
}

/**
 * Hook for streaming chat completions
 *
 * @example
 * ```tsx
 * const streaming = useStreamingChatCompletion()
 *
 * const sessionId = await streaming.mutateAsync({
 *   namespace: 'llamafarm',
 *   projectId: 'project_seed',
 *   request: { messages: [{ role: 'user', content: 'Hello' }] },
 *   sessionId: 'existing-session-id',
 *   options: {
 *     onChunk: (chunk) => console.log('Chunk:', chunk),
 *     onComplete: () => console.log('Done'),
 *   }
 * })
 * ```
 */
export function useStreamingChatCompletion() {
  const queryClient = useQueryClient()

  return useMutation<string, Error, StreamingChatCompletionParams>({
    mutationFn: async ({
      namespace,
      projectId,
      request,
      sessionId,
      options,
    }) => {
      return await streamChatCompletion(
        namespace,
        projectId,
        request,
        sessionId,
        options
      )
    },
    onSuccess: (sessionId, variables) => {
      // Invalidate completion queries for this project
      queryClient.invalidateQueries({
        queryKey: chatCompletionsKeys.completions(
          variables.namespace,
          variables.projectId
        ),
      })

      // Update session cache if we have a session ID
      if (sessionId) {
        queryClient.invalidateQueries({
          queryKey: chatCompletionsKeys.session(
            variables.namespace,
            variables.projectId,
            sessionId
          ),
        })
      }
    },
    onError: error => {
      console.error('Failed to stream chat completion:', error)
    },
  })
}

/**
 * Convenience hook for sending a simple text message (non-streaming)
 *
 * @example
 * ```tsx
 * const sendMessage = useChatCompletionMessage()
 *
 * const result = await sendMessage.mutateAsync({
 *   namespace: 'llamafarm',
 *   projectId: 'project_seed',
 *   message: 'Hello',
 *   sessionId: 'existing-session-id'
 * })
 * ```
 */
export function useChatCompletionMessage() {
  const queryClient = useQueryClient()

  return useMutation<
    ChatCompletionResult,
    Error,
    {
      namespace: string
      projectId: string
      message: string
      sessionId?: string
      options?: Partial<ChatRequest>
    }
  >({
    mutationFn: async ({
      namespace,
      projectId,
      message,
      sessionId,
      options,
    }) => {
      const request = createChatCompletionRequest(message, options)
      return await sendChatCompletion(namespace, projectId, request, sessionId)
    },
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({
        queryKey: chatCompletionsKeys.completions(
          variables.namespace,
          variables.projectId
        ),
      })

      if (data.sessionId) {
        queryClient.invalidateQueries({
          queryKey: chatCompletionsKeys.session(
            variables.namespace,
            variables.projectId,
            data.sessionId
          ),
        })
      }
    },
    retry: false,
    onError: error => {
      console.error('Failed to send chat completion message:', error)
    },
  })
}

/**
 * Convenience hook for sending a streaming text message
 *
 * @example
 * ```tsx
 * const streamMessage = useStreamingChatCompletionMessage()
 *
 * const sessionId = await streamMessage.mutateAsync({
 *   namespace: 'llamafarm',
 *   projectId: 'project_seed',
 *   message: 'Hello',
 *   sessionId: 'existing-session-id',
 *   streamingOptions: {
 *     onChunk: (chunk) => console.log('Chunk:', chunk),
 *   }
 * })
 * ```
 */
export function useStreamingChatCompletionMessage() {
  const queryClient = useQueryClient()

  return useMutation<
    string,
    Error,
    {
      namespace: string
      projectId: string
      message: string
      sessionId?: string
      requestOptions?: Partial<ChatRequest>
      streamingOptions?: StreamingChatOptions
    }
  >({
    mutationFn: async ({
      namespace,
      projectId,
      message,
      sessionId,
      requestOptions,
      streamingOptions,
    }) => {
      const request = createChatCompletionRequest(message, requestOptions)
      return await streamChatCompletion(
        namespace,
        projectId,
        request,
        sessionId,
        streamingOptions
      )
    },
    onSuccess: (sessionId, variables) => {
      queryClient.invalidateQueries({
        queryKey: chatCompletionsKeys.completions(
          variables.namespace,
          variables.projectId
        ),
      })

      if (sessionId) {
        queryClient.invalidateQueries({
          queryKey: chatCompletionsKeys.session(
            variables.namespace,
            variables.projectId,
            sessionId
          ),
        })
      }
    },
    onError: error => {
      console.error('Failed to stream chat completion message:', error)
    },
  })
}

/**
 * Main unified hook that provides both streaming and non-streaming capabilities
 *
 * @example
 * ```tsx
 * const chat = useChatCompletions()
 *
 * // Non-streaming
 * const result = await chat.completion.mutateAsync({
 *   namespace: 'llamafarm',
 *   projectId: 'project_seed',
 *   request: { messages: [...] }
 * })
 *
 * // Streaming
 * const sessionId = await chat.streaming.mutateAsync({
 *   namespace: 'llamafarm',
 *   projectId: 'project_seed',
 *   request: { messages: [...] },
 *   options: { onChunk: (chunk) => ... }
 * })
 * ```
 */
export function useChatCompletions() {
  const completion = useChatCompletion()
  const streaming = useStreamingChatCompletion()
  const message = useChatCompletionMessage()
  const streamingMessage = useStreamingChatCompletionMessage()

  return {
    /**
     * Non-streaming completion mutation
     */
    completion,

    /**
     * Streaming completion mutation
     */
    streaming,

    /**
     * Non-streaming message mutation (convenience wrapper)
     */
    message,

    /**
     * Streaming message mutation (convenience wrapper)
     */
    streamingMessage,

    /**
     * Combined loading state
     */
    isLoading:
      completion.isPending ||
      streaming.isPending ||
      message.isPending ||
      streamingMessage.isPending,

    /**
     * Combined error state
     */
    error:
      completion.error ||
      streaming.error ||
      message.error ||
      streamingMessage.error,
  }
}

export default {
  useChatCompletion,
  useStreamingChatCompletion,
  useChatCompletionMessage,
  useStreamingChatCompletionMessage,
  useChatCompletions,
  chatCompletionsKeys,
}
