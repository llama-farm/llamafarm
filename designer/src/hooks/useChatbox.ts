/**
 * Unified Chatbox Hook
 *
 * Provides a single configurable hook for managing chatbox state and API interactions.
 * Supports both basic session management and project session management with tool calls.
 * Extracts chat logic from the Chatbox component for better reusability and testability.
 */

import { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useProject } from './useProjects'
import { useActiveProject } from './useActiveProject'
import { parsePromptSets } from '../utils/promptSets'
import { useProjectSession } from './useProjectSession'
import { useChatSession } from './useChatSession'
import { ChatboxMessage } from '../types/chatbox'
import { ChatStreamChunk, NetworkError, ChatMessage } from '../types/chat'
import { generateMessageId } from '../utils/idGenerator'
import {
  useStreamingChatCompletionMessage,
  useChatCompletionMessage,
} from './useChatCompletions'
import { createChatCompletionRequest } from '../api/chatCompletionsService'
import { DEV_CHAT_NAMESPACE, DEV_CHAT_PROJECT_ID } from '../constants/chat'

/**
 * Configuration options for the unified chatbox hook
 */
export interface UseChatboxOptions {
  /**
   * Use project session management (for Designer Chat with tool calls)
   * If false, uses simple session management (for basic chat)
   * @default false
   */
  useProjectSession?: boolean

  /**
   * Enable streaming responses
   * @default true
   */
  enableStreaming?: boolean

  /**
   * Initial session ID (only used for simple session mode)
   */
  initialSessionId?: string

  /**
   * Namespace for chat API (defaults to DEV_CHAT_NAMESPACE)
   */
  namespace?: string

  /**
   * Project ID for chat API (defaults to DEV_CHAT_PROJECT_ID)
   */
  projectId?: string

  /**
   * Chat service for project session (only used when useProjectSession is true)
   * @default 'designer'
   */
  chatService?: 'designer' | 'project'

  /**
   * Auto-create session (only used when useProjectSession is true)
   * @default false
   */
  autoCreateSession?: boolean
}

/**
 * Convert project session message to chatbox message format
 */
function projectSessionToChatboxMessage(msg: {
  id: string
  role: 'user' | 'assistant' | 'tool'
  content: string
  timestamp: string
  tool_call_id?: string
}): ChatboxMessage {
  let type: 'user' | 'assistant' | 'tool' | 'error' = 'assistant'
  if (msg.role === 'user') {
    type = 'user'
  } else if (msg.role === 'tool') {
    type = 'tool'
  }

  return {
    id: msg.id,
    type,
    content: msg.content,
    timestamp: new Date(msg.timestamp),
    tool_call_id: msg.tool_call_id,
  }
}

/**
 * Unified chatbox hook that supports both simple and project session management
 *
 * @example Basic usage (simple session)
 * ```tsx
 * const chat = useChatbox()
 * ```
 *
 * @example With project session and tool calls (Designer Chat)
 * ```tsx
 * const chat = useChatbox({ useProjectSession: true })
 * ```
 *
 * @example Custom namespace/project
 * ```tsx
 * const chat = useChatbox({
 *   namespace: 'my-namespace',
 *   projectId: 'my-project'
 * })
 * ```
 */
export function useChatbox(options: UseChatboxOptions = {}) {
  const {
    useProjectSession: useProjectSessionMode = false,
    enableStreaming = true,
    initialSessionId,
    namespace = DEV_CHAT_NAMESPACE,
    projectId = DEV_CHAT_PROJECT_ID,
    chatService = 'designer',
    autoCreateSession = false,
  } = options

  const streamingEnabled =
    enableStreaming && !import.meta.env.VITE_DISABLE_STREAMING

  // Session management - conditional based on mode
  const simpleSession = useChatSession(
    useProjectSessionMode ? undefined : initialSessionId
  )
  const projectSession = useProjectSession({
    chatService,
    autoCreate: autoCreateSession,
  })

  // Load project config for prompt sets (only when using project session)
  const activeProject = useActiveProject()
  const { data: projectResponse } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    useProjectSessionMode &&
      !!activeProject?.namespace &&
      !!activeProject?.project
  )

  // UI state
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  // Local messages state (for simple session mode)
  const [localMessages, setLocalMessages] = useState<ChatboxMessage[]>([])
  const [hasInitialSync, setHasInitialSync] = useState(false)

  // Streaming messages state (for project session mode)
  const [streamingMessages, setStreamingMessages] = useState<ChatboxMessage[]>(
    []
  )

  // Refs for streaming and debounced save
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const streamingAbortControllerRef = useRef<AbortController | null>(null)
  const fallbackTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const sessionReconciliationTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)
  const streamingMessagesRef = useRef<ChatboxMessage[]>([])

  // Keep ref in sync with state
  useEffect(() => {
    streamingMessagesRef.current = streamingMessages
  }, [streamingMessages])

  // Refs for accumulated streaming content and tool calls (project session mode only)
  const accumulatedContentRef = useRef<Record<string, string>>({})
  const toolCallsRef = useRef<
    Record<string, Array<{ name: string; arguments: string; id?: string }>>
  >({})
  const savedToolCallIdsRef = useRef<Set<string>>(new Set())

  // API hooks - using unified chat completions interface
  const queryClient = useQueryClient()
  const streamingChat = useStreamingChatCompletionMessage()
  const nonStreamingChat = useChatCompletionMessage()

  // Get current session ID and messages based on mode
  const currentSessionId = useProjectSessionMode
    ? projectSession.sessionId
    : simpleSession.currentSessionId

  const projectSessionMessages = useMemo(() => {
    if (!useProjectSessionMode) return []
    return projectSession.messages.map(projectSessionToChatboxMessage)
  }, [useProjectSessionMode, projectSession.messages])

  // Sync persisted messages with local state (simple session mode only)
  useEffect(() => {
    if (useProjectSessionMode) return

    // Reset state when session changes (project switching)
    setHasInitialSync(false)

    // Load messages from new session immediately
    if (simpleSession.messages.length > 0) {
      setLocalMessages(simpleSession.messages)
    } else {
      setLocalMessages([])
    }
    setHasInitialSync(true)
  }, [
    useProjectSessionMode,
    simpleSession.currentSessionId,
    simpleSession.messages,
  ])

  // Debounced save function (simple session mode only)
  const debouncedSave = useCallback(
    (sessionId: string, messages: ChatboxMessage[]) => {
      if (useProjectSessionMode) return

      // Clear existing timeout
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current)
      }

      // Set new timeout for debounced save
      saveTimeoutRef.current = setTimeout(() => {
        simpleSession.saveSessionMessages(sessionId, messages)
      }, 500)
    },
    [useProjectSessionMode, simpleSession]
  )

  // Save messages to persistence when they change (simple session mode only)
  useEffect(() => {
    if (useProjectSessionMode) return

    // Save if we have a valid session ID and either:
    // 1. We've done initial sync (loaded from persistence), OR
    // 2. We have messages to save (new session with messages)
    if (currentSessionId && (hasInitialSync || localMessages.length > 0)) {
      debouncedSave(currentSessionId, localMessages)
    }
  }, [
    useProjectSessionMode,
    localMessages,
    currentSessionId,
    debouncedSave,
    hasInitialSync,
    queryClient,
  ])

  // Cleanup timeout and abort streaming on unmount
  useEffect(() => {
    isMountedRef.current = true

    return () => {
      isMountedRef.current = false
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current)
      }
      if (fallbackTimeoutRef.current) {
        clearTimeout(fallbackTimeoutRef.current)
      }
      if (sessionReconciliationTimeoutRef.current) {
        clearTimeout(sessionReconciliationTimeoutRef.current)
      }
      if (streamingAbortControllerRef.current) {
        streamingAbortControllerRef.current.abort()
      }
    }
  }, [])

  // Helper function to prepend prompt sets to chat request (project session mode only)
  const prependActiveSet = useCallback(
    (chatRequest: { messages: ChatMessage[] }) => {
      if (!useProjectSessionMode) return

      const projectPrompts = projectResponse?.project?.config
        ?.prompts as Array<{
        name: string
        messages: Array<{ role?: string; content: string }>
      }>
      if (Array.isArray(projectPrompts) && projectPrompts.length > 0) {
        // Get messages from the first prompt set
        const sets = parsePromptSets(projectPrompts)
        if (sets.length > 0 && sets[0].items.length > 0) {
          const systemMessages = sets[0].items.map(item => ({
            role: item.role,
            content: item.content,
          })) as ChatMessage[]
          chatRequest.messages = [...systemMessages, ...chatRequest.messages]
        }
      }
    },
    [useProjectSessionMode, projectResponse?.project?.config]
  )

  // Helper function to execute fallback non-streaming request
  const executeFallbackRequest = useCallback(
    async (
      messageContent: string,
      currentSessionId: string,
      onSuccess: (response: {
        data: { choices: Array<{ message: { content: string } }> }
        sessionId: string
      }) => void,
      onError: (error: Error) => void
    ) => {
      if (!isMountedRef.current) return

      try {
        const chatRequest = createChatCompletionRequest(messageContent)
        prependActiveSet(chatRequest)

        const result = await nonStreamingChat.mutateAsync({
          namespace,
          projectId,
          message: messageContent,
          sessionId: currentSessionId || undefined,
          options: chatRequest,
        })

        if (!isMountedRef.current) return

        // Convert to expected format for compatibility
        onSuccess({
          data: result.response,
          sessionId: result.sessionId,
        })
      } catch (fallbackError) {
        if (!isMountedRef.current) return

        console.error('Fallback request failed:', fallbackError)
        onError(
          fallbackError instanceof Error
            ? fallbackError
            : new Error('Unknown fallback error')
        )
      }
    },
    [nonStreamingChat, namespace, projectId, prependActiveSet]
  )

  // Add message to state
  const addMessage = useCallback(
    (message: Omit<ChatboxMessage, 'id'>) => {
      const newMessage: ChatboxMessage = {
        ...message,
        id: generateMessageId(),
      }

      if (useProjectSessionMode) {
        // Check if this is a placeholder message that should skip project session
        const isThinkingPlaceholder =
          message.content === 'Thinking...' && message.type === 'assistant'
        const shouldSkipProjectSession =
          isThinkingPlaceholder && !projectSession.isTemporaryMode

        if (!shouldSkipProjectSession) {
          try {
            projectSession.addMessage(
              message.content,
              message.type === 'user' ? 'user' : 'assistant'
            )
          } catch (err) {
            console.error('Failed to add message to project session:', err)
            throw err
          }
        }
      } else {
        // Simple session mode - add to local state
        setLocalMessages(prev => [...prev, newMessage])
      }

      return newMessage.id
    },
    [useProjectSessionMode, projectSession]
  )

  // Update message by ID
  const updateMessage = useCallback(
    (id: string, updates: Partial<ChatboxMessage>) => {
      if (useProjectSessionMode) {
        // For project session, maintain temporary streaming messages
        setStreamingMessages(prev => {
          const existing = prev.find(msg => msg.id === id)
          if (existing) {
            return prev.map(msg =>
              msg.id === id ? { ...msg, ...updates } : msg
            )
          } else {
            return [
              ...prev,
              {
                id,
                type: 'assistant',
                content: '',
                timestamp: new Date(),
                ...updates,
              } as ChatboxMessage,
            ]
          }
        })
      } else {
        // Simple session mode - update local messages
        setLocalMessages(prev =>
          prev.map(msg => (msg.id === id ? { ...msg, ...updates } : msg))
        )
      }
    },
    [useProjectSessionMode]
  )

  // Combine messages based on mode
  const currentMessages = useMemo(() => {
    if (useProjectSessionMode) {
      const combined = [...projectSessionMessages, ...streamingMessages]
      // Filter out "Thinking..." placeholder messages
      return combined.filter(msg => {
        const isThinkingPlaceholder =
          msg.type === 'assistant' &&
          msg.content === 'Thinking...' &&
          !msg.isStreaming &&
          !msg.isLoading
        return !isThinkingPlaceholder
      })
    }
    return localMessages
  }, [
    useProjectSessionMode,
    projectSessionMessages,
    streamingMessages,
    localMessages,
  ])

  // Handle sending message with streaming or non-streaming
  const sendMessage = useCallback(
    async (messageContent: string) => {
      if (
        !messageContent.trim() ||
        streamingChat.isPending ||
        nonStreamingChat.isPending ||
        isStreaming
      )
        return false

      // Cancel any existing streaming request
      if (streamingAbortControllerRef.current) {
        streamingAbortControllerRef.current.abort()
        streamingAbortControllerRef.current = null
      }

      setError(null)

      // Add user message immediately
      addMessage({
        type: 'user',
        content: messageContent,
        timestamp: new Date(),
      })

      // Add loading assistant message
      const assistantMessageId = generateMessageId()

      if (useProjectSessionMode && streamingEnabled) {
        // For project session streaming, add to streamingMessages
        setStreamingMessages(prev => [
          ...prev,
          {
            id: assistantMessageId,
            type: 'assistant',
            content: 'Thinking...',
            timestamp: new Date(),
            isLoading: false,
            isStreaming: true,
          },
        ])
      } else {
        // For simple session or non-streaming, add via addMessage
        addMessage({
          type: 'assistant',
          content: 'Thinking...',
          timestamp: new Date(),
          isLoading: true,
        })
      }

      let timeoutId: NodeJS.Timeout | undefined
      let accumulatedContent = ''

      try {
        const chatRequest = createChatCompletionRequest(messageContent)
        prependActiveSet(chatRequest)

        if (streamingEnabled) {
          // Streaming path
          setIsStreaming(true)

          const abortController = new AbortController()
          streamingAbortControllerRef.current = abortController

          timeoutId = setTimeout(() => {
            console.log('Streaming request timed out after 1 minute')
            abortController.abort()
          }, 60000)

          // Initialize refs for this message (project session mode only)
          if (useProjectSessionMode) {
            accumulatedContentRef.current[assistantMessageId] = ''
            toolCallsRef.current[assistantMessageId] = []
          }

          const deferredSessionIdRef: { current: string | null } = {
            current: null,
          }

          const responseSessionId = await streamingChat.mutateAsync({
            namespace,
            projectId,
            message: messageContent,
            sessionId: currentSessionId || undefined,
            requestOptions: {
              ...chatRequest,
              stream: undefined,
            },
            streamingOptions: {
              signal: abortController.signal,
              onChunk: (chunk: ChatStreamChunk) => {
                if (!isMountedRef.current) return

                const choice = chunk.choices?.[0]
                if (!choice) return

                const delta = choice.delta

                // Handle tool calls (project session mode only)
                if (
                  useProjectSessionMode &&
                  delta.tool_calls &&
                  delta.tool_calls.length > 0
                ) {
                  const messageToolCalls =
                    toolCallsRef.current[assistantMessageId] || []
                  for (const toolCall of delta.tool_calls) {
                    const toolIndex = toolCall.index ?? 0

                    if (toolCall.function?.name) {
                      if (!messageToolCalls[toolIndex]) {
                        messageToolCalls[toolIndex] = {
                          name: toolCall.function.name,
                          arguments: toolCall.function.arguments || '',
                          id: toolCall.id,
                        }
                      } else {
                        if (toolCall.function.arguments) {
                          messageToolCalls[toolIndex].arguments +=
                            toolCall.function.arguments
                        }
                      }

                      const toolCallMsg = messageToolCalls[toolIndex]
                      const toolContent = `🔧 Calling tool: ${toolCallMsg.name}${toolCallMsg.arguments ? `\n\nArguments: ${toolCallMsg.arguments}` : ''}`
                      const toolMessageId = `tool_${assistantMessageId}_${toolIndex}`
                      const toolCallId = toolCallMsg.id || toolMessageId

                      setStreamingMessages(prev => {
                        const existing = prev.find(
                          msg => msg.id === toolMessageId
                        )
                        if (existing) {
                          // Update existing tool message
                          return prev.map(msg =>
                            msg.id === toolMessageId
                              ? { ...msg, content: toolContent }
                              : msg
                          )
                        } else {
                          // Insert tool message BEFORE any assistant message with the same ID
                          const assistantIndex = prev.findIndex(
                            msg => msg.id === assistantMessageId
                          )
                          const newToolMessage = {
                            id: toolMessageId,
                            type: 'tool' as const,
                            content: toolContent,
                            timestamp: new Date(),
                            tool_call_id: toolCallMsg.id,
                          }

                          if (assistantIndex >= 0) {
                            // Insert before assistant message
                            return [
                              ...prev.slice(0, assistantIndex),
                              newToolMessage,
                              ...prev.slice(assistantIndex),
                            ]
                          } else {
                            // No assistant message yet, append to end
                            return [...prev, newToolMessage]
                          }
                        }
                      })

                      // Track tool call ID to save complete content later in onComplete
                      // This prevents duplicate messages from being saved for each streaming chunk
                      if (!savedToolCallIdsRef.current.has(toolCallId)) {
                        savedToolCallIdsRef.current.add(toolCallId)
                      }
                    }
                  }
                  toolCallsRef.current[assistantMessageId] = messageToolCalls
                  return
                }

                // Handle content chunks
                if (delta.content) {
                  if (useProjectSessionMode) {
                    const currentContent =
                      accumulatedContentRef.current[assistantMessageId] || ''
                    const newContent = currentContent + delta.content
                    accumulatedContentRef.current[assistantMessageId] =
                      newContent

                    setStreamingMessages(prev => {
                      const existing = prev.find(
                        msg => msg.id === assistantMessageId
                      )
                      if (!existing) {
                        return [
                          ...prev,
                          {
                            id: assistantMessageId,
                            type: 'assistant',
                            content: newContent,
                            timestamp: new Date(),
                            isStreaming: true,
                          },
                        ]
                      }
                      return prev.map(msg =>
                        msg.id === assistantMessageId
                          ? { ...msg, content: newContent, isStreaming: true }
                          : msg
                      )
                    })
                  } else {
                    // Simple session mode
                    accumulatedContent += delta.content
                    updateMessage(assistantMessageId, {
                      content: accumulatedContent,
                      isLoading: false,
                      isStreaming: true,
                    })
                  }
                }

                // Check if streaming is complete
                if (choice.finish_reason) {
                  if (!useProjectSessionMode) {
                    updateMessage(assistantMessageId, {
                      content: accumulatedContent || 'Response completed',
                      isLoading: false,
                      isStreaming: false,
                    })
                  }
                }
              },
              onError: error => {
                console.error('Streaming error:', error)
                if (!isMountedRef.current) return

                setIsStreaming(false)
                if (timeoutId) clearTimeout(timeoutId)

                // Remove streaming message
                if (useProjectSessionMode) {
                  setStreamingMessages(prev =>
                    prev.filter(msg => msg.id !== assistantMessageId)
                  )
                } else {
                  setLocalMessages(prev =>
                    prev.filter(msg => msg.id !== assistantMessageId)
                  )
                }

                // Handle fallback for non-abort errors
                const isUserCancellation =
                  error instanceof Error && error.name === 'AbortError'
                const isNetworkError =
                  error instanceof NetworkError &&
                  (error.message.includes('cancelled') ||
                    error.message.includes('aborted'))

                if (isNetworkError && !isUserCancellation) {
                  if (fallbackTimeoutRef.current) {
                    clearTimeout(fallbackTimeoutRef.current)
                  }

                  fallbackTimeoutRef.current = setTimeout(() => {
                    fallbackTimeoutRef.current = null
                    executeFallbackRequest(
                      messageContent,
                      currentSessionId || '',
                      response => {
                        if (
                          response.data.choices &&
                          response.data.choices.length > 0
                        ) {
                          const assistantResponse =
                            response.data.choices[0].message.content
                          if (assistantResponse && assistantResponse.trim()) {
                            addMessage({
                              type: 'assistant',
                              content: assistantResponse,
                              timestamp: new Date(),
                            })
                          } else {
                            addMessage({
                              type: 'assistant',
                              content:
                                "Sorry, I didn't receive a proper response.",
                              timestamp: new Date(),
                            })
                          }
                        }
                      },
                      fallbackError => {
                        const errorMessage =
                          fallbackError instanceof Error
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
                  const errorMessage = isUserCancellation
                    ? 'Request was cancelled'
                    : error instanceof NetworkError
                      ? error.message
                      : 'Streaming connection failed'

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
                if (!isMountedRef.current) return

                setIsStreaming(false)
                if (timeoutId) clearTimeout(timeoutId)

                if (useProjectSessionMode) {
                  const finalContent =
                    accumulatedContentRef.current[assistantMessageId] || ''

                  // Save all tool call messages with complete content to project session
                  // Use ref to get current state (avoids stale closure)
                  const toolCallPattern = `tool_${assistantMessageId}_`
                  const toolCallMessages = streamingMessagesRef.current.filter(
                    msg => msg.id.startsWith(toolCallPattern)
                  )

                  for (const toolMsg of toolCallMessages) {
                    try {
                      projectSession.addMessage(
                        toolMsg.content,
                        'tool',
                        toolMsg.tool_call_id
                      )
                    } catch (err) {
                      console.warn('Failed to save complete tool message:', err)
                    }
                  }

                  // Clean up refs
                  delete accumulatedContentRef.current[assistantMessageId]
                  delete toolCallsRef.current[assistantMessageId]
                  for (const savedId of Array.from(
                    savedToolCallIdsRef.current
                  )) {
                    if (savedId.startsWith(toolCallPattern)) {
                      savedToolCallIdsRef.current.delete(savedId)
                    }
                  }

                  // Remove tool call messages from streaming state since they're now in persistent storage
                  setStreamingMessages(prev =>
                    prev.filter(msg => !msg.id.startsWith(toolCallPattern))
                  )

                  if (finalContent && finalContent.trim()) {
                    try {
                      projectSession.addMessage(finalContent, 'assistant')
                      setTimeout(() => {
                        setStreamingMessages(prev =>
                          prev.filter(msg => msg.id !== assistantMessageId)
                        )
                      }, 100)
                    } catch (err) {
                      console.warn('Failed to save to project session:', err)
                      setStreamingMessages(prev =>
                        prev.map(msg =>
                          msg.id === assistantMessageId
                            ? {
                                ...msg,
                                content: finalContent,
                                isStreaming: false,
                                isLoading: false,
                              }
                            : msg
                        )
                      )
                    }
                  } else {
                    // No content - attempt fallback
                    setStreamingMessages(prev =>
                      prev.filter(msg => msg.id !== assistantMessageId)
                    )

                    if (fallbackTimeoutRef.current) {
                      clearTimeout(fallbackTimeoutRef.current)
                    }

                    fallbackTimeoutRef.current = setTimeout(() => {
                      fallbackTimeoutRef.current = null
                      executeFallbackRequest(
                        messageContent,
                        currentSessionId || '',
                        response => {
                          if (
                            response.data.choices &&
                            response.data.choices.length > 0
                          ) {
                            const assistantResponse =
                              response.data.choices[0].message.content
                            if (assistantResponse && assistantResponse.trim()) {
                              addMessage({
                                type: 'assistant',
                                content: assistantResponse,
                                timestamp: new Date(),
                              })
                            }
                          }
                        },
                        () => {
                          addMessage({
                            type: 'error',
                            content: 'Error: Failed to get response',
                            timestamp: new Date(),
                          })
                        }
                      )
                    }, 100)
                  }
                }
              },
            },
          })

          // Store session ID for deferred processing (project session mode)
          if (useProjectSessionMode && responseSessionId) {
            deferredSessionIdRef.current = responseSessionId

            // Handle session reconciliation after streaming completes
            // Clear any existing timeout first
            if (sessionReconciliationTimeoutRef.current) {
              clearTimeout(sessionReconciliationTimeoutRef.current)
            }

            // Schedule reconciliation with tracked timeout
            sessionReconciliationTimeoutRef.current = setTimeout(() => {
              if (!isMountedRef.current) return

              sessionReconciliationTimeoutRef.current = null

              if (deferredSessionIdRef.current) {
                try {
                  const existingSessionId =
                    currentSessionId || projectSession.sessionId
                  if (existingSessionId) {
                    if (existingSessionId !== deferredSessionIdRef.current) {
                      projectSession.reconcileWithServer(
                        existingSessionId,
                        deferredSessionIdRef.current
                      )
                    }
                  } else {
                    projectSession.createSessionFromServer(
                      deferredSessionIdRef.current
                    )
                  }
                } catch (sessionError) {
                  console.error('Session management error:', sessionError)
                }
              }
            }, 10)
          }

          // Update session ID (simple session mode)
          if (
            !useProjectSessionMode &&
            responseSessionId &&
            responseSessionId !== currentSessionId
          ) {
            simpleSession.setSessionId(responseSessionId)
            if (!hasInitialSync) {
              setHasInitialSync(true)
            }
          }

          setIsStreaming(false)
          if (timeoutId) clearTimeout(timeoutId)
          return true
        } else {
          // Non-streaming path
          const result = await nonStreamingChat.mutateAsync({
            namespace,
            projectId,
            message: messageContent,
            sessionId: currentSessionId || undefined,
            options: chatRequest,
          })

          // Handle session management
          if (result.sessionId) {
            if (useProjectSessionMode) {
              try {
                const existingSessionId =
                  currentSessionId || projectSession.sessionId
                if (existingSessionId) {
                  if (existingSessionId !== result.sessionId) {
                    projectSession.reconcileWithServer(
                      existingSessionId,
                      result.sessionId
                    )
                  }
                } else {
                  projectSession.createSessionFromServer(result.sessionId)
                }
              } catch (sessionError) {
                console.error('Session management error:', sessionError)
              }
            } else {
              if (result.sessionId !== currentSessionId) {
                simpleSession.setSessionId(result.sessionId)
                if (!hasInitialSync) {
                  setHasInitialSync(true)
                }
              }
            }
          }

          // Update assistant message with response
          if (result.response.choices && result.response.choices.length > 0) {
            const assistantResponse = result.response.choices[0].message.content

            if (!assistantResponse || !assistantResponse.trim()) {
              updateMessage(assistantMessageId, {
                content: "Sorry, I didn't receive a proper response.",
                isLoading: false,
              })
            } else {
              if (useProjectSessionMode) {
                try {
                  projectSession.addMessage(assistantResponse, 'assistant')
                  setStreamingMessages(prev =>
                    prev.filter(msg => msg.id !== assistantMessageId)
                  )
                } catch (err) {
                  console.warn('Failed to save to project session:', err)
                  updateMessage(assistantMessageId, {
                    content: assistantResponse,
                    isLoading: false,
                  })
                }
              } else {
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
        if (timeoutId) clearTimeout(timeoutId)

        // Check if this was an abort error (user cancelled)
        if (error instanceof Error && error.name === 'AbortError') {
          updateMessage(assistantMessageId, {
            content:
              (useProjectSessionMode
                ? accumulatedContentRef.current[assistantMessageId]
                : accumulatedContent) || 'Request was cancelled',
            isLoading: false,
            isStreaming: false,
          })
          return false
        }

        // Remove loading message
        if (useProjectSessionMode) {
          setStreamingMessages(prev =>
            prev.filter(msg => msg.id !== assistantMessageId)
          )
        } else {
          setLocalMessages(prev =>
            prev.filter(msg => msg.id !== assistantMessageId)
          )
        }

        // Set error message
        const errorMessage =
          error instanceof Error
            ? error.message
            : 'An unexpected error occurred'
        setError(errorMessage)

        // Add error message to chat
        addMessage({
          type: 'error',
          content: `Error: ${errorMessage}`,
          timestamp: new Date(),
        })

        return false
      } finally {
        streamingAbortControllerRef.current = null
        if (timeoutId) {
          clearTimeout(timeoutId)
        }
      }
    },
    [
      streamingChat,
      nonStreamingChat,
      currentSessionId,
      addMessage,
      updateMessage,
      isStreaming,
      hasInitialSync,
      streamingEnabled,
      executeFallbackRequest,
      namespace,
      projectId,
      prependActiveSet,
      useProjectSessionMode,
      projectSession,
      simpleSession,
      streamingMessages,
    ]
  )

  // Handle clear chat
  const clearChat = useCallback(async () => {
    try {
      if (useProjectSessionMode) {
        projectSession.clearHistory()
        setStreamingMessages([])
      } else {
        setLocalMessages([])
        setHasInitialSync(false)
        simpleSession.createNewSession()
      }
      setError(null)
      return true
    } catch (error) {
      console.error('Clear chat error:', error)
      const errorMessage =
        error instanceof Error ? error.message : 'Failed to clear chat'
      setError(errorMessage)
      return false
    }
  }, [useProjectSessionMode, projectSession, simpleSession])

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
      if (useProjectSessionMode) {
        setStreamingMessages(prev =>
          prev.map(msg =>
            msg.isStreaming
              ? {
                  ...msg,
                  isStreaming: false,
                  content: msg.content + ' [Cancelled]',
                }
              : msg
          )
        )
      } else {
        setLocalMessages(prev =>
          prev.map(msg =>
            msg.isStreaming
              ? {
                  ...msg,
                  isStreaming: false,
                  content: msg.content + ' [Cancelled]',
                }
              : msg
          )
        )
      }
    }
  }, [isStreaming, useProjectSessionMode])

  // Reset to new session
  const resetSession = useCallback(() => {
    // Cancel any active streaming first
    if (isStreaming) {
      cancelStreaming()
    }

    if (useProjectSessionMode) {
      setStreamingMessages([])
    } else {
      const newSessionId = simpleSession.createNewSession()
      setLocalMessages([])
      setHasInitialSync(false)
      setInputValue('')
      setError(null)
      return newSessionId
    }

    setInputValue('')
    setError(null)
    return ''
  }, [isStreaming, cancelStreaming, useProjectSessionMode, simpleSession])

  return {
    // State
    sessionId: currentSessionId,
    messages: currentMessages,
    inputValue,
    error: error || (useProjectSessionMode ? projectSession.error : null),

    // Loading states
    isSending:
      streamingChat.isPending || nonStreamingChat.isPending || isStreaming,
    isStreaming,
    isClearing: false,
    isLoadingSession: useProjectSessionMode ? false : simpleSession.isLoading,

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
    canSend:
      !streamingChat.isPending &&
      !nonStreamingChat.isPending &&
      !isStreaming &&
      inputValue.trim().length > 0,
  }
}

// Export default for backward compatibility
export default useChatbox

// Export backward compatible variants
/**
 * Backward compatible wrapper for simple chatbox usage
 * @deprecated Use useChatbox() directly instead
 */
export function useChatboxSimple(
  initialSessionId?: string,
  enableStreaming: boolean = true
) {
  return useChatbox({
    useProjectSession: false,
    enableStreaming,
    initialSessionId,
  })
}

/**
 * Backward compatible wrapper for project session chatbox usage
 * @deprecated Use useChatbox({ useProjectSession: true }) instead
 */
export function useChatboxWithProjectSession(enableStreaming: boolean = true) {
  return useChatbox({
    useProjectSession: true,
    enableStreaming,
    chatService: 'designer',
  })
}
