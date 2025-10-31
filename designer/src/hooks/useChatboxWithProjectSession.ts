/**
 * Enhanced Chatbox Hook with Project Session Management
 *
 * Integrates with the project session manager for Designer Chat service
 * Maintains backward compatibility while adding project context session management
 */

import { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { useProject } from './useProjects'
import { useActiveProject } from './useActiveProject'
import { parsePromptSets } from '../utils/promptSets'
import { useProjectSession } from './useProjectSession'
import { ChatboxMessage } from '../types/chatbox'
import { ChatStreamChunk, NetworkError, ChatMessage } from '../types/chat'
import { generateMessageId } from '../utils/idGenerator'
import {
  useStreamingChatCompletionMessage,
  useChatCompletionMessage,
} from './useChatCompletions'
import { createChatCompletionRequest } from '../api/chatCompletionsService'
import { DEV_CHAT_NAMESPACE, DEV_CHAT_PROJECT_ID } from '../constants/chat'

// Export for backward compatibility
export const PROJECT_SEED_NAMESPACE = DEV_CHAT_NAMESPACE
export const PROJECT_SEED_PROJECT = DEV_CHAT_PROJECT_ID

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
 * Enhanced chatbox hook with project session management for Designer Chat
 */
export function useChatboxWithProjectSession(enableStreaming: boolean = true) {
  const streamingEnabled =
    enableStreaming && !import.meta.env.VITE_DISABLE_STREAMING

  // Project session management for Designer Chat
  const projectSession = useProjectSession({
    chatService: 'designer',
    autoCreate: false, // Sessions created on first message
  })

  // Load project config to include active prompt set
  const activeProject = useActiveProject()
  const { data: projectResponse } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject?.namespace && !!activeProject?.project
  )

  // UI state
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  // Refs for streaming
  const streamingAbortControllerRef = useRef<AbortController | null>(null)
  const fallbackTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)
  // Refs for accumulated streaming content and tool calls
  const accumulatedContentRef = useRef<Record<string, string>>({})
  const toolCallsRef = useRef<
    Record<string, Array<{ name: string; arguments: string; id?: string }>>
  >({})
  // Track which tool calls have been saved to project session (to prevent duplicates)
  const savedToolCallIdsRef = useRef<Set<string>>(new Set())

  // API hooks - using unified chat completions interface
  // Dev Chat uses hardcoded namespace/project for project_seed
  const streamingChat = useStreamingChatCompletionMessage()
  const nonStreamingChat = useChatCompletionMessage()

  // Get current state from project session system (always used)
  const currentSessionId = projectSession.sessionId
  const projectSessionMessages = useMemo(() => {
    return projectSession.messages.map(projectSessionToChatboxMessage)
  }, [projectSession.messages, currentSessionId])
  const isLoadingSession = false // Session loading is now synchronous

  // Cleanup timeout and abort streaming on unmount
  useEffect(() => {
    // Set mounted flag on mount
    isMountedRef.current = true

    return () => {
      // Only set to false on actual unmount, and abort any active streams
      isMountedRef.current = false
      if (fallbackTimeoutRef.current) {
        clearTimeout(fallbackTimeoutRef.current)
      }
      if (streamingAbortControllerRef.current) {
        streamingAbortControllerRef.current.abort()
      }
    }
  }, [])

  // Helper function to prepend prompt sets to chat request
  const prependActiveSet = useCallback(
    (chatRequest: { messages: ChatMessage[] }) => {
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
    [projectResponse?.project?.config]
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
      // Check if component is still mounted before proceeding
      if (!isMountedRef.current) {
        return
      }

      try {
        const chatRequest = createChatCompletionRequest(messageContent)
        // Prepend active prompt set once
        prependActiveSet(chatRequest)
        const result = await nonStreamingChat.mutateAsync({
          namespace: DEV_CHAT_NAMESPACE,
          projectId: DEV_CHAT_PROJECT_ID,
          message: messageContent,
          sessionId: currentSessionId,
          options: chatRequest,
        })

        // Check if component is still mounted before updating state
        if (!isMountedRef.current) {
          return
        }

        // Convert to expected format for compatibility
        onSuccess({
          data: result.response,
          sessionId: result.sessionId,
        })
      } catch (fallbackError) {
        // Check if component is still mounted before updating state
        if (!isMountedRef.current) {
          return
        }

        console.error('Fallback request also failed:', fallbackError)
        onError(
          fallbackError instanceof Error
            ? fallbackError
            : new Error('Unknown fallback error')
        )
      }
    },
    [nonStreamingChat, prependActiveSet]
  )

  // Add message to both streaming state and project session
  const addMessage = useCallback(
    (message: Omit<ChatboxMessage, 'id'>) => {
      const newMessage: ChatboxMessage = {
        ...message,
        id: generateMessageId(),
      }

      // Check if this is a placeholder message that should skip project session in persistent mode
      const isThinkingPlaceholder =
        message.content === 'Thinking...' && message.type === 'assistant'
      const shouldSkipProjectSession =
        isThinkingPlaceholder && !projectSession.isTemporaryMode

      if (!shouldSkipProjectSession) {
        // Add to project session system - it will create temporary session if needed
        try {
          projectSession.addMessage(
            message.content,
            message.type === 'user' ? 'user' : 'assistant'
          )
        } catch (err) {
          console.error('Failed to add message to project session:', err)
          // Don't fail silently - this is a critical error
          throw err
        }
      }

      return newMessage.id
    },
    [projectSession]
  )

  // Update message helper (for streaming updates before final save to project session)
  const [streamingMessages, setStreamingMessages] = useState<ChatboxMessage[]>(
    []
  )
  const updateMessage = useCallback(
    (id: string, updates: Partial<ChatboxMessage>) => {
      // For project session system, we maintain temporary streaming messages
      // These are later replaced when final message is saved to project session
      setStreamingMessages(prev => {
        const existing = prev.find(msg => msg.id === id)
        if (existing) {
          return prev.map(msg => (msg.id === id ? { ...msg, ...updates } : msg))
        } else {
          // Add new streaming message
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
    },
    []
  )

  // Combine project session messages with temporary streaming messages
  const currentMessages = useMemo(() => {
    const combined = [...projectSessionMessages, ...streamingMessages]

    // Filter out "Thinking..." placeholder messages for UI display (but keep streaming ones)
    const filtered = combined.filter(msg => {
      const isThinkingPlaceholder =
        msg.type === 'assistant' &&
        msg.content === 'Thinking...' &&
        !msg.isStreaming &&
        !msg.isLoading
      return !isThinkingPlaceholder
    })

    return filtered
  }, [projectSessionMessages, streamingMessages])

  // Handle sending message with streaming or non-streaming API integration
  const sendMessage = useCallback(
    async (messageContent: string) => {
      // Validate input
      if (!messageContent || messageContent.trim() === '') {
        return false
      }

      if (
        streamingChat.isPending ||
        nonStreamingChat.isPending ||
        isStreaming
      ) {
        return false
      }

      messageContent = messageContent.trim()

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
      // For streaming, add directly to streamingMessages so we can update it in real-time
      const assistantMessageId = generateMessageId()
      console.log('Creating streaming message:', assistantMessageId, {
        streamingEnabled,
      })
      if (streamingEnabled) {
        setStreamingMessages(prev => {
          const newMessage = {
            id: assistantMessageId,
            type: 'assistant' as const,
            content: 'Thinking...',
            timestamp: new Date(),
            isLoading: false,
            isStreaming: true,
          }
          console.log('Adding to streamingMessages:', {
            assistantMessageId,
            currentCount: prev.length,
            newMessage,
          })
          return [...prev, newMessage]
        })
      } else {
        // For non-streaming, add to project session
        addMessage({
          type: 'assistant',
          content: 'Thinking...',
          timestamp: new Date(),
          isLoading: true,
          isStreaming: false,
        })
      }

      let timeoutId: NodeJS.Timeout | undefined

      try {
        // Dev Chat uses hardcoded namespace/project
        // Create chat request
        const chatRequest = createChatCompletionRequest(messageContent)

        // Prepend active prompt set once
        prependActiveSet(chatRequest)

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

          // Initialize refs for this message
          const messageContentRef =
            accumulatedContentRef.current[assistantMessageId] || ''
          accumulatedContentRef.current[assistantMessageId] = messageContentRef
          const deferredSessionIdRef: { current: string | null } = {
            current: null,
          }
          if (!toolCallsRef.current[assistantMessageId]) {
            toolCallsRef.current[assistantMessageId] = []
          }

          const responseSessionId = await streamingChat.mutateAsync({
            namespace: DEV_CHAT_NAMESPACE,
            projectId: DEV_CHAT_PROJECT_ID,
            message: messageContent,
            sessionId: currentSessionId || undefined,
            requestOptions: {
              ...chatRequest,
              // Remove stream since it's handled by the streaming function
              stream: undefined,
            },
            streamingOptions: {
              signal: abortController.signal,
              onChunk: (chunk: ChatStreamChunk) => {
                console.log('🔥 onChunk CALLBACK INVOKED in hook:', {
                  assistantMessageId,
                  chunkId: chunk.id,
                  hasChoices: !!chunk.choices,
                  choicesLength: chunk.choices?.length || 0,
                  isMounted: isMountedRef.current,
                })

                // Don't check isMounted - we want to process chunks even if component is unmounting
                // The cleanup will handle aborting the stream if needed

                const choice = chunk.choices?.[0]
                if (!choice) {
                  console.warn('⚠️ No choice in chunk, skipping')
                  return
                }

                const delta = choice.delta

                // Handle tool calls
                if (delta.tool_calls && delta.tool_calls.length > 0) {
                  const messageToolCalls =
                    toolCallsRef.current[assistantMessageId] || []
                  for (const toolCall of delta.tool_calls) {
                    const toolIndex = toolCall.index ?? 0

                    if (toolCall.function?.name) {
                      // Initialize or update tool call
                      if (!messageToolCalls[toolIndex]) {
                        messageToolCalls[toolIndex] = {
                          name: toolCall.function.name,
                          arguments: toolCall.function.arguments || '',
                          id: toolCall.id,
                        }
                      } else {
                        // Accumulate arguments for this tool call
                        if (toolCall.function.arguments) {
                          messageToolCalls[toolIndex].arguments +=
                            toolCall.function.arguments
                        }
                      }

                      // Display tool call as a simple message
                      const toolCallMsg = messageToolCalls[toolIndex]
                      const toolContent = `🔧 Calling tool: ${toolCallMsg.name}${toolCallMsg.arguments ? `\n\nArguments: ${toolCallMsg.arguments}` : ''}`
                      const toolMessageId = `tool_${assistantMessageId}_${toolIndex}`
                      const toolCallId = toolCallMsg.id || toolMessageId

                      // Update or create tool message in streaming state (for display during streaming)
                      setStreamingMessages(prev => {
                        const existing = prev.find(
                          msg => msg.id === toolMessageId
                        )
                        if (existing) {
                          return prev.map(msg =>
                            msg.id === toolMessageId
                              ? { ...msg, content: toolContent }
                              : msg
                          )
                        } else {
                          return [
                            ...prev,
                            {
                              id: toolMessageId,
                              type: 'tool' as const,
                              content: toolContent,
                              timestamp: new Date(),
                              tool_call_id: toolCallMsg.id,
                            },
                          ]
                        }
                      })

                      // Save tool message to project session for persistence (only once per tool call)
                      if (!savedToolCallIdsRef.current.has(toolCallId)) {
                        savedToolCallIdsRef.current.add(toolCallId)
                        try {
                          projectSession.addMessage(
                            toolContent,
                            'tool',
                            toolCallMsg.id
                          )
                        } catch (err) {
                          console.warn(
                            'Failed to save tool message to project session:',
                            err
                          )
                          // Remove from saved set if save failed so we can retry
                          savedToolCallIdsRef.current.delete(toolCallId)
                        }
                      }
                    }
                  }
                  toolCallsRef.current[assistantMessageId] = messageToolCalls
                  return
                }

                // Handle role assignment (first chunk) - just log it, don't skip
                if (delta.role) {
                  console.log('Role delta received:', delta.role)
                  // Don't return - continue to process content if present
                }

                // Handle content chunks
                if (delta.content) {
                  console.log('📝 Processing content chunk:', {
                    assistantMessageId,
                    deltaContent: delta.content,
                    deltaContentLength: delta.content.length,
                  })

                  const currentContent =
                    accumulatedContentRef.current[assistantMessageId] || ''
                  const newContent = currentContent + delta.content
                  accumulatedContentRef.current[assistantMessageId] = newContent

                  console.log('📝 Updating message with content:', {
                    assistantMessageId,
                    currentLength: currentContent.length,
                    deltaLength: delta.content.length,
                    newLength: newContent.length,
                    currentContent: currentContent.substring(0, 50),
                    newContentPreview: newContent.substring(0, 100),
                    refContent: accumulatedContentRef.current[
                      assistantMessageId
                    ]?.substring(0, 50),
                  })

                  // Update streaming message directly
                  setStreamingMessages(prev => {
                    const existing = prev.find(
                      msg => msg.id === assistantMessageId
                    )
                    if (!existing) {
                      console.warn(
                        'Message not found in streamingMessages, adding:',
                        assistantMessageId
                      )
                      return [
                        ...prev,
                        {
                          id: assistantMessageId,
                          type: 'assistant' as const,
                          content: newContent,
                          timestamp: new Date(),
                          isStreaming: true,
                        },
                      ]
                    }

                    const updated = prev.map(msg =>
                      msg.id === assistantMessageId
                        ? { ...msg, content: newContent, isStreaming: true }
                        : msg
                    )

                    console.log('Message updated:', {
                      before: existing.content?.substring(0, 50),
                      after: newContent.substring(0, 50),
                      updatedCount: updated.length,
                    })

                    return updated
                  })
                }
              },
              onError: (error: Error) => {
                console.error('Streaming error:', error)
                clearTimeout(timeoutId)
                setIsStreaming(false)

                // Remove streaming message
                setStreamingMessages(prev =>
                  prev.filter(msg => msg.id !== assistantMessageId)
                )

                // Only attempt fallback for network errors that are NOT user-initiated cancellations
                // User cancellations (AbortError) should not trigger fallback
                const isUserCancellation =
                  error instanceof Error && error.name === 'AbortError'
                const isNetworkError =
                  error instanceof NetworkError &&
                  (error.message.includes('cancelled') ||
                    error.message.includes('aborted'))

                if (isNetworkError && !isUserCancellation) {
                  // Clear any existing fallback timeout
                  if (fallbackTimeoutRef.current) {
                    clearTimeout(fallbackTimeoutRef.current)
                  }

                  // Set up tracked fallback timeout
                  fallbackTimeoutRef.current = setTimeout(() => {
                    fallbackTimeoutRef.current = null

                    executeFallbackRequest(
                      messageContent,
                      currentSessionId || '',
                      response => {
                        // Add the response as a new message
                        if (
                          response.data.choices &&
                          response.data.choices.length > 0
                        ) {
                          const assistantResponse =
                            response.data.choices[0].message.content

                          // Skip empty responses
                          if (
                            !assistantResponse ||
                            assistantResponse.trim() === ''
                          ) {
                            addMessage({
                              type: 'assistant',
                              content:
                                "Sorry, I didn't receive a proper response.",
                              timestamp: new Date(),
                            })
                          } else {
                            addMessage({
                              type: 'assistant',
                              content: assistantResponse,
                              timestamp: new Date(),
                            })
                          }
                        } else {
                          addMessage({
                            type: 'assistant',
                            content:
                              "Sorry, I didn't receive a proper response.",
                            timestamp: new Date(),
                          })
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
                  // For user cancellations or other errors, show error message
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
                clearTimeout(timeoutId)
                setIsStreaming(false)

                // If we got content, finalize the message
                const finalContent =
                  accumulatedContentRef.current[assistantMessageId] || ''
                // Clean up refs for this message
                delete accumulatedContentRef.current[assistantMessageId]
                delete toolCallsRef.current[assistantMessageId]
                // Clean up saved tool call tracking for this message
                const toolCallPattern = `tool_${assistantMessageId}_`
                for (const savedId of Array.from(savedToolCallIdsRef.current)) {
                  if (
                    savedId.startsWith(toolCallPattern) ||
                    savedId === assistantMessageId
                  ) {
                    savedToolCallIdsRef.current.delete(savedId)
                  }
                }

                console.log('onComplete called:', {
                  assistantMessageId,
                  finalContent: finalContent?.substring(0, 100),
                  finalContentLength: finalContent?.length || 0,
                })

                if (finalContent && finalContent.trim()) {
                  // Save final message to project session and remove temporary streaming message
                  try {
                    console.log(
                      'Saving final message to project session:',
                      finalContent.substring(0, 100)
                    )
                    // Add final response to project session (will go to temp messages since streaming happens before session transfer)
                    projectSession.addMessage(finalContent, 'assistant')

                    // Remove the temporary streaming message AFTER a small delay to ensure project session has updated
                    setTimeout(() => {
                      console.log(
                        'Removing streaming message:',
                        assistantMessageId
                      )
                      setStreamingMessages(prev => {
                        const filtered = prev.filter(
                          msg => msg.id !== assistantMessageId
                        )
                        console.log('Streaming messages after removal:', {
                          before: prev.length,
                          after: filtered.length,
                          removedId: assistantMessageId,
                        })
                        return filtered
                      })
                    }, 100)

                    // NOW handle session creation/reconciliation after all messages are added
                    // Use a small delay to ensure the addMessage state update has completed
                    setTimeout(() => {
                      if (deferredSessionIdRef.current) {
                        try {
                          // Check if we have any existing session
                          const existingSessionId =
                            currentSessionId || projectSession.sessionId

                          if (existingSessionId) {
                            // Check if reconciliation is actually needed
                            if (
                              existingSessionId !== deferredSessionIdRef.current
                            ) {
                              // Session IDs differ, reconciliation needed
                              projectSession.reconcileWithServer(
                                existingSessionId,
                                deferredSessionIdRef.current
                              )
                            }
                          } else {
                            // Truly no existing session, create new one with all temp messages
                            projectSession.createSessionFromServer(
                              deferredSessionIdRef.current
                            )
                          }
                        } catch (sessionError) {
                          console.error(
                            'Failed to handle deferred session creation:',
                            sessionError
                          )
                          // Don't fail the whole request for session management errors
                        }
                      }
                    }, 10) // Small delay to ensure state updates have completed
                  } catch (err) {
                    console.warn('Failed to save to project session:', err)
                    // Keep the message in streaming state with final content
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
                  // No content received, try non-streaming fallback
                  setStreamingMessages(prev =>
                    prev.filter(msg => msg.id !== assistantMessageId)
                  )

                  // Handle deferred session even without content
                  if (deferredSessionIdRef.current) {
                    try {
                      const existingSessionId =
                        currentSessionId || projectSession.sessionId
                      if (!existingSessionId) {
                        projectSession.createSessionFromServer(
                          deferredSessionIdRef.current
                        )
                      }
                    } catch (sessionError) {
                      console.error(
                        'Failed to handle deferred session after streaming failure:',
                        sessionError
                      )
                    }
                  }

                  // Clear any existing fallback timeout
                  if (fallbackTimeoutRef.current) {
                    clearTimeout(fallbackTimeoutRef.current)
                  }

                  // Set up tracked fallback timeout
                  fallbackTimeoutRef.current = setTimeout(() => {
                    fallbackTimeoutRef.current = null

                    executeFallbackRequest(
                      messageContent,
                      currentSessionId || '',
                      response => {
                        // Add the response as a new message
                        if (
                          response.data.choices &&
                          response.data.choices.length > 0
                        ) {
                          const assistantResponse =
                            response.data.choices[0].message.content

                          // Skip empty responses
                          if (
                            !assistantResponse ||
                            assistantResponse.trim() === ''
                          ) {
                            addMessage({
                              type: 'assistant',
                              content:
                                "Sorry, I didn't receive a proper response.",
                              timestamp: new Date(),
                            })
                          } else {
                            addMessage({
                              type: 'assistant',
                              content: assistantResponse,
                              timestamp: new Date(),
                            })
                          }
                        } else {
                          addMessage({
                            type: 'assistant',
                            content:
                              "Sorry, I didn't receive a proper response.",
                            timestamp: new Date(),
                          })
                        }
                      },
                      fallbackError => {
                        console.error(
                          'Fallback request also failed:',
                          fallbackError
                        )
                        addMessage({
                          type: 'error',
                          content: 'Error: Failed to get response',
                          timestamp: new Date(),
                        })
                      }
                    )
                  }, 100)
                }
              },
            },
          })

          // Store session ID for deferred processing after all messages are added
          if (responseSessionId) {
            deferredSessionIdRef.current = responseSessionId
          }

          // For streaming, we return true immediately as the request is initiated
          // The actual success/failure will be handled by the streaming callbacks
          return true
        } else {
          // Non-streaming path - using unified interface
          const result = await nonStreamingChat.mutateAsync({
            namespace: DEV_CHAT_NAMESPACE,
            projectId: DEV_CHAT_PROJECT_ID,
            message: messageContent,
            sessionId: currentSessionId || undefined,
            options: chatRequest,
          })

          // Handle session reconciliation if we got a session ID from server
          if (result.sessionId) {
            try {
              // Check if we have any existing session (even if currentSessionId is null)
              const existingSessionId =
                currentSessionId || projectSession.sessionId

              if (existingSessionId) {
                // Check if reconciliation is actually needed
                if (existingSessionId !== result.sessionId) {
                  // Session IDs differ, reconciliation needed
                  projectSession.reconcileWithServer(
                    existingSessionId,
                    result.sessionId
                  )
                }
              } else {
                // Truly no existing session, create new one
                projectSession.createSessionFromServer(result.sessionId)
              }
            } catch (sessionError) {
              console.error(
                'Failed to handle session from server response:',
                sessionError
              )
              // Don't fail the whole request for session management errors
            }
          }

          // Update assistant message with response
          if (result.response.choices && result.response.choices.length > 0) {
            const assistantResponse = result.response.choices[0].message.content

            // Skip empty responses
            if (!assistantResponse || assistantResponse.trim() === '') {
              updateMessage(assistantMessageId, {
                content: "Sorry, I didn't receive a proper response.",
                isLoading: false,
              })
            } else {
              // Save final message to project session and remove temporary one
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
        setStreamingMessages(prev =>
          prev.filter(msg => msg.id !== assistantMessageId)
        )

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
        // Clear the abort controller reference and timeout
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
      streamingEnabled,
      isStreaming,
      projectSession,
      executeFallbackRequest,
      activeProject,
    ]
  )

  // Handle clear chat
  const clearChat = useCallback(async () => {
    try {
      // Use project session system
      projectSession.clearHistory()
      // Also clear any temporary streaming messages
      setStreamingMessages([])
      setError(null)
      return true
    } catch (error) {
      console.error('Clear chat error:', error)
      const errorMessage =
        error instanceof Error ? error.message : 'Failed to clear chat'
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

  const result = {
    // State
    sessionId: currentSessionId,
    messages: currentMessages,
    inputValue,
    error: error || projectSession.error,

    // Loading states
    isSending:
      streamingChat.isPending || nonStreamingChat.isPending || isStreaming,
    isStreaming,
    isClearing: false,
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
    canSend:
      !streamingChat.isPending &&
      !nonStreamingChat.isPending &&
      !isStreaming &&
      inputValue.trim().length > 0,
  }

  return result
}

export default useChatboxWithProjectSession
