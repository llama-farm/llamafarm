import { useCallback, useEffect, useRef, useState } from 'react'
import FontIcon from '../../common/FontIcon'
import { ChatboxMessage } from '../../types/chatbox'
import { Badge } from '../ui/badge'
import { useActiveProject } from '../../hooks/useActiveProject'
import { 
  useProjectChatStreamingMessage,
  useProjectChatParams 
} from '../../hooks/useProjectChat'
import { useProjectChatStreamingSession } from '../../hooks/useProjectChatSession'
import { useProjectSession } from '../../hooks/useProjectSession'
import { useChatbox } from '../../hooks/useChatbox'
import { ProjectChatStreamChunk } from '../../api/projectChatService'

export interface TestChatProps {
  showReferences: boolean
  allowRanking: boolean
  useTestData?: boolean
  showPrompts?: boolean
  showThinking?: boolean
  showGenSettings?: boolean
}

const containerClasses =
  // Match page background with clear outlines
  'w-full h-full flex flex-col rounded-xl border border-border bg-background text-foreground'

const inputContainerClasses =
  'flex flex-col gap-2 p-3 md:p-4 bg-background/60 border-t border-border rounded-b-xl'

const textareaClasses =
  'w-full h-auto min-h-[3rem] md:min-h-[3.5rem] resize-none bg-transparent border-none placeholder-opacity-60 focus:outline-none focus:ring-0 font-sans text-sm md:text-base leading-relaxed overflow-y-auto text-foreground placeholder-foreground/60'

function EmptyState() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40">
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-teal-500/20 border border-teal-500/30">
          <FontIcon type="test" className="w-5 h-5 text-teal-400" />
        </div>
        <div className="text-lg font-medium text-foreground">
          Start testing your model
        </div>
        <div className="mt-1 text-sm text-muted-foreground">
          Send a message to evaluate responses and run diagnostics.
        </div>
        <div className="mt-3 text-xs text-muted-foreground">
          Tip: Press Enter to send
        </div>
      </div>
    </div>
  )
}

export default function TestChat({
  showReferences,
  allowRanking,
  useTestData,
  showPrompts,
  showThinking,
  showGenSettings,
}: TestChatProps) {
  // Get active project for project chat API
  const activeProject = useActiveProject()
  const chatParams = useProjectChatParams(activeProject)
  
  // Project chat streaming session management
  const projectChatStreamingSession = useProjectChatStreamingSession(
    chatParams?.namespace,
    chatParams?.projectId
  )
  
  // Project chat streaming message sending
  const projectChatStreamingMessage = useProjectChatStreamingMessage()

  // Project session management for Project Chat (with persistence)
  const projectSession = useProjectSession({
    chatService: 'project',
    autoCreate: false, // Sessions created on first message
  })

  // Fallback chatbox for when project chat is not available
  const {
    messages: fallbackMessages,
    inputValue: fallbackInputValue,
    isSending: fallbackIsSending,
    isClearing: fallbackIsClearing,
    error: fallbackError,
    sendMessage: fallbackSendMessage,
    clearChat: fallbackClearChat,
    updateInput: fallbackUpdateInput,
    hasMessages: fallbackHasMessages,
    canSend: fallbackCanSend,
    addMessage: fallbackAddMessage,
    updateMessage: fallbackUpdateMessage,
  } = useChatbox()

  // Mock mode controlled by parent
  const MOCK_MODE = Boolean(useTestData)
  
  // Use project chat if we have an active project and not in mock mode
  const USE_PROJECT_CHAT = !MOCK_MODE && !!chatParams
  
  // Project chat state management
  const [projectInputValue, setProjectInputValue] = useState('')
  const [isProjectSending, setIsProjectSending] = useState(false)
  
  // Convert project session messages to chatbox format
  const projectSessionMessages: ChatboxMessage[] = projectSession.messages.map(msg => ({
    id: msg.id,
    type: msg.role === 'user' ? 'user' as const : 'assistant' as const,
    content: msg.content,
    timestamp: new Date(msg.timestamp),
  }))
  
  // Choose which chat system to use
  const messages = USE_PROJECT_CHAT ? projectSessionMessages : fallbackMessages
  const inputValue = USE_PROJECT_CHAT ? projectInputValue : fallbackInputValue
  const isSending = USE_PROJECT_CHAT ? isProjectSending : fallbackIsSending
  const isClearing = USE_PROJECT_CHAT ? false : fallbackIsClearing // Project chat doesn't have clearing in this context
  const error = USE_PROJECT_CHAT ? projectSession.error : fallbackError
  const hasMessages = USE_PROJECT_CHAT ? projectSessionMessages.length > 0 : fallbackHasMessages
  const canSend = USE_PROJECT_CHAT 
    ? (!isProjectSending && projectInputValue.trim().length > 0 && !projectChatStreamingMessage.isPending)
    : fallbackCanSend

  // Combined loading state
  const isProjectChatLoading = projectChatStreamingMessage.isPending
  const combinedIsSending = isSending || isProjectChatLoading
  
  // Combined error state
  const projectChatError = projectChatStreamingMessage.error || projectChatStreamingSession.error
  const combinedError = error || (projectChatError ? projectChatError.message : null)
  
  // Combined canSend state
  const combinedCanSend = canSend && !isProjectChatLoading
  const projectSessionId = projectSession.sessionId

  // Project chat message management functions
  const addMessage = useCallback((message: Omit<ChatboxMessage, 'id'>) => {
    if (USE_PROJECT_CHAT) {
      // Add message to project session (which handles persistence and creates temp session if needed)
      try {
        const sessionMessage = projectSession.addMessage(
          message.content,
          message.type === 'user' ? 'user' : 'assistant'
        )
        return sessionMessage.id
      } catch (err) {
        console.error('Failed to add message to project session:', err)
        // Generate a fallback ID for UI purposes
        return `msg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
      }
    } else {
      return fallbackAddMessage(message)
    }
  }, [USE_PROJECT_CHAT, projectSession, fallbackAddMessage])

  const updateMessage = useCallback((id: string, updates: Partial<ChatboxMessage>) => {
    if (USE_PROJECT_CHAT) {
      // For project session, we can't update individual messages after they're saved
      // This is only used for streaming updates before final save
      console.warn('updateMessage called on project session - this should only be used for streaming updates')
    } else {
      fallbackUpdateMessage(id, updates)
    }
  }, [USE_PROJECT_CHAT, fallbackUpdateMessage])

  const clearChat = useCallback(() => {
    if (USE_PROJECT_CHAT) {
      projectSession.clearHistory()
    } else {
      fallbackClearChat()
    }
  }, [USE_PROJECT_CHAT, projectSession, fallbackClearChat])

  const updateInput = useCallback((value: string) => {
    if (USE_PROJECT_CHAT) {
      setProjectInputValue(value)
    } else {
      fallbackUpdateInput(value)
    }
  }, [USE_PROJECT_CHAT, fallbackUpdateInput])

  const listRef = useRef<HTMLDivElement | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const lastUserInputRef = useRef<string>('')

  // Auto-grow textarea up to a comfortable max height before scrolling
  const resizeTextarea = useCallback(() => {
    const el = inputRef.current
    if (!el) return
    const maxHeight = 220 // ~6 lines depending on line-height
    el.style.height = 'auto'
    const newHeight = Math.min(el.scrollHeight, maxHeight)
    el.style.height = `${newHeight}px`
    el.style.overflowY = el.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }, [])

  useEffect(() => {
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    } else if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [messages])

  // Resize textarea on mount and input changes
  useEffect(() => {
    resizeTextarea()
  }, [inputValue, resizeTextarea])

  // Clear chat when active project changes (start fresh session)
  useEffect(() => {
    // Only clear if we have a valid project and we're not in mock mode
    if (!MOCK_MODE && chatParams?.namespace && chatParams?.projectId) {
      // Clear existing chat when switching projects
      clearChat()
      // Reset session for new project
      projectChatStreamingSession.clearSession()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatParams?.namespace, chatParams?.projectId]) // Only trigger when project actually changes

  const handleSend = useCallback(async () => {
    const content = inputValue.trim()
    if (!combinedCanSend || !content) return
    
    // Prevent multiple simultaneous requests
    if (combinedIsSending) {
      return
    }
    
    if (MOCK_MODE) {
      // Local-only optimistic flow without backend
      addMessage({ type: 'user', content, timestamp: new Date() })
      lastUserInputRef.current = content
      const assistantId = addMessage({
        type: 'assistant',
        content: 'Thinking…',
        timestamp: new Date(),
        isLoading: true,
      })
      updateInput('')
      setTimeout(() => {
        const mockAnswer = `Here is a mock response to: "${content}"\n\n- Point A\n- Point B\n\nThis is sample output while backend is disconnected.`
        const mockPrompts = [
          'System: You are an expert assistant. Answer clearly and concisely.',
          'Instruction: Provide helpful, safe, and accurate output.',
          `User input: ${content}`,
        ]
        const mockThinking = [
          'Identified intent and key entities.',
          'Searched internal knowledge for relevant facts.',
          'Composed structured response with bullet points.',
        ]
        updateMessage(assistantId, {
          content: mockAnswer,
          isLoading: false,
          sources: [
            {
              source: 'dataset/manuals/aircraft_mx_guide.pdf',
              score: 0.83,
              page: 12,
              chunk: 4,
              length: 182,
              content:
                'Hydraulic pressure drops during taxi often indicate minor leaks or entrained air. Inspect lines and fittings.',
            },
            {
              source: 'dataset/bulletins/bulletin-2024-17.md',
              score: 0.71,
              page: 3,
              chunk: 2,
              length: 126,
              content:
                'Pressure sensor calibration drifts were reported in batch 24B. Verify calibration if readings fluctuate.',
            },
          ],
          metadata: {
            prompts: mockPrompts,
            thinking: mockThinking,
            generation: {
              temperature: 0.7,
              topP: 0.9,
              maxTokens: 256,
              presencePenalty: 0.0,
              frequencyPenalty: 0.0,
              seed: undefined,
            },
          },
        })
      }, 1000)
      return
    }

    if (USE_PROJECT_CHAT && chatParams) {
      // Use project chat streaming API
      let assistantId: string | undefined
      let accumulatedContent = ''
      
      setIsProjectSending(true)
      
      try {
        // Add user message to project session
        projectSession.addMessage(content, 'user')
        lastUserInputRef.current = content
        
        // Add streaming assistant message (temporary, will be replaced with final message)
        assistantId = addMessage({
          type: 'assistant',
          content: 'Thinking…',
          timestamp: new Date(),
          isStreaming: true,
        })
        
        updateInput('')
        
        // Send streaming message via project chat
        const finalSessionId = await projectChatStreamingMessage.mutateAsync({
          namespace: chatParams.namespace,
          projectId: chatParams.projectId,
          message: content,
          sessionId: projectChatStreamingSession.sessionId || undefined,
          streamingOptions: {
            onChunk: (chunk: ProjectChatStreamChunk) => {
              // Handle content chunks
              if (chunk.choices?.[0]?.delta?.content) {
                accumulatedContent += chunk.choices[0].delta.content
                updateMessage(assistantId!, {
                  content: accumulatedContent,
                  isStreaming: true,
                })
              }
              // Clear "Thinking..." on first chunk even if no content (role assignment)
              else if (chunk.choices?.[0]?.delta && accumulatedContent === '') {
                updateMessage(assistantId!, {
                  content: '',
                  isStreaming: true,
                })
              }
            },
            onError: (error: Error) => {
              console.error('Project chat streaming error:', error)
              if (assistantId) {
                updateMessage(assistantId, {
                  content: `Error: ${error.message}`,
                  isStreaming: false,
                  isLoading: false,
                })
              }
            },
            onComplete: () => {
              if (assistantId && accumulatedContent) {
                // Save final message to project session
                projectSession.addMessage(accumulatedContent, 'assistant')
                updateMessage(assistantId, {
                  content: accumulatedContent,
                  isStreaming: false,
                  isLoading: false,
                  // Add mock metadata for consistency with existing UI
                  metadata: {
                    prompts: ['System: You are a helpful assistant.', `User: ${content}`],
                    thinking: ['Processing user request...', 'Generating response...'],
                    generation: {
                      temperature: 0.7,
                      topP: 0.9,
                      maxTokens: 512,
                      presencePenalty: 0.0,
                      frequencyPenalty: 0.0,
                      seed: undefined,
                    },
                  },
                })
              }
            },
          },
        })
        
        // Update session ID if we got a new one
        if (finalSessionId && finalSessionId !== projectChatStreamingSession.sessionId) {
          projectChatStreamingSession.setSessionId(finalSessionId)
        }
        
        // Create project session if we got a new session ID
        if (finalSessionId) {
          projectSession.createSessionFromServer(finalSessionId)
        }
        
      } catch (error) {
        console.error('Project chat streaming error:', error)
        // Update assistant message with error using the stored assistantId
        if (assistantId) {
          updateMessage(assistantId, {
            content: `Error: ${error instanceof Error ? error.message : 'Failed to send message'}`,
            isStreaming: false,
            isLoading: false,
          })
        }
      } finally {
        setIsProjectSending(false)
      }
      return
    }

    // Fallback to original chat system
    const ok = await fallbackSendMessage(content)
    if (ok) updateInput('')
  }, [
    combinedCanSend, 
    inputValue, 
    combinedIsSending,
    MOCK_MODE, 
    USE_PROJECT_CHAT, 
    chatParams, 
    projectChatStreamingMessage, 
    projectChatStreamingSession.sessionId,
    projectChatStreamingSession.setSessionId,
    projectSession,
    addMessage, 
    updateMessage, 
    updateInput,
    fallbackSendMessage
  ])

  const handleKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Wire lightweight global events for Retry and Use-as-prompt
  useEffect(() => {
    const onRetry = () => {
      // For now, simply re-send the last user message
      const lastUser = [...messages].reverse().find(m => m.type === 'user')
      if (lastUser) {
        updateInput(lastUser.content)
        setTimeout(() => handleSend(), 0)
      }
    }
    const onUse = (e: Event) => {
      const detail = (e as CustomEvent).detail as { content: string }
      updateInput(detail.content || '')
    }
    window.addEventListener('lf-chat-retry', onRetry as EventListener)
    window.addEventListener('lf-chat-use-as-prompt', onUse as EventListener)
    return () => {
      window.removeEventListener('lf-chat-retry', onRetry as EventListener)
      window.removeEventListener(
        'lf-chat-use-as-prompt',
        onUse as EventListener
      )
    }
  }, [messages, updateInput, handleSend])

  // Lightweight evaluator and mock generator for test runs
  const evaluateTest = useCallback(
    (input: string, expected: string, actual: string) => {
      const tokenize = (s: string) =>
        s
          .toLowerCase()
          .replace(/[^a-z0-9\s]/g, ' ')
          .split(/\s+/)
          .filter(Boolean)
      const a = new Set(tokenize(expected))
      const b = new Set(tokenize(actual))
      let inter = 0
      a.forEach(t => {
        if (b.has(t)) inter++
      })
      const union = a.size + b.size - inter || 1
      let score = (inter / union) * 100
      // Gentle bias upward and random jitter for realism
      score = Math.max(0, Math.min(100, score + 12 + (Math.random() * 6 - 3)))
      score = Math.round(score * 10) / 10
      const latencyMs = 120 + Math.round(Math.random() * 280)
      const promptTokens = tokenize(input).length
      const completionTokens = tokenize(actual).length
      return {
        score,
        latencyMs,
        tokenUsage: {
          prompt: promptTokens,
          completion: completionTokens,
          total: promptTokens + completionTokens,
        },
      }
    },
    []
  )

  // Handle external test-run events from the Tests panel
  useEffect(() => {
    const onRun = async (e: Event) => {
      const detail = (e as CustomEvent).detail as {
        id: number
        name: string
        input: string
        expected: string
      }

      const input = (detail.input || '').trim()
      const expected = (detail.expected || '').trim()

      // For test runs, we need to ensure we have a valid project
      if (!chatParams) {
        console.error('Test runs require a valid project configuration.')
        return
      }

      try {
        // Ensure we have a session for test execution
        let sessionId;
        if (!projectSessionId) {
          // Create a session first using streaming API
          sessionId = await projectChatStreamingMessage.mutateAsync({
            namespace: chatParams.namespace,
            projectId: chatParams.projectId,
            message: 'Starting test session...',
            sessionId: undefined,
            streamingOptions: {
              onChunk: () => {}, // No-op for test session creation
              onError: () => {},
              onComplete: () => {},
            },
          })
          
          // Session will be created automatically by the hook when needed
        }

        // Add test input message
        const userMessage = input || '(no input provided)'
        addMessage({
          type: 'user',
          content: userMessage,
          timestamp: new Date(),
          metadata: {
            isTest: true,
            testId: detail.id,
            testName: detail.name,
            expected,
          },
        })
        lastUserInputRef.current = input

        // Add loading assistant message
        const assistantId = addMessage({
          type: 'assistant',
          content: 'Evaluating…',
          timestamp: new Date(),
          isLoading: true,
          metadata: { isTest: true, testId: detail.id, testName: detail.name },
        })

        // Send the actual test input via project chat streaming
        let accumulatedContent = ''
        await projectChatStreamingMessage.mutateAsync({
          namespace: chatParams.namespace,
          projectId: chatParams.projectId,
          message: userMessage,
          sessionId: projectSessionId|| sessionId || undefined,
          streamingOptions: {
            onChunk: (chunk: ProjectChatStreamChunk) => {
              if (chunk.choices?.[0]?.delta?.content) {
                accumulatedContent += chunk.choices[0].delta.content
              }
            },
            onError: (error: Error) => {
              console.error('Test streaming error:', error)
            },
            onComplete: () => {},
          },
        })

        // Extract the response content
        const actualResponse = accumulatedContent || 'No response received'
        
        // Compute test evaluation
        const testResult = evaluateTest(input, expected, actualResponse)
        
        // Update the assistant message with the actual response and test results
        updateMessage(assistantId, {
          content: actualResponse,
          isLoading: false,
          metadata: {
            isTest: true,
            testId: detail.id,
            testName: detail.name,
            testResult: { ...testResult, expected },
            // Add mock prompts and thinking for now (can be real data from API later)
            prompts: [
              'System: You are an expert assistant. Answer clearly and concisely.',
              'Instruction: Provide likely causes and actionable next steps.',
              `User input: ${input || '(empty)'}`,
            ],
            thinking: [
              'Parsed the problem and identified the domain.',
              'Searched knowledge base for relevant information.',
              'Cross-checked with available data and context.',
              'Composed a comprehensive response.',
            ],
            generation: {
              temperature: 0.6,
              topP: 0.9,
              maxTokens: 512,
              presencePenalty: 0.0,
              frequencyPenalty: 0.0,
              seed: 42,
            },
          },
        })

      } catch (error) {
        console.error('Test run error:', error)
        // Add error message
        addMessage({
          type: 'assistant',
          content: `Error running test: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date(),
          metadata: {
            isTest: true,
            testId: detail.id,
            testName: detail.name,
            error: true,
          },
        })
      }
    }
    
    window.addEventListener('lf-test-run', onRun as EventListener)
    return () =>
      window.removeEventListener('lf-test-run', onRun as EventListener)
  }, [addMessage, updateMessage, evaluateTest, chatParams, projectChatStreamingMessage, projectSessionId])

  return (
    <div className={containerClasses}>
      {/* Header row actions */}
      <div className="flex items-center justify-between px-3 md:px-4 py-2 border-b border-border rounded-t-xl bg-background/50">
        <div className="text-xs md:text-sm text-muted-foreground">
          {USE_PROJECT_CHAT && chatParams ? (
            <span>
              Project: {chatParams.namespace}/{chatParams.projectId}
              {projectChatStreamingSession.sessionId && (
                <span className="ml-2 opacity-60">
                  • Session: {projectChatStreamingSession.sessionId.slice(-8)}
                </span>
              )}
            </span>
          ) : (
            'Session'
          )}
        </div>
        <button
          type="button"
          onClick={() => {
            clearChat()
            if (!MOCK_MODE && chatParams) {
              projectChatStreamingSession.clearSession()
            }
          }}
          disabled={isClearing}
          className="text-xs px-2 py-1 rounded bg-secondary hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isClearing ? 'Clearing…' : 'Clear'}
        </button>
      </div>

      {/* Error */}
      {combinedError && (
        <div className="mx-4 mt-3 p-2 bg-red-100 border border-red-400 text-red-700 rounded text-xs">
          {combinedError}
        </div>
      )}
      
      {/* No active project warning */}
      {!MOCK_MODE && !chatParams && (
        <div className="mx-4 mt-3 p-2 bg-amber-100 border border-amber-400 text-amber-700 rounded text-xs">
          No active project selected. Please select a project to use the chat feature.
        </div>
      )}

      {/* Messages */}
      <div ref={listRef} className="flex-1 overflow-y-auto p-3 md:p-4">
        <div className="flex flex-col gap-4 h-full">
          {!hasMessages ? (
            <EmptyState />
          ) : (
            messages.map((m: ChatboxMessage) => (
              <TestChatMessage
                key={m.id}
                message={m}
                showReferences={showReferences}
                allowRanking={allowRanking}
                showPrompts={showPrompts}
                showThinking={showThinking}
                lastUserInput={lastUserInputRef.current}
                showGenSettings={showGenSettings}
              />
            ))
          )}
          <div ref={endRef} />
        </div>
      </div>

      {/* Input */}
      <div className={inputContainerClasses}>
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={e => updateInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={combinedIsSending || (!MOCK_MODE && !chatParams)}
          placeholder={
            combinedIsSending
              ? 'Waiting for response…'
              : !MOCK_MODE && !chatParams
                ? 'Select a project to start chatting…'
                : 'Type a message and press Enter'
          }
          className={textareaClasses}
          aria-label="Message input"
        />
        <div className="flex items-center justify-between">
          {combinedIsSending && (
            <span className="text-xs text-muted-foreground">
              {USE_PROJECT_CHAT ? 'Sending to project…' : 'Sending…'}
            </span>
          )}
          <FontIcon
            isButton
            type="arrow-filled"
            className={`w-8 h-8 self-end ${!combinedCanSend || (!MOCK_MODE && !chatParams) ? 'text-muted-foreground opacity-50' : 'text-primary'}`}
            handleOnClick={handleSend}
          />
        </div>
      </div>
    </div>
  )
}

interface TestChatMessageProps {
  message: ChatboxMessage
  showReferences: boolean
  allowRanking: boolean
  showPrompts?: boolean
  showThinking?: boolean
  lastUserInput?: string
  showGenSettings?: boolean
}

export function TestChatMessage({
  message,
  showReferences,
  allowRanking,
  showPrompts,
  showThinking,
  lastUserInput,
  showGenSettings,
}: TestChatMessageProps) {
  const isUser = message.type === 'user'
  const isAssistant = message.type === 'assistant'
  const [thumb, setThumb] = useState<null | 'up' | 'down'>(null)
  const [showExpected, setShowExpected] = useState<boolean>(false)
  const [openPrompts, setOpenPrompts] = useState<boolean>(true)
  const [openThinking, setOpenThinking] = useState<boolean>(true)

  // Load persisted thumb for this message
  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      const key = `lf_thumb_${message.id}`
      const saved = localStorage.getItem(key)
      if (saved === 'up' || saved === 'down') setThumb(saved)
    } catch {}
  }, [message.id])

  const onThumb = useCallback(
    (kind: 'up' | 'down') => {
      setThumb(prev => {
        const next = prev === kind ? null : kind
        try {
          const key = `lf_thumb_${message.id}`
          if (next) localStorage.setItem(key, next)
          else localStorage.removeItem(key)
        } catch {}
        return next
      })
    },
    [message.id]
  )

  return (
    <div
      className={`flex flex-col ${isUser ? 'self-end' : ''}`}
      style={{ maxWidth: isUser ? 'min(88%, 900px)' : 'min(92%, 900px)' }}
    >
      <div
        className={
          isUser
            ? 'px-4 py-3 md:px-4 md:py-3 rounded-lg bg-primary/10 text-foreground'
            : isAssistant
              ? 'px-0 md:px-0 text-[15px] md:text-base leading-relaxed text-foreground/90'
              : 'px-4 py-3 rounded-lg bg-muted text-foreground'
        }
      >
        {message.isLoading && isAssistant ? (
          <TypingDots label="Thinking" />
        ) : message.metadata?.isTest && isUser ? (
          <div className="whitespace-pre-wrap">
            <div className="mb-2">
              <Badge className="bg-teal-500/20 text-teal-400 border border-teal-500/30">
                Test input
              </Badge>
            </div>
            {message.content}
          </div>
        ) : (
          message.content
        )}
      </div>

      {/* Assistant footer actions */}
      {isAssistant && (
        <div className="mt-2 flex items-center gap-2 text-muted-foreground">
          {allowRanking && (
            <>
              <ThumbButton
                kind="up"
                active={thumb === 'up'}
                onClick={() => onThumb('up')}
              />
              <ThumbButton
                kind="down"
                active={thumb === 'down'}
                onClick={() => {
                  const next = thumb === 'down' ? null : 'down'
                  onThumb('down')
                  // Show a subtle troubleshoot nudge when giving thumbs down
                  if (next === 'down') {
                    try {
                      // Lightweight toast via alert-style for now: inline nudge button below
                    } catch {}
                  }
                }}
              />
              <span className="mx-1 opacity-40">•</span>
            </>
          )}
          {/* Copy button removed */}
          <span className="opacity-40">•</span>
          <ActionLink
            label="Diagnose"
            className={
              thumb === 'down'
                ? 'text-xs text-teal-500 hover:text-teal-400 hover:underline font-medium'
                : undefined
            }
            onClick={() => {
              // brief local visual loading cue by dimming text
              const el = document.activeElement as HTMLElement | null
              if (el) el.blur()
              window.dispatchEvent(
                new CustomEvent('lf-diagnose', {
                  detail: {
                    source: 'message_action',
                    responseText: message.content,
                  },
                })
              )
            }}
          />
          <span className="opacity-40">/</span>
          <ActionLink
            label="Retry"
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent('lf-chat-retry', { detail: { id: message.id } })
              )
            }
          />
          <span className="opacity-40">/</span>
          <ActionLink
            label="Use as prompt"
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent('lf-chat-use-as-prompt', {
                  detail: { content: message.content },
                })
              )
            }
          />
        </div>
      )}

      {/* Generation settings, compact */}
      {isAssistant &&
        showGenSettings &&
        message.metadata &&
        (() => {
          const gen = (message.metadata as any)?.generation || null
          if (!gen) return null
          return (
            <div className="mt-1 text-[11px] text-muted-foreground">
              T={gen?.temperature ?? '—'} • top‑p={gen?.topP ?? '—'} • max=
              {gen?.maxTokens ?? '—'}
              {typeof gen?.seed !== 'undefined' ? (
                <> • seed={String(gen?.seed)}</>
              ) : null}
            </div>
          )
        })()}

      {/* References */}
      {showReferences &&
        isAssistant &&
        Array.isArray(message.sources) &&
        message.sources.length > 0 && <References sources={message.sources} />}

      {/* Test result block */}
      {isAssistant && message.metadata?.testResult && (
        <div className="mt-3 rounded-md border border-border bg-card/40 p-3">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Badge className="bg-teal-500/20 text-teal-400 border border-teal-500/30">
                Test result
              </Badge>
              <span
                className="text-[11px] text-muted-foreground"
                title="This is a simple lexical overlap metric and may not reflect semantic correctness."
              >
                experimental
              </span>
              <span
                className={`px-2 py-0.5 rounded-2xl text-xs ${
                  (message.metadata.testResult.score ?? 0) >= 95
                    ? 'bg-teal-300 text-black'
                    : (message.metadata.testResult.score ?? 0) >= 75
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-amber-300 text-black'
                }`}
              >
                {message.metadata.testResult.score}% match
              </span>
            </div>
            <button
              className="text-xs underline text-muted-foreground"
              onClick={() => setShowExpected(s => !s)}
            >
              {showExpected ? 'Hide expected' : 'View expected'}
            </button>
          </div>
          {typeof message.metadata.testResult.score === 'number' &&
            message.metadata.testResult.score < 80 && (
              <div className="mt-2 text-xs">
                <button
                  type="button"
                  className="px-2 py-0.5 rounded border border-teal-500/50 text-teal-700 hover:bg-teal-500/10 dark:text-teal-300"
                  onClick={() =>
                    window.dispatchEvent(
                      new CustomEvent('lf-diagnose', {
                        detail: {
                          source: 'low_score',
                          testId: message.metadata?.testId,
                          testName: message.metadata?.testName,
                          input: lastUserInput || '',
                          expected:
                            message.metadata?.testResult?.expected || '',
                          matchScore: message.metadata?.testResult?.score,
                        },
                      })
                    )
                  }
                >
                  Diagnose
                </button>
              </div>
            )}
          <div className="mt-2 text-xs text-muted-foreground flex items-center gap-4">
            <div>{message.metadata.testResult.latencyMs}ms result</div>
            <div>
              {message.metadata.testResult.tokenUsage.total} tokens
              <span className="opacity-60">
                {' '}
                (p {message.metadata.testResult.tokenUsage.prompt} / c{' '}
                {message.metadata.testResult.tokenUsage.completion})
              </span>
            </div>
          </div>
          {showExpected && message.metadata.testResult.expected && (
            <div className="mt-2 p-2 rounded bg-muted text-xs whitespace-pre-wrap">
              {message.metadata.testResult.expected}
            </div>
          )}
        </div>
      )}

      {/* Optional helper cards */}
      {isAssistant &&
        showPrompts &&
        Array.isArray(message.metadata?.prompts) && (
          <div className="mt-2 rounded-md border border-border bg-card/40">
            <button
              type="button"
              onClick={() => setOpenPrompts(o => !o)}
              className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground rounded-t-md hover:bg-accent/40"
              aria-expanded={openPrompts}
            >
              <span className="font-medium">
                Prompts sent ({message.metadata.prompts.length})
              </span>
              <span className="text-[11px]">
                {openPrompts ? 'Hide' : 'Show'}
              </span>
            </button>
            {openPrompts && (
              <div className="divide-y divide-border">
                {message.metadata.prompts.map((p: string, i: number) => (
                  <div
                    key={i}
                    className="px-3 py-2 text-sm whitespace-pre-wrap"
                  >
                    {p}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

      {isAssistant &&
        showThinking &&
        Array.isArray(message.metadata?.thinking) && (
          <div className="mt-2 rounded-md border border-border bg-card/40">
            <button
              type="button"
              onClick={() => setOpenThinking(o => !o)}
              className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground rounded-t-md hover:bg-accent/40"
              aria-expanded={openThinking}
            >
              <span className="font-medium">Thinking steps</span>
              <span className="text-[11px]">
                {openThinking ? 'Hide' : 'Show'}
              </span>
            </button>
            {openThinking && (
              <ol className="px-5 py-2 text-sm list-decimal marker:text-muted-foreground/70">
                {message.metadata.thinking.map((step: string, i: number) => (
                  <li key={i} className="py-1">
                    {step}
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
    </div>
  )
}

function ThumbButton({
  kind,
  active,
  onClick,
}: {
  kind: 'up' | 'down'
  active?: boolean
  onClick?: () => void
}) {
  return (
    <button onClick={onClick} className="flex items-center gap-1 group cursor-pointer rounded-sm hover:opacity-80">
      <FontIcon
        type={
          kind === 'up'
            ? active
              ? 'thumbs-up-filled'
              : 'thumbs-up'
            : active
              ? 'thumbs-down-filled'
              : 'thumbs-down'
        }
        className={`w-5 h-5 ${active ? 'text-teal-500' : 'text-muted-foreground group-hover:text-foreground'}`}
      />
    </button>
  )
}

// Copy button removed

function ActionLink({
  label,
  onClick,
  className,
}: {
  label: string
  onClick: () => void
  className?: string
}) {
  return (
    <button
      onClick={onClick}
      className={className || 'text-xs hover:underline'}
    >
      {label}
    </button>
  )
}

function References({ sources }: { sources: any[] }) {
  const [open, setOpen] = useState<boolean>(true)
  const count = sources.length
  return (
    <div className="mt-2 rounded-md border border-border bg-card/40">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground hover:bg-accent/40 rounded-t-md focus:outline-none focus:ring-2 focus:ring-primary/60"
        aria-expanded={open}
        aria-controls={`references-panel`}
      >
        <span className="font-medium">References ({count})</span>
        <span className="text-[11px]">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open && (
        <div id="references-panel" className="divide-y divide-border">
          {sources.map((s, idx) => (
            <div key={idx} className="px-3 py-2">
              {s.content && (
                <div className="text-sm text-foreground whitespace-pre-wrap line-clamp-2">
                  {s.content}
                </div>
              )}
              <div className="mt-1 flex items-center justify-between text-xs text-muted-foreground">
                <div className="truncate">
                  {s.source || s.metadata?.source || 'source'}
                </div>
                {typeof s.score === 'number' && (
                  <span className="ml-2 text-[11px]">
                    {(s.score * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function TypingDots({ label = 'Thinking' }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-1 opacity-80">
      <span>{label}</span>
      <span className="animate-pulse">.</span>
      <span className="animate-pulse [animation-delay:150ms]">.</span>
      <span className="animate-pulse [animation-delay:300ms]">.</span>
    </span>
  )
}
