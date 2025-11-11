import { useEffect, useRef, useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Message from './Message'
import FontIcon from '../../common/FontIcon'
import { useChatbox } from '../../hooks/useChatbox'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useMobileView } from '../../contexts/MobileViewContext'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../ui/tooltip'
import { ClassifiedError } from '../../types/chat'
import { getHealthSummary } from '../../utils/recoveryCommands'

interface ChatboxProps {
  isPanelOpen: boolean
  setIsPanelOpen: (isOpen: boolean) => void
  initialMessage?: string | null
}

/**
 * Copy button component for command boxes
 */
function CopyButton({ text, onCopy }: { text: string; onCopy?: () => void }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      onCopy?.()
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  return (
    <>
      <button
        onClick={handleCopy}
        className="p-1 hover:bg-muted/50 rounded transition-colors"
        title={copied ? 'Copied!' : 'Copy to clipboard'}
        aria-label={copied ? 'Copied to clipboard' : 'Copy to clipboard'}
      >
        <FontIcon
          type={copied ? 'checkmark-filled' : 'copy'}
          className={`w-4 h-4 ${copied ? 'text-green-500' : 'text-muted-foreground'}`}
        />
      </button>
      {/* Screen reader announcement */}
      <span aria-live="polite" className="sr-only">
        {copied ? 'Command copied to clipboard.' : ''}
      </span>
    </>
  )
}

function Chatbox({
  isPanelOpen,
  setIsPanelOpen,
  initialMessage,
}: ChatboxProps) {
  const navigate = useNavigate()
  const { markUserChoice } = useMobileView()
  const [isDiagnosing, setIsDiagnosing] = useState<boolean>(false)
  const [hasProcessedInitialMessage, setHasProcessedInitialMessage] =
    useState(false)
  const diagnosingInFlightRef = useRef<boolean>(false)
  const [isTooltipOpen, setIsTooltipOpen] = useState<boolean>(false)
  // Use the unified chatbox hook with project session management for Designer Chat
  const {
    messages,
    inputValue,
    error,
    isSending,
    isStreaming,
    isClearing,
    sendMessage,
    clearChat,
    updateInput,
    cancelStreaming,
    hasMessages,
    canSend,
    sessionId,
  } = useChatbox({ useProjectSession: true })

  const activeProject = useActiveProject()
  const activeProjectName = activeProject?.project || ''
  // Session list UI removed: single session per project

  // Refs for auto-scroll
  const listRef = useRef<HTMLDivElement | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)
  const rafRef = useRef<number | null>(null)

  // Sticky scroll state
  const BOTTOM_THRESHOLD = 24 // pixels from bottom to consider "at bottom"
  const [isUserAtBottom, setIsUserAtBottom] = useState(true)
  const [wantsAutoScroll, setWantsAutoScroll] = useState(true)

  // Check if user is at the bottom of the scroll container
  const checkIfAtBottom = useCallback(() => {
    if (!listRef.current) return false
    const { scrollTop, scrollHeight, clientHeight } = listRef.current
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight
    return distanceFromBottom <= BOTTOM_THRESHOLD
  }, [BOTTOM_THRESHOLD])

  // Handle scroll events with RAF debouncing
  const handleScroll = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
    }

    rafRef.current = requestAnimationFrame(() => {
      const atBottom = checkIfAtBottom()
      setIsUserAtBottom(atBottom)

      // If user scrolled back to bottom, resume auto-scroll
      if (atBottom) {
        setWantsAutoScroll(true)
      } else {
        // User scrolled up, pause auto-scroll
        setWantsAutoScroll(false)
      }
    })
  }, [checkIfAtBottom])

  // Cleanup RAF on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
      }
    }
  }, [])

  // Handle window resize - recompute scroll position
  useEffect(() => {
    const handleResize = () => {
      // Recompute if we're at bottom after resize
      const atBottom = checkIfAtBottom()
      setIsUserAtBottom(atBottom)
      
      // If we were wanting to auto-scroll and we're now at bottom, maintain that
      if (wantsAutoScroll && atBottom && listRef.current) {
        listRef.current.scrollTo({
          top: listRef.current.scrollHeight,
          behavior: 'auto',
        })
      }
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [checkIfAtBottom, wantsAutoScroll])

  // Conditional auto-scroll effect
  useEffect(() => {
    if (wantsAutoScroll && listRef.current) {
      // Use scrollTo for better control during streaming
      listRef.current.scrollTo({
        top: listRef.current.scrollHeight,
        behavior: 'auto', // 'auto' prevents jank during streaming
      })
    }
  }, [messages, wantsAutoScroll])

  // Handle initial message from home page project creation
  useEffect(() => {
    if (
      initialMessage &&
      !hasProcessedInitialMessage &&
      !hasMessages && // Only if no existing messages
      sessionId === null // Only if no existing session
    ) {
      // Use the existing sendMessage function - this will trigger normal session creation
      const briefKickoff = `${initialMessage}\n\nPlease answer briefly (one or two sentences).`
      sendMessage(briefKickoff)
      setHasProcessedInitialMessage(true)
    }
  }, [
    initialMessage,
    hasProcessedInitialMessage,
    hasMessages,
    sessionId,
    sendMessage,
  ])

  // Handle sending message
  const handleSendClick = useCallback(async () => {
    const messageContent = inputValue.trim()
    if (!canSend || !messageContent) return

    // Optionally scroll to bottom when sending a new message
    setWantsAutoScroll(true)
    setIsUserAtBottom(true)

    // Send message using the hook
    const success = await sendMessage(messageContent)

    // Clear input on successful send
    if (success) {
      updateInput('')
    }
  }, [inputValue, canSend, sendMessage, updateInput])

  // Handle "Jump to latest" button click
  const handleJumpToLatest = useCallback(() => {
    if (listRef.current) {
      listRef.current.scrollTo({
        top: listRef.current.scrollHeight,
        behavior: 'smooth',
      })
      setWantsAutoScroll(true)
      setIsUserAtBottom(true)
    }
  }, [])

  // Handle clear chat
  const handleClearChat = useCallback(async () => {
    await clearChat()
  }, [clearChat])

  const handleKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendClick()
    }
  }

  // Compose an auto-message for diagnose intents
  const composeDiagnoseMessage = useCallback((detail: any) => {
    const lines: string[] = []
    lines.push(
      'Help me diagnose this. Identify issues and propose fixes (prompt, retrieval, tools, or settings).'
    )
    if (detail?.source === 'low_score') {
      const scoreText =
        typeof detail?.matchScore === 'number' ? `${detail.matchScore}%` : '—'
      const testLabel = detail?.testName ? `"${detail.testName}"` : ''
      lines.push(`Context: Test ${testLabel} scored ${scoreText}.`)
    }
    if (
      detail?.source === 'message_action' ||
      detail?.source === 'thumbs_down'
    ) {
      lines.push('Context: Diagnosing a specific response.')
    }
    if (detail?.input) {
      lines.push(`Input:\n${detail.input}`)
    }
    if (detail?.expected) {
      lines.push(`Expected:\n${detail.expected}`)
    }
    if (detail?.responseText) {
      lines.push(`Response:\n${detail.responseText}`)
    }
    // Note: We intentionally do not dump prompts/thinking here.
    return lines.join('\n\n')
  }, [])

  // Global diagnose listener: open panel and auto-send composed message
  useEffect(() => {
    const onDiagnose = async (e: Event) => {
      const detail = (e as CustomEvent).detail as any
      try {
        // Prevent overlapping diagnoses if events fire rapidly
        if (diagnosingInFlightRef.current) return
        diagnosingInFlightRef.current = true
        setIsDiagnosing(true)
        setIsPanelOpen(true)
        const message = composeDiagnoseMessage(detail)
        if (message && message.trim()) {
          await sendMessage(message)
          updateInput('')
        }
        // keep the diagnose indicator visible briefly
        setTimeout(() => {
          setIsDiagnosing(false)
          diagnosingInFlightRef.current = false
        }, 800)
      } catch {}
    }

    window.addEventListener('lf-diagnose', onDiagnose as EventListener)
    return () =>
      window.removeEventListener('lf-diagnose', onDiagnose as EventListener)
  }, [sendMessage, updateInput, setIsPanelOpen, composeDiagnoseMessage])

  return (
    <TooltipProvider>
      <div className="w-full h-full flex flex-col transition-colors bg-card text-foreground">
        {isPanelOpen && (
          <div className="px-4 pt-3 pb-2 border-b border-border">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FontIcon
                  type="build-assistant"
                  className="w-5 h-5 text-foreground"
                />
                <span className="text-base font-medium">Build assistant</span>
                <Tooltip open={isTooltipOpen} onOpenChange={setIsTooltipOpen}>
                  <TooltipTrigger asChild>
                    <button
                      className="inline-flex items-center"
                      onClick={() => setIsTooltipOpen(!isTooltipOpen)}
                    >
                      <FontIcon
                        type="info"
                        className="w-4 h-4 text-muted-foreground hover:text-foreground"
                      />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-[280px]">
                    <p>
                      This chat helps you build and configure your project. To
                      test your AI project outputs, open the{' '}
                      <button
                        onClick={() => {
                          markUserChoice('project')
                          navigate('/chat/test', {
                            state: { focusInput: true },
                          })
                          setIsTooltipOpen(false)
                        }}
                        className="font-semibold underline hover:opacity-80"
                      >
                        Test tab
                      </button>
                      .
                    </p>
                  </TooltipContent>
                </Tooltip>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleClearChat}
                  disabled={isClearing}
                  className="text-xs px-2 py-1 rounded bg-secondary hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isClearing ? 'Clearing...' : 'Clear'}
                </button>
                <div className="hidden md:flex md:items-center">
                  <FontIcon
                    isButton
                    type="close-panel"
                    className="w-6 h-6 text-primary hover:opacity-80"
                    handleOnClick={() => setIsPanelOpen(false)}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {!isPanelOpen && (
          <div className="relative flex justify-center mt-3">
            <div className="hidden md:block">
              <FontIcon
                isButton
                type="open-panel"
                className="w-6 h-6 text-primary hover:opacity-80"
                handleOnClick={() => setIsPanelOpen(true)}
              />
            </div>
          </div>
        )}

        {/* Centered project title on mobile */}
        {isPanelOpen && (
          <span className="md:hidden absolute left-1/2 top-3 -translate-x-1/2 text-sm text-muted-foreground truncate max-w-[60vw] pointer-events-none z-10">
            {activeProjectName}
          </span>
        )}
        {isDiagnosing && (
          <div className="mx-4 mb-2 flex items-center gap-2 text-xs text-muted-foreground">
            <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-teal-500 border-t-transparent" />
            <span>Diagnosing…</span>
          </div>
        )}

        {/* Enhanced error banner with recovery commands */}
        {error && isPanelOpen && (() => {
          // Handle both ClassifiedError and legacy string errors
          const classifiedError = typeof error === 'string' 
            ? null 
            : (error as ClassifiedError)
          const errorTitle = classifiedError?.title || 'Error'
          const errorMessage = classifiedError?.message || (typeof error === 'string' ? error : 'An error occurred')
          const recoveryCommands = classifiedError?.recoveryCommands || []
          const healthStatus = classifiedError?.healthStatus

          return (
            <div className="mx-4 mb-2 rounded-xl border-2 border-red-500/40 bg-card/40 p-3">
              {/* Error header */}
              <div className="flex items-start gap-3 mb-2">
                <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-red-500/15 text-red-400 border border-red-500/30 flex-shrink-0 mt-0.5">
                  <FontIcon type="info" className="w-4 h-4" />
                </span>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-foreground text-sm">
                    {errorTitle}
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    {errorMessage}
                  </div>
                  {/* Health summary */}
                  {healthStatus && (
                    <div className="text-xs text-muted-foreground mt-1 italic">
                      {getHealthSummary(healthStatus)}
                    </div>
                  )}
                </div>
              </div>

              {/* Recovery commands */}
              {recoveryCommands.length > 0 && (
                <div className="space-y-2">
                  <div className="text-xs text-muted-foreground font-medium">
                    {recoveryCommands.length === 1 ? 'Fix:' : 'Try these steps:'}
                  </div>
                  {recoveryCommands.map((cmd, idx) => (
                    <div key={idx} className="space-y-1">
                      {cmd.description && (
                        <div className="text-xs text-muted-foreground">
                          {recoveryCommands.length > 1 ? `${idx + 1}. ${cmd.description}` : cmd.description}
                        </div>
                      )}
                      <div className="bg-muted/50 border border-border rounded-lg px-3 py-2 flex items-center justify-between gap-2">
                        <code className="font-mono text-sm text-foreground flex-1 overflow-x-auto">
                          {cmd.command}
                        </code>
                        <CopyButton text={cmd.command} />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Action buttons */}
              <div className="flex items-center gap-2 mt-3">
                <button
                  onClick={() =>
                    window.open(
                      'https://github.com/llama-farm/llamafarm#quickstart',
                      '_blank'
                    )
                  }
                  className="text-xs px-2 py-1 rounded bg-secondary hover:bg-secondary/80"
                >
                  View Docs
                </button>
              </div>
            </div>
          )
        })()}

        <div
          className={`flex flex-col h-full p-4 pt-2 overflow-hidden ${isPanelOpen ? 'flex' : 'hidden'}`}
        >
          <div className="relative flex-1 min-h-0">
            <div
              ref={listRef}
              onScroll={handleScroll}
              className="absolute inset-0 overflow-y-auto flex flex-col gap-5 pr-1"
            >
              {!hasMessages ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40 max-w-[560px]">
                    <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-primary/15 border border-primary/30">
                      <span className="text-primary text-lg">💬</span>
                    </div>
                    <div className="text-base font-medium">
                      Start a conversation
                    </div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      Type a message below to chat with your model.
                    </div>
                    {error && (
                      <div className="mt-3 text-xs text-red-400">
                        Set up a project config first to get responses.
                      </div>
                    )}
                    <div className="mt-3 text-xs text-muted-foreground">
                      Tip: Press Enter to send
                    </div>
                  </div>
                </div>
              ) : (
                messages.map(message => (
                  <Message key={message.id} message={message} />
                ))
              )}
              <div ref={endRef} />
            </div>

            {/* Jump to latest pill - shown when user scrolls up */}
            {!isUserAtBottom && hasMessages && (
              <button
                onClick={handleJumpToLatest}
                className="absolute bottom-4 right-6 z-10 flex items-center gap-2 px-3 py-2 rounded-full bg-primary/90 hover:bg-primary text-primary-foreground shadow-lg transition-all hover:shadow-xl"
                aria-label="Jump to latest message"
              >
                <span className="text-sm font-medium">Jump to latest</span>
                <svg
                  viewBox="0 0 24 24"
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="7 13 12 18 17 13" />
                  <polyline points="7 6 12 11 17 6" />
                </svg>
              </button>
            )}
          </div>
          <div className="flex flex-col gap-3 p-3 rounded-lg bg-secondary mt-auto sticky bottom-4">
            <div className="relative">
              <textarea
                value={inputValue}
                onChange={e => updateInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isSending}
                className="w-full h-10 pr-10 resize-none bg-transparent border-none placeholder-opacity-60 focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed overflow-hidden text-foreground placeholder-foreground/60 disabled:opacity-50"
                placeholder={
                  isStreaming
                    ? 'Streaming response...'
                    : isSending
                      ? 'Waiting for response...'
                      : 'Type here...'
                }
              />
              {/* Action button overlays top-right inside the input area, same spot for send/stop */}
              {isStreaming ? (
                <button
                  onClick={cancelStreaming}
                  className="absolute right-2 top-2 z-10 w-8 h-8 rounded-full flex items-center justify-center text-primary hover:opacity-80"
                  aria-label="Stop response"
                >
                  <svg
                    viewBox="0 0 24 24"
                    className="w-6 h-6"
                    aria-hidden="true"
                  >
                    <rect
                      x="6"
                      y="6"
                      width="12"
                      height="12"
                      rx="2"
                      className="fill-current"
                    />
                  </svg>
                </button>
              ) : (
                <FontIcon
                  isButton
                  type="arrow-filled"
                  className={`absolute right-2 top-2 z-10 w-8 h-8 ${
                    inputValue.trim().length === 0
                      ? 'text-muted-foreground opacity-50 cursor-not-allowed'
                      : 'text-primary hover:opacity-80'
                  }`}
                  handleOnClick={handleSendClick}
                />
              )}
            </div>
            <div className="flex justify-between items-center">
              {(isSending || isStreaming) && (
                <span className="text-xs text-muted-foreground flex items-center gap-2">
                  {isStreaming ? (
                    <>
                      <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-teal-500" />
                      Streaming response...
                    </>
                  ) : (
                    'Sending message...'
                  )}
                </span>
              )}
            </div>
          </div>

          {/* Footer */}
          <div className="pt-3 text-sm text-muted-foreground flex items-center justify-center gap-2 flex-wrap">
            <span>Testing project outputs? Open the</span>
            <button
              onClick={() => {
                markUserChoice('project')
                navigate('/chat/test', { state: { focusInput: true } })
              }}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-teal-500/10 hover:bg-teal-500/20 transition-colors border border-teal-500/30"
            >
              <FontIcon
                type="test"
                className="w-4 h-4 text-teal-600 dark:text-teal-400"
              />
              <span className="text-teal-700 dark:text-teal-300 font-medium">
                Test
              </span>
            </button>
            <span>tab.</span>
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}

export default Chatbox
