import { useEffect, useRef, useCallback, useState } from 'react'
import Message from './Message'
import FontIcon from '../../common/FontIcon'
import useChatboxWithProjectSession from '../../hooks/useChatboxWithProjectSession'
import { useActiveProject } from '../../hooks/useActiveProject'

interface ChatboxProps {
  isPanelOpen: boolean
  setIsPanelOpen: (isOpen: boolean) => void
  initialMessage?: string | null
}

function Chatbox({
  isPanelOpen,
  setIsPanelOpen,
  initialMessage,
}: ChatboxProps) {
  const [isDiagnosing, setIsDiagnosing] = useState<boolean>(false)
  const [hasProcessedInitialMessage, setHasProcessedInitialMessage] =
    useState(false)
  const diagnosingInFlightRef = useRef<boolean>(false)
  // Use the enhanced chatbox hook with project session management for Designer Chat
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
  } = useChatboxWithProjectSession()

  const activeProject = useActiveProject()
  const activeProjectName = activeProject?.project || ''

  // Refs for auto-scroll
  const listRef = useRef<HTMLDivElement | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    // Scroll to bottom on mount and whenever messages change
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    } else if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [messages])

  // Handle initial message from home page project creation
  useEffect(() => {
    if (
      initialMessage &&
      !hasProcessedInitialMessage &&
      !hasMessages && // Only if no existing messages
      sessionId === null // Only if no existing session
    ) {
      // Use the existing sendMessage function - this will trigger normal session creation
      sendMessage(initialMessage)
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

    // Send message using the hook
    const success = await sendMessage(messageContent)

    // Clear input on successful send
    if (success) {
      updateInput('')
    }
  }, [inputValue, canSend, sendMessage, updateInput])

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
    <div className="w-full h-full flex flex-col transition-colors bg-card text-foreground">
      <div
        className={`relative flex ${isPanelOpen ? 'justify-between items-center mr-1 mt-1' : 'justify-center mt-3'}`}
      >
        <div className="flex items-center gap-2">
          {isPanelOpen && (
            <button
              onClick={handleClearChat}
              disabled={isClearing}
              className="text-xs px-2 py-1 rounded bg-secondary hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isClearing ? 'Clearing...' : 'Clear'}
            </button>
          )}
        </div>
        {/* Centered project title on mobile */}
        <span className="md:hidden absolute left-1/2 -translate-x-1/2 text-sm text-muted-foreground truncate max-w-[60vw] pointer-events-none">
          {activeProjectName}
        </span>
        {/* Hide collapse toggle on mobile when chat is full-screen */}
        <div className="hidden md:block">
          <FontIcon
            isButton
            type={isPanelOpen ? 'close-panel' : 'open-panel'}
            className="w-6 h-6 text-primary hover:opacity-80"
            handleOnClick={() => setIsPanelOpen(!isPanelOpen)}
          />
        </div>
      </div>
      {isDiagnosing && (
        <div className="mx-4 mb-2 flex items-center gap-2 text-xs text-muted-foreground">
          <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-teal-500 border-t-transparent" />
          <span>Diagnosing…</span>
        </div>
      )}

      {/* Error display */}
      {error && isPanelOpen && (
        <div className="mx-4 mb-2 p-2 bg-red-100 border border-red-400 text-red-700 rounded text-sm">
          {error}
        </div>
      )}

      <div
        className={`flex flex-col h-full p-4 pt-2 overflow-hidden ${isPanelOpen ? 'flex' : 'hidden'}`}
      >
        <div
          ref={listRef}
          className="flex-1 overflow-y-auto flex flex-col gap-5 pr-1"
        >
          {!hasMessages ? (
            <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
              Start a conversation...
            </div>
          ) : (
            messages.map(message => (
              <Message key={message.id} message={message} />
            ))
          )}
          <div ref={endRef} />
        </div>
        <div className="flex flex-col gap-3 p-3 rounded-lg bg-secondary mt-auto sticky bottom-4">
          <textarea
            value={inputValue}
            onChange={e => updateInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isSending}
            className="w-full h-10 resize-none bg-transparent border-none placeholder-opacity-60 focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed overflow-hidden text-foreground placeholder-foreground/60 disabled:opacity-50"
            placeholder={
              isStreaming
                ? 'Streaming response...'
                : isSending
                  ? 'Waiting for response...'
                  : 'Type here...'
            }
          />
          <div className="flex justify-between items-center">
            {(isSending || isStreaming) && (
              <span className="text-xs text-muted-foreground flex items-center gap-2">
                {isStreaming ? (
                  <>
                    <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-teal-500" />
                    Streaming response...
                    <button
                      onClick={cancelStreaming}
                      className="ml-2 text-xs px-2 py-1 rounded bg-red-500 hover:bg-red-600 text-white"
                    >
                      Cancel
                    </button>
                  </>
                ) : (
                  'Sending message...'
                )}
              </span>
            )}
            <FontIcon
              isButton
              type="arrow-filled"
              className={`w-8 h-8 self-end ${!canSend ? 'text-muted-foreground opacity-50' : 'text-primary'}`}
              handleOnClick={handleSendClick}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default Chatbox
