import { useEffect, useRef, useCallback } from 'react'
import Message from './Message'
import FontIcon from '../../common/FontIcon'
import useChatbox from '../../hooks/useChatbox'

interface ChatboxProps {
  isPanelOpen: boolean
  setIsPanelOpen: (isOpen: boolean) => void
}

function Chatbox({ isPanelOpen, setIsPanelOpen }: ChatboxProps) {
  // Use the custom chatbox hook for all chat logic
  const {
    messages,
    inputValue,
    error,
    isSending,
    isClearing,
    isStreaming,
    sendMessage,
    clearChat,
    updateInput,
    hasMessages,
    canSend,
    hasActiveProject,
    projectInfo,
    abortStreaming
  } = useChatbox()
  

  
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
  
  // Handle abort streaming
  const handleAbortStreaming = useCallback(() => {
    abortStreaming()
  }, [abortStreaming])

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

  return (
    <div className="w-full h-full flex flex-col transition-colors bg-card text-foreground">
      <div
        className={`flex ${isPanelOpen ? 'justify-between items-center mr-1 mt-1' : 'justify-center mt-3'}`}
      >
        {isPanelOpen && (
          <div className="flex items-center gap-2">
            <button
              onClick={handleClearChat}
              disabled={isClearing}
              className="text-xs px-2 py-1 rounded bg-secondary hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isClearing ? 'Clearing...' : 'Clear'}
            </button>
            {isStreaming && (
              <button
                onClick={handleAbortStreaming}
                className="text-xs px-2 py-1 rounded bg-red-100 hover:bg-red-200 text-red-700"
              >
                Stop
              </button>
            )}
          </div>
        )}
        <FontIcon
          isButton
          type={isPanelOpen ? 'close-panel' : 'open-panel'}
          className="w-6 h-6 text-primary hover:opacity-80"
          handleOnClick={() => setIsPanelOpen(!isPanelOpen)}
        />
      </div>
      
      {/* Project info header */}
      {isPanelOpen && (
        <div className="px-4 py-2 border-b border-border">
          {hasActiveProject && projectInfo ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>📁</span>
              <span>Project: {projectInfo}</span>
            </div>
          ) : (
            <div className="text-sm text-muted-foreground text-center py-1">
              Please select a project to start chatting
            </div>
          )}
        </div>
      )}
      
      {/* Error display */}
      {error && isPanelOpen && (
        <div className="mx-4 mb-2 p-2 bg-red-100 border border-red-400 text-red-700 rounded text-sm">
          {error}
        </div>
      )}
      
      <div
        className={`flex flex-col h-full p-4 overflow-hidden ${isPanelOpen ? 'flex' : 'hidden'}`}
      >
        <div
          ref={listRef}
          className="flex-1 overflow-y-auto flex flex-col gap-5 pr-1"
        >
          {!hasActiveProject ? (
            <div className="flex items-center justify-center h-full text-muted-foreground text-sm text-center">
              <div>
                <div className="mb-2">Please select a project to start chatting</div>
                <div className="text-xs opacity-75">
                  Chat requires an active project for context and session management
                </div>
              </div>
            </div>
          ) : !hasMessages ? (
            <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
              Start a conversation...
            </div>
          ) : (
            messages.map((message) => (
              <Message key={message.id} message={message} />
            ))
          )}
          <div ref={endRef} />
        </div>
        <div className="flex flex-col gap-3 p-3 rounded-lg bg-secondary">
          <textarea
            value={inputValue}
            onChange={e => updateInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isSending || !hasActiveProject}
            aria-disabled={isSending || !hasActiveProject}
            aria-label={
              !hasActiveProject 
                ? "Chat input - Please select a project first" 
                : isSending 
                  ? "Chat input - Sending message..."
                  : "Chat input - Type your message here"
            }
            aria-describedby="chat-input-help"
            className="w-full h-10 resize-none bg-transparent border-none placeholder-opacity-60 focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed overflow-hidden text-foreground placeholder-foreground/60 disabled:opacity-50"
            placeholder={
              !hasActiveProject 
                ? "Select a project to start chatting..." 
                : isSending 
                  ? isStreaming 
                    ? "Streaming response..." 
                    : "Waiting for response..."
                  : "Type here..."
            }
          />
          {!hasActiveProject && (
            <div id="chat-input-help" className="text-xs text-muted-foreground sr-only">
              You must select a project before you can start chatting. Use the project switcher in the header to choose a project.
            </div>
          )}
          <div className="flex justify-between items-center">
            {isSending && (
              <span className="text-xs text-muted-foreground">
                {isStreaming ? (
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1">
                      <div className="w-1 h-1 bg-current rounded-full animate-pulse"></div>
                      <div className="w-1 h-1 bg-current rounded-full animate-pulse [animation-delay:0.2s]"></div>
                      <div className="w-1 h-1 bg-current rounded-full animate-pulse [animation-delay:0.4s]"></div>
                    </div>
                    <span>Thinking...</span>
                  </div>
                ) : (
                  "Sending message..."
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
