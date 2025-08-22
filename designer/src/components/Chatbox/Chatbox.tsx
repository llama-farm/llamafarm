import { useEffect, useRef, useState, useCallback } from 'react'
import Message from './Message'
import FontIcon from '../../common/FontIcon'
import { useChatInference, useDeleteChatSession } from '../../hooks/useChat'
import { createChatRequest } from '../../api/chatService'

export interface Message {
  id: string
  type: 'user' | 'assistant' | 'system' | 'error'
  content: string
  sources?: any[]
  metadata?: any
  timestamp: Date
  isLoading?: boolean
}

// Generate unique session ID
const generateSessionId = (): string => {
  return `chat-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

// Generate unique message ID
const generateMessageId = (): string => {
  return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

interface ChatboxProps {
  isPanelOpen: boolean
  setIsPanelOpen: (isOpen: boolean) => void
}

function Chatbox({ isPanelOpen, setIsPanelOpen }: ChatboxProps) {
  // Session and message state
  const [sessionId, setSessionId] = useState<string>(generateSessionId())
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  
  // Refs for auto-scroll
  const listRef = useRef<HTMLDivElement | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)
  
  // API hooks
  const chatMutation = useChatInference()
  const deleteSessionMutation = useDeleteChatSession()

  useEffect(() => {
    // Scroll to bottom on mount and whenever messages change
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    } else if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [messages])

  // Add message to state
  const addMessage = useCallback((message: Omit<Message, 'id'>) => {
    const newMessage: Message = {
      ...message,
      id: generateMessageId()
    }
    setMessages(prev => [...prev, newMessage])
    return newMessage.id
  }, [])

  // Update message by ID
  const updateMessage = useCallback((id: string, updates: Partial<Message>) => {
    setMessages(prev => prev.map(msg => 
      msg.id === id ? { ...msg, ...updates } : msg
    ))
  }, [])

  // Handle sending message with API integration
  const handleSendClick = useCallback(async () => {
    const messageContent = inputValue.trim()
    if (!messageContent || chatMutation.isPending) return

    // Clear any previous errors
    setError(null)

    // Add user message immediately (optimistic update)
    addMessage({
      type: 'user',
      content: messageContent,
      timestamp: new Date()
    })

    // Clear input
    setInputValue('')

    // Add loading assistant message
    const assistantMessageId = addMessage({
      type: 'assistant',
      content: 'Thinking...',
      timestamp: new Date(),
      isLoading: true
    })

    try {
      // Create chat request
      const chatRequest = createChatRequest(messageContent)

      // Send to API
      const response = await chatMutation.mutateAsync({
        chatRequest,
        sessionId
      })

      // Update assistant message with response
      if (response.choices && response.choices.length > 0) {
        const assistantResponse = response.choices[0].message.content
        updateMessage(assistantMessageId, {
          content: assistantResponse,
          isLoading: false
        })
      } else {
        updateMessage(assistantMessageId, {
          content: 'Sorry, I didn\'t receive a proper response.',
          isLoading: false
        })
      }

    } catch (error) {
      console.error('Chat error:', error)

      // Remove loading message
      setMessages(prev => prev.filter(msg => msg.id !== assistantMessageId))

      // Set error message
      const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred'
      setError(errorMessage)

      // Add error message to chat
      addMessage({
        type: 'error',
        content: `Error: ${errorMessage}`,
        timestamp: new Date()
      })
    }
  }, [inputValue, chatMutation, sessionId, addMessage, updateMessage])

  // Handle clear chat
  const handleClearChat = useCallback(async () => {
    if (deleteSessionMutation.isPending) return

    try {
      await deleteSessionMutation.mutateAsync(sessionId)

      // Clear messages and errors
      setMessages([])
      setError(null)

      // Generate new session ID
      const newSessionId = generateSessionId()
      setSessionId(newSessionId)

    } catch (error) {
      console.error('Delete session error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to clear chat'
      setError(errorMessage)
    }
  }, [deleteSessionMutation, sessionId])

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
          <button
            onClick={handleClearChat}
            disabled={deleteSessionMutation.isPending}
            className="text-xs px-2 py-1 rounded bg-secondary hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {deleteSessionMutation.isPending ? 'Clearing...' : 'Clear'}
          </button>
        )}
        <FontIcon
          isButton
          type={isPanelOpen ? 'close-panel' : 'open-panel'}
          className="w-6 h-6 text-primary hover:opacity-80"
          handleOnClick={() => setIsPanelOpen(!isPanelOpen)}
        />
      </div>
      
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
          {messages.length === 0 ? (
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
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={chatMutation.isPending}
            className="w-full h-10 resize-none bg-transparent border-none placeholder-opacity-60 focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed overflow-hidden text-foreground placeholder-foreground/60 disabled:opacity-50"
            placeholder={chatMutation.isPending ? "Waiting for response..." : "Type here..."}
          />
          <div className="flex justify-between items-center">
            {chatMutation.isPending && (
              <span className="text-xs text-muted-foreground">Sending message...</span>
            )}
            <FontIcon
              isButton
              type="arrow-filled"
              className={`w-8 h-8 self-end ${chatMutation.isPending || !inputValue.trim() ? 'text-muted-foreground opacity-50' : 'text-primary'}`}
              handleOnClick={handleSendClick}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default Chatbox
