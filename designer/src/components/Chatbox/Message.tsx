import { useState, useEffect, useMemo } from 'react'
import { ChatboxMessage } from '../../types/chatbox'
import Markdown from '../../utils/renderMarkdown'
import { parseMessageContentMemo } from '../../utils/messageParser'

export interface MessageProps {
  message: ChatboxMessage
}

const Message: React.FC<MessageProps> = ({ message }) => {
  const { type, content, isLoading, isStreaming, cancelled } = message
  const [isArgsExpanded, setIsArgsExpanded] = useState(false)
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(true) // Always expanded by default

  // Reset expansion state when message changes
  useEffect(() => {
    setIsArgsExpanded(false)
    setIsThinkingExpanded(true) // Always start expanded
  }, [message.id])

  // Parse message content using memoized utility
  const { thinking, contentWithoutThinking } = useMemo(
    () => parseMessageContentMemo(content, type, message.id),
    [content, type, message.id]
  )

  // Show typing indicator while assistant is preparing/streaming with no content yet
  const showTypingIndicator =
    type === 'assistant' &&
    (isLoading || isStreaming) &&
    (!content || content.trim() === '' || content === 'Thinking...')

  const getMessageStyles = (): string => {
    const baseStyles = 'flex flex-col mb-4'

    switch (type) {
      case 'user':
        return `${baseStyles} self-end max-w-[80%] md:max-w-[90%]`
      default:
        return baseStyles
    }
  }

  const getContentStyles = (): string => {
    const baseBubble = 'px-4 py-3 md:px-4 md:py-3 rounded-lg'

    switch (type) {
      case 'user':
        return `${baseBubble} bg-secondary text-foreground text-base leading-relaxed`
      case 'assistant':
        return 'text-[15px] md:text-base leading-relaxed text-foreground/90'
      case 'tool':
        return `${baseBubble} bg-muted/50 border border-border`
      case 'system':
        return `${baseBubble} bg-green-500 text-white rounded-2xl border-green-500 italic`
      case 'error':
        return `${baseBubble} bg-red-500 text-white rounded-2xl rounded-bl-sm border-red-500`
      default:
        return `${baseBubble} bg-muted text-foreground`
    }
  }

  // Parse tool call content for better display
  const parseToolContent = (content: string) => {
    const toolMatch = content.match(
      /🔧 Calling tool: (.+?)(?:\n\nArguments: (.+))?$/s
    )
    if (toolMatch) {
      return {
        toolName: toolMatch[1],
        arguments: toolMatch[2] || null,
      }
    }
    return null
  }

  return (
    <div className={getMessageStyles()}>
      {/* Thinking steps section - collapsible and smaller font */}
      {type === 'assistant' && thinking && !isLoading && (
        <div className="mb-2 rounded-md border border-border bg-card/40">
          <button
            type="button"
            onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
            className="w-full flex items-center justify-between px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted/30 transition-colors"
            aria-expanded={isThinkingExpanded}
          >
            <span className="font-medium">Thinking steps</span>
            <span className="text-[11px]">
              {isThinkingExpanded ? 'Hide' : 'Show'}
            </span>
          </button>
          {isThinkingExpanded && (
            <div className="px-3 py-2 border-t border-border">
              <Markdown
                className="text-xs leading-relaxed text-muted-foreground break-words [&>p]:my-1 [&>p:first-child]:mt-0 [&>p:last-child]:mb-0 [&>h1]:mt-2 [&>h1]:mb-1 [&>h2]:mt-2 [&>h2]:mb-1 [&>h3]:mt-1 [&>h3]:mb-0.5 [&>ul]:my-1 [&>ol]:my-1 [&>li]:my-0.5"
                content={thinking}
              />
            </div>
          )}
        </div>
      )}
      
      <div className={getContentStyles()}>
        {type === 'assistant' ? (
          isLoading ? (
            <span className="italic opacity-70">{content}</span>
          ) : (
            <Markdown
              className="leading-relaxed break-words [&>p]:my-2 [&>p:first-child]:mt-0 [&>p:last-child]:mb-0 [&>h1]:mt-4 [&>h1]:mb-2 [&>h2]:mt-3 [&>h2]:mb-2 [&>h3]:mt-2 [&>h3]:mb-1 [&>ul]:my-2 [&>ol]:my-2"
              content={contentWithoutThinking}
            />
          )
        ) : type === 'tool' ? (
          (() => {
            const parsed = parseToolContent(content)
            if (parsed) {
              let parsedArgs = null
              try {
                if (parsed.arguments) {
                  parsedArgs = JSON.parse(parsed.arguments)
                }
              } catch {
                // If JSON parsing fails, use raw string
                parsedArgs = parsed.arguments
              }

              return (
                <div className="flex flex-col gap-2">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">🔧</span>
                      <div className="flex flex-col">
                        <span className="text-[10px] uppercase tracking-wide text-muted-foreground/60 font-medium">
                          Tool call
                        </span>
                        <span className="text-sm font-semibold text-foreground">
                          {parsed.toolName}
                        </span>
                      </div>
                    </div>
                    {parsed.arguments && (
                      <button
                        onClick={() => setIsArgsExpanded(!isArgsExpanded)}
                        className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
                      >
                        <span className="font-medium">
                          {isArgsExpanded ? 'Hide' : 'Show'} details
                        </span>
                        <svg
                          className={`w-3 h-3 transition-transform ${isArgsExpanded ? 'rotate-90' : ''}`}
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 5l7 7-7 7"
                          />
                        </svg>
                      </button>
                    )}
                  </div>
                  {parsed.arguments && isArgsExpanded && (
                    <pre className="text-[11px] leading-relaxed bg-card border border-border rounded px-2.5 py-2 overflow-x-auto text-foreground/80 font-mono">
                      {typeof parsedArgs === 'object' && parsedArgs !== null
                        ? JSON.stringify(parsedArgs, null, 2)
                        : parsed.arguments}
                    </pre>
                  )}
                </div>
              )
            }
            // Fallback to plain text if parsing fails
            return (
              <span className="whitespace-pre-wrap text-sm text-muted-foreground">
                {content}
              </span>
            )
          })()
        ) : (
          <span className="whitespace-pre-wrap">{content}</span>
        )}
        {cancelled && type === 'assistant' && (
          <div className="mt-1 text-xs text-muted-foreground">
            (response cancelled)
          </div>
        )}
        {showTypingIndicator && (
          <span
            className="inline-flex items-center gap-1 ml-1 align-middle"
            aria-label="Assistant is typing"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-foreground/70 animate-bounce [animation-delay:-0.2s]" />
            <span className="w-1.5 h-1.5 rounded-full bg-foreground/70 animate-bounce [animation-delay:-0.1s]" />
            <span className="w-1.5 h-1.5 rounded-full bg-foreground/70 animate-bounce" />
          </span>
        )}
      </div>
    </div>
  )
}

export default Message
