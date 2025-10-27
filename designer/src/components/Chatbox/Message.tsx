import { ChatboxMessage } from '../../types/chatbox'
import Markdown from '../../utils/renderMarkdown'

export interface MessageProps {
  message: ChatboxMessage
}

const Message: React.FC<MessageProps> = ({ message }) => {
  const { type, content, isLoading, isStreaming, cancelled } = message

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
      case 'system':
        return `${baseBubble} bg-green-500 text-white rounded-2xl border-green-500 italic`
      case 'error':
        return `${baseBubble} bg-red-500 text-white rounded-2xl rounded-bl-sm border-red-500`
      default:
        return `${baseBubble} bg-muted text-foreground`
    }
  }

  return (
    <div className={getMessageStyles()}>
      <div className={getContentStyles()}>
        {type === 'assistant' ? (
          isLoading ? (
            <span className="italic opacity-70">{content}</span>
          ) : (
            <Markdown
              className="leading-relaxed break-words [&>p]:my-2 [&>h1]:mt-4 [&>h1]:mb-2 [&>h2]:mt-3 [&>h2]:mb-2 [&>h3]:mt-2 [&>h3]:mb-1 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-1"
              content={content}
            />
          )
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
