import { Message as MessageType } from './Chatbox'

export interface MessageProps {
  message: MessageType
}

const Message: React.FC<MessageProps> = ({ message }) => {
  const { type, content } = message

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
      <div className={getContentStyles()}>{content}</div>
    </div>
  )
}

export default Message
