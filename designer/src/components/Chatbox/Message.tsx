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
        return `${baseBubble} bg-[#F4F4F4] text-[#252525] dark:bg-blue-600 dark:text-white text-base leading-relaxed`
      case 'assistant':
        // Light mode: dark text; Dark mode: softer near-white text
        return 'text-[15px] md:text-base leading-relaxed text-gray-800 dark:text-white/90'
      case 'system':
        return `${baseBubble} bg-green-500 text-white rounded-2xl border-green-500 italic`
      case 'error':
        return `${baseBubble} bg-red-500 text-white rounded-2xl rounded-bl-sm border-red-500`
      default:
        return `${baseBubble} bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200`
    }
  }

  return (
    <div className={getMessageStyles()}>
      <div className={getContentStyles()}>{content}</div>
    </div>
  )
}

export default Message
