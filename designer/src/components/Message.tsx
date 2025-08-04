import { Message as MessageType } from './Chatbox'

export interface MessageProps {
  message: MessageType
}

const Message: React.FC<MessageProps> = ({ message }) => {
  const { type, content } = message

  const getMessageStyles = (): string => {
    const baseStyles = 'flex flex-col  mb-4'

    switch (type) {
      case 'user':
        return `${baseStyles} self-end max-w-[80%] md:max-w-[90%]`
      default:
        return baseStyles
    }
  }

  const getContentStyles = (): string => {
    const baseStyles = 'px-3 py-2 md:px-3 md:py-2'

    switch (type) {
      case 'user':
        return `${baseStyles} bg-blue-600 text-white rounded-lg`
      case 'assistant':
        return `bg-transparent`
      case 'system':
        return `${baseStyles} bg-green-500 text-white rounded-2xl border-green-500 italic`
      case 'error':
        return `${baseStyles} bg-red-500 text-white rounded-2xl rounded-bl-sm border-red-500`
      default:
        return `${baseStyles} bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200 rounded-2xl`
    }
  }

  return (
    <div className={getMessageStyles()}>
      <div className={getContentStyles()}>{content}</div>
    </div>
  )
}

export default Message
