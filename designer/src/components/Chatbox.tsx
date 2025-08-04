import { useState } from 'react'
import Message from './Message'
import FontIcon from '../common/FontIcon'

export interface Message {
  type: 'user' | 'assistant' | 'system' | 'error'
  content: string
  sources?: any[]
  metadata?: any
  timestamp: Date
  isLoading?: boolean
}

interface ChatboxProps {
  isPanelOpen: boolean
  setIsPanelOpen: (isOpen: boolean) => void
}

function Chatbox({ isPanelOpen, setIsPanelOpen }: ChatboxProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      type: 'user',
      content: 'Aircraft maintenance app',
      timestamp: new Date(),
    },
    {
      type: 'assistant',
      content:
        "Great start! Before we dive in, we'll need to take a look at your data. Do you have any aircraft logs or other context we can work with?",
      timestamp: new Date(),
    },
  ])

  const [inputValue, setInputValue] = useState('')

  const handleSendClick = () => {
    setMessages([
      ...messages,
      { type: 'user', content: inputValue, timestamp: new Date() },
    ])
  }

  return (
    <div className="w-full h-full flex flex-col transition-colors bg-gray-100 text-gray-900 dark:bg-blue-500 dark:text-white">
      <div
        className={`flex  ${
          isPanelOpen ? 'justify-end mr-1 mt-1' : 'justify-center mt-3'
        }`}
      >
        <FontIcon
          isButton
          type={isPanelOpen ? 'close-panel' : 'open-panel'}
          className="w-6 h-6 text-gray-600 hover:text-gray-800 dark:text-blue-300 dark:hover:text-blue-100"
          handleOnClick={() => setIsPanelOpen(!isPanelOpen)}
        />
      </div>
      <div
        className={`flex flex-col justify-between h-full p-4 ${
          isPanelOpen ? 'flex' : 'hidden'
        }`}
      >
        <div className="flex flex-col gap-4">
          {messages.map((message, index) => (
            <Message key={index} message={message} />
          ))}
        </div>
        <div className="flex flex-col gap-2 p-2 rounded-lg bg-white shadow-md dark:bg-blue-700">
          <textarea
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            className="w-full h-8 resize-none bg-transparent border-none placeholder-opacity-60 focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed overflow-hidden text-gray-900 placeholder-gray-500 dark:text-white dark:placeholder-white"
            placeholder="Type here..."
          />
          <FontIcon
            isButton
            type="arrow-filled"
            className="w-8 h-8 self-end text-gray-600 dark:text-blue-100"
            handleOnClick={handleSendClick}
          />
        </div>
      </div>
    </div>
  )
}

export default Chatbox
