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
    <div className="bg-blue-500 w-full h-full flex flex-col text-white">
      <div
        className={`flex  ${
          isPanelOpen ? 'justify-end mr-1 mt-1' : 'justify-center mt-3'
        }`}
      >
        <FontIcon
          isButton
          type={isPanelOpen ? 'close-panel' : 'open-panel'}
          className={`w-6 h-6 text-blue-300 ${
            isPanelOpen ? 'text-blue-300' : 'text-blue-100'
          }`}
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
        <div className="bg-blue-700 flex flex-col gap-2 p-2 rounded-lg">
          <textarea
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            className="w-full h-8 resize-none bg-transparent border-none text-white placeholder-white focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed overflow-hidden"
            //   className="w-full h-[78px] bg-transparent border-none resize-none p-4 pr-12 text-white placeholder-white/60 focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed"
            placeholder="Type here..."
          />
          {/* <button
          onClick={handleSendClick}
          className="w-8 h-8 bg-blue-600 hover:bg-blue-500 rounded-full flex items-center justify-center text-[#040D1D] transition-colors duration-200 shadow-sm hover:shadow-md self-end"
        > */}
          <FontIcon
            isButton
            type="arrow-filled"
            className="w-8 h-8 text-blue-100 self-end"
            handleOnClick={handleSendClick}
          />
          {/* </button> */}
        </div>
      </div>
    </div>
  )
}

export default Chatbox
