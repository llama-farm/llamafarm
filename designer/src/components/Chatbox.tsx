import { useState } from 'react'
import Message from './Message'

export interface Message {
  type: 'user' | 'assistant' | 'system' | 'error'
  content: string
  sources?: any[]
  metadata?: any
  timestamp: Date
  isLoading?: boolean
}
function Chatbox() {
  const [messages, setMessages] = useState<Message[]>([
    {
      type: 'user',
      content: 'Aircraft maintenance app',
      timestamp: new Date(),
    },
    {
      type: 'assistant',
      content:
        'Great start! Before we dive in, we’ll need to take a look at your data. Do you have any aircraft logs or other context we can work with?',
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
    <div className="bg-[#131E45] w-1/4 h-full p-4 text-white flex flex-col justify-between">
      <div className="flex flex-col gap-4">
        {messages.map((message, index) => (
          <Message key={index} message={message} />
        ))}
      </div>
      <div className="bg-[#040D1D] flex flex-col gap-2 p-2 rounded-lg">
        <textarea
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          className="w-full h-8 resize-none bg-transparent border-none text-white placeholder-white focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed overflow-hidden"
          //   className="w-full h-[78px] bg-transparent border-none resize-none p-4 pr-12 text-white placeholder-white/60 focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed"
          placeholder="Type here..."
        />
        <button
          onClick={handleSendClick}
          className="w-8 h-8 bg-blue-600 hover:bg-blue-500 rounded-full flex items-center justify-center text-[#040D1D] transition-colors duration-200 shadow-sm hover:shadow-md self-end"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
            />
          </svg>
        </button>
      </div>
    </div>
  )
}

export default Chatbox
