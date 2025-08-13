import { useState } from 'react'
import Message from './Message'
import FontIcon from '../../common/FontIcon'

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
    {
      type: 'user',
      content: 'I have aircraft logs in PDFs',
      timestamp: new Date(),
    },
    {
      type: 'assistant',
      content:
        'Fantastic! Please bear with us as we process your data. Background tasks in progress: Parsing PDFs: We are utilizing **PDFParserPro** to extract data from your files. This tool was selected for its accuracy and efficiency in handling complex PDF structures. Chunking Data: Next, we will segment the extracted data into manageable pieces to facilitate further analysis. While we work on this, please share where you plan to deploy your aircraft maintenance application.',
      timestamp: new Date(),
    },
  ])

  const [inputValue, setInputValue] = useState('')

  const handleSendClick = () => {
    const text = inputValue.trim()
    if (!text) return
    setMessages(prev => [
      ...prev,
      { type: 'user', content: text, timestamp: new Date() },
    ])
    setInputValue('')
  }

  const handleKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendClick()
    }
  }

  return (
    <div className="w-full h-full flex flex-col transition-colors bg-[#FFFFFF] text-gray-900 dark:bg-blue-500 dark:text-white">
      <div
        className={`flex  ${
          isPanelOpen ? 'justify-end mr-1 mt-1' : 'justify-center mt-3'
        }`}
      >
        <FontIcon
          isButton
          type={isPanelOpen ? 'close-panel' : 'open-panel'}
          className="w-6 h-6 text-blue-200 hover:text-gray-800 dark:text-blue-300 dark:hover:text-blue-100"
          handleOnClick={() => setIsPanelOpen(!isPanelOpen)}
        />
      </div>
      <div
        className={`flex flex-col justify-between h-full p-4 ${
          isPanelOpen ? 'flex' : 'hidden'
        }`}
      >
        <div className="flex flex-col gap-5">
          {messages.map((message, index) => (
            <Message key={index} message={message} />
          ))}
        </div>
        <div className="flex flex-col gap-3 p-3 rounded-lg bg-[#F4F4F4] dark:bg-blue-700">
          <textarea
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full h-10 resize-none bg-transparent border-none placeholder-opacity-60 focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed overflow-hidden text-gray-900 placeholder-gray-500 dark:text-white dark:placeholder-white"
            placeholder="Type here..."
          />
          <FontIcon
            isButton
            type="arrow-filled"
            className="w-8 h-8 self-end text-blue-200 dark:text-green-100"
            handleOnClick={handleSendClick}
          />
        </div>
      </div>
    </div>
  )
}

export default Chatbox
