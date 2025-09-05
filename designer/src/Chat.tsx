import { useState } from 'react'
import Chatbox from './components/Chatbox/Chatbox'
import { Outlet } from 'react-router-dom'

function Chat() {
  const [isPanelOpen, setIsPanelOpen] = useState<boolean>(true)

  return (
    <div className="w-full h-full flex transition-colors bg-gray-200 dark:bg-blue-800 pt-12">
      <div
        className={`h-full transition-all duration-300 ${isPanelOpen ? 'w-1/4' : 'w-[47px]'}`}
      >
        <Chatbox isPanelOpen={isPanelOpen} setIsPanelOpen={setIsPanelOpen} />
      </div>

      <div
        className={`h-full ${isPanelOpen ? 'w-3/4' : 'flex-1'} text-gray-900 dark:text-white px-6 pt-6 overflow-hidden`}
      >
        <Outlet />
      </div>
    </div>
  )
}

export default Chat
