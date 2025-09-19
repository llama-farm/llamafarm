import { useEffect, useRef, useState } from 'react'
import Chatbox from './components/Chatbox/Chatbox'
import { Outlet, useLocation } from 'react-router-dom'

function Chat() {
  const [isPanelOpen, setIsPanelOpen] = useState<boolean>(true)
  const location = useLocation()
  const prevPanelOpenRef = useRef<boolean>(true)

  // Close the designer chat panel when viewing the Test page, and restore previous state when leaving
  useEffect(() => {
    if (location.pathname.startsWith('/chat/test')) {
      // Save current state before closing
      prevPanelOpenRef.current = isPanelOpen
      setIsPanelOpen(false)
    } else {
      // Restore previous state when leaving /chat/test
      setIsPanelOpen(prevPanelOpenRef.current)
    }
  }, [location.pathname])

  return (
    <div className="w-full h-full flex transition-colors bg-gray-200 dark:bg-blue-800 pt-12">
      <div
        className={`h-full transition-all duration-300 ${isPanelOpen ? 'w-1/4' : 'w-[47px]'}`}
      >
        <Chatbox isPanelOpen={isPanelOpen} setIsPanelOpen={setIsPanelOpen} />
      </div>

      <div
        className={`h-full ${isPanelOpen ? 'w-3/4' : 'flex-1'} text-gray-900 dark:text-white px-6 pt-6 overflow-auto min-h-0`}
      >
        <Outlet />
      </div>
    </div>
  )
}

export default Chat
