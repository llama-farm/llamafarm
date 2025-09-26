import { useEffect, useRef, useState } from 'react'
import Chatbox from './components/Chatbox/Chatbox'
import { Outlet, useLocation, useSearchParams } from 'react-router-dom'
import { decodeMessageFromUrl } from './utils/homePageUtils'

function Chat() {
  const [isPanelOpen, setIsPanelOpen] = useState<boolean>(true)
  const [initialMessage, setInitialMessage] = useState<string | null>(null)
  const location = useLocation()
  const [searchParams, setSearchParams] = useSearchParams()
  const prevPanelOpenRef = useRef<boolean>(true)

  // Handle initial message from URL parameters (from home page project creation)
  useEffect(() => {
    const initialMessageParam = searchParams.get('initialMessage')
    if (initialMessageParam) {
      const decodedMessage = decodeMessageFromUrl(initialMessageParam)
      if (decodedMessage) {
        setInitialMessage(decodedMessage)
        
        // Clear the URL parameter after handling to clean up the URL
        setSearchParams(params => {
          params.delete('initialMessage')
          return params
        })
      }
    }
  }, [searchParams, setSearchParams])

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
        <Chatbox 
          isPanelOpen={isPanelOpen} 
          setIsPanelOpen={setIsPanelOpen}
          initialMessage={initialMessage}
        />
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
