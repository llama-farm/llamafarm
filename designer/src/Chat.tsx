import Chatbox from './components/Chatbox'
import { Outlet } from 'react-router-dom'

function Chat() {
  return (
    <div className="w-full h-full flex pt-12 bg-[#000B1B]">
      <div className="w-1/4 h-full">
        <Chatbox />
      </div>
      <div className="w-3/4 h-full text-white">
        <Outlet />
      </div>
    </div>
  )
}

export default Chat
