import Chatbox from './components/Chatbox'
import BuildArea from './components/BuildArea'

function Chat() {
  return (
    <div className="w-full h-full flex pt-12 bg-[#000B1B]">
      <div className="w-1/4 h-full">
        <Chatbox />
      </div>
      <div className="w-3/4 h-full">
        <BuildArea />
      </div>
    </div>
  )
}

export default Chat
