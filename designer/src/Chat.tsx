import Chatbox from './components/Chatbox'
import BuildArea from './components/BuildArea'

function Chat() {
  return (
    <div className="w-full h-full flex flex-row pt-10 bg-[#000B1B]">
      <Chatbox />
      <BuildArea />
    </div>
  )
}

export default Chat
