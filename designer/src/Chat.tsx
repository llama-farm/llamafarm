import Chatbox from './components/Chatbox'
import BuildArea from './components/BuildArea'

function Chat() {
  return (
    <div className="h-screen w-full flex flex-row">
      <Chatbox />
      <BuildArea />
    </div>
  )
}

export default Chat
