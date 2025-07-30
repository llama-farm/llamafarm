import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Home from './Home'
import Chat from './Chat'

function App() {
  return (
    <div className="min-h-screen">
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/chat" element={<Chat />} />
      </Routes>
    </div>
  )
}

export default App
