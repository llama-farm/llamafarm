import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Home from './Home'
import Chat from './Chat'

function App() {
  return (
    <main className="h-screen w-full">
      <Header />
      <div className="h-full w-full">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/chat" element={<Chat />} />
        </Routes>
      </div>
    </main>
  )
}

export default App
