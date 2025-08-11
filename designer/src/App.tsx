import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Home from './Home'
import Chat from './Chat'
import Data from './components/Data/Data'
import Prompt from './components/Prompt/Prompt'
import Test from './components/Test'
import Dashboard from './components/Dashboard/Dashboard'

function App() {
  return (
    <main className="h-screen w-full">
      <Header />
      <div className="h-full w-full">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/chat" element={<Chat />}>
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="data" element={<Data />} />
            <Route path="prompt" element={<Prompt />} />
            <Route path="test" element={<Test />} />
          </Route>
        </Routes>
      </div>
    </main>
  )
}

export default App
