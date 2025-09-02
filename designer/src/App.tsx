import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import { ToastProvider } from './components/ui/toast'
import Home from './Home'
import Chat from './Chat'
import Data from './components/Data/Data'
import DatasetView from './components/Data/DatasetView'
import Prompt from './components/Prompt/Prompt'
import Test from './components/Test'
import Dashboard from './components/Dashboard/Dashboard'
import Models from './components/Models/Models'
// Projects standalone page removed; Home now hosts projects section

function App() {
  return (
    <main className="h-screen w-full">
      <ToastProvider>
        <Header />
        <div className="h-full w-full">
          <Routes>
            <Route path="/" element={<Home />} />
            {/* Redirect '/projects' to Home; Home will scroll to projects */}
            <Route path="/projects" element={<Home />} />
            <Route path="/chat" element={<Chat />}>
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="data" element={<Data />} />
              <Route path="data/:datasetId" element={<DatasetView />} />
              <Route path="models" element={<Models />} />
              <Route path="prompt" element={<Prompt />} />
              <Route path="test" element={<Test />} />
            </Route>
          </Routes>
        </div>
      </ToastProvider>
    </main>
  )
}

export default App
