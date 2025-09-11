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
import Versions from './components/Dashboard/Versions'
import Models from './components/Models/Models'
import Rag from './components/Rag/Rag'
import StrategyView from './components/Rag/StrategyView'
import ChangeEmbeddingModel from './components/Rag/ChangeEmbeddingModel'
import ExtractionSettings from './components/Rag/ExtractionSettings'
import ParsingStrategy from './components/Rag/ParsingStrategy'
import RetrievalMethod from './components/Rag/RetrievalMethod'
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
              <Route path="versions" element={<Versions />} />
              <Route path="data" element={<Data />} />
              <Route path="data/:datasetId" element={<DatasetView />} />
              <Route path="models" element={<Models />} />
              <Route path="rag" element={<Rag />} />
              <Route path="rag/:strategyId" element={<StrategyView />} />
              <Route
                path="rag/:strategyId/change-embedding"
                element={<ChangeEmbeddingModel />}
              />
              <Route
                path="rag/:strategyId/extraction"
                element={<ExtractionSettings />}
              />
              <Route
                path="rag/:strategyId/parsing"
                element={<ParsingStrategy />}
              />
              <Route
                path="rag/:strategyId/retrieval"
                element={<RetrievalMethod />}
              />
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
