import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import { ToastProvider } from './components/ui/toast'
import ProjectModal from './components/Project/ProjectModal'
import {
  ProjectModalProvider,
  useProjectModalContext,
} from './contexts/ProjectModalContext'
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
// @ts-ignore - component is TSX local file
import AddEmbeddingStrategy from './components/Rag/AddEmbeddingStrategy'
// Removed legacy per-strategy pages in favor of unified StrategyView
import RetrievalMethod from './components/Rag/RetrievalMethod'
// @ts-ignore - component is TSX local file
import AddRetrievalStrategy from './components/Rag/AddRetrievalStrategy'
// Projects standalone page removed; Home now hosts projects section

function ProjectModalRoot() {
  const modal = useProjectModalContext()
  return (
    <ProjectModal
      isOpen={modal.isModalOpen}
      mode={modal.modalMode}
      initialName={modal.projectName}
      initialDescription={''}
      onClose={modal.closeModal}
      onSave={modal.saveProject}
      onDelete={modal.modalMode === 'edit' ? modal.deleteProject : undefined}
      isLoading={modal.isLoading}
    />
  )
}

function App() {
  return (
    <main className="h-screen w-full">
      <ToastProvider>
        <ProjectModalProvider>
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
                {/* Project-level pages */}
                <Route
                  path="rag/add-embedding"
                  element={<AddEmbeddingStrategy />}
                />
                <Route
                  path="rag/add-retrieval"
                  element={<AddRetrievalStrategy />}
                />
                <Route path="rag/processing" element={<StrategyView />} />
                <Route path="rag/:strategyId" element={<StrategyView />} />
                <Route
                  path="rag/:strategyId/change-embedding"
                  element={<ChangeEmbeddingModel />}
                />
                {/* Legacy routes above remain; new entries reuse same components */}
                <Route
                  path="rag/:strategyId/retrieval"
                  element={<RetrievalMethod />}
                />
                <Route path="prompt" element={<Prompt />} />
                <Route path="test" element={<Test />} />
              </Route>
            </Routes>
          </div>
          <ProjectModalRoot />
        </ProjectModalProvider>
      </ToastProvider>
    </main>
  )
}

export default App
