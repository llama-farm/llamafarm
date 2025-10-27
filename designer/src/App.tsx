import {
  Routes,
  Route,
  Navigate,
  useLocation,
  useParams,
} from 'react-router-dom'
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
import Models from './components/Models/Models'
import Databases from './components/Rag/Databases'
import StrategyView from './components/Rag/StrategyView'
import ChangeEmbeddingModel from './components/Rag/ChangeEmbeddingModel'
import SampleProjects from './components/Samples/SampleProjects'
// @ts-ignore - component is TSX local file
import AddEmbeddingStrategy from './components/Rag/AddEmbeddingStrategy'
// Removed legacy per-strategy pages in favor of unified StrategyView
import RetrievalMethod from './components/Rag/RetrievalMethod'
// @ts-ignore - component is TSX local file
import AddRetrievalStrategy from './components/Rag/AddRetrievalStrategy'
// Projects standalone page removed; Home now hosts projects section
import { HomeUpgradeBanner } from './components/common/UpgradeBanners'
import { useUpgradeAvailability } from './hooks/useUpgradeAvailability'
import { MobileViewProvider } from './contexts/MobileViewContext'
import NotFound from './components/NotFound'

// Redirect component for dynamic routes from /rag to /databases
function RagRedirect({ path }: { path: string }) {
  const params = useParams()
  const newPath = path.replace(':strategyId', params.strategyId || '')
  return <Navigate to={newPath} replace />
}

function ProjectModalRoot() {
  const modal = useProjectModalContext()
  // Only render the modal for edit mode; create is handled by Home form
  if (modal.modalMode !== 'edit') return null
  // Derive initial details from current project config
  const cfg = (modal.currentProject?.config || {}) as Record<string, any>
  const projectBrief = (cfg?.project_brief || {}) as Record<string, any>
  const initialBrief = {
    what: projectBrief?.what || '',
    goals: projectBrief?.goals || '',
    audience: projectBrief?.audience || '',
  }
  return (
    <ProjectModal
      isOpen={modal.isModalOpen}
      mode={modal.modalMode}
      initialName={modal.projectName}
      initialBrief={initialBrief}
      onClose={modal.closeModal}
      onSave={modal.saveProject}
      onDelete={modal.modalMode === 'edit' ? modal.deleteProject : undefined}
      isLoading={modal.isLoading}
    />
  )
}

function App() {
  const location = useLocation()
  const isHome = location.pathname === '/'
  const { currentVersion } = useUpgradeAvailability()
  return (
    <main className="h-screen w-full">
      <ToastProvider>
        <ProjectModalProvider>
          <MobileViewProvider>
            <Header currentVersion={currentVersion} />
            {isHome ? <HomeUpgradeBanner /> : null}
            <div className="h-full w-full">
              <Routes>
                <Route path="/" element={<Home />} />
                {/* Redirect '/projects' to Home; Home will scroll to projects */}
                <Route path="/projects" element={<Home />} />
                <Route path="/samples" element={<SampleProjects />} />
                <Route path="/chat" element={<Chat />}>
                  <Route
                    index
                    element={<Navigate to="/chat/dashboard" replace />}
                  />
                  <Route path="dashboard" element={<Dashboard />} />
                  <Route
                    path="versions"
                    element={<Navigate to="/chat/dashboard" replace />}
                  />
                  <Route path="data" element={<Data />} />
                  <Route path="data/:datasetId" element={<DatasetView />} />
                  {/* Processing strategies routes - now under data */}
                  <Route
                    path="data/strategies/processing"
                    element={<StrategyView />}
                  />
                  <Route
                    path="data/strategies/:strategyId"
                    element={<StrategyView />}
                  />
                  <Route path="models" element={<Models />} />
                  {/* Redirect old /rag routes to /databases */}
                  <Route
                    path="rag"
                    element={<Navigate to="/chat/databases" replace />}
                  />
                  <Route
                    path="rag/add-embedding"
                    element={
                      <Navigate to="/chat/databases/add-embedding" replace />
                    }
                  />
                  <Route
                    path="rag/add-retrieval"
                    element={
                      <Navigate to="/chat/databases/add-retrieval" replace />
                    }
                  />
                  <Route
                    path="rag/:strategyId/change-embedding"
                    element={
                      <RagRedirect path="/chat/databases/:strategyId/change-embedding" />
                    }
                  />
                  <Route
                    path="rag/:strategyId/retrieval"
                    element={
                      <RagRedirect path="/chat/databases/:strategyId/retrieval" />
                    }
                  />
                  <Route path="databases" element={<Databases />} />
                  {/* Database-level embedding and retrieval strategy pages */}
                  <Route
                    path="databases/add-embedding"
                    element={<AddEmbeddingStrategy />}
                  />
                  <Route
                    path="databases/add-retrieval"
                    element={<AddRetrievalStrategy />}
                  />
                  <Route
                    path="databases/:strategyId/change-embedding"
                    element={<ChangeEmbeddingModel />}
                  />
                  <Route
                    path="databases/:strategyId/retrieval"
                    element={<RetrievalMethod />}
                  />
                  <Route path="prompt" element={<Prompt />} />
                  <Route path="test" element={<Test />} />
                  {/* Catch-all for unknown /chat routes */}
                  <Route path="*" element={<NotFound />} />
                </Route>
                {/* Catch-all for unknown top-level routes */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </div>
            <ProjectModalRoot />
          </MobileViewProvider>
        </ProjectModalProvider>
      </ToastProvider>
    </main>
  )
}

export default App
