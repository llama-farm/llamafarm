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
import CreateProjectModal from './components/Project/CreateProjectModal'
import DeleteProjectModal from './components/Project/DeleteProjectModal'
import {
  ProjectModalProvider,
  useProjectModalContext,
} from './contexts/ProjectModalContext'
import { useProjects } from './hooks/useProjects'
import { getProjectsList } from './utils/projectConstants'
import { ModeResetProvider } from './contexts/ModeContext'
import { UnsavedChangesProvider } from './contexts/UnsavedChangesContext'
import Home from './Home'
import Chat from './Chat'
import Data from './components/Data/Data'
import DatasetView from './components/Data/DatasetView'
import Prompt from './components/Prompt/Prompt'
import Test from './components/Test'
import Dashboard from './components/Dashboard/Dashboard'
import Models from './components/Models/Models'
import AddInferenceModels from './components/Models/AddInferenceModels'
import AnomalyModel from './components/Models/AnomalyModel'
import ClassifierModel from './components/Models/ClassifierModel'
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
import EditRetrievalStrategy from './components/Rag/EditRetrievalStrategy'
// Projects standalone page removed; Home now hosts projects section
import { HomeUpgradeBanner } from './components/common/UpgradeBanners'
import {
  useUpgradeAvailability,
  UpgradeAvailabilityProvider,
} from './hooks/useUpgradeAvailability'
import { MobileViewProvider } from './contexts/MobileViewContext'
import NotFound from './components/NotFound'
import { DemoModalProvider, useDemoModal } from './contexts/DemoModalContext'
import { DemoModal } from './components/Demo/DemoModal'
import { getCurrentNamespace } from './utils/namespaceUtils'

// Redirect component for dynamic routes from /rag to /databases
function RagRedirect({ path }: { path: string }) {
  const params = useParams()
  const newPath = path.replace(':strategyId', params.strategyId || '')
  return <Navigate to={newPath} replace />
}

function ProjectModalRoot() {
  const modal = useProjectModalContext()
  const namespace = getCurrentNamespace()
  const { data: projectsResponse } = useProjects(namespace)
  const availableProjects = getProjectsList(projectsResponse)

  return (
    <>
      {/* Create modal - shown when modalMode is 'create' */}
      {modal.modalMode === 'create' && (
        <CreateProjectModal
          isOpen={modal.isModalOpen}
          availableProjects={availableProjects}
          copyFromProject={modal.copyFromProject}
          onClose={modal.closeModal}
          onCreate={modal.createProject}
          isLoading={modal.isLoading}
          projectError={modal.projectError}
          onNameChange={modal.validateName}
        />
      )}

      {/* Edit modal - shown when modalMode is 'edit' */}
      {modal.modalMode === 'edit' && (
        <>
          {/* Derive initial details from current project config */}
          {(() => {
            const cfg = (modal.currentProject?.config || {}) as Record<
              string,
              any
            >
            const projectBrief = (cfg?.project_brief || {}) as Record<
              string,
              any
            >
            const initialBrief = {
              what: projectBrief?.what || '',
            }
            return (
              <ProjectModal
                isOpen={modal.isModalOpen}
                mode={modal.modalMode}
                initialName={modal.projectName}
                initialBrief={initialBrief}
                onClose={modal.closeModal}
                onSave={modal.saveProject}
                onOpenDelete={modal.openDeleteModal}
                onCopy={() => modal.openCreateModal(modal.projectName)}
                isLoading={modal.isLoading}
                projectError={modal.projectError}
                onNameChange={modal.validateName}
              />
            )
          })()}
        </>
      )}

      <DeleteProjectModal
        isOpen={modal.isDeleteModalOpen}
        projectName={modal.projectName}
        onClose={modal.closeDeleteModal}
        onConfirm={modal.deleteProject}
        isLoading={modal.isLoading}
      />
    </>
  )
}

function DemoModalRoot() {
  const demoModal = useDemoModal()
  const namespace = getCurrentNamespace()

  return (
    <DemoModal
      isOpen={demoModal.isOpen}
      onClose={demoModal.closeModal}
      namespace={namespace}
      autoStartDemoId={demoModal.autoStartDemoId}
    />
  )
}

function AppContent() {
  const location = useLocation()
  const isHome = location.pathname === '/'
  const { currentVersion } = useUpgradeAvailability()
  return (
    <>
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
                        <Route
                          path="data/:datasetId"
                          element={<DatasetView />}
                        />
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
                        <Route path="models/add" element={<AddInferenceModels />} />
                        <Route path="models/train/anomaly/new" element={<AnomalyModel />} />
                        <Route path="models/train/anomaly/:id" element={<AnomalyModel />} />
                        <Route path="models/train/classifier/new" element={<ClassifierModel />} />
                        <Route path="models/train/classifier/:id" element={<ClassifierModel />} />
                        {/* Redirect old /rag routes to /databases */}
                        <Route
                          path="rag"
                          element={<Navigate to="/chat/databases" replace />}
                        />
                        <Route
                          path="rag/add-embedding"
                          element={
                            <Navigate
                              to="/chat/databases/add-embedding"
                              replace
                            />
                          }
                        />
                        <Route
                          path="rag/add-retrieval"
                          element={
                            <Navigate
                              to="/chat/databases/add-retrieval"
                              replace
                            />
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
                          path="add-embedding-strategy"
                          element={<AddEmbeddingStrategy />}
                        />
                        <Route
                          path="add-retrieval-strategy"
                          element={<AddRetrievalStrategy />}
                        />
                        <Route
                          path="edit-retrieval-strategy"
                          element={<EditRetrievalStrategy />}
                        />
                        <Route
                          path="databases/:strategyId/change-embedding"
                          element={<ChangeEmbeddingModel />}
                        />
                        <Route
                          path="change-embedding-model"
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
      <DemoModalRoot />
    </>
  )
}

function App() {
  return (
    <main className="h-screen w-full">
      <ToastProvider>
        <UpgradeAvailabilityProvider>
          <ProjectModalProvider>
            <DemoModalProvider>
              <ModeResetProvider>
                <MobileViewProvider>
                  <UnsavedChangesProvider>
                    <AppContent />
                  </UnsavedChangesProvider>
                </MobileViewProvider>
              </ModeResetProvider>
            </DemoModalProvider>
          </ProjectModalProvider>
        </UpgradeAvailabilityProvider>
      </ToastProvider>
    </main>
  )
}

export default App
