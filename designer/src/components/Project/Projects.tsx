import { useEffect, useMemo, useState } from 'react'
import FontIcon from '../../common/FontIcon'
import { useNavigate } from 'react-router-dom'
import ProjectModal from './ProjectModal'
import { useProjects } from '../../hooks/useProjects'
import { useProjectModal } from '../../hooks/useProjectModal'
import { getCurrentNamespace } from '../../utils/namespaceUtils'
import { getProjectsForUI, filterProjectsBySearch, getProjectsList } from '../../utils/projectConstants'
import Loader from '../../common/Loader'

const Projects = () => {
  const [search, setSearch] = useState('')
  const navigate = useNavigate()
  const namespace = getCurrentNamespace()
  
  // API hooks
  const { data: projectsResponse, isLoading, error } = useProjects(namespace)
  
  // Get existing project names for validation
  const existingProjects = useMemo(() => getProjectsList(projectsResponse), [projectsResponse])
  
  // Shared modal hook
  const projectModal = useProjectModal({
    namespace,
    existingProjects,
    onSuccess: (_, mode) => {
      if (mode === 'create') {
        navigate('/chat/dashboard')
      }
    }
  })

  // Open create modal if signaled by header
  useEffect(() => {
    const flag = localStorage.getItem('openCreateProjectModal')
    if (flag === '1') {
      localStorage.removeItem('openCreateProjectModal')
      projectModal.openCreateModal()
    }
    const editName = localStorage.getItem('openEditProject')
    if (editName) {
      localStorage.removeItem('openEditProject')
      projectModal.openEditModal(editName)
    }
  }, [projectModal])

  // Convert API projects to UI format
  const projects = useMemo(() => {
    return getProjectsForUI(projectsResponse)
  }, [projectsResponse])
  
  const filteredProjects = useMemo(() => {
    return filterProjectsBySearch(projects, search)
  }, [projects, search])

  const openProject = (name: string) => {
    localStorage.setItem('activeProject', name)
    navigate('/chat/dashboard')
  }





  if (error) {
    return (
      <div className="w-full h-full transition-colors bg-background pt-16">
        <div className="max-w-6xl mx-auto px-6 flex flex-col gap-6">
          <div className="text-center py-8">
            <p className="text-destructive">Failed to load projects: {error.message}</p>
            <p className="text-muted-foreground mt-2">Showing default projects instead.</p>
          </div>
        </div>
      </div>
    )
  }
  
  return (
    <div className="w-full h-full transition-colors bg-background pt-16">
      <div className="max-w-6xl mx-auto px-6 flex flex-col gap-6">
        {isLoading && (
          <div className="flex justify-center py-8">
            <Loader />
          </div>
        )}
        <div className="flex items-center justify-between">
          <h2 className="text-2xl text-foreground">Projects</h2>
          <div className="flex items-center gap-2">
            <button className="px-3 py-2 rounded-lg border border-input text-primary hover:bg-accent/20">
              Explore public projects
            </button>
            <button
              className="px-3 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90"
              onClick={projectModal.openCreateModal}
            >
              New project
            </button>
          </div>
        </div>

        <div className="w-full flex items-center bg-card rounded-lg px-3 py-2 border border-input">
          <FontIcon type="search" className="w-4 h-4 text-foreground" />
          <input
            className="w-full bg-transparent border-none focus:outline-none px-2 text-sm text-foreground"
            placeholder="Search projects"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-8">
          {!isLoading && filteredProjects.map(p => (
            <div
              key={p.id}
              className="group w-full rounded-lg p-4 bg-card border border-border cursor-pointer"
              onClick={() => openProject(p.name)}
            >
              <div className="flex items-start justify-between">
                <div className="text-base text-foreground">{p.name}</div>
                <FontIcon type="arrow-right" className="w-5 h-5 text-primary" />
              </div>
              <div className="mt-3">
                <span className="text-xs text-primary-foreground bg-primary rounded-xl px-3 py-0.5">
                  {p.model}
                </span>
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                Last edited on {p.lastEdited}
              </div>
              <div className="mt-4 flex justify-end">
                <button
                  className="flex items-center gap-1 text-primary hover:opacity-80"
                  onClick={e => {
                    e.stopPropagation()
                    projectModal.openEditModal(p.name)
                  }}
                >
                  <FontIcon type="edit" className="w-5 h-5 text-primary" />
                  <span className="text-sm">Edit</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      <ProjectModal
        isOpen={projectModal.isModalOpen}
        mode={projectModal.modalMode}
        initialName={projectModal.projectName}
        initialDescription={''}
        onClose={projectModal.closeModal}
        onSave={projectModal.saveProject}
        onDelete={projectModal.modalMode === 'edit' ? projectModal.deleteProject : undefined}
        isLoading={projectModal.isLoading}
      />
    </div>
  )
}

export default Projects
