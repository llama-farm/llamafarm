import { useEffect, useMemo, useState } from 'react'
import FontIcon from '../../common/FontIcon'
import { useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
// Modal rendered globally in App
import { useProjects, projectKeys } from '../../hooks/useProjects'
import { useProjectModalContext } from '../../contexts/ProjectModalContext'
import { getCurrentNamespace } from '../../utils/namespaceUtils'
import {
  getProjectsForUI,
  filterProjectsBySearch,
} from '../../utils/projectConstants'
import Loader from '../../common/Loader'

const Projects = () => {
  const [search, setSearch] = useState('')
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const namespace = getCurrentNamespace()

  // API hooks
  const { data: projectsResponse, isLoading, error, refetch } = useProjects(namespace)

  const handleRefresh = async () => {
    queryClient.invalidateQueries({ queryKey: projectKeys.list(namespace) })
    try {
      await refetch()
    } catch (err) {
      // Error handling is done by React Query
    }
  }

  // Get existing project names for validation
  // existingProjects used via centralized context provider

  // Shared modal hook
  const projectModal = useProjectModalContext()

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
  // Never show fallback projects - always show empty state when no projects exist
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

  const handleCreateProject = () => {
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' })
    // Open create modal
    projectModal.openCreateModal()
  }

  // Show empty state when no projects exist (and not loading, not error)
  const showEmptyState = !isLoading && filteredProjects.length === 0 && !error

  return (
    <div className="w-full h-full transition-colors bg-background pt-16">
      <div className="max-w-6xl mx-auto px-6 flex flex-col gap-6">
        {isLoading && (
          <div className="flex justify-center py-8">
            <Loader />
          </div>
        )}
        {error && (
          <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
            <p className="text-destructive text-sm">
              ⚠️ Failed to load projects: {error.message}
            </p>
            <p className="text-muted-foreground text-xs mt-1">
              Please check that the server is running and try refreshing.
            </p>
          </div>
        )}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl text-foreground">Projects</h2>
            <p className="text-xs text-muted-foreground mt-1">
              Namespace: <code className="px-1 py-0.5 bg-muted rounded">{namespace}</code>
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              className="px-3 py-2 rounded-lg border border-input text-primary hover:bg-accent/20 flex items-center gap-2"
              onClick={handleRefresh}
              disabled={isLoading}
              title="Refresh projects list"
            >
              <FontIcon type="recently-viewed" className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            <button className="px-3 py-2 rounded-lg border border-input text-primary hover:bg-accent/20">
              Explore public projects
            </button>
            <button
              className="px-3 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90"
              onClick={() => projectModal.openCreateModal()}
            >
              New project
            </button>
          </div>
        </div>

        {!showEmptyState && (
          <div className="w-full flex items-center bg-card rounded-lg px-3 py-2 border border-input">
            <FontIcon type="search" className="w-4 h-4 text-foreground" />
            <input
              className="w-full bg-transparent border-none focus:outline-none px-2 text-sm text-foreground"
              placeholder="Search projects"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
        )}

        {showEmptyState ? (
          <div className="flex items-center justify-center min-h-[60vh]">
            <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40 max-w-md">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/15 border border-primary/30">
                <FontIcon type="folder" className="w-6 h-6 text-primary" />
              </div>
              <div className="text-lg font-medium text-foreground mb-2">
                No projects yet
              </div>
              <div className="text-sm text-muted-foreground mb-6">
                Create your first project to start building AI-powered applications.
                Each project can have its own models, prompts, and data.
              </div>
              <button
                className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
                onClick={handleCreateProject}
              >
                Create your first project
              </button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-8">
            {!isLoading &&
              filteredProjects.map(p => (
              <div
                key={p.id}
                className="group w-full rounded-lg p-4 bg-card border border-border cursor-pointer"
                onClick={() => openProject(p.name)}
              >
                <div className="flex items-start justify-between">
                  <div className="text-base text-foreground">{p.name}</div>
                  <FontIcon
                    type="arrow-right"
                    className="w-5 h-5 text-primary"
                  />
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
        )}
      </div>

      {/* Modal rendered globally in App */}
    </div>
  )
}

export default Projects
