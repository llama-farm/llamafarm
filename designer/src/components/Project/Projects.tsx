import { useEffect, useMemo, useState } from 'react'
import FontIcon from '../../common/FontIcon'
import { useNavigate } from 'react-router-dom'
import ProjectModal, { ProjectModalMode } from './ProjectModal'
import {
  useProjects,
  useProject,
  useCreateProject,
  useUpdateProject,
  useDeleteProject,
} from '../../hooks/useProjects'
import {
  DEFAULT_PROJECTS,
  apiProjectsToProjectItems,
  filterProjectsBySearch,
  setActiveProject,
} from '../../utils/projectUtils'
import { getCurrentNamespace } from '../../utils/namespaceUtils'
import { validateProjectConfig, mergeProjectConfig } from '../../utils/projectConfigUtils'
import Loader from '../../common/Loader'

const Projects = () => {
  const [search, setSearch] = useState('')
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [modalMode, setModalMode] = useState<ProjectModalMode>('create')
  const [modalProject, setModalProject] = useState<{
    name: string
    config?: Record<string, any>
  }>({ name: '' })
  const navigate = useNavigate()
  const namespace = getCurrentNamespace()
  
  // API hooks
  const { data: projectsResponse, isLoading, error } = useProjects(namespace)
  const { data: currentProjectResponse, isLoading: isLoadingProject } = useProject(
    namespace, 
    modalProject.name, 
    modalMode === 'edit' && !!modalProject.name && isModalOpen
  )
  const createProjectMutation = useCreateProject()
  const updateProjectMutation = useUpdateProject()
  const deleteProjectMutation = useDeleteProject()

  // Open create modal if signaled by header
  useEffect(() => {
    const flag = localStorage.getItem('openCreateProjectModal')
    if (flag === '1') {
      localStorage.removeItem('openCreateProjectModal')
      setModalMode('create')
      setModalProject({ name: '' })
      setIsModalOpen(true)
    }
    const editName = localStorage.getItem('openEditProject')
    if (editName) {
      localStorage.removeItem('openEditProject')
      setModalMode('edit')
      setModalProject({ name: editName })
      setIsModalOpen(true)
    }
  }, [])

  // Update modal project state when current project is loaded
  useEffect(() => {
    if (modalMode === 'edit' && currentProjectResponse?.project) {
      setModalProject(prev => ({
        ...prev,
        config: currentProjectResponse.project.config
      }))
    }
  }, [modalMode, currentProjectResponse])

  // Convert API projects to UI format
  const projects = useMemo(() => {
    if (projectsResponse?.projects) {
      return apiProjectsToProjectItems(projectsResponse.projects)
    }
    return DEFAULT_PROJECTS
  }, [projectsResponse])
  
  const filteredProjects = useMemo(() => {
    return filterProjectsBySearch(projects, search)
  }, [projects, search])

  const openProject = (name: string) => {
    setActiveProject(name)
    navigate('/chat/dashboard')
  }

  const openCreate = () => {
    setModalMode('create')
    setModalProject({ name: '' })
    setIsModalOpen(true)
  }

  const openEdit = (name: string) => {
    setModalMode('edit')
    setModalProject({ name })
    setIsModalOpen(true)
  }



  const handleSave = async (name: string) => {
    if (modalMode === 'create') {
      try {
        await createProjectMutation.mutateAsync({
          namespace,
          request: { name, config_template: 'default' }
        })
        setActiveProject(name)
        setIsModalOpen(false)
        navigate('/chat/dashboard')
      } catch (error) {
        console.error('Failed to create project:', error)
        // Error handling is done by React Query and apiClient
      }
    } else {
      try {
        if (!modalProject.config) {
          console.error('Cannot update project: config not loaded')
          return
        }
        
        // Update the config with the new name while preserving all other properties
        const updatedConfig = mergeProjectConfig(modalProject.config, {
          name: name.trim(),
          namespace: namespace
        })
        
        // Validate the config before sending
        if (!validateProjectConfig(updatedConfig)) {
          console.error('Invalid project config, aborting update')
          return
        }
        
        await updateProjectMutation.mutateAsync({
          namespace,
          projectId: modalProject.name,
          request: { config: updatedConfig }
        })
        setActiveProject(name)
        setIsModalOpen(false)
      } catch (error) {
        console.error('Failed to update project:', error)
      }
    }
  }

  const handleDelete = async () => {
    try {
      await deleteProjectMutation.mutateAsync({
        namespace,
        projectId: modalProject.name
      })
      setIsModalOpen(false)
    } catch (error) {
      console.error('Failed to delete project:', error)
    }
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
              onClick={openCreate}
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
                    openEdit(p.name)
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
        isOpen={isModalOpen}
        mode={modalMode}
        initialName={modalProject.name}
        initialDescription={''}
        onClose={() => setIsModalOpen(false)}
        onSave={handleSave}
        onDelete={modalMode === 'edit' ? handleDelete : undefined}
        isLoading={createProjectMutation.isPending || updateProjectMutation.isPending || deleteProjectMutation.isPending || isLoadingProject}
      />
    </div>
  )
}

export default Projects
