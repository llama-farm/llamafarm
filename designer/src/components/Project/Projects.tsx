import { useEffect, useMemo, useState } from 'react'
import FontIcon from '../../common/FontIcon'
import { useNavigate } from 'react-router-dom'
import ProjectModal, { ProjectModalMode } from './ProjectModal'
import {
  DEFAULT_PROJECTS,
  getProjectsList,
  saveProjectsList,
  setActiveProject,
  namesToProjectItems,
  filterProjectsBySearch,
  updateProjectInList,
  removeProjectFromList,
  addProjectToList,
} from '../../utils/projectUtils'

const Projects = () => {
  const [search, setSearch] = useState('')
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [modalMode, setModalMode] = useState<ProjectModalMode>('create')
  const [modalProject, setModalProject] = useState<{
    name: string
  }>({ name: '' })
  const navigate = useNavigate()

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

  const projectsList = getProjectsList()
  const projects = useMemo(() => namesToProjectItems(projectsList), [projectsList])
  const filteredProjects = useMemo(() => {
    const base = projects.length > 0 ? projects : DEFAULT_PROJECTS
    return filterProjectsBySearch(base, search)
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



  const handleSave = (name: string) => {
    if (modalMode === 'create') {
      const updated = addProjectToList(projectsList, name)
      saveProjectsList(updated)
      setActiveProject(name)
      setIsModalOpen(false)
      navigate('/chat/dashboard')
    } else {
      const updated = updateProjectInList(projectsList, modalProject.name, name)
      saveProjectsList(updated)
      setActiveProject(name)
      setIsModalOpen(false)
    }
  }

  const handleDelete = () => {
    const updated = removeProjectFromList(projectsList, modalProject.name)
    saveProjectsList(updated)
    setIsModalOpen(false)
  }

  return (
    <div className="w-full h-full transition-colors bg-background pt-16">
      <div className="max-w-6xl mx-auto px-6 flex flex-col gap-6">
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
          {filteredProjects.map(p => (
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
      />
    </div>
  )
}

export default Projects
