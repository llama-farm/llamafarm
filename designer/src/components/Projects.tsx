import { useEffect, useMemo, useState } from 'react'
import FontIcon from '../common/FontIcon'
import { useNavigate } from 'react-router-dom'
import ProjectModal, { ProjectModalMode } from './ProjectModal'

interface ProjectItem {
  id: number
  name: string
  model: string
  lastEdited: string
  description?: string
}

const defaultProjectNames = [
  'aircraft-mx-flow',
  'Option 1',
  'Option 2',
  'Option 3',
  'Option 4',
]
const defaultProjects: ProjectItem[] = [
  { id: 1, name: 'Aircraft MX', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 2, name: 'SkyGuard', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 3, name: 'FalconEye', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 4, name: 'EagleVision', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 5, name: 'ThunderStrike', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 6, name: 'ViperWatch', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 7, name: 'HawkEye', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 8, name: 'StealthOps MX', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 9, name: 'JetStream', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 10, name: 'RaptorControl', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 11, name: 'AeroSentinel', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 12, name: 'CloudSurge', model: 'TinyLama', lastEdited: '8/15/2025' },
]

const Projects = () => {
  const [search, setSearch] = useState('')
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [modalMode, setModalMode] = useState<ProjectModalMode>('create')
  const [modalProject, setModalProject] = useState<{
    name: string
    description: string
  }>({ name: '', description: '' })
  const navigate = useNavigate()

  // Open create modal if signaled by header
  useEffect(() => {
    const flag = localStorage.getItem('openCreateProjectModal')
    if (flag === '1') {
      localStorage.removeItem('openCreateProjectModal')
      setModalMode('create')
      setModalProject({ name: '', description: '' })
      setIsModalOpen(true)
    }
  }, [])

  // Pull list from localStorage to mirror dropdown, fallback to defaults
  const listFromStorage = (() => {
    try {
      const stored = localStorage.getItem('projectsList')
      if (stored) return JSON.parse(stored) as string[]
    } catch {}
    return defaultProjectNames
  })()

  const mirroredProjects = useMemo<ProjectItem[]>(() => {
    // Map the names to card objects; if name not in demo list, still show it
    return listFromStorage.map((name, idx) => ({
      id: idx + 1,
      name,
      model: 'TinyLama',
      lastEdited: '8/15/2025',
    }))
  }, [listFromStorage])

  const filtered = useMemo(() => {
    const base =
      mirroredProjects.length > 0 ? mirroredProjects : defaultProjects
    if (!search) return base
    return base.filter(p => p.name.toLowerCase().includes(search.toLowerCase()))
  }, [mirroredProjects, search])

  const openProject = (name: string) => {
    localStorage.setItem('activeProject', name)
    navigate('/chat/dashboard')
  }

  const openCreate = () => {
    setModalMode('create')
    setModalProject({ name: '', description: '' })
    setIsModalOpen(true)
  }

  const openEdit = (name: string) => {
    setModalMode('edit')
    setModalProject({ name, description: '' })
    setIsModalOpen(true)
  }

  const saveProjectsList = (names: string[]) => {
    try {
      localStorage.setItem('projectsList', JSON.stringify(names))
    } catch {}
  }

  const handleSave = (name: string, /* description: string */) => {
    if (modalMode === 'create') {
      const exists = listFromStorage.includes(name)
      const updated = exists ? listFromStorage : [...listFromStorage, name]
      saveProjectsList(updated)
      localStorage.setItem('activeProject', name)
      setIsModalOpen(false)
      navigate('/chat/dashboard')
    } else {
      // edit: rename in list
      const updated = listFromStorage.map(n =>
        n === modalProject.name ? name : n
      )
      saveProjectsList(updated)
      localStorage.setItem('activeProject', name)
      setIsModalOpen(false)
    }
  }

  const handleDelete = () => {
    const updated = listFromStorage.filter(n => n !== modalProject.name)
    saveProjectsList(updated)
    setIsModalOpen(false)
  }

  return (
    <div className="w-full h-full transition-colors bg-gray-200 dark:bg-blue-800 pt-16">
      <div className="max-w-6xl mx-auto px-6 flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl text-gray-900 dark:text-blue-50">Projects</h2>
          <div className="flex items-center gap-2">
            <button className="px-3 py-2 rounded-lg border border-blue-50 text-blue-50 hover:bg-blue-50 hover:text-white dark:border-blue-400 dark:text-blue-100 dark:hover:bg-blue-600/40">
              Explore public projects
            </button>
            <button
              className="px-3 py-2 rounded-lg bg-blue-200 text-white hover:opacity-90"
              onClick={openCreate}
            >
              New project
            </button>
          </div>
        </div>

        <div className="w-full flex items-center bg-white dark:bg-blue-500 rounded-lg px-3 py-2 border border-blue-50 dark:border-blue-100">
          <FontIcon
            type="search"
            className="w-4 h-4 text-gray-700 dark:text-white"
          />
          <input
            className="w-full bg-transparent border-none focus:outline-none px-2 text-sm text-gray-800 dark:text-white"
            placeholder="Search projects"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-8">
          {filtered.map(p => (
            <div
              key={p.id}
              className="group w-full rounded-lg p-4 bg-white dark:bg-blue-600 border border-blue-50 dark:border-blue-600 cursor-pointer"
              onClick={() => openProject(p.name)}
            >
              <div className="flex items-start justify-between">
                <div className="text-base text-gray-900 dark:text-blue-50">
                  {p.name}
                </div>
                <FontIcon
                  type="arrow-right"
                  className="w-5 h-5 text-blue-200 dark:text-blue-100"
                />
              </div>
              <div className="mt-3">
                <span className="text-xs text-white bg-blue-100 dark:bg-blue-200 rounded-xl px-3 py-0.5">
                  {p.model}
                </span>
              </div>
              <div className="text-xs text-blue-100 mt-2">
                Last edited on {p.lastEdited}
              </div>
              <div className="mt-4 flex justify-end">
                <button
                  className="flex items-center gap-1 text-green-100 hover:opacity-80"
                  onClick={e => {
                    e.stopPropagation()
                    openEdit(p.name)
                  }}
                >
                  <FontIcon type="edit" className="w-5 h-5 text-green-100" />
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
        initialDescription={modalProject.description}
        onClose={() => setIsModalOpen(false)}
        onSave={handleSave}
        onDelete={modalMode === 'edit' ? handleDelete : undefined}
      />
    </div>
  )
}

export default Projects
