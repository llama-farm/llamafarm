import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
// removed decorative llama image
import FontIcon from './common/FontIcon'
import ProjectModal, { ProjectModalMode } from './components/ProjectModal'

function Home() {
  const [inputValue, setInputValue] = useState('')
  const [search, setSearch] = useState('')
  const navigate = useNavigate()

  const projectOptions = [
    { id: 1, text: 'AI Agent for Enterprise Product' },
    { id: 2, text: 'AI-Powered Chatbot for Customer Support' },
    { id: 3, text: 'AI Model for Predicting Equipment Failures' },
    { id: 4, text: 'Recommendation System for E-commerce' },
  ]

  const defaultProjectNames = [
    'aircraft-mx-flow',
    'Option 1',
    'Option 2',
    'Option 3',
    'Option 4',
  ]

  const [projectsList, setProjectsList] = useState<string[]>(() => {
    try {
      const stored = localStorage.getItem('projectsList')
      if (stored) return JSON.parse(stored) as string[]
    } catch {}
    return defaultProjectNames
  })

  const filteredProjectNames = useMemo(() => {
    if (!search) return projectsList
    return projectsList.filter(name =>
      name.toLowerCase().includes(search.toLowerCase())
    )
  }, [projectsList, search])

  const handleOptionClick = (option: { id: number; text: string }) => {
    setInputValue(option.text)
  }

  const handleSendClick = () => {
    navigate('/chat/data')
  }

  const openProject = (name: string) => {
    try {
      localStorage.setItem('activeProject', name)
    } catch {}
    navigate('/chat/dashboard')
  }

  const openCreate = () => {
    setModalMode('create')
    setModalProjectName('')
    setIsModalOpen(true)
  }

  // Local edit modal state for Home
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [modalProjectName, setModalProjectName] = useState('')
  const [modalMode, setModalMode] = useState<ProjectModalMode>('edit')

  const handleEditClick = (name: string) => {
    setModalProjectName(name)
    setModalMode('edit')
    setIsModalOpen(true)
  }

  // Listen for header-triggered create intent and scroll
  useEffect(() => {
    // Support router state-based control from Header
    try {
      // @ts-ignore - history state type
      const state = window.history.state && window.history.state.usr
      if (state?.openCreate) {
        setModalMode('create')
        setModalProjectName('')
        setIsModalOpen(true)
      }
      if (state?.scrollTo === 'projects') {
        const el = document.getElementById('projects')
        el?.scrollIntoView({ behavior: 'smooth' })
      }
    } catch {}
  }, [])

  return (
    <div className="min-h-screen flex flex-col items-stretch px-4 sm:px-6 lg:px-8 pt-24 md:pt-28 pb-8 bg-gradient-to-b from-blue-50 to-blue-100 dark:from-[#0c1634] dark:to-[#0b122b]">
      <div className="max-w-4xl w-full mx-auto text-center space-y-8">
        <div className="space-y-4">
          <p className="text-sm font-medium tracking-wide text-gray-700 dark:text-white">
            Welcome to LlamaFarm 🦙
          </p>

          <h1 className="font-serif text-2xl sm:text-3xl lg:text-4xl font-normal leading-tight text-gray-900 dark:text-white">
            What are you building?
          </h1>
        </div>
        <div className="max-w-3xl mx-auto">
          <div className="backdrop-blur-sm rounded-lg border-2 p-1 relative bg-white/80 border-gray-300 shadow-lg dark:bg-black/10 dark:border-blue-300/40 focus-within:border-blue-200 dark:focus-within:border-blue-300 transition-colors">
            <textarea
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              className="w-full h-24 sm:h-28 bg-transparent border-none resize-none p-4 pr-12 placeholder-opacity-60 focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed text-gray-900 placeholder-gray-500 dark:text-white dark:placeholder-white/60"
              placeholder="I'm building an agent that will work with my app..."
            />
            <button
              onClick={handleSendClick}
              className="absolute bottom-2 right-2 p-0 bg-transparent text-blue-200 hover:opacity-90"
              aria-label="Send"
            >
              <FontIcon type="arrow-filled" className="w-6 h-6 text-blue-200" />
            </button>
          </div>
        </div>

        <p className="max-w-2xl mx-auto text-sm sm:text-base leading-relaxed text-gray-600 dark:text-white/90">
          We'll help you bring your AI project dreams to life, all while showing
          you how we're doing it.
        </p>

        {/* Project option buttons */}
        <div className="max-w-4xl mx-auto space-y-4">
          {/* First row - stacks on mobile */}
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center">
            <button
              onClick={() => handleOptionClick(projectOptions[0])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-white/60 border-gray-300 text-gray-700 hover:bg-white/80 hover:border-gray-400 dark:bg-slate-800/20 dark:border-blue-300/40 dark:text-white dark:hover:bg-slate-700/30 dark:hover:border-blue-300/60"
            >
              {projectOptions[0].text}
            </button>

            <button
              onClick={() => handleOptionClick(projectOptions[1])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-white/60 border-gray-300 text-gray-700 hover:bg-white/80 hover:border-gray-400 dark:bg-slate-800/20 dark:border-blue-300/40 dark:text-white dark:hover:bg-slate-700/30 dark:hover:border-blue-300/60"
            >
              {projectOptions[1].text}
            </button>
          </div>

          {/* Second row */}
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center">
            <button
              onClick={() => handleOptionClick(projectOptions[2])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-white/60 border-gray-300 text-gray-700 hover:bg-white/80 hover:border-gray-400 dark:bg-slate-800/20 dark:border-blue-300/40 dark:text-white dark:hover:bg-slate-700/30 dark:hover:border-blue-300/60"
            >
              {projectOptions[2].text}
            </button>

            <button
              onClick={() => handleOptionClick(projectOptions[3])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-white/60 border-gray-300 text-gray-700 hover:bg-white/80 hover:border-gray-400 dark:bg-slate-800/20 dark:border-blue-300/40 dark:text-white dark:hover:bg-slate-700/30 dark:hover:border-blue-300/60"
            >
              {projectOptions[3].text}
            </button>
          </div>
        </div>

        {/* Your projects removed here to place outside the narrow container */}
      </div>

      {/* Your projects (moved outside to align with Resources width) */}
      <div
        id="projects"
        className="w-full max-w-6xl mx-auto px-6 mt-16 lg:mt-24"
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl text-blue-200 dark:text-white text-left">
            Your projects
          </h3>
          <div className="hidden md:flex items-center gap-2 shrink-0">
            <button className="px-3 py-2 rounded-lg border border-blue-200 text-blue-200 hover:bg-blue-200 hover:text-white dark:border-blue-400 dark:text-blue-100 dark:hover:bg-blue-600/40">
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
        {/* Controls for small screens */}
        <div className="md:hidden mb-4 flex items-center justify-between gap-3">
          <button className="flex-1 px-3 py-2 rounded-lg border border-blue-200 text-blue-200 hover:bg-blue-200 hover:text-white dark:border-blue-400 dark:text-blue-100 dark:hover:bg-blue-600/40">
            Explore public projects
          </button>
          <button
            className="px-3 py-2 rounded-lg bg-blue-200 text-white hover:opacity-90"
            onClick={openCreate}
          >
            New project
          </button>
        </div>

        {/* Search */}
        <div className="mb-4 w-full flex items-center bg-white dark:bg-blue-500 rounded-lg px-3 py-2 border border-blue-50 dark:border-blue-100">
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

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 pb-8">
          {filteredProjectNames.map(name => (
            <div
              key={name}
              className="group w-full rounded-lg p-4 bg-white dark:bg-blue-600 border border-blue-50 dark:border-blue-600 cursor-pointer"
              onClick={() => openProject(name)}
            >
              <div className="flex items-start justify-between">
                <div className="flex flex-col">
                  <div className="text-base text-gray-900 dark:text-blue-50">
                    {name}
                  </div>
                  <div className="mt-3">
                    <span className="text-xs text-white bg-blue-100 dark:bg-blue-200 rounded-xl px-3 py-0.5">
                      TinyLama
                    </span>
                  </div>
                  <div className="text-xs text-blue-100 mt-2">
                    Last edited on N/A
                  </div>
                </div>
                <FontIcon
                  type="arrow-right"
                  className="w-5 h-5 text-blue-200 dark:text-blue-100"
                />
              </div>
              <div className="mt-4 flex justify-end">
                <button
                  className="flex items-center gap-1 text-blue-200 hover:text-blue-300 dark:text-green-100 hover:opacity-80"
                  onClick={e => {
                    e.stopPropagation()
                    handleEditClick(name)
                  }}
                >
                  <FontIcon
                    type="edit"
                    className="w-5 h-5 text-blue-200 dark:text-green-100"
                  />
                  <span className="text-sm">Edit</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Resources footer-like section */}
      <div
        id="resources"
        className="w-full max-w-6xl mx-auto px-6 mt-20 lg:mt-28"
      >
        <h3 className="text-xl text-white mb-4">Resources</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <a
            href="https://github.com/llama-farm/llamafarm"
            target="_blank"
            rel="noreferrer"
            className="block rounded-lg p-4 bg-white dark:bg-blue-600 border border-blue-50 dark:border-blue-600 hover:shadow-md transition-shadow"
          >
            <div className="text-base text-gray-900 dark:text-blue-50">
              GitHub
            </div>
            <div className="text-sm text-gray-600 dark:text-blue-100">
              Source code and issues
            </div>
            <div className="mt-2 text-xs text-blue-200 dark:text-blue-100">
              github.com/llama-farm/llamafarm
            </div>
          </a>
          <a
            href="https://docs.llamafarm.dev/"
            target="_blank"
            rel="noreferrer"
            className="block rounded-lg p-4 bg-white dark:bg-blue-600 border border-blue-50 dark:border-blue-600 hover:shadow-md transition-shadow"
          >
            <div className="text-base text-gray-900 dark:text-blue-50">
              Documentation
            </div>
            <div className="text-sm text-gray-600 dark:text-blue-100">
              Guides and API references
            </div>
            <div className="mt-2 text-xs text-blue-200 dark:text-blue-100">
              docs.llamafarm.dev
            </div>
          </a>
          <a
            href="https://llamafarm.dev/"
            target="_blank"
            rel="noreferrer"
            className="block rounded-lg p-4 bg-white dark:bg-blue-600 border border-blue-50 dark:border-blue-600 hover:shadow-md transition-shadow"
          >
            <div className="text-base text-gray-900 dark:text-blue-50">
              Website
            </div>
            <div className="text-sm text-gray-600 dark:text-blue-100">
              Overview and updates
            </div>
            <div className="mt-2 text-xs text-blue-200 dark:text-blue-100">
              llamafarm.dev
            </div>
          </a>
        </div>
      </div>
      {/* Project edit modal over Home */}
      <ProjectModal
        isOpen={isModalOpen}
        mode={modalMode}
        initialName={modalProjectName}
        initialDescription={''}
        onClose={() => setIsModalOpen(false)}
        onSave={(name: string /* desc */) => {
          try {
            const stored = localStorage.getItem('projectsList')
            const list = stored ? (JSON.parse(stored) as string[]) : []
            if (list.includes(name) && name !== modalProjectName) {
              setIsModalOpen(false)
              return
            }
            const updated = list.map(n => (n === modalProjectName ? name : n))
            localStorage.setItem('projectsList', JSON.stringify(updated))
            localStorage.setItem('activeProject', name)
            // Update local state for immediate UI sync
            setProjectsList(updated)
            // Best-effort refresh via event for header/project dropdown consumers
            try {
              window.dispatchEvent(
                new CustomEvent<string>('lf-active-project', { detail: name })
              )
            } catch {}
            setIsModalOpen(false)
          } catch {
            setIsModalOpen(false)
          }
        }}
        onDelete={() => {
          try {
            const stored = localStorage.getItem('projectsList')
            const list = stored ? (JSON.parse(stored) as string[]) : []
            const updated = list.filter(n => n !== modalProjectName)
            localStorage.setItem('projectsList', JSON.stringify(updated))
            const next = updated[0] || 'aircraft-mx-flow'
            localStorage.setItem('activeProject', next)
            setProjectsList(updated)
            try {
              window.dispatchEvent(
                new CustomEvent<string>('lf-active-project', { detail: next })
              )
            } catch {}
            setIsModalOpen(false)
          } catch {
            setIsModalOpen(false)
          }
        }}
      />
    </div>
  )
}

export default Home

// Modal mount appended at end of component
