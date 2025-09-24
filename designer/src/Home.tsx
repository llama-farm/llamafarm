import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
// removed decorative llama image
import FontIcon from './common/FontIcon'
// Modal rendered globally in App
import { useProjects } from './hooks/useProjects'
import { useProjectModalContext } from './contexts/ProjectModalContext'
import {
  filterProjectsBySearch,
  getProjectsList,
} from './utils/projectConstants'
import { getCurrentNamespace } from './utils/namespaceUtils'
import { 
  createProjectFromChat, 
  encodeMessageForUrl 
} from './utils/homePageUtils'

function Home() {
  const [inputValue, setInputValue] = useState('')
  const [search, setSearch] = useState('')
  const [isCreatingProject, setIsCreatingProject] = useState(false)
  const navigate = useNavigate()

  // Enhanced chat functionality for project creation

  const projectOptions = [
    { id: 1, text: 'AI Agent for Enterprise Product' },
    { id: 2, text: 'AI-Powered Chatbot for Customer Support' },
    { id: 3, text: 'AI Model for Predicting Equipment Failures' },
    { id: 4, text: 'Recommendation System for E-commerce' },
  ]

  const namespace = getCurrentNamespace()

  // API hooks
  const { data: projectsResponse } = useProjects(namespace)

  // Convert API projects to project names for UI compatibility
  const projectsList = useMemo(
    () => getProjectsList(projectsResponse),
    [projectsResponse]
  )

  // Shared modal hook
  const projectModal = useProjectModalContext()

  const filteredProjectNames = useMemo(() => {
    return filterProjectsBySearch(
      projectsList.map(name => ({ name })),
      search
    ).map(item => item.name)
  }, [projectsList, search])

  const handleOptionClick = (option: { id: number; text: string }) => {
    setInputValue(option.text)
  }

  const handleSendClick = async () => {
    const messageContent = inputValue.trim()

    // Validate message for project creation
    if (!messageContent || messageContent.length < 3) {
      console.warn('Message too short for project creation:', messageContent)
      return
    }

    setIsCreatingProject(true)
    
    try {
      
      // 1. Create the project using the existing API
      const { projectName } = await createProjectFromChat(messageContent)
      
      
      // 2. Set the active project (using existing localStorage mechanism)
      localStorage.setItem('activeProject', projectName)
      
      // 3. Navigate to the new project with the message as a URL parameter
      // This allows the project page to pick up the conversation
      const encodedMessage = encodeMessageForUrl(messageContent)
      navigate(`/chat/dashboard?initialMessage=${encodedMessage}`)
      
      // Clear input on successful creation
      setInputValue('')
      
    } catch (error) {
      console.error('❌ Failed to create project from chat:', error)
      // Show error to user - you might want to add a toast notification here
      alert(`Failed to create project: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsCreatingProject(false)
    }
  }

  const openProject = (name: string) => {
    localStorage.setItem('activeProject', name)
    navigate('/chat/dashboard')
  }

  // Listen for header-triggered create intent and scroll (run once on mount)
  useEffect(() => {
    // Support router state-based control from Header
    try {
      // @ts-ignore - history state type
      const state = window.history.state && window.history.state.usr
      let usedState = false
      if (state?.openCreate) {
        projectModal.openCreateModal()
        usedState = true
      }
      if (state?.scrollTo === 'projects') {
        const el = document.getElementById('projects')
        el?.scrollIntoView({ behavior: 'smooth' })
        usedState = true
      }
      // Clear the one-time state so the modal doesn't immediately re-open after closing
      if (usedState) {
        navigate('.', { replace: true, state: undefined as any })
      }
      // Fallback: check localStorage hint
      const createFlag = localStorage.getItem('homeOpenCreate')
      if (createFlag === '1') {
        localStorage.removeItem('homeOpenCreate')
        projectModal.openCreateModal()
        const el = document.getElementById('projects')
        el?.scrollIntoView({ behavior: 'smooth' })
      }
    } catch {}
  }, [])

  return (
    <div className="min-h-screen flex flex-col items-stretch px-4 sm:px-6 lg:px-8 pt-24 md:pt-28 pb-8 bg-background">
      <div className="max-w-4xl w-full mx-auto text-center space-y-8">
        <div className="space-y-4">
          <p className="text-sm font-medium tracking-wide text-foreground/80">
            Welcome to LlamaFarm 🦙
          </p>

          <h1 className="font-serif text-2xl sm:text-3xl lg:text-4xl font-normal leading-tight text-foreground">
            What are you building?
          </h1>
        </div>
        <div className="max-w-3xl mx-auto">
          <div className="backdrop-blur-sm rounded-lg border-2 p-1 relative bg-card/90 border-input shadow-lg focus-within:border-primary transition-colors">
            <textarea
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSendClick()
                }
              }}
              className="w-full h-24 sm:h-28 bg-transparent border-none resize-none p-4 pr-12 placeholder-opacity-60 focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed text-foreground placeholder-foreground/50"
              placeholder={
                isCreatingProject 
                  ? "Creating your project..." 
                  : "I'm building an agent that will work with my app..."
              }
              disabled={isCreatingProject}
            />
            <button
              onClick={handleSendClick}
              disabled={isCreatingProject || !inputValue.trim()}
              className="absolute bottom-2 right-2 p-0 bg-transparent text-primary hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label={
                isCreatingProject 
                  ? 'Creating project...' 
                  : inputValue.trim() 
                    ? 'Create project and start chat' 
                    : 'Enter a message to create project'
              }
            >
              {isCreatingProject ? (
                <div className="w-6 h-6 flex items-center justify-center">
                  <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                </div>
              ) : (
                <FontIcon type="arrow-filled" className="w-6 h-6 text-primary" />
              )}
            </button>
          </div>
        </div>

        <p className="max-w-2xl mx-auto text-sm sm:text-base leading-relaxed text-foreground/80">
          {isCreatingProject ? (
            "Creating your project and setting up the chat environment..."
          ) : (
            "Describe what you're building and we'll create a project for you to start chatting about it right away."
          )}
        </p>

        {/* Project option buttons */}
        <div className="max-w-4xl mx-auto space-y-4">
          {/* First row - stacks on mobile */}
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center">
            <button
              onClick={() => handleOptionClick(projectOptions[0])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-card/60 border-input text-foreground hover:bg-card/80"
            >
              {projectOptions[0].text}
            </button>

            <button
              onClick={() => handleOptionClick(projectOptions[1])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-card/60 border-input text-foreground hover:bg-card/80"
            >
              {projectOptions[1].text}
            </button>
          </div>

          {/* Second row */}
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center">
            <button
              onClick={() => handleOptionClick(projectOptions[2])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-card/60 border-input text-foreground hover:bg-card/80"
            >
              {projectOptions[2].text}
            </button>

            <button
              onClick={() => handleOptionClick(projectOptions[3])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-card/60 border-input text-foreground hover:bg-card/80"
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
          <h3 className="text-xl text-primary text-left">Your projects</h3>
          <div className="hidden md:flex items-center gap-2 shrink-0">
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
        {/* Controls for small screens */}
        <div className="md:hidden mb-4 flex items-center justify-between gap-3">
          <button className="flex-1 px-3 py-2 rounded-lg border border-input text-primary hover:bg-accent/20">
            Explore public projects
          </button>
          <button
            className="px-3 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90"
            onClick={projectModal.openCreateModal}
          >
            New project
          </button>
        </div>

        {/* Search */}
        <div className="mb-4 w-full flex items-center bg-card rounded-lg px-3 py-2 border border-input">
          <FontIcon type="search" className="w-4 h-4 text-foreground" />
          <input
            className="w-full bg-transparent border-none focus:outline-none px-2 text-sm text-foreground"
            placeholder="Search projects"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 pb-8">
          {filteredProjectNames.map(name => (
            <div
              key={name}
              className="group w-full rounded-lg p-4 bg-card border border-border cursor-pointer"
              onClick={() => openProject(name)}
            >
              <div className="flex items-start justify-between">
                <div className="flex flex-col">
                  <div className="text-base text-foreground">{name}</div>
                  <div className="mt-3">
                    <span className="text-xs text-primary-foreground bg-primary rounded-xl px-3 py-0.5">
                      TinyLama
                    </span>
                  </div>
                  <div className="text-xs text-foreground/60 mt-2">
                    Last edited on N/A
                  </div>
                </div>
                <FontIcon type="arrow-right" className="w-5 h-5 text-primary" />
              </div>
              <div className="mt-4 flex justify-end">
                <button
                  className="flex items-center gap-1 text-primary hover:opacity-80"
                  onClick={e => {
                    e.stopPropagation()
                    projectModal.openEditModal(name)
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
            className="block rounded-lg p-4 bg-secondary border border-input hover:shadow-md transition-shadow"
          >
            <div className="text-base text-foreground">GitHub</div>
            <div className="text-sm text-muted-foreground">
              Source code and issues
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              github.com/llama-farm/llamafarm
            </div>
          </a>
          <a
            href="https://docs.llamafarm.dev/"
            target="_blank"
            rel="noreferrer"
            className="block rounded-lg p-4 bg-secondary border border-input hover:shadow-md transition-shadow"
          >
            <div className="text-base text-foreground">Documentation</div>
            <div className="text-sm text-muted-foreground">
              Guides and API references
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              docs.llamafarm.dev
            </div>
          </a>
          <a
            href="https://llamafarm.dev/"
            target="_blank"
            rel="noreferrer"
            className="block rounded-lg p-4 bg-secondary border border-input hover:shadow-md transition-shadow"
          >
            <div className="text-base text-foreground">Website</div>
            <div className="text-sm text-muted-foreground">
              Overview and updates
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              llamafarm.dev
            </div>
          </a>
        </div>
      </div>
      {/* Project edit modal over Home */}
      {/* Modal rendered globally in App */}
    </div>
  )
}

export default Home

// Modal mount appended at end of component
