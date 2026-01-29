import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
// removed decorative llama image
import FontIcon from './common/FontIcon'
// Modal rendered globally in App
import { useProjects, projectKeys } from './hooks/useProjects'
import { useQueryClient } from '@tanstack/react-query'
import { useProjectModalContext } from './contexts/ProjectModalContext'
import {
  filterProjectsBySearch,
  getProjectsList,
} from './utils/projectConstants'
import {
  getModelNames,
  formatLastModified,
  parseTimestamp,
} from './utils/projectHelpers'
import { getCurrentNamespace } from './utils/namespaceUtils'
import { setActiveProject } from './utils/projectUtils'
import projectService from './api/projectService'
import { mergeProjectConfig } from './utils/projectConfigUtils'
import {
  sanitizeProjectName,
  checkForDuplicateName,
  validateProjectName,
} from './utils/projectValidation'
import { Label } from './components/ui/label'
import { Input } from './components/ui/input'
import { useDemoModal } from './contexts/DemoModalContext'
import { getFileBasedDemos } from './config/demos'
import { useGitHubStars } from './hooks/useGitHubStars'
import { Star } from 'lucide-react'

function Home() {
  // Demo modal context
  const demoModal = useDemoModal()

  // Get file-based demos (for the demo section)
  const fileBasedDemos = getFileBasedDemos()
  const llamaDemo = fileBasedDemos[0] // First demo is Llama & Alpaca

  // Form state
  const [projectName, setProjectName] = useState('')
  const [deployment, setDeployment] = useState<'local' | 'cloud' | 'unsure'>(
    'local'
  )
  const [projectNameError, setProjectNameError] = useState<string | null>(null)
  const [generalError, setGeneralError] = useState<string | null>(null)

  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState<
    'newest' | 'oldest' | 'a-z' | 'z-a' | 'model'
  >('newest')
  const [isCreatingProject, setIsCreatingProject] = useState(false)
  const [loadingMsgIndex, setLoadingMsgIndex] = useState(0)
  const [fakeProgress, setFakeProgress] = useState(0)
  // Counter to force projectsList recalculation when localStorage changes
  const [projectsRefreshKey, setProjectsRefreshKey] = useState(0)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  // Enhanced chat functionality for project creation

  // Removed quick-pick pills to keep the form minimal

  const namespace = getCurrentNamespace()

  // API hooks
  const { data: projectsResponse } = useProjects(namespace)
  const { data: githubData } = useGitHubStars()

  // Convert API projects to project names for UI compatibility
  // projectsRefreshKey forces recalculation when localStorage changes (e.g., ghost project deleted)
  const projectsList = useMemo(
    () => getProjectsList(projectsResponse),
    [projectsResponse, projectsRefreshKey]
  )

  // Determine view mode based on project count
  const hasManyProjects = projectsList.length > 2

  // Get full project objects from API with precomputed sort keys
  // Uses namespace+name as key to avoid potential collisions
  const fullProjects = useMemo(() => {
    const apiProjects = projectsResponse?.projects || []
    return new Map(
      apiProjects.map(p => {
        const key = `${p.namespace}/${p.name}`
        return [
          key,
          {
            ...p,
            // Precompute sort keys for performance
            _sortTimestamp: parseTimestamp(p.last_modified),
            _sortModels: getModelNames(p.config),
          },
        ]
      })
    )
  }, [projectsResponse])

  // Shared modal hook
  const projectModal = useProjectModalContext()

  // create-form scroll handler no longer used (buttons removed)

  const filteredAndSortedProjectNames = useMemo(() => {
    // First, filter by search
    const filtered = filterProjectsBySearch(
      projectsList.map(name => ({ name })),
      search
    ).map(item => item.name)

    // Get current namespace for key lookup
    const currentNamespace = namespace

    // Then sort based on sortBy selection (inline immediately returned variable)
    return [...filtered].sort((a, b) => {
      // Use namespace+name composite key for lookup
      const projectA = fullProjects.get(`${currentNamespace}/${a}`)
      const projectB = fullProjects.get(`${currentNamespace}/${b}`)

      switch (sortBy) {
        case 'newest':
          // Use precomputed timestamps
          return (
            (projectB?._sortTimestamp || 0) - (projectA?._sortTimestamp || 0)
          )
        case 'oldest':
          // Use precomputed timestamps
          return (
            (projectA?._sortTimestamp || 0) - (projectB?._sortTimestamp || 0)
          )
        case 'a-z':
          return a.localeCompare(b)
        case 'z-a':
          return b.localeCompare(a)
        case 'model': {
          // Use precomputed model lists
          const modelA = projectA?._sortModels?.[0] || 'zzz'
          const modelB = projectB?._sortModels?.[0] || 'zzz'
          return modelA.localeCompare(modelB)
        }
        default:
          return 0
      }
    })
  }, [projectsList, search, sortBy, fullProjects, namespace])

  // No-op: pills removed

  const loadingMessages = [
    'Creating project…',
    'Saddling up the llama…',
    'Packing prompts in the saddlebag…',
    'Fluffing the retrieval fleece…',
    'Warming the stable for your model…',
    'Trotting to the chat pasture…',
  ]

  useEffect(() => {
    if (!isCreatingProject) {
      setLoadingMsgIndex(0)
      setFakeProgress(0)
      return
    }
    const msgTimer = setInterval(() => {
      setLoadingMsgIndex(i => (i + 1) % loadingMessages.length)
    }, 2800)
    const progTimer = setInterval(() => {
      setFakeProgress(p => {
        if (p >= 95) return 95
        const inc = Math.floor(1 + Math.random() * 3)
        return Math.min(p + inc, 95)
      })
    }, 180)
    return () => {
      clearInterval(msgTimer)
      clearInterval(progTimer)
    }
  }, [isCreatingProject])

  const hasAnyInput = projectName.trim().length > 0

  const handleCreateProject = async () => {
    const MIN_LOADING_MS = 3000

    // Validate and sanitize project name
    const sanitizedName = sanitizeProjectName(projectName)
    const validation = validateProjectName(sanitizedName)

    if (!validation.isValid) {
      setProjectNameError(validation.error || 'Invalid project name')
      return
    }

    // Check for duplicate name
    if (checkForDuplicateName(sanitizedName, projectsList)) {
      setProjectNameError('A project with this name already exists')
      return
    }

    setProjectNameError(null)
    setGeneralError(null)
    setIsCreatingProject(true)
    const startedAt = performance.now()

    try {
      // 1) Create the base project
      await projectService.createProject(namespace, {
        name: sanitizedName,
        config_template: 'default',
      })

      // 2) Save deployment preference
      if (deployment) {
        const brief: { deployment?: string } = {}
        brief.deployment = deployment

        // Get current config
        const currentProject = await projectService.getProject(
          namespace,
          sanitizedName
        )

        const mergedConfig = mergeProjectConfig(
          currentProject.project.config || {},
          {
            project_brief: brief,
          }
        )
        try {
          await projectService.updateProject(namespace, sanitizedName, {
            config: mergedConfig,
          })
        } catch (e) {
          console.error('Failed to update project brief:', e)
          // Non-critical, continue anyway
        }
      }

      // 3) Activate and navigate to dashboard
      setActiveProject(sanitizedName)

      // Optimistically update caches
      try {
        queryClient.invalidateQueries({ queryKey: projectKeys.list(namespace) })
      } catch {}

      // Persist in local fallback list
      try {
        const raw = localStorage.getItem('lf_custom_projects')
        const arr: string[] = raw ? JSON.parse(raw) : []
        if (!arr.includes(sanitizedName)) {
          localStorage.setItem(
            'lf_custom_projects',
            JSON.stringify([...arr, sanitizedName])
          )
        }
      } catch {}

      // Ensure the loading overlay is visible for at least MIN_LOADING_MS
      const elapsed = performance.now() - startedAt
      if (elapsed < MIN_LOADING_MS) {
        await new Promise(resolve =>
          setTimeout(resolve, MIN_LOADING_MS - elapsed)
        )
      }

      navigate('/chat/dashboard')
    } catch (error) {
      console.error('❌ Failed to create project:', error)
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error'
      setGeneralError(`Failed to create project: ${errorMessage}`)
    } finally {
      setIsCreatingProject(false)
    }
  }

  const openProject = (name: string) => {
    setActiveProject(name)
    navigate('/chat/dashboard')
  }

  // Start the llama demo directly - modal will auto-start it and show progress
  const handleStartLlamaDemo = () => {
    if (llamaDemo) {
      // Open modal with auto-start demo ID
      demoModal.openModal(llamaDemo.id)
    }
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
      if (state?.scrollTo === 'home-create-form') {
        // Scroll to the top to ensure the header and title are visible
        window.scrollTo({ top: 0, behavior: 'smooth' })
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

  // React to project deletions fired from the modal to update UI immediately
  useEffect(() => {
    const onDeleted = (event: Event) => {
      const deletedProjectName = (event as CustomEvent<string>).detail
      // Force refetch of projects list to ensure UI is updated
      queryClient.invalidateQueries({ queryKey: projectKeys.list(namespace) })
      // Increment refresh key to force useMemo recalculation (for localStorage-only projects)
      setProjectsRefreshKey(k => k + 1)

      // Clear active project if it was the one deleted
      try {
        const active = localStorage.getItem('activeProject')
        if (active === deletedProjectName) {
          localStorage.removeItem('activeProject')
          window.dispatchEvent(
            new CustomEvent<string>('lf-active-project', { detail: '' })
          )
        }
      } catch {}
    }
    window.addEventListener('lf-project-deleted' as any, onDeleted as any)
    return () =>
      window.removeEventListener('lf-project-deleted' as any, onDeleted as any)
  }, [namespace, queryClient])

  return (
    <div className="min-h-screen flex flex-col items-stretch pt-24 md:pt-28 pb-8 bg-background">
      <div className="max-w-6xl w-full mx-auto px-6 text-center space-y-8">
        {!hasManyProjects && (
          <>
            <div className="space-y-4">
              <div className="flex items-center justify-center gap-2">
                <p className="text-sm font-medium tracking-wide text-foreground/80">
                  Welcome to LlamaFarm
                </p>
                {githubData && (
                  <a
                    href="https://github.com/llama-farm/llamafarm"
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md border border-input bg-secondary hover:bg-accent/20 transition-colors text-xs text-foreground"
                  >
                    <span>GitHub</span>
                    <Star className="w-3 h-3 text-primary fill-primary" />
                    <span className="font-medium">
                      {githubData.stargazers_count.toLocaleString()}
                    </span>
                  </a>
                )}
              </div>

              <h1 className="font-serif text-2xl sm:text-3xl lg:text-4xl font-normal leading-tight text-foreground">
                Create a new project
              </h1>
            </div>

            {/* Split Screen Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
              {/* Left Side: Quick Start Demo */}
              <div className="rounded-xl border-2 border-primary/40 bg-card p-6 flex flex-col relative">
                {/* Center Recommended Tag */}
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-primary text-primary-foreground px-3 py-1 rounded-md text-xs font-semibold">
                  Recommended
                </div>

                <div className="mb-4 text-center">
                  <h2 className="text-xl font-semibold text-foreground mb-1">
                    Quick start demo
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    Learn LlamaFarm by exploring a pre-configured project.
                  </p>
                </div>

                {/* Llama Demo Card */}
                {llamaDemo && (
                  <div className="mb-6 rounded-lg border border-input bg-accent/50 p-6">
                    <div className="flex flex-col items-center text-center gap-3">
                      <div className="text-5xl">{llamaDemo.icon}</div>
                      <div className="flex flex-col gap-1">
                        <h3 className="font-semibold text-foreground text-base">
                          Llama & Alpaca Care
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          Chat with an encyclopedia about llama and alpaca care.
                        </p>
                      </div>
                      <span className="px-2 py-0.5 rounded-md text-xs bg-primary/10 text-primary">
                        Demo project
                      </span>
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex gap-3 mt-auto">
                  <button
                    onClick={() => demoModal.openModal()}
                    className="flex-1 px-4 py-3 rounded-lg border border-input bg-background text-foreground hover:bg-accent/20 font-medium transition-colors"
                  >
                    Explore demos
                  </button>
                  <button
                    onClick={handleStartLlamaDemo}
                    className="flex-1 px-4 py-3 rounded-lg bg-primary text-primary-foreground hover:opacity-90 font-medium transition-opacity"
                  >
                    Start
                  </button>
                </div>
              </div>

              {/* OR Divider */}
              <div className="lg:hidden text-center">
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-border"></div>
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="bg-background px-3 py-1 text-foreground font-medium">
                      OR
                    </span>
                  </div>
                </div>
              </div>

              {/* Right Side: Custom Project */}
              <div className="rounded-xl border-2 border-border bg-card p-6 flex flex-col">
                <div className="mb-4">
                  <h2 className="text-xl font-semibold text-foreground mb-1">
                    Custom project
                  </h2>
                </div>

                {/* Custom Project Form */}
                <div id="home-create-form" className="flex-1 flex flex-col">
                  {generalError && (
                    <div className="mb-4 text-red-600 bg-red-100 border border-red-300 rounded p-3 text-sm">
                      {generalError}
                    </div>
                  )}
                  <div className="grid gap-4 text-left flex-1 relative">
                    <div className="grid gap-2.5">
                      <Label htmlFor="projectName">Project name</Label>
                      <Input
                        id="projectName"
                        value={projectName}
                        onChange={e => {
                          setProjectName(e.target.value)
                          if (projectNameError) setProjectNameError(null)
                          if (generalError) setGeneralError(null)
                        }}
                        placeholder="my-project"
                        disabled={isCreatingProject}
                        className={projectNameError ? 'border-destructive' : ''}
                      />
                      {projectNameError && (
                        <p className="text-xs text-destructive">
                          {projectNameError}
                        </p>
                      )}
                      <p className="text-xs text-muted-foreground">
                        Only letters, numbers, underscores (_), and hyphens (-)
                        allowed. No spaces.
                      </p>
                    </div>

                    <div className="grid gap-2.5">
                      <Label>Where do you plan to deploy this?</Label>
                      <div className="flex flex-row gap-3">
                        <label className="flex-1 inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 h-10 hover:bg-accent/20 cursor-pointer">
                          <input
                            type="radio"
                            name="deploy"
                            className="h-4 w-4"
                            checked={deployment === 'local'}
                            onChange={() => setDeployment('local')}
                            disabled={isCreatingProject}
                          />
                          <span className="text-sm">Local machine</span>
                        </label>
                        <label className="flex-1 inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 h-10 hover:bg-accent/20 cursor-pointer">
                          <input
                            type="radio"
                            name="deploy"
                            className="h-4 w-4"
                            checked={deployment === 'cloud'}
                            onChange={() => setDeployment('cloud')}
                            disabled={isCreatingProject}
                          />
                          <span className="text-sm">Cloud</span>
                        </label>
                        <label className="flex-1 inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 h-10 hover:bg-accent/20 cursor-pointer">
                          <input
                            type="radio"
                            name="deploy"
                            className="h-4 w-4"
                            checked={deployment === 'unsure'}
                            onChange={() => setDeployment('unsure')}
                            disabled={isCreatingProject}
                          />
                          <span className="text-sm">Not sure</span>
                        </label>
                      </div>
                    </div>

                    <div className="pt-1 mt-auto">
                      <button
                        onClick={handleCreateProject}
                        disabled={isCreatingProject || !hasAnyInput}
                        className="w-full px-6 py-3 rounded-lg bg-muted text-foreground hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-opacity"
                        aria-label={
                          isCreatingProject
                            ? 'Creating project...'
                            : hasAnyInput
                              ? 'Create new project'
                              : 'Enter a project name to create'
                        }
                      >
                        {isCreatingProject ? 'Creating…' : 'Create project'}
                      </button>
                    </div>

                    {isCreatingProject && (
                      <div className="absolute inset-0 rounded-lg bg-background/70 backdrop-blur-[2px] flex flex-col items-center justify-center gap-4">
                        <div className="w-9 h-9 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                        <div className="text-sm font-serif text-foreground text-center px-4">
                          {loadingMessages[loadingMsgIndex]}
                        </div>
                        <div className="w-56 h-2 rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full bg-primary/70 transition-all duration-200"
                            style={{ width: `${fakeProgress}%` }}
                          />
                        </div>
                        <div className="flex items-center gap-1 mt-1">
                          <span
                            className="w-2 h-2 rounded-full bg-primary animate-bounce"
                            style={{ animationDelay: '0ms' }}
                          />
                          <span
                            className="w-2 h-2 rounded-full bg-primary animate-bounce"
                            style={{ animationDelay: '150ms' }}
                          />
                          <span
                            className="w-2 h-2 rounded-full bg-primary animate-bounce"
                            style={{ animationDelay: '300ms' }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {isCreatingProject && (
              <p className="max-w-2xl mx-auto text-xs sm:text-sm leading-relaxed text-foreground/80">
                Creating your project and setting up the dashboard...
              </p>
            )}
            {/* Your projects removed here to place outside the narrow container */}
          </>
        )}

        {/* Condensed view for >2 projects */}
        {hasManyProjects && (
          <div className="space-y-6">
            <div className="flex items-center justify-center gap-2">
              <p className="text-sm font-medium tracking-wide text-foreground/80">
                Welcome to LlamaFarm
              </p>
              {githubData && (
                <a
                  href="https://github.com/llama-farm/llamafarm"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md border border-input bg-secondary hover:bg-accent/20 transition-colors text-xs text-foreground"
                >
                  <span>GitHub</span>
                  <Star className="w-3 h-3 text-primary fill-primary" />
                  <span className="font-medium">
                    {githubData.stargazers_count.toLocaleString()}
                  </span>
                </a>
              )}
            </div>
            <h1 className="font-serif text-2xl sm:text-3xl lg:text-4xl font-normal leading-tight text-foreground">
              Your projects
            </h1>
            <div className="flex items-center justify-center gap-3 pt-2">
              <button
                className="flex-1 max-w-[200px] px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90 font-medium transition-opacity"
                onClick={() => projectModal.openCreateModal()}
              >
                Create new
              </button>
              <button
                className="flex-1 max-w-[200px] px-4 py-2 rounded-lg border border-input bg-background text-foreground hover:bg-accent/20 font-medium transition-colors"
                onClick={() => demoModal.openModal()}
              >
                Explore demo projects
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Your projects (moved outside to align with Resources width) */}
      <div
        id="projects"
        className={`w-full max-w-6xl mx-auto px-6 ${hasManyProjects ? 'mt-8' : 'mt-8 lg:mt-12'}`}
      >
        {!hasManyProjects && (
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl text-primary text-left">Your projects</h3>
            <div className="hidden md:flex items-center gap-2 shrink-0">
              <button
                className="px-3 py-2 rounded-lg border border-input text-primary hover:bg-accent/20"
                onClick={() => demoModal.openModal()}
              >
                Explore demo projects
              </button>
            </div>
          </div>
        )}
        {!hasManyProjects && (
          <div className="md:hidden mb-4 flex items-center justify-between gap-3">
            <button
              className="flex-1 px-3 py-2 rounded-lg border border-input text-primary hover:bg-accent/20"
              onClick={() => demoModal.openModal()}
            >
              Explore demo projects
            </button>
          </div>
        )}

        {/* Search and Sort - only show if there are projects */}
        {projectsList.length > 0 && (
          <div className="mb-4 flex flex-col sm:flex-row gap-3">
            <div className="flex-1 flex items-center bg-card rounded-lg px-3 py-2 border border-input">
              <FontIcon type="search" className="w-4 h-4 text-foreground" />
              <input
                className="w-full bg-transparent border-none focus:outline-none px-2 text-sm text-foreground"
                placeholder="Search projects"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <select
              className="px-3 py-2 rounded-lg border border-input bg-card text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              value={sortBy}
              onChange={e => setSortBy(e.target.value as any)}
            >
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="a-z">A-Z</option>
              <option value="z-a">Z-A</option>
              <option value="model">By Model</option>
            </select>
          </div>
        )}

        {/* Empty state when no projects or no search results */}
        {filteredAndSortedProjectNames.length === 0 ? (
          <div className="w-full">
            <div className="text-center px-6 py-8 rounded-xl border border-border bg-card/40">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/15 border border-primary/30">
                <FontIcon type="folder" className="w-6 h-6 text-primary" />
              </div>
              <div className="text-lg font-medium text-foreground mb-2">
                {projectsList.length === 0
                  ? 'No projects yet'
                  : 'No projects match your search'}
              </div>
              <div className="text-sm text-muted-foreground mb-6">
                {projectsList.length === 0
                  ? 'Create your first project to start building AI-powered applications. Each project can have its own models, prompts, and data.'
                  : "Try adjusting your search query to find what you're looking for."}
              </div>
              <button
                className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
                onClick={() => {
                  window.scrollTo({ top: 0, behavior: 'smooth' })
                  projectModal.openCreateModal()
                }}
              >
                Create your first project
              </button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 pb-8">
            {filteredAndSortedProjectNames.map(name => {
              const project = fullProjects.get(`${namespace}/${name}`)
              const modelNames = project?._sortModels || []
              const hasValidationError = project?.validation_error

              // Extract just the part after "/" from model names
              const getModelDisplayName = (modelName: string) => {
                const parts = modelName.split('/')
                return parts.length > 1 ? parts.slice(1).join('/') : modelName
              }

              // Show first 1 model, then "+N" for additional
              const visibleModels = modelNames
                .slice(0, 1)
                .map(getModelDisplayName)
              const additionalCount = modelNames.length - 1

              return (
                <div
                  key={name}
                  className="group w-full rounded-lg p-4 bg-card border border-border cursor-pointer flex flex-col"
                  onClick={() => openProject(name)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex flex-col flex-1 min-w-0">
                      <div className="text-base text-foreground line-clamp-2 break-words">
                        {name}
                      </div>
                      <div className="mt-3 flex flex-col gap-2 w-full">
                        {visibleModels.length > 0 ? (
                          <>
                            {visibleModels.map((model, idx) => (
                              <span
                                key={idx}
                                className="text-xs text-foreground/70 bg-muted rounded-xl px-3 py-0.5 w-full"
                              >
                                {model}
                              </span>
                            ))}
                            {additionalCount > 0 && (
                              <span
                                className="text-xs text-foreground/70 bg-muted rounded-xl px-3 py-0.5 w-full"
                                title={modelNames
                                  .slice(1)
                                  .map(getModelDisplayName)
                                  .join(', ')}
                              >
                                +{additionalCount}
                              </span>
                            )}
                          </>
                        ) : (
                          <span className="text-xs text-foreground/60 bg-muted rounded-xl px-3 py-0.5 w-full">
                            No model
                          </span>
                        )}
                        {hasValidationError && (
                          <span
                            className="text-xs text-red-100 bg-red-600 rounded-xl px-3 py-0.5 w-full"
                            title={hasValidationError}
                          >
                            Validation Error
                          </span>
                        )}
                      </div>
                    </div>
                    <FontIcon
                      type="arrow-right"
                      className="w-5 h-5 text-primary shrink-0 ml-2"
                    />
                  </div>
                  <div className="mt-auto pt-4 flex items-center justify-between">
                    <div className="text-xs text-foreground/60">
                      {formatLastModified(project?.last_modified)}
                    </div>
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
              )
            })}
          </div>
        )}
      </div>

      {/* Resources footer-like section */}
      <div
        id="resources"
        className="w-full max-w-6xl mx-auto px-6 mt-20 lg:mt-28"
      >
        <h3 className="text-xl text-primary mb-4">Resources</h3>
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
      {/* Demo Modal rendered globally in App via DemoModalRoot */}
    </div>
  )
}

export default Home

// Modal mount appended at end of component
