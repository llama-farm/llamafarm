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
import { getCurrentNamespace } from './utils/namespaceUtils'
import { encodeMessageForUrl } from './utils/homePageUtils'
import projectService from './api/projectService'
import { mergeProjectConfig } from './utils/projectConfigUtils'
import {
  sanitizeProjectName,
  checkForDuplicateName,
} from './utils/projectValidation'
import { Label } from './components/ui/label'
import { Input } from './components/ui/input'
import { Textarea } from './components/ui/textarea'

function Home() {
  // Form state
  const [what, setWhat] = useState('')
  const [goals, setGoals] = useState('')
  const [audience, setAudience] = useState('')
  const [deployment, setDeployment] = useState<'local' | 'cloud' | 'unsure'>(
    'local'
  )

  const [search, setSearch] = useState('')
  const [isCreatingProject, setIsCreatingProject] = useState(false)
  const [loadingMsgIndex, setLoadingMsgIndex] = useState(0)
  const [fakeProgress, setFakeProgress] = useState(0)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  // Enhanced chat functionality for project creation

  // Removed quick-pick pills to keep the form minimal

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

  // create-form scroll handler no longer used (buttons removed)

  const filteredProjectNames = useMemo(() => {
    return filterProjectsBySearch(
      projectsList.map(name => ({ name })),
      search
    ).map(item => item.name)
  }, [projectsList, search])

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

  const summarizeWhatToSlug = (text: string): string => {
    const stopwords = new Set([
      'the',
      'a',
      'an',
      'and',
      'or',
      'for',
      'to',
      'with',
      'of',
      'on',
      'in',
      'into',
      'by',
      'from',
      'about',
      'this',
      'that',
      'these',
      'those',
      'is',
      'am',
      'are',
      'be',
      'being',
      'been',
      'it',
      'its',
      'my',
      'our',
      'your',
      'their',
      'his',
      'her',
      'as',
      'at',
      'we',
      'i',
      'you',
      'they',
      'what',
      'which',
      'who',
      'whom',
      'will',
      'can',
      'could',
      'should',
      'would',
      'may',
      'might',
      'just',
      'like',
      'make',
      'makes',
      'made',
      'build',
      'building',
      'create',
      'creating',
      'new',
      'project',
    ])
    const tokens = text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(Boolean)
      .filter(t => t.length > 2 && !stopwords.has(t))
    const unique: string[] = []
    for (const t of tokens) {
      if (!unique.includes(t)) unique.push(t)
    }
    const picked = unique.slice(0, 3)
    return picked.join('-') || 'project'
  }

  const composeInitialMessage = (): string => {
    const parts: string[] = []
    if (what.trim()) parts.push(`What: ${what.trim()}`)
    if (goals.trim()) parts.push(`Goals: ${goals.trim()}`)
    if (audience.trim()) parts.push(`Users: ${audience.trim()}`)
    if (deployment)
      parts.push(
        `Deployment: ${deployment === 'local' ? 'Local machine' : deployment === 'cloud' ? 'Cloud' : 'Not sure'}`
      )
    return parts.join('\n')
  }

  const hasAnyInput =
    what.trim().length > 0 ||
    goals.trim().length > 0 ||
    audience.trim().length > 0

  const handleCreateProject = async () => {
    const MIN_LOADING_MS = 3000
    // Autogenerate project name from "what" or generic fallback
    const baseFromWhat = summarizeWhatToSlug(what || goals || audience || '')
    let desiredName = sanitizeProjectName(baseFromWhat).replace(/\s+/g, '-')

    // Ensure uniqueness optimistically against current list
    let finalName = desiredName
    const allNames = projectsList
    if (checkForDuplicateName(finalName, allNames)) {
      const suffix = Date.now().toString().slice(-3)
      finalName = `${finalName}-${suffix}`
    }

    setIsCreatingProject(true)
    const startedAt = performance.now()
    try {
      // 1) Create the project
      const created = await projectService.createProject(namespace, {
        name: finalName,
        config_template: 'default',
      })

      // 2) Save brief answers into config
      const brief = {
        what: what || undefined,
        goals: goals || undefined,
        audience: audience || undefined,
        deployment,
      }
      const mergedConfig = mergeProjectConfig(created.project.config || {}, {
        project_brief: brief,
      })
      try {
        await projectService.updateProject(namespace, created.project.name, {
          config: mergedConfig,
        })
      } catch (e) {
        // Notify the user that config update failed, but project was created
        try {
          window.alert(
            'Project was created, but saving project details failed. Some information may not be saved.'
          )
        } catch {}
        console.error('Failed to update project config:', e)
      }

      // Also persist brief locally for resilience against schema mismatches or offline use
      try {
        const briefKey = `lf_project_brief_${namespace}_${created.project.name}`
        localStorage.setItem(briefKey, JSON.stringify(brief))
      } catch {}

      // 3) Activate and navigate with initial message
      localStorage.setItem('activeProject', created.project.name)
      // Optimistically update caches so it appears in dropdowns/lists immediately
      try {
        queryClient.setQueryData(
          projectKeys.detail(namespace, created.project.name),
          {
            project: {
              namespace,
              name: created.project.name,
              config: mergedConfig,
            },
          }
        )
        queryClient.invalidateQueries({ queryKey: projectKeys.list(namespace) })
      } catch {}

      // Also persist in local fallback list used elsewhere
      try {
        const raw = localStorage.getItem('lf_custom_projects')
        const arr: string[] = raw ? JSON.parse(raw) : []
        if (!arr.includes(created.project.name)) {
          localStorage.setItem(
            'lf_custom_projects',
            JSON.stringify([...arr, created.project.name])
          )
        }
      } catch {}
      // Ensure the fun loading overlay is visible for at least MIN_LOADING_MS
      const elapsed = performance.now() - startedAt
      if (elapsed < MIN_LOADING_MS) {
        await new Promise(resolve =>
          setTimeout(resolve, MIN_LOADING_MS - elapsed)
        )
      }

      const initialMessage = composeInitialMessage()
      const encoded = encodeMessageForUrl(initialMessage)
      navigate(`/chat/dashboard?initialMessage=${encoded}`)
    } catch (error) {
      console.error('❌ Failed to create project:', error)
      alert(
        `Failed to create project: ${error instanceof Error ? error.message : 'Unknown error'}`
      )
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
    const onDeleted = () => {
      // Invalidate local derived list by forcing a refilter
      setSearch(s => s + '')
    }
    window.addEventListener('lf-project-deleted' as any, onDeleted as any)
    return () =>
      window.removeEventListener('lf-project-deleted' as any, onDeleted as any)
  }, [])

  return (
    <div className="min-h-screen flex flex-col items-stretch px-4 sm:px-6 lg:px-8 pt-24 md:pt-28 pb-8 bg-background">
      <div className="max-w-4xl w-full mx-auto text-center space-y-8">
        <div className="space-y-4">
          <p className="text-sm font-medium tracking-wide text-foreground/80">
            Welcome to LlamaFarm 🦙
          </p>

          <h1 className="font-serif text-2xl sm:text-3xl lg:text-4xl font-normal leading-tight text-foreground">
            Tell us about your new project
          </h1>
        </div>
        <div id="home-create-form" className="max-w-3xl mx-auto">
          <div className="rounded-lg border p-4 sm:p-5 bg-card border-input shadow-sm relative">
            <div className="grid gap-4 text-left">
              <div className="grid gap-2.5">
                <Label htmlFor="what">What are you building?</Label>
                <Textarea
                  id="what"
                  value={what}
                  onChange={e => setWhat(e.target.value)}
                  placeholder="A customer support chatbot, inventory system, data dashboard..."
                  className="min-h-[72px]"
                  disabled={isCreatingProject}
                />
              </div>

              <div className="grid gap-2.5">
                <Label htmlFor="goals">What do you hope to achieve?</Label>
                <Textarea
                  id="goals"
                  value={goals}
                  onChange={e => setGoals(e.target.value)}
                  placeholder="Reduce response times, automate tasks, improve satisfaction..."
                  className="min-h-[72px]"
                  disabled={isCreatingProject}
                />
              </div>

              <div className="grid gap-2.5">
                <Label htmlFor="audience">Who will use this?</Label>
                <Input
                  id="audience"
                  value={audience}
                  onChange={e => setAudience(e.target.value)}
                  placeholder="Support team, end customers, internal employees..."
                  disabled={isCreatingProject}
                />
              </div>

              <div className="grid gap-2.5">
                <Label>Where do you plan to deploy this?</Label>
                <div className="flex flex-col sm:flex-row gap-3 sm:gap-4">
                  <label className="inline-flex items-center gap-2 rounded-md border border-input bg-card px-3 py-2 hover:bg-accent/20">
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
                  <label className="inline-flex items-center gap-2 rounded-md border border-input bg-card px-3 py-2 hover:bg-accent/20">
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
                  <label className="inline-flex items-center gap-2 rounded-md border border-input bg-card px-3 py-2 hover:bg-accent/20">
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

              <div className="flex justify-end pt-1">
                <button
                  onClick={handleCreateProject}
                  disabled={isCreatingProject || !hasAnyInput}
                  className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
                  aria-label={
                    isCreatingProject
                      ? 'Creating project...'
                      : hasAnyInput
                        ? 'Create new project'
                        : 'Fill at least one field to create a project'
                  }
                >
                  {isCreatingProject ? 'Creating…' : 'Create new project'}
                </button>
              </div>
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

        <p className="max-w-2xl mx-auto text-xs sm:text-sm leading-relaxed text-foreground/80">
          {isCreatingProject
            ? 'Creating your project and setting up the chat environment...'
            : 'Provide at least one detail above (what, goals, or users) to create your project.'}
        </p>
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
            <button
              className="px-3 py-2 rounded-lg border border-input text-primary hover:bg-accent/20"
              onClick={() => navigate('/samples')}
            >
              Explore sample projects
            </button>
            {/* New project button removed per design */}
          </div>
        </div>
        {/* Controls for small screens */}
        <div className="md:hidden mb-4 flex items-center justify-between gap-3">
          <button
            className="flex-1 px-3 py-2 rounded-lg border border-input text-primary hover:bg-accent/20"
            onClick={() => navigate('/samples')}
          >
            Explore sample projects
          </button>
          {/* New project button removed per design */}
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
              className="group w-full rounded-lg p-4 bg-card border border-border cursor-pointer flex flex-col"
              onClick={() => openProject(name)}
            >
              <div className="flex items-start justify-between">
                <div className="flex flex-col flex-1 min-w-0">
                  <div className="text-base text-foreground line-clamp-2 break-words">
                    {name}
                  </div>
                  <div className="mt-3">
                    <span className="text-xs text-primary-foreground bg-primary rounded-xl px-3 py-0.5">
                      TinyLama
                    </span>
                  </div>
                  <div className="text-xs text-foreground/60 mt-2">
                    Last edited on N/A
                  </div>
                </div>
                <FontIcon
                  type="arrow-right"
                  className="w-5 h-5 text-primary shrink-0 ml-2"
                />
              </div>
              <div className="mt-auto pt-4 flex justify-end">
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
    </div>
  )
}

export default Home

// Modal mount appended at end of component
