import { useEffect, useRef, useState, useMemo } from 'react'
import FontIcon from '../common/FontIcon'
import { useLocation, useNavigate } from 'react-router-dom'
import { useTheme } from '../contexts/ThemeContext'
import { useModeReset } from '../contexts/ModeContext'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu'
import { useProjects } from '../hooks/useProjects'
import { useProjectModalContext } from '../contexts/ProjectModalContext'
import {
  setActiveProject as setActiveProjectUtil,
  getActiveProject,
} from '../utils/projectUtils'
import { getCurrentNamespace } from '../utils/namespaceUtils'
import { getProjectsList } from '../utils/projectConstants'
import { useQueryClient } from '@tanstack/react-query'
import { VersionDetailsDialog } from './common/VersionDetailsDialog'
import UpgradeModal from './common/UpgradeModal'
import { getInjectedImageTag } from '../utils/versionUtils'
import { projectKeys } from '../hooks/useProjects'
import { Button } from './ui/button'
import { useMobileView } from '../contexts/MobileViewContext'

type HeaderProps = { currentVersion?: string }

function Header({ currentVersion }: HeaderProps) {
  const [isBuilding, setIsBuilding] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const isSelected = location.pathname.split('/')[2]
  const { theme, setTheme } = useTheme()
  const { triggerReset } = useModeReset()
  const [versionDialogOpen, setVersionDialogOpen] = useState(false)
  const [upgradeModalOpen, setUpgradeModalOpen] = useState(false)
  const [effectiveVersion, setEffectiveVersion] = useState<string>('0.0.0')
  const { isMobile, mobileView, markUserChoice } = useMobileView()

  // Project dropdown state
  const [isProjectOpen, setIsProjectOpen] = useState(false)
  const [activeProject, setActiveProject] = useState<string>(() =>
    getActiveProject()
  )
  const namespace = getCurrentNamespace()

  // API hooks
  const { data: projectsResponse } = useProjects(namespace)
  const queryClient = useQueryClient()
  // Convert API projects to project names for dropdown with fallback
  const projects = useMemo(() => {
    return getProjectsList(projectsResponse)
  }, [projectsResponse, isProjectOpen])
  const projectRef = useRef<HTMLDivElement>(null)
  // Project modal from centralized context
  const projectModal = useProjectModalContext()

  // Page switching overlay (fade only)
  const [isSwitching, setIsSwitching] = useState(false)

  useEffect(() => {
    // Show middle nav only on /chat routes
    setIsBuilding(location.pathname.startsWith('/chat'))
  }, [location.pathname])

  const emitSetMobileView = (v: 'chat' | 'project') => {
    markUserChoice(v)
  }

  // Navigation items for header tabs
  const navigationItems = [
    { key: 'dashboard', label: 'Dashboard', icon: 'dashboard' },
    { key: 'prompt', label: 'Prompts', icon: 'prompt' },
    { key: 'data', label: 'Data', icon: 'data' },
    { key: 'databases', label: 'Databases', icon: 'rag' },
    { key: 'models', label: 'Models', icon: 'model' },
    { key: 'test', label: 'Test', icon: 'test' },
  ] as const

  // Mapping for mobile project dropdown (labels + icons)
  const pageDefs: Record<
    string,
    { label: string; icon: string; path: string }
  > = {
    dashboard: {
      label: 'Dashboard',
      icon: 'dashboard',
      path: '/chat/dashboard',
    },
    prompt: { label: 'Prompts', icon: 'prompt', path: '/chat/prompt' },
    data: { label: 'Data', icon: 'data', path: '/chat/data' },
    databases: { label: 'Databases', icon: 'rag', path: '/chat/databases' },
    models: { label: 'Models', icon: 'model', path: '/chat/models' },
    test: { label: 'Test', icon: 'test', path: '/chat/test' },
  }
  // Resolve effective version: prefer injected image tag, then prop
  useEffect(() => {
    let alive = true
    const tag = getInjectedImageTag()
    const normalized =
      tag && typeof tag === 'string' && tag.trim() !== ''
        ? tag.startsWith('v')
          ? tag.slice(1)
          : tag
        : (currentVersion || '0.0.0').replace(/^v/, '')
    if (alive) setEffectiveVersion(normalized)
    return () => {
      alive = false
    }
  }, [currentVersion])

  // Keep activeProject in sync with localStorage when route changes (e.g., from Projects click)
  useEffect(() => {
    const stored = getActiveProject()
    if (stored && stored !== activeProject) {
      setActiveProject(stored)
    }
  }, [location.pathname, activeProject])

  // Close dropdown on outside click
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (!projectRef.current) return
      if (!projectRef.current.contains(e.target as Node)) {
        setIsProjectOpen(false)
      }
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [])

  // React to global active project change events and storage updates
  useEffect(() => {
    const onActiveProject = (e: Event) => {
      try {
        const { detail } = e as CustomEvent<string>
        if (detail && detail !== activeProject) {
          setActiveProject(detail)
        }
      } catch {}
    }
    const onStorage = (e: StorageEvent) => {
      if (e.key === 'activeProject' && typeof e.newValue === 'string') {
        if (e.newValue !== activeProject) setActiveProject(e.newValue)
      }
    }
    window.addEventListener('lf-active-project' as any, onActiveProject as any)
    window.addEventListener('storage', onStorage)
    return () => {
      window.removeEventListener(
        'lf-active-project' as any,
        onActiveProject as any
      )
      window.removeEventListener('storage', onStorage)
    }
  }, [activeProject])

  // Synchronize animation with query loading state
  useEffect(() => {
    if (!isSwitching) return

    const currentProjectKey = projectKeys.detail(namespace, activeProject)
    const isLoading =
      queryClient.isFetching({ queryKey: currentProjectKey }) > 0

    if (!isLoading) {
      // End animation when data is loaded
      const timer = setTimeout(() => setIsSwitching(false), 100) // Small delay for smoother transition
      return () => clearTimeout(timer)
    }
  }, [queryClient, namespace, activeProject, isSwitching])

  // (removed unused persistProjects and handleCreateProject)

  const handleSelectProject = (name: string) => {
    const isDifferent = name !== activeProject

    if (isDifferent) {
      // Invalidate the current project query to force refetch
      const currentProjectKey = projectKeys.detail(namespace, activeProject)
      queryClient.invalidateQueries({ queryKey: currentProjectKey })

      // Set the new active project
      setActiveProject(name)
      setActiveProjectUtil(name)

      // Show switching animation - will be ended by useEffect when loading completes
      setIsSwitching(true)
    }

    setIsProjectOpen(false)
  }

  const isHomePage = location.pathname === '/'
  const isHomeLike =
    isHomePage ||
    location.pathname === '/samples' ||
    location.pathname.startsWith('/samples/')

  return (
    <header className="fixed top-0 left-0 z-50 w-full border-b transition-colors bg-background border-border">
      {/* Fade overlay (below header) */}
      {isSwitching && (
        <div className="fixed z-40 top-12 left-0 right-0 bottom-0 bg-background/60 backdrop-blur-[2px] page-fade-overlay"></div>
      )}

      <div className="w-full flex items-center h-12 relative">
        <div className="w-auto sm:w-1/4 pl-3 flex items-center gap-1.5 min-w-0 relative">
          {isHomeLike ? (
            <button
              className="font-serif text-base text-foreground"
              onClick={() => navigate('/')}
              aria-label="LlamaFarm Home"
            >
              <img
                src={
                  theme === 'dark'
                    ? '/logotype-long-tan.svg'
                    : '/logotype-long tan-navy.svg'
                }
                alt="LlamaFarm"
                className="h-7 md:h-8 w-auto"
              />
            </button>
          ) : (
            <div ref={projectRef} className="flex items-center gap-2 min-w-0">
              <button
                onClick={() => navigate('/')}
                aria-label="LlamaFarm Home"
                className="hover:opacity-90 transition-opacity shrink-0"
              >
                <img
                  src={
                    theme === 'dark'
                      ? '/llama-head-tan-dark.svg'
                      : '/llama-head-tan-light.svg'
                  }
                  alt="LlamaFarm logo"
                  className="h-7 md:h-8 w-auto shrink-0"
                />
              </button>
              <DropdownMenu
                open={isProjectOpen}
                onOpenChange={setIsProjectOpen}
              >
                <DropdownMenuTrigger asChild>
                  <button
                    className="flex items-center justify-center sm:justify-start gap-1.5 sm:gap-2 px-2 sm:px-3 h-8 rounded-lg bg-secondary text-secondary-foreground hover:bg-secondary/80 min-w-0 overflow-hidden w-8 sm:w-auto max-w-[28vw] sm:max-w-[22vw] md:max-w-[24vw] lg:max-w-[28vw] xl:max-w-[320px]"
                    aria-haspopup="listbox"
                    aria-expanded={isProjectOpen}
                    aria-label={activeProject}
                    title={activeProject}
                  >
                    <span className="hidden sm:inline font-serif text-base whitespace-nowrap text-foreground truncate max-w-full">
                      {activeProject}
                    </span>
                    <FontIcon
                      type="chevron-down"
                      className={`w-4 h-4 ${isProjectOpen ? 'rotate-180' : ''}`}
                    />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-72 max-h-[60vh] overflow-auto rounded-lg border border-border bg-popover text-popover-foreground">
                  {projects.map(name => (
                    <DropdownMenuItem
                      key={name}
                      className={`px-4 py-3 transition-colors hover:bg-accent/20 ${
                        name === activeProject ? 'opacity-100' : 'opacity-90'
                      }`}
                      onClick={() => handleSelectProject(name)}
                    >
                      <div className="w-full border-b border-border pb-3 last:border-b-0">
                        {name}
                      </div>
                    </DropdownMenuItem>
                  ))}
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    className="px-0"
                    onSelect={() => {
                      setIsProjectOpen(false)
                      projectModal.openCreateModal()
                    }}
                  >
                    <div className="w-full flex items-center justify-center gap-2 rounded-md border border-input text-primary hover:bg-primary hover:text-primary-foreground transition-colors px-3 py-2">
                      <FontIcon type="add" className="w-4 h-4" />
                      <span>Create new project</span>
                    </div>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    className="px-0"
                    onSelect={() => {
                      setIsProjectOpen(false)
                      navigate('/', { state: { scrollTo: 'projects' } })
                      setTimeout(() => {
                        const el = document.getElementById('projects')
                        el?.scrollIntoView({ behavior: 'smooth' })
                      }, 0)
                    }}
                  >
                    <div className="w-full flex items-center justify-center gap-2 rounded-md text-primary hover:bg-accent/20 px-3 py-2">
                      All projects
                    </div>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          )}
        </div>

        {isBuilding && (
          <div className="hidden md:flex absolute left-1/2 -translate-x-1/2 items-center gap-3 z-10">
            {navigationItems.map(item => (
              <button
                key={item.key}
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg py-2 pl-2 pr-3 ${
                  isSelected === item.key
                    ? 'bg-primary text-primary-foreground'
                    : 'text-foreground hover:bg-secondary/80'
                }`}
                onClick={() => {
                  if (isSelected === item.key) {
                    triggerReset()
                  }
                  navigate(`/chat/${item.key}`)
                }}
              >
                <FontIcon
                  type={item.icon as any}
                  className="w-6 h-6 shrink-0"
                />
                <span className="hidden lg:inline">{item.label}</span>
              </button>
            ))}
          </div>
        )}

        <div className="flex items-center gap-3 justify-end absolute right-4 top-1/2 -translate-y-1/2">
          {/* Version pill on Home only - place to the left of the theme toggle */}
          {isHomeLike ? (
            <button
              className="hidden sm:inline-flex items-center rounded-full border border-input text-foreground text-xs h-7 px-2.5"
              onClick={() => setVersionDialogOpen(true)}
              title="Version details"
            >
              <span className="font-mono">v{effectiveVersion}</span>
            </button>
          ) : null}
          <div className="flex rounded-lg overflow-hidden border border-border">
            <button
              className={`w-8 h-7 flex items-center justify-center transition-colors ${
                theme === 'light'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
              }`}
              onClick={() => setTheme('light')}
              aria-pressed={theme === 'light'}
              title="Light mode"
            >
              <FontIcon type="sun" className="w-4 h-4" />
            </button>
            <button
              className={`w-8 h-7 flex items-center justify-center transition-colors ${
                theme === 'dark'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
              }`}
              onClick={() => setTheme('dark')}
              aria-pressed={theme === 'dark'}
              title="Dark mode"
            >
              <FontIcon type="moon-filled" className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Mobile chat/project switcher in header when on /chat */}
      {isBuilding && isMobile ? (
        <div className="md:hidden absolute left-1/2 -translate-x-1/2 top-1.5 z-10">
          <div className="rounded-full border border-border bg-card shadow p-1 flex gap-1">
            <Button
              variant={mobileView === 'chat' ? 'secondary' : 'ghost'}
              size="sm"
              className={`rounded-full px-4 ${
                mobileView === 'chat'
                  ? 'bg-primary text-primary-foreground'
                  : ''
              }`}
              onClick={() => emitSetMobileView('chat')}
              aria-pressed={mobileView === 'chat'}
            >
              Chat
            </Button>
            <Button
              variant={mobileView === 'project' ? 'secondary' : 'ghost'}
              size="sm"
              className={`rounded-full px-4 ${
                mobileView === 'project'
                  ? 'bg-primary text-primary-foreground'
                  : ''
              }`}
              onClick={() => emitSetMobileView('project')}
              aria-pressed={mobileView === 'project'}
            >
              Project
            </Button>
          </div>
        </div>
      ) : null}

      {/* Mobile second-row nav when Project view is selected */}
      {isBuilding && isMobile && mobileView === 'project' ? (
        <div className="md:hidden fixed top-12 left-0 right-0 z-40 bg-background border-b border-border">
          <div className="py-2 px-3">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="w-full px-4 py-2 rounded-lg bg-secondary text-secondary-foreground hover:bg-secondary/80 flex items-center justify-between gap-3">
                  <span className="flex items-center gap-2 font-serif text-base">
                    <FontIcon
                      type={(pageDefs[isSelected]?.icon || 'dashboard') as any}
                      className="w-5 h-5"
                    />
                    {pageDefs[isSelected]?.label || 'Dashboard'}
                  </span>
                  <FontIcon type="chevron-down" className="w-4 h-4" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-[calc(100vw-24px)] max-w-none rounded-lg border border-border bg-popover text-popover-foreground">
                {Object.values(pageDefs).map(def => (
                  <DropdownMenuItem
                    key={def.path}
                    onClick={() => navigate(def.path)}
                    className="flex items-center gap-2"
                  >
                    <FontIcon type={def.icon as any} className="w-4 h-4" />
                    <span>{def.label}</span>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      ) : null}
      {/* Version dialog mounted at header scope */}
      <VersionDetailsDialog
        open={versionDialogOpen}
        onOpenChange={setVersionDialogOpen}
        onRequestUpgrade={() => {
          setVersionDialogOpen(false)
          setUpgradeModalOpen(true)
        }}
      />
      <UpgradeModal
        open={upgradeModalOpen}
        onOpenChange={setUpgradeModalOpen}
      />
      {/* Modal is rendered by ProjectModalRoot in App */}
    </header>
  )
}

// moved dialog control into Header component state for clearer boundaries

export default Header
