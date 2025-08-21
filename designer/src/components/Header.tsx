import { useEffect, useRef, useState } from 'react'
import FontIcon from '../common/FontIcon'
import { useLocation, useNavigate } from 'react-router-dom'
import { useTheme } from '../contexts/ThemeContext'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu'

function Header() {
  const [isBuilding, setIsBuilding] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const isSelected = location.pathname.split('/')[2]
  const { theme, setTheme } = useTheme()

  // Project dropdown state
  const defaultProjectNames = [
    'aircraft-mx-flow',
    'Option 1',
    'Option 2',
    'Option 3',
    'Option 4',
  ]
  const [isProjectOpen, setIsProjectOpen] = useState(false)
  const [projects /* setProjects */] = useState<string[]>(() => {
    try {
      const stored = localStorage.getItem('projectsList')
      if (stored) return JSON.parse(stored)
    } catch (err) {
      console.error('Failed to read projectsList from localStorage:', err)
    }
    return defaultProjectNames
  })
  const [activeProject, setActiveProject] = useState<string>(
    () => localStorage.getItem('activeProject') ?? 'aircraft-mx-flow'
  )
  const projectRef = useRef<HTMLDivElement>(null)

  // Page switching overlay (fade only)
  const [isSwitching, setIsSwitching] = useState(false)

  useEffect(() => {
    // Show middle nav only on /chat routes
    setIsBuilding(location.pathname.startsWith('/chat'))
  }, [location.pathname])

  // Keep activeProject in sync with localStorage when route changes (e.g., from Projects click)
  useEffect(() => {
    const stored = localStorage.getItem('activeProject')
    if (stored && stored !== activeProject) {
      setActiveProject(stored)
    }
  }, [location.pathname])

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

  // const persistProjects = (list: string[]) => {
  //   try {
  //     localStorage.setItem('projectsList', JSON.stringify(list))
  //   } catch {}
  // }

  // (removed unused handleCreateProject)

  const handleSelectProject = (name: string) => {
    const isDifferent = name !== activeProject
    setActiveProject(name)
    localStorage.setItem('activeProject', name)
    try {
      window.dispatchEvent(
        new CustomEvent<string>('lf-active-project', { detail: name })
      )
    } catch (err) {
      console.error('Failed to dispatch lf-active-project event:', err)
    }
    setIsProjectOpen(false)
    if (isDifferent) {
      setIsSwitching(true)
      setTimeout(() => setIsSwitching(false), 900)
    }
  }

  const isHomePage = location.pathname === '/'

  return (
    <header className="fixed top-0 left-0 z-50 w-full border-b transition-colors bg-background border-border">
      {/* Fade overlay (below header) */}
      {isSwitching && (
        <div className="fixed z-40 top-12 left-0 right-0 bottom-0 bg-background/60 backdrop-blur-[2px] page-fade-overlay"></div>
      )}

      <div className="w-full flex items-center h-12">
        <div className="w-1/4 pl-4 flex items-center gap-2">
          {isHomePage ? (
            <button
              className="font-serif text-base text-foreground"
              onClick={() => navigate('/')}
              aria-label="LlamaFarm Home"
            >
              <img
                src={
                  theme === 'dark'
                    ? '/logotype-long-tan.svg'
                    : '/logotype-long-tan-navy.svg'
                }
                alt="LlamaFarm"
                className="h-5 md:h-6 w-auto"
              />
            </button>
          ) : (
            <div ref={projectRef} className="flex items-center gap-2">
              <button
                onClick={() => navigate('/')}
                aria-label="LlamaFarm Home"
                className="hover:opacity-90 transition-opacity"
              >
                <img
                  src={
                    theme === 'dark'
                      ? '/llama-head-tan-dark.svg'
                      : '/llama-head-tan-light.svg'
                  }
                  alt="LlamaFarm logo"
                  className="h-5 md:h-6 w-auto"
                />
              </button>
              <DropdownMenu
                open={isProjectOpen}
                onOpenChange={setIsProjectOpen}
              >
                <DropdownMenuTrigger asChild>
                  <button
                    className="flex items-center gap-2 px-3 h-8 rounded-lg bg-secondary text-secondary-foreground hover:bg-secondary/80"
                    aria-haspopup="listbox"
                    aria-expanded={isProjectOpen}
                  >
                    <span className="font-serif text-base whitespace-nowrap text-foreground">
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
                      navigate('/', {
                        state: { openCreate: true, scrollTo: 'projects' },
                      })
                      try {
                        localStorage.setItem('homeOpenCreate', '1')
                      } catch {}
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

        <div
          className={`flex items-center w-3/4 justify-end pr-4 ${
            isBuilding ? 'justify-between' : 'justify-end'
          }`}
        >
          {isBuilding && (
            <div className="flex items-center gap-4 w-2/3">
              <button
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 ${
                  isSelected === 'dashboard'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-foreground hover:bg-secondary/80'
                }`}
                onClick={() => navigate('/chat/dashboard')}
              >
                <FontIcon type="dashboard" className="w-6 h-6" />
                <span>Dashboard</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 ${
                  isSelected === 'data'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-foreground hover:bg-secondary/80'
                }`}
                onClick={() => navigate('/chat/data')}
              >
                <FontIcon type="data" className="w-6 h-6" />
                <span>Data</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 ${
                  isSelected === 'models'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-foreground hover:bg-secondary/80'
                }`}
                onClick={() => navigate('/chat/models')}
              >
                <FontIcon type="model" className="w-6 h-6" />
                <span>Models</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 ${
                  isSelected === 'prompt'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-foreground hover:bg-secondary/80'
                }`}
                onClick={() => navigate('/chat/prompt')}
              >
                <FontIcon type="prompt" className="w-6 h-6" />
                <span>Prompt</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 ${
                  isSelected === 'test'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-foreground hover:bg-secondary/80'
                }`}
                onClick={() => navigate('/chat/test')}
              >
                <FontIcon type="test" className="w-6 h-6" />
                <span>Test</span>
              </button>
            </div>
          )}

          <div className="flex items-center gap-3 justify-end">
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
            <FontIcon type="user-avatar" className="w-6 h-6 text-foreground" />
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
