import { useEffect, useRef, useState } from 'react'
import FontIcon from '../common/FontIcon'
import { useLocation, useNavigate } from 'react-router-dom'
import { useTheme } from '../contexts/ThemeContext'

function Header() {
  const [isBuilding, setIsBuilding] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const isSelected = location.pathname.split('/')[2]
  const { setTheme } = useTheme()

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

  const handleCreateProject = () => {
    setIsProjectOpen(false)
    navigate('/')
    // open create modal on Home and scroll to projects
    try {
      localStorage.setItem('homeOpenCreate', '1')
    } catch {}
    setTimeout(() => {
      const el = document.getElementById('projects')
      el?.scrollIntoView({ behavior: 'smooth' })
      window.dispatchEvent(new Event('home-open-create'))
    }, 0)
  }

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
    <header className="fixed top-0 left-0 z-50 w-full border-b transition-colors bg-white border-gray-200 dark:bg-blue-700 dark:border-blue-400/30">
      {/* Fade overlay (below header) */}
      {isSwitching && (
        <div className="fixed z-40 top-12 left-0 right-0 bottom-0 bg-white/60 dark:bg-blue-800/60 backdrop-blur-[2px] page-fade-overlay"></div>
      )}

      <div className="w-full flex items-center h-12">
        <div className="w-1/4 pl-4 flex items-center gap-2">
          <div
            className="font-serif text-base font-medium select-none text-gray-900 dark:text-white"
            aria-hidden
          >
            🦙
          </div>
          {isHomePage ? (
            <button
              className="font-serif text-base text-gray-900 dark:text-white"
              onClick={() => navigate('/')}
            >
              LlamaFarm
            </button>
          ) : (
            <div className="relative" ref={projectRef}>
              <button
                className="flex items-center gap-2 px-3 h-8 rounded-lg bg-gray-200 text-gray-800 hover:bg-gray-300 dark:bg-blue-500/40 dark:text-white dark:hover:bg-blue-500/60"
                onClick={() => setIsProjectOpen(p => !p)}
                aria-haspopup="listbox"
                aria-expanded={isProjectOpen}
              >
                <span className="font-serif text-base whitespace-nowrap">
                  {activeProject}
                </span>
                <FontIcon
                  type="chevron-down"
                  className={`w-4 h-4 ${isProjectOpen ? 'rotate-180' : ''}`}
                />
              </button>

              {isProjectOpen && (
                <div className="absolute mt-2 w-72 max-h-[60vh] overflow-auto rounded-lg shadow-lg border border-blue-400/30 bg-white text-gray-900 dark:bg-blue-700 dark:text-white">
                  {/* Options */}
                  {projects.map(name => (
                    <button
                      key={name}
                      className={`w-full text-left px-4 py-3 transition-colors hover:bg-blue-600/20 dark:hover:bg-blue-600/40 ${
                        name === activeProject ? 'opacity-100' : 'opacity-90'
                      }`}
                      onClick={() => handleSelectProject(name)}
                    >
                      <div className="border-b border-blue-400/20 pb-3 last:border-b-0">
                        {name}
                      </div>
                    </button>
                  ))}
                  <div className="px-4 py-3 border-t border-blue-400/20">
                    <button
                      className="w-full flex items-center justify-center gap-2 rounded-md border border-green-100 text-green-100 hover:bg-green-100 hover:text-blue-700 transition-colors px-3 py-2"
                      onClick={handleCreateProject}
                    >
                      <FontIcon type="add" className="w-4 h-4" />
                      <span>Create new project</span>
                    </button>
                    <button
                      className="w-full mt-2 flex items-center justify-center gap-2 rounded-md text-blue-100 hover:bg-blue-600/30 px-3 py-2"
                      type="button"
                      onClick={() => {
                        setIsProjectOpen(false)
                        navigate('/')
                        setTimeout(() => {
                          const el = document.getElementById('projects')
                          el?.scrollIntoView({ behavior: 'smooth' })
                        }, 0)
                      }}
                    >
                      All projects
                    </button>
                  </div>
                </div>
              )}
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
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 hover:bg-blue-100 dark:hover:bg-blue-600 dark:hover:opacity-80 ${
                  isSelected === 'dashboard'
                    ? 'bg-blue-50 dark:bg-blue-600'
                    : ''
                }`}
                onClick={() => navigate('/chat/dashboard')}
              >
                <FontIcon
                  type="dashboard"
                  className="w-6 h-6 text-gray-700 dark:text-white"
                />
                <span className="text-gray-700 dark:text-white">Dashboard</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 hover:bg-blue-100 dark:hover:bg-blue-600 dark:hover:opacity-80 ${
                  isSelected === 'data' ? 'bg-blue-50 dark:bg-blue-600' : ''
                }`}
                onClick={() => navigate('/chat/data')}
              >
                <FontIcon
                  type="data"
                  className="w-6 h-6 text-gray-700 dark:text-white"
                />
                <span className="text-gray-700 dark:text-white">Data</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 hover:bg-blue-100 dark:hover:bg-blue-600 dark:hover:opacity-80 ${
                  isSelected === 'prompt' ? 'bg-blue-50 dark:bg-blue-600' : ''
                }`}
                onClick={() => navigate('/chat/prompt')}
              >
                <FontIcon
                  type="prompt"
                  className="w-6 h-6 text-gray-700 dark:text-white"
                />
                <span className="text-gray-700 dark:text-white">Prompt</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 hover:bg-blue-100 dark:hover:bg-blue-600 dark:hover:opacity-80 ${
                  isSelected === 'test' ? 'bg-blue-50 dark:bg-blue-600' : ''
                }`}
                onClick={() => navigate('/chat/test')}
              >
                <FontIcon
                  type="test"
                  className="w-6 h-6 text-gray-700 dark:text-white"
                />
                <span className="text-gray-700 dark:text-white">Test</span>
              </button>
            </div>
          )}

          <div className="flex items-center gap-3 justify-end">
            <div className="flex rounded-lg overflow-hidden border-gray-300 dark:border-blue-400/50 dark:border">
              <button
                className={`w-8 h-7 flex items-center justify-center transition-colors bg-blue-100 text-white hover:bg-gray-200 dark:bg-transparent dark:text-blue-100 dark:hover:bg-blue-700`}
                onClick={() => setTheme('light')}
              >
                <FontIcon type="sun" className="w-4 h-4" />
              </button>
              <button
                className={`w-8 h-7 flex items-center justify-center transition-colors text-gray-100 bg-[#F4F4F4] hover:text-white hover:bg-gray-100 dark:text-white dark:bg-blue-400  dark:hover:bg-blue-800/50`}
                onClick={() => setTheme('dark')}
              >
                <FontIcon type="moon-filled" className="w-4 h-4" />
              </button>
            </div>
            <FontIcon
              type="user-avatar"
              className="w-6 h-6 text-gray-700 dark:text-white"
            />
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
