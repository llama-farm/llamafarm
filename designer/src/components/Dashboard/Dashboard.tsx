import FontIcon from '../../common/FontIcon'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Mode } from '../ModeToggle'
import PageActions from '../common/PageActions'
import DataCards from './DataCards'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import { useProjectModalContext } from '../../contexts/ProjectModalContext'
import { useProject } from '../../hooks/useProjects'
// import { getCurrentNamespace } from '../../utils/namespaceUtils'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useListDatasets } from '../../hooks/useDatasets'

const Dashboard = () => {
  const navigate = useNavigate()
  // const namespace = getCurrentNamespace()
  const activeProject = useActiveProject()

  // All state declarations first
  const [mode, setMode] = useState<Mode>('designer')
  const [projectName, setProjectName] = useState<string>('Dashboard')
  // Datasets list for Data card
  const { data: apiDatasets, isLoading: isDatasetsLoading } = useListDatasets(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject?.namespace && !!activeProject?.project }
  )

  // Active project details for description and project brief fields
  const { data: projectDetail } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject?.namespace && !!activeProject?.project
  )

  const { brief } = useMemo(() => {
    const cfg = (projectDetail?.project?.config || {}) as Record<string, any>
    const project_brief = (cfg?.project_brief || {}) as Record<string, any>
    // Fallback to localStorage cache if server-side brief not present yet
    if (!project_brief || Object.keys(project_brief).length === 0) {
      try {
        const ns = activeProject?.namespace || ''
        const pid = activeProject?.project || ''
        if (ns && pid) {
          const briefKey = `lf_project_brief_${ns}_${pid}`
          const cached = localStorage.getItem(briefKey)
          if (cached) {
            const parsed = JSON.parse(cached)
            return {
              brief: {
                what: parsed?.what || '',
                goals: parsed?.goals || '',
                audience: parsed?.audience || '',
              },
            }
          }
        }
      } catch {}
    }
    return {
      brief: {
        what: project_brief?.what || '',
        goals: project_brief?.goals || '',
        audience: project_brief?.audience || '',
      },
    }
  }, [projectDetail, activeProject?.namespace, activeProject?.project])

  const datasets = useMemo(() => {
    // Only return datasets from the API, no localStorage fallback
    if (apiDatasets?.datasets && apiDatasets.datasets.length > 0) {
      return apiDatasets.datasets.map(dataset => ({
        id: dataset.name,
        name: dataset.name,
        lastRun: new Date(),
      }))
    }
    return [] as Array<{ id: string; name: string; lastRun: string | Date }>
  }, [apiDatasets])

  // Calculate dashboard stats
  const filesProcessed = useMemo(() => {
    if (apiDatasets?.datasets && apiDatasets.datasets.length > 0) {
      return apiDatasets.datasets.reduce((sum, dataset) => {
        return sum + (dataset.files?.length || 0)
      }, 0)
    }
    return 0
  }, [apiDatasets])

  const databaseCount = useMemo(() => {
    const databases = projectDetail?.project?.config?.rag?.databases
    return Array.isArray(databases) ? databases.length : 0
  }, [projectDetail])

  const modelsCount = useMemo(() => {
    return projectDetail?.project?.config?.runtime?.model ? 1 : 0
  }, [projectDetail])

  // Shared modal hook
  const projectModal = useProjectModalContext()

  useEffect(() => {
    const refresh = () => {
      try {
        const stored = localStorage.getItem('activeProject')
        if (stored) setProjectName(stored)
      } catch {}
    }
    refresh()
    const handler = (e: Event) => {
      // @ts-ignore custom event detail
      const detailName = (e as CustomEvent<string>).detail
      if (detailName) setProjectName(detailName)
      else refresh()
    }
    window.addEventListener('lf-active-project', handler as EventListener)
    return () =>
      window.removeEventListener('lf-active-project', handler as EventListener)
  }, [])

  // Keep default project model in sync (listen for updates)
  const [defaultModelName, setDefaultModelName] = useState<string>(() => {
    try {
      const raw = localStorage.getItem('lf_default_project_model')
      if (raw) {
        const parsed = JSON.parse(raw)
        return parsed?.name || 'TinyLlama'
      }
    } catch {}
    return 'TinyLlama'
  })
  useEffect(() => {
    const load = () => {
      try {
        const raw = localStorage.getItem('lf_default_project_model')
        if (raw) {
          const parsed = JSON.parse(raw)
          setDefaultModelName(parsed?.name || 'TinyLlama')
        }
      } catch {}
    }
    const handler = () => load()
    window.addEventListener(
      'lf:defaultProjectModelUpdated',
      handler as EventListener
    )
    window.addEventListener('storage', handler)
    return () => {
      window.removeEventListener(
        'lf:defaultProjectModelUpdated',
        handler as EventListener
      )
      window.removeEventListener('storage', handler)
    }
  }, [])

  return (
    <>
      <div className="w-full h-full flex flex-col">
        <div className="flex items-center justify-between mb-4 flex-shrink-0">
          <div className="flex items-center gap-2">
            <h2 className="text-2xl break-words ">
              {mode === 'designer' ? projectName : 'Config editor'}
            </h2>
            {mode === 'designer' && (
              <button
                className="rounded-sm hover:opacity-80"
                onClick={() => {
                  projectModal.openEditModal(projectName)
                }}
              >
                <FontIcon type="edit" className="w-5 h-5 text-primary" />
              </button>
            )}
          </div>
          <PageActions mode={mode} onModeChange={setMode} />
        </div>
        {mode !== 'designer' ? (
          <div className="flex-1 min-h-0 overflow-hidden pb-6">
            <ConfigEditor className="h-full" />
          </div>
        ) : (
          <>
            {/* Project details card */}
            <div className="w-full flex flex-col mb-4">
              <div className="h-[40px] px-2 flex items-center justify-between rounded-tl-lg rounded-tr-lg bg-card border-b border-border">
                <div className="flex flex-row gap-2 items-center text-foreground pl-2">
                  Project details
                </div>
                <button
                  className="text-xs text-primary flex flex-row gap-1 items-center pr-3"
                  onClick={() => projectModal.openEditModal(projectName)}
                >
                  <FontIcon type="edit" className="w-4 h-4" />
                  Edit
                </button>
              </div>
              <div className="p-6 flex flex-col gap-4 rounded-b-lg bg-card">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <div className="text-xs text-muted-foreground">
                      What are you building?
                    </div>
                    <div className="text-foreground break-words">
                      {brief.what && brief.what.trim().length > 0
                        ? brief.what
                        : '—'}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">
                      What do you hope to achieve?
                    </div>
                    <div className="text-foreground break-words">
                      {brief.goals && brief.goals.trim().length > 0
                        ? brief.goals
                        : '—'}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">
                      Who will use this?
                    </div>
                    <div className="text-foreground break-words">
                      {brief.audience && brief.audience.trim().length > 0
                        ? brief.audience
                        : '—'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <DataCards
              filesProcessed={filesProcessed}
              databaseCount={databaseCount}
              modelsCount={modelsCount}
            />
            <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              {/* Data (1/3) */}
              <div className="flex flex-col min-w-0 overflow-hidden">
                <div className="flex flex-row gap-2 items-center h-[40px] px-2 rounded-tl-lg rounded-tr-lg justify-between bg-card border-b border-border">
                  <div className="flex flex-row gap-2 items-center text-foreground">
                    <FontIcon type="data" className="w-4 h-4" />
                    Data
                  </div>
                  <button
                    className="text-xs text-primary"
                    onClick={() => navigate('/chat/data')}
                  >
                    View and add
                  </button>
                </div>
                <div className="p-4 md:p-6 flex flex-col gap-2 rounded-b-lg bg-card md:min-h-[260px]">
                  {isDatasetsLoading ? (
                    <div className="text-xs text-muted-foreground">
                      Loading…
                    </div>
                  ) : datasets.length === 0 ? (
                    <div className="text-xs text-muted-foreground">
                      No datasets yet
                    </div>
                  ) : (
                    <>
                      {datasets.slice(0, 8).map(d => (
                        <div
                          key={d.id}
                          className="py-1 px-2 rounded-lg flex flex-row gap-2 items-center justify-between bg-secondary cursor-pointer hover:bg-accent/30 min-w-0 overflow-hidden"
                          onClick={() =>
                            navigate(`/chat/data/${encodeURIComponent(d.name)}`)
                          }
                          role="button"
                          aria-label={`Open dataset ${d.name}`}
                        >
                          <div className="text-foreground truncate min-w-0">
                            {d.name}
                          </div>
                          <div className="text-xs text-muted-foreground whitespace-nowrap shrink-0 hidden lg:inline">
                            {(() => {
                              const dt = new Date(d.lastRun)
                              return `Updated ${dt.toLocaleDateString()} ${dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
                            })()}
                          </div>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              </div>

              {/* Models (1/3) */}
              <div className="min-w-0 overflow-hidden">
                <div className="h-[40px] px-2 flex items-center rounded-tl-lg rounded-tr-lg bg-card border-b border-border">
                  <div className="flex flex-row gap-2 items-center text-foreground">
                    <FontIcon type="model" className="w-4 h-4" />
                    Models
                  </div>
                </div>
                <div className="p-4 md:p-6 flex flex-col justify-start md:justify-between rounded-b-lg bg-card md:min-h-[260px]">
                  <div className="flex flex-col gap-3">
                    <div>
                      <label className="text-xs text-muted-foreground flex items-center gap-2">
                        Default inference model
                        <div className="relative group">
                          <FontIcon
                            type="info"
                            className="w-3.5 h-3.5 text-muted-foreground"
                          />
                          <div className="pointer-events-none absolute left-1/2 -translate-x-1/2 mt-2 w-64 rounded-md border border-border bg-popover p-2 text-xs text-popover-foreground opacity-0 group-hover:opacity-100 transition-opacity">
                            Generates AI responses to your questions using the
                            relevant documents as context.
                          </div>
                        </div>
                      </label>
                      <div className="mt-2 rounded-xl border border-primary/50 bg-background px-4 py-2 text-base font-medium text-foreground">
                        {defaultModelName}
                      </div>
                    </div>
                  </div>
                  <div className="pt-4">
                    <button
                      className="w-full text-primary border border-primary rounded-lg py-2 text-base hover:bg-primary/10"
                      onClick={() => navigate('/chat/models')}
                    >
                      Go to models
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
      {/* Modal rendered globally in App */}
    </>
  )
}

export default Dashboard
