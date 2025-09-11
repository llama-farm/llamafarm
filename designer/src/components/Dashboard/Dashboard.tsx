import FontIcon from '../../common/FontIcon'
import GitlabLogoDark from '../../assets/logos/gitlab-logo-dark.svg'
import GithubLogoDark from '../../assets/logos/github-logo-dark.svg'
import SlackLogoDark from '../../assets/logos/slack-logo-dark.svg'
import GitlabLogoLight from '../../assets/logos/gitlab-logo-light.svg'
import GithubLogoLight from '../../assets/logos/github-logo-light.svg'
import SlackLogoLight from '../../assets/logos/slack-logo-light.svg'
import { useTheme } from '../../contexts/ThemeContext'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Mode } from '../ModeToggle'
import PageActions from '../common/PageActions'
import DataCards from './DataCards'
import ProjectModal from '../../components/Project/ProjectModal'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import { useProjectModal } from '../../hooks/useProjectModal'
import { getCurrentNamespace } from '../../utils/namespaceUtils'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useListDatasets } from '../../hooks/useDatasets'

const Dashboard = () => {
  const { theme } = useTheme()
  const navigate = useNavigate()
  const namespace = getCurrentNamespace()
  const activeProject = useActiveProject()

  // All state declarations first
  const [mode, setMode] = useState<Mode>('designer')
  const [projectName, setProjectName] = useState<string>('Dashboard')
  const [versions, setVersions] = useState<
    Array<{
      id: string
      name: string
      description: string
      date: string
      isCurrent?: boolean
    }>
  >([])
  // Datasets list for Data card
  const { data: apiDatasets, isLoading: isDatasetsLoading } = useListDatasets(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject?.namespace && !!activeProject?.project }
  )

  const datasets = useMemo(() => {
    if (apiDatasets?.datasets && apiDatasets.datasets.length > 0) {
      return apiDatasets.datasets.map(dataset => ({
        id: dataset.name,
        name: dataset.name,
        lastRun: new Date(),
      }))
    }
    try {
      const stored = localStorage.getItem('lf_demo_datasets')
      if (stored) {
        const parsed = JSON.parse(stored)
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed.map((d: any) => ({
            id: d.id || d.name,
            name: d.name || d.id,
            lastRun: d.lastRun || new Date(),
          }))
        }
      }
    } catch {}
    return [] as Array<{ id: string; name: string; lastRun: string | Date }>
  }, [apiDatasets])

  // Shared modal hook
  const projectModal = useProjectModal({
    namespace,
    existingProjects: [], // Dashboard doesn't need duplicate checking since it edits current project
    onSuccess: (newProjectName, mode) => {
      if (mode === 'edit' && newProjectName) {
        setProjectName(newProjectName)
      }
    },
  })

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

  // Load and keep versions list in sync with Versions page/localStorage
  useEffect(() => {
    const load = () => {
      try {
        const raw = localStorage.getItem('lf_versions')
        if (raw) {
          const arr = JSON.parse(raw)
          if (Array.isArray(arr)) setVersions(arr)
          else setVersions([])
        } else setVersions([])
      } catch {
        setVersions([])
      }
    }
    load()
    const onUpdate = () => load()
    window.addEventListener('lf_versions_updated', onUpdate as EventListener)
    window.addEventListener('storage', onUpdate)
    return () => {
      window.removeEventListener(
        'lf_versions_updated',
        onUpdate as EventListener
      )
      window.removeEventListener('storage', onUpdate)
    }
  }, [])

  return (
    <>
      <div className="w-full h-full flex flex-col">
        <div className="flex items-center justify-between mb-4 flex-shrink-0">
          <div className="flex items-center gap-2">
            <h2 className="text-2xl ">
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
            <DataCards />
            <div className="w-full flex flex-row gap-4 mt-4">
              <div className="w-3/5 flex flex-col gap-4">
                <div className="flex flex-col">
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
                  <div className="p-6 flex flex-col gap-2 rounded-b-lg bg-card">
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
                            className="py-1 px-2 rounded-lg flex flex-row gap-2 items-center justify-between bg-secondary cursor-pointer hover:bg-accent/30"
                            onClick={() =>
                              navigate(
                                `/chat/data/${encodeURIComponent(d.name)}`
                              )
                            }
                            role="button"
                            aria-label={`Open dataset ${d.name}`}
                          >
                            <div className="text-foreground truncate">
                              {d.name}
                            </div>
                            <div className="text-xs text-muted-foreground whitespace-nowrap">
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
                <div className="flex flex-row gap-4 rounded-lg">
                  <div className="w-1/2">
                    <div className="h-[40px] px-2 flex items-center rounded-tl-lg rounded-tr-lg bg-card border-b border-border">
                      <div className="flex flex-row gap-2 items-center text-foreground">
                        <FontIcon type="model" className="w-4 h-4" />
                        Models
                      </div>
                    </div>
                    <div className="p-6 flex flex-col min-h-[325px] justify-between rounded-b-lg bg-card">
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
                                Generates AI responses to your questions using
                                the relevant documents as context.
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
                  <div className="w-1/2">
                    <div className="flex flex-row gap-2 items-center justify-between h-[40px] px-2 rounded-tl-lg rounded-tr-lg bg-card border-b border-border">
                      <div className="flex flex-row gap-2 items-center text-foreground">
                        <FontIcon type="integration" className="w-4 h-4" />
                        Integrations
                      </div>
                      <button className="text-xs text-primary flex flex-row gap-1 items-center">
                        Add
                        <FontIcon type="add" className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="p-6 flex flex-col min-h-[325px] justify-between rounded-b-lg bg-card">
                      <div className="flex flex-col gap-2">
                        <div className="flex flex-row gap-2 items-center border border-input rounded-lg py-1 px-2 justify-between bg-card">
                          <div className="flex flex-row gap-2 items-center text-foreground">
                            <img
                              src={
                                theme === 'dark'
                                  ? GitlabLogoDark
                                  : GitlabLogoLight
                              }
                              alt="integrations"
                              className="w-4 h-4"
                            />
                            <div>Gitlab</div>
                          </div>
                          <div className="flex flex-row gap-1 items-center">
                            <div className="w-2 h-2 bg-primary rounded-full"></div>
                            <div className="text-xs text-primary">
                              Connected
                            </div>
                          </div>
                        </div>
                        <div className="flex flex-row gap-2 items-center border border-input rounded-lg py-1 px-2 justify-between bg-card">
                          <div className="flex flex-row gap-2 items-center text-foreground">
                            <img
                              src={
                                theme === 'dark'
                                  ? GithubLogoDark
                                  : GithubLogoLight
                              }
                              alt="integrations"
                              className="w-4 h-4"
                            />
                            <div>Github</div>
                          </div>
                          <div className="flex flex-row gap-1 items-center">
                            <div className="w-2 h-2 bg-primary rounded-full"></div>
                            <div className="text-xs text-primary">
                              Connected
                            </div>
                          </div>
                        </div>
                        <div className="flex flex-row gap-2 items-center border border-input rounded-lg py-1 px-2 justify-between bg-card">
                          <div className="flex flex-row gap-2 items-center text-foreground">
                            <img
                              src={
                                theme === 'dark'
                                  ? SlackLogoDark
                                  : SlackLogoLight
                              }
                              alt="integrations"
                              className="w-4 h-4"
                            />
                            <div>Slack</div>
                          </div>
                          <div className="flex flex-row gap-1 items-center">
                            <div className="w-2 h-2 bg-primary rounded-full"></div>
                            <div className="text-xs text-primary">
                              Connected
                            </div>
                          </div>
                        </div>
                      </div>
                      <div>
                        <button className="text-xs text-primary flex flex-row gap-1 items-center justify-center">
                          Edit
                          <FontIcon type="edit" className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="w-2/5">
                <div className="flex flex-row gap-2 items-center justify-between h-[40px] px-2 rounded-tl-lg rounded-tr-lg bg-card border-b border-border">
                  <span className="text-foreground">Project versions</span>
                </div>
                <div className="p-6 flex flex-col min-h-[325px] justify-between rounded-b-lg bg-card">
                  <div className="flex flex-col gap-2 flex-1 overflow-y-auto">
                    {versions.length === 0 ? (
                      <div className="text-xs text-muted-foreground">
                        No versions yet
                      </div>
                    ) : (
                      versions.slice(0, 10).map((v, index) => (
                        <div
                          key={`${v.id}_${index}`}
                          className="flex flex-col mb-2"
                        >
                          <div className="flex flex-row gap-2 items-center justify-between">
                            <div className="text-foreground flex items-center gap-2">
                              <span>{v.name}</span>
                              {v.isCurrent ? (
                                <span className="px-2 py-0.5 rounded-2xl text-[10px] border border-teal-200 text-teal-700 bg-teal-50 dark:border-teal-800 dark:text-teal-300 dark:bg-teal-900/30">
                                  current
                                </span>
                              ) : null}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {v.date}
                            </div>
                          </div>
                          {v.description ? (
                            <div className="text-xs text-muted-foreground">
                              {v.description}
                            </div>
                          ) : null}
                        </div>
                      ))
                    )}
                  </div>
                  <div className="w-full flex justify-center items-center mt-4">
                    <button
                      className="w-full rounded-lg py-1 border flex flex-row items-center justify-center border-input text-primary hover:bg-accent/20"
                      onClick={() => navigate('/chat/versions')}
                    >
                      View all versions
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
      <ProjectModal
        isOpen={projectModal.isModalOpen}
        mode={projectModal.modalMode}
        initialName={projectModal.projectName}
        initialDescription={''}
        onClose={projectModal.closeModal}
        onSave={projectModal.saveProject}
        onDelete={
          projectModal.modalMode === 'edit'
            ? async () => {
                await projectModal.deleteProject()
                // Navigate to projects page after deletion
                navigate('/chat/projects')
              }
            : undefined
        }
        isLoading={projectModal.isLoading}
      />
    </>
  )
}

export default Dashboard
