import FontIcon from '../../common/FontIcon'
import GitlabLogoDark from '../../assets/logos/gitlab-logo-dark.svg'
import GithubLogoDark from '../../assets/logos/github-logo-dark.svg'
import SlackLogoDark from '../../assets/logos/slack-logo-dark.svg'
import GitlabLogoLight from '../../assets/logos/gitlab-logo-light.svg'
import GithubLogoLight from '../../assets/logos/github-logo-light.svg'
import SlackLogoLight from '../../assets/logos/slack-logo-light.svg'
import { useTheme } from '../../contexts/ThemeContext'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import ModeToggle, { Mode } from '../ModeToggle'
import DataCards from './DataCards'
import ProjectModal from '../../components/Project/ProjectModal'  
import ConfigEditor from '../ConfigEditor'
import { useProjectModal } from '../../hooks/useProjectModal'
import { getCurrentNamespace } from '../../utils/namespaceUtils'

const Dashboard = () => {
  const { theme } = useTheme()
  const navigate = useNavigate()
  const namespace = getCurrentNamespace()
  
  // All state declarations first
  const [mode, setMode] = useState<Mode>('designer')
  const [projectName, setProjectName] = useState<string>('Dashboard')
  
  // Shared modal hook
  const projectModal = useProjectModal({
    namespace,
    existingProjects: [], // Dashboard doesn't need duplicate checking since it edits current project
    onSuccess: (newProjectName, mode) => {
      if (mode === 'edit' && newProjectName) {
        setProjectName(newProjectName)
      }
    }
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

  return (
    <>
      <div className="w-full flex flex-col">
        <div className="flex items-center justify-between mb-4">
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
          <div className="flex items-center gap-3">
            <ModeToggle mode={mode} onToggle={setMode} />
            <button className="opacity-50 cursor-not-allowed text-sm px-3 py-2 rounded-lg border border-input text-muted-foreground">
              Deploy
            </button>
          </div>
        </div>
        {mode !== 'designer' ? (
          <ConfigEditor />
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
                    <button className="text-xs text-primary">
                      View and add
                    </button>
                  </div>
                  <div className="p-6 flex flex-col gap-2 rounded-b-lg bg-card">
                    <div className="py-1 px-2 rounded-lg flex flex-row gap-2 items-center justify-between bg-secondary">
                      <div className="text-foreground">
                        dataset-1-aircraft-logs
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Updated 3 hours ago
                      </div>
                    </div>
                    <div className="py-1 px-2 rounded-lg flex flex-row gap-2 items-center justify-between bg-secondary">
                      <div className="text-foreground">
                        data-set-2-aircraft-maintenance-rules
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Updated 6 hours ago
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      3 datasets total, showing last updated
                    </div>
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
                            TinyLlama
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
                  <FontIcon type="recently-viewed" className="w-4 h-4" />
                </div>
                <div className="p-6 flex flex-col rounded-b-lg bg-card">
                  <div className="flex flex-col gap-2 h-[400px] overflow-y-auto">
                    <div className="text-xs text-muted-foreground">Today</div>
                    {Array.from({ length: 4 }).map((_, index) => (
                      <div key={index} className="flex flex-col mb-2">
                        <div className="flex flex-row gap-2 items-center justify-between">
                          <div className="text-foreground">Version 1.0.0</div>
                          <div className="text-muted-foreground">7:45PM</div>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          RAG and prompt model updates
                        </div>
                      </div>
                    ))}
                    <div className="text-xs text-muted-foreground">July 30</div>
                    {Array.from({ length: 2 }).map((_, index) => (
                      <div key={index} className="flex flex-col">
                        <div className="flex flex-row gap-2 items-center justify-between">
                          <div className="text-foreground">Version 1.0.0</div>
                          <div className="text-muted-foreground">7:45PM</div>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          RAG and prompt model updates
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="w-full flex justify-center items-center mt-4">
                    <button className="w-full rounded-lg py-1 border flex flex-row gap-2 items-center justify-center border-input text-primary hover:bg-accent/20">
                      View config
                      <FontIcon type="code" className="w-4 h-4" />
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
        onDelete={projectModal.modalMode === 'edit' ? async () => {
          await projectModal.deleteProject()
          // Navigate to projects page after deletion
          navigate('/chat/projects')
        } : undefined}
        isLoading={projectModal.isLoading}
      />
    </>
  )
}

export default Dashboard
