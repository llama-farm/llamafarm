import FontIcon from '../../common/FontIcon'
import GitlabLogoDark from '../../assets/logos/gitlab-logo-dark.svg'
import GithubLogoDark from '../../assets/logos/github-logo-dark.svg'
import SlackLogoDark from '../../assets/logos/slack-logo-dark.svg'
import GitlabLogoLight from '../../assets/logos/gitlab-logo-light.svg'
import GithubLogoLight from '../../assets/logos/github-logo-light.svg'
import SlackLogoLight from '../../assets/logos/slack-logo-light.svg'
import { useTheme } from '../../contexts/ThemeContext'
import { useEffect, useState } from 'react'
import ModeToggle, { Mode } from '../ModeToggle'
import DataCards from './DataCards'
import ProjectModal, { ProjectModalMode } from '../ProjectModal'
import ConfigEditor from '../ConfigEditor'

const Dashboard = () => {
  const { theme } = useTheme()
  const [mode, setMode] = useState<Mode>('designer')
  const [projectName, setProjectName] = useState<string>('Dashboard')
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [modalMode, setModalMode] = useState<ProjectModalMode>('edit')

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
                  setModalMode('edit')
                  setIsModalOpen(true)
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
                      <span className="text-foreground">Models</span>
                    </div>
                    <div className="p-6 flex flex-col min-h-[325px] rounded-b-lg bg-card">
                      <div className="mb-4">
                        <label className="text-xs text-muted-foreground">
                          Current model
                        </label>
                        <div className="w-full flex flex-row gap-2 items-center justify-between">
                          <div className="rounded-xl px-3 py-1 my-1 w-full bg-secondary">
                            <span className="text-foreground">TinyLlama</span>
                          </div>
                          <FontIcon
                            type="edit"
                            className="w-6 h-6 text-primary"
                          />
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Why TinyLama?
                        </div>
                      </div>
                      <div>
                        <label className="text-xs text-muted-foreground">
                          OpenAI API Key
                        </label>
                        <div className="w-full flex flex-row gap-2 items-center justify-between my-1">
                          <div className="rounded-xl px-3 py-1 w-full bg-secondary">
                            <span className="text-muted-foreground">
                              Enter here
                            </span>
                          </div>
                          <button className="rounded-lg p-1 w-10 h-8 flex items-center justify-center bg-primary">
                            <FontIcon
                              type="add"
                              className="w-4 h-4 text-primary-foreground"
                            />
                          </button>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Connect your project to OpenAI
                        </div>
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
        isOpen={isModalOpen}
        mode={modalMode}
        initialName={projectName}
        initialDescription={''}
        onClose={() => setIsModalOpen(false)}
        onSave={(name: string) => {
          try {
            const stored = localStorage.getItem('projectsList')
            const list = stored ? (JSON.parse(stored) as string[]) : []
            // prevent duplicate names
            if (list.includes(name) && name !== projectName) {
              console.warn('Project rename skipped: duplicate name', name)
              setIsModalOpen(false)
              return
            }
            const updated = list.map(n => (n === projectName ? name : n))
            localStorage.setItem('projectsList', JSON.stringify(updated))
            localStorage.setItem('activeProject', name)
            setProjectName(name)
            try {
              window.dispatchEvent(
                new CustomEvent<string>('lf-active-project', { detail: name })
              )
            } catch (err) {
              console.error('Failed to dispatch lf-active-project event:', err)
            }
          } catch (err) {
            console.error('Failed to update project in localStorage:', err)
          }
          setIsModalOpen(false)
        }}
        onDelete={() => {
          try {
            const stored = localStorage.getItem('projectsList')
            const list = stored ? (JSON.parse(stored) as string[]) : []
            const updated = list.filter(n => n !== projectName)
            localStorage.setItem('projectsList', JSON.stringify(updated))
            // pick a fallback active project if any
            const next = updated[0] || 'aircraft-mx-flow'
            localStorage.setItem('activeProject', next)
            setProjectName(next)
            try {
              window.dispatchEvent(
                new CustomEvent<string>('lf-active-project', { detail: next })
              )
            } catch (err) {
              console.error('Failed to dispatch lf-active-project event:', err)
            }
          } catch (err) {
            console.error('Failed to delete project from localStorage:', err)
          }
          setIsModalOpen(false)
        }}
      />
    </>
  )
}

export default Dashboard
