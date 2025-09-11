import { useState } from 'react'
import ModeToggle, { Mode } from '../ModeToggle'
import { Button } from '../ui/button'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import GeneratedOutputs from './GeneratedOutput/GeneratedOutputs'
import { usePackageModal } from '../../contexts/PackageModalContext'

const Prompt = () => {
  // Persist onboarding state per active project
  const [hasGeneratedOutputs, setHasGeneratedOutputs] = useState<boolean>(
    () => {
      try {
        const project = localStorage.getItem('activeProject') || 'default'
        const key = `prompt.onboarding.completed:${project}`
        return localStorage.getItem(key) === '1'
      } catch {
        return false
      }
    }
  )
  const [mode, setMode] = useState<Mode>('designer')
  const { openPackageModal } = usePackageModal()

  return (
    <div className="h-full w-full flex flex-col">
      <div className="flex items-center justify-between mb-4 flex-shrink-0">
        <h2 className="text-2xl ">
          {mode === 'designer' ? 'Prompt' : 'Config editor'}
        </h2>
        <div className="flex items-center gap-3">
          <ModeToggle mode={mode} onToggle={setMode} />
          <Button variant="outline" size="sm" onClick={openPackageModal}>
            Package
          </Button>
        </div>
      </div>
      {mode === 'designer' ? (
        hasGeneratedOutputs ? (
          <GeneratedOutputs />
        ) : (
          <div className="flex-1 min-h-0 pb-6 overflow-auto">
            <div className="bg-card rounded-lg p-4 flex flex-col">
              <div className="mb-2">Expected outputs</div>
              <div className="text-sm mb-6">
                Help us generate better results, faster
              </div>
              <label className="text-sm mb-2">
                What kind of output are you hoping for?
              </label>
              <textarea
                className="w-full h-18 bg-secondary border-none rounded-lg px-4 py-2 text-lg mb-6 code-like text-foreground"
                placeholder="e.g. “Summarize each entry,” “Generate a visual timeline,” “Classify logs,” “Extract sensor anomalies,” “Create a chart,” “Turn this into a checklist,” etc."
              />
              <label className="text-xs text-muted-foreground mb-2">
                Example input
              </label>
              <input
                className="w-full h-18 bg-secondary border-none rounded-lg px-4 py-2 mb-1 code-like text-foreground"
                placeholder="Enter here"
              />
              <label className="text-xs text-muted-foreground mb-4">
                Connect your projects to OpenAI
              </label>
              <div className="mb-3">
                What are some expected outputs you'd hope to see?
              </div>
              <label className="text-xs text-muted-foreground mb-2">
                Example ideal outputs
              </label>
              <div className="flex flex-col gap-2 mb-6">
                <input
                  className="w-full h-18 bg-secondary border-none rounded-lg px-4 py-2 code-like text-foreground"
                  placeholder="Enter here"
                />
                <input
                  className="w-full h-18 bg-secondary border-none rounded-lg px-4 py-2 code-like text-foreground"
                  placeholder="Enter here"
                />
                <input
                  className="w-full h-18 bg-secondary border-none rounded-lg px-4 py-2 code-like text-foreground"
                  placeholder="Enter here"
                />
              </div>
              <button
                className="bg-primary text-primary-foreground rounded-lg px-4 py-2 w-fit self-end"
                onClick={() => {
                  setHasGeneratedOutputs(true)
                  try {
                    const project =
                      localStorage.getItem('activeProject') || 'default'
                    const key = `prompt.onboarding.completed:${project}`
                    localStorage.setItem(key, '1')
                  } catch {}
                }}
              >
                Generate outputs
              </button>
            </div>
          </div>
        )
      ) : (
        <div className="flex-1 min-h-0 overflow-hidden pb-6">
          <ConfigEditor className="h-full" />
        </div>
      )}
    </div>
  )
}

export default Prompt
