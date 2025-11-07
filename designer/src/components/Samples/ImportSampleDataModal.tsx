import { useEffect, useMemo, useState } from 'react'
import type { SampleProject } from '../../data/sampleProjects'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'

type SubmitPayload = {
  target: 'new' | 'existing'
  name: string
  description: string
  includeStrategies: boolean
}

type Props = {
  open: boolean
  sample: SampleProject | null
  isSubmitting?: boolean
  projects: string[]
  onOpenChange: (open: boolean) => void
  onSubmit: (payload: SubmitPayload) => void
}

function ImportSampleDataModal({
  open,
  sample,
  isSubmitting = false,
  projects,
  onOpenChange,
  onSubmit,
}: Props) {
  const [mode, setMode] = useState<'new' | 'existing'>('new')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [existingName, setExistingName] = useState('')
  const [includeStrategies, setIncludeStrategies] = useState(true)

  useEffect(() => {
    if (!sample) return
    setName(`${sample.slug}-sample`)
    setDescription(sample.description)
  }, [sample])

  useEffect(() => {
    if (projects && projects.length > 0 && !existingName) {
      setExistingName(projects[0])
    }
  }, [projects, existingName])

  const cta = useMemo(() => {
    return includeStrategies ? 'Import sample data' : 'Import raw data'
  }, [includeStrategies])

  const resolvedName = mode === 'new' ? name : existingName

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Import sample data</DialogTitle>
        </DialogHeader>
        <div className="text-sm space-y-4">
          {sample ? (
            <div className="rounded-md border border-border bg-card p-3">
              <div className="text-foreground font-medium">{sample.title}</div>
              <div className="mt-1 text-xs text-muted-foreground line-clamp-2">
                {sample.description}
              </div>
              <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                <span className="inline-flex items-center justify-center gap-1.5 px-3 py-0.5 rounded-2xl border border-input bg-background">
                  <span className="text-[11px]">Datasets</span>
                  <span className="text-foreground font-mono">
                    {sample.datasetCount ?? '-'}
                  </span>
                </span>
                <span className="inline-flex items-center justify-center gap-1.5 px-3 py-0.5 rounded-2xl border border-input bg-background">
                  <span className="text-[11px]">Data</span>
                  <span className="text-foreground font-mono">
                    {sample.dataSize}
                  </span>
                </span>
              </div>
            </div>
          ) : null}

          <div className="flex items-center gap-3">
            <input
              id="includeStrategies"
              type="checkbox"
              className="h-4 w-4"
              checked={includeStrategies}
              onChange={e => setIncludeStrategies(e.target.checked)}
            />
            <label htmlFor="includeStrategies" className="text-sm">
              Include processing strategies with data
            </label>
          </div>

          <div className="space-y-2">
            <div className="text-xs text-muted-foreground">
              New or existing project?
            </div>
            <div className="flex items-center gap-6">
              <label className="inline-flex items-center gap-2">
                <input
                  type="radio"
                  name="target-project"
                  checked={mode === 'new'}
                  onChange={() => setMode('new')}
                />
                <span>New project</span>
              </label>
              <label className="inline-flex items-center gap-2">
                <input
                  type="radio"
                  name="target-project"
                  checked={mode === 'existing'}
                  onChange={() => setMode('existing')}
                />
                <span>Existing project</span>
              </label>
            </div>
          </div>

          {mode === 'new' ? (
            <>
              <div className="space-y-2">
                <label className="text-xs text-muted-foreground">
                  Project name
                </label>
                <input
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  placeholder="Enter a name for the new project"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs text-muted-foreground">
                  Project description
                </label>
                <textarea
                  className="w-full min-h-24 rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={description}
                  onChange={e => setDescription(e.target.value)}
                  placeholder="Optional description"
                />
              </div>
            </>
          ) : (
            <div className="space-y-2">
              <label className="text-xs text-muted-foreground">
                Choose project
              </label>
              {projects.length === 0 ? (
                <div className="text-sm text-muted-foreground">
                  No existing projects available.
                </div>
              ) : (
                <select
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={existingName}
                  onChange={e => setExistingName(e.target.value)}
                  disabled={projects.length === 0}
                >
                  {projects.map(name => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              )}
            </div>
          )}
        </div>
        <DialogFooter>
          <button
            type="button"
            className="px-3 py-2 rounded-md text-sm border border-input hover:bg-accent/30"
            onClick={() => onOpenChange(false)}
            disabled={isSubmitting}
          >
            Cancel
          </button>
          <button
            type="button"
            className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-60"
            onClick={() =>
              onSubmit({
                target: mode,
                name: resolvedName,
                description,
                includeStrategies,
              })
            }
            disabled={isSubmitting || !resolvedName.trim()}
          >
            {isSubmitting ? 'Importing…' : cta}
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default ImportSampleDataModal
