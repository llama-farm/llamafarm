import { useEffect, useState } from 'react'
import { SampleProject } from '../../data/sampleProjects'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'

type Props = {
  open: boolean
  sample: SampleProject | null
  isSubmitting?: boolean
  onOpenChange: (open: boolean) => void
  onSubmit: (payload: { name: string; description: string }) => void
}

function ImportSampleProjectModal({
  open,
  sample,
  isSubmitting = false,
  onOpenChange,
  onSubmit,
}: Props) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')

  useEffect(() => {
    if (!sample) return
    setName(`${sample.title} (sample project)`)
    setDescription(sample.description)
  }, [sample])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Import sample project</DialogTitle>
        </DialogHeader>
        <div className="text-sm space-y-4">
          <div className="text-muted-foreground">
            Import this sample project into your LlamaFarm.
          </div>
          {sample ? (
            <div className="rounded-md border border-border bg-card p-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-foreground font-medium">
                    {sample.title}
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground line-clamp-2">
                    {sample.description}
                  </div>
                  <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="inline-flex items-center justify-center gap-1.5 px-3 py-0.5 rounded-2xl border border-input bg-background">
                      <span className="text-[11px]">Project</span>
                      <span className="text-foreground font-mono">
                        {sample.downloadSize}
                      </span>
                    </span>
                    <span className="inline-flex items-center justify-center gap-1.5 px-3 py-0.5 rounded-2xl border border-input bg-background">
                      <span className="text-[11px]">Data</span>
                      <span className="text-foreground font-mono">
                        {sample.dataSize}
                      </span>
                    </span>
                    {typeof sample.datasetCount === 'number' ? (
                      <span className="inline-flex items-center justify-center gap-1.5 px-3 py-0.5 rounded-2xl border border-input bg-background">
                        <span className="text-[11px]">Datasets</span>
                        <span className="text-foreground font-mono">
                          {sample.datasetCount}
                        </span>
                      </span>
                    ) : null}
                  </div>
                </div>
              </div>
            </div>
          ) : null}

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
            onClick={() => onSubmit({ name, description })}
            disabled={isSubmitting || !name.trim()}
          >
            {isSubmitting ? 'Importing…' : 'Import'}
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default ImportSampleProjectModal
