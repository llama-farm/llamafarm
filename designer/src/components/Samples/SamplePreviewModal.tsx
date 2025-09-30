import { SampleProject } from '../../data/sampleProjects'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '../ui/dialog'

type Props = {
  open: boolean
  sample: SampleProject | null
  onOpenChange: (open: boolean) => void
  onImportProject: (sample: SampleProject) => void
  onImportData: (sample: SampleProject) => void
}

function SamplePreviewModal({
  open,
  sample,
  onOpenChange,
  onImportProject,
  onImportData,
}: Props) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl -translate-y-[58%] md:-translate-y-[60%]">
        <DialogHeader>
          <DialogTitle>{sample?.title ?? 'Preview'}</DialogTitle>
        </DialogHeader>
        <div className="text-sm">
          {sample ? (
            <div className="space-y-3">
              <div className="text-muted-foreground">{sample.description}</div>
              {sample.samplePrompt ? (
                <div className="rounded-md border border-border bg-card p-3 text-xs">
                  <div className="text-muted-foreground mb-1">
                    Sample prompt
                  </div>
                  <div className="text-foreground">{sample.samplePrompt}</div>
                </div>
              ) : null}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div className="rounded-md border border-border bg-card p-3">
                  <div className="text-xs text-muted-foreground">
                    Download size
                  </div>
                  <div className="mt-1 text-foreground">
                    {sample.downloadSize}
                  </div>
                </div>
                <div className="rounded-md border border-border bg-card p-3">
                  <div className="text-xs text-muted-foreground">Data size</div>
                  <div className="mt-1 text-foreground">{sample.dataSize}</div>
                </div>
                {sample.primaryModel ? (
                  <div className="rounded-md border border-border bg-card p-3">
                    <div className="text-xs text-muted-foreground">
                      Inference model
                    </div>
                    <div className="mt-1 text-foreground">
                      {sample.primaryModel}
                    </div>
                  </div>
                ) : null}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {sample.embeddingStrategy ? (
                  <div className="rounded-md border border-border bg-card p-3">
                    <div className="text-xs text-muted-foreground">
                      Embedding strategy
                    </div>
                    <div className="mt-1 text-foreground">
                      {sample.embeddingStrategy}
                    </div>
                  </div>
                ) : null}
                {sample.retrievalStrategy ? (
                  <div className="rounded-md border border-border bg-card p-3">
                    <div className="text-xs text-muted-foreground">
                      Retrieval strategy
                    </div>
                    <div className="mt-1 text-foreground">
                      {sample.retrievalStrategy}
                    </div>
                  </div>
                ) : null}
                {typeof sample.datasetCount === 'number' ? (
                  <div className="rounded-md border border-border bg-card p-3">
                    <div className="text-xs text-muted-foreground">
                      Datasets
                    </div>
                    <div className="mt-1 text-foreground">
                      {sample.datasetCount}
                    </div>
                  </div>
                ) : null}
              </div>
              {sample.tags && sample.tags.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {sample.tags.map((t, i) => (
                    <span
                      key={`${sample?.id}-tag-${t}-${i}`}
                      className="px-2 py-0.5 rounded-2xl border border-input text-[11px]"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
        <DialogFooter>
          <button
            type="button"
            className="px-3 py-2 rounded-md text-sm border border-input hover:bg-accent/30"
            onClick={() => sample && onImportData(sample)}
            disabled={!sample}
          >
            Import data only
          </button>
          <button
            type="button"
            className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90"
            onClick={() => sample && onImportProject(sample)}
            disabled={!sample}
          >
            Import project
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default SamplePreviewModal
