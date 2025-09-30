import { SampleProject } from '../../data/sampleProjects'

type Props = {
  sample: SampleProject
  onPreview: (sample: SampleProject) => void
}

function SampleCard({ sample, onPreview }: Props) {
  return (
    <button
      type="button"
      className="group w-full text-left rounded-lg p-4 bg-card border border-border hover:bg-accent/20"
      onClick={() => onPreview(sample)}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="text-base text-foreground truncate">
            {sample.title}
          </div>
          <div className="mt-2 flex items-center gap-2 text-xs">
            {sample.primaryModel ? (
              <span className="text-[11px] text-primary-foreground bg-primary rounded-xl px-3 py-0.5">
                {sample.primaryModel}
              </span>
            ) : null}
            {sample.tags && sample.tags.length > 0 ? (
              <div className="flex items-center gap-1.5 text-muted-foreground">
                {sample.tags.slice(0, 2).map((t, i) => (
                  <span
                    key={`${sample.id}-tag-${t}-${i}`}
                    className="px-3 py-0.5 rounded-2xl border border-input text-xs"
                  >
                    {t}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
          <div className="mt-2 text-xs text-muted-foreground line-clamp-2">
            {sample.description}
          </div>
          <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
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
          </div>
          <div className="text-xs text-foreground/60 mt-2">
            Last edited on {new Date(sample.updatedAt).toLocaleDateString()}
          </div>
        </div>
        <span className="px-3 py-1.5 rounded-md border border-input text-sm text-primary bg-background group-hover:bg-accent/30 whitespace-nowrap">
          Preview
        </span>
      </div>
    </button>
  )
}

export default SampleCard
