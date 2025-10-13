import { useMemo, useState, useEffect } from 'react'
import { Input } from '../ui/input'
import { Button } from '../ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '../ui/dialog'
import { Badge } from '../ui/badge'
import { type SuggestedDataset } from '../../data/sampleProjects'
import { useExampleDatasets } from '../../hooks/useExamples'

type Kind = NonNullable<SuggestedDataset['kind']>

type FlattenedDataset = {
  uid: string
  id: string
  name: string
  kind: Kind | undefined
  size?: string
  projectId: string
  projectTitle: string
  defaultStrategy: string
}

type Props = {
  open: boolean
  onOpenChange: (open: boolean) => void
  onImport: (payload: {
    name: string
    rag_strategy: string
    sourceProjectId: string
  }) => void
}

const mapKindToStrategy = (kind?: Kind): string => {
  switch (kind) {
    case 'pdf':
      return 'PDF Simple'
    case 'markdown':
      return 'Markdown'
    case 'csv':
      return 'Tabular (CSV)'
    case 'images':
      return 'Image OCR'
    case 'json':
      return 'JSON (records)'
    case 'timeseries':
      return 'Timeseries (basic)'
    default:
      return 'default'
  }
}

function ImportSampleDatasetModal({ open, onOpenChange, onImport }: Props) {
  const [search, setSearch] = useState('')
  const [kind] = useState<'all' | Kind>('all')
  const [selected, setSelected] = useState<string>('')
  const [includeStrategy, setIncludeStrategy] = useState(true)

  // Reset transient state whenever the modal opens
  useEffect(() => {
    if (open) {
      setSearch('')
      setSelected('')
    }
  }, [open])

  const { data, isLoading, isError, refetch } = useExampleDatasets()

  const allDatasets: FlattenedDataset[] = useMemo(() => {
    const rows = (data?.datasets || []) as any[]
    return rows.map(row => {
      const kind = (row.kind || undefined) as Kind | undefined
      return {
        uid: `${row.example_id}:${row.name}`,
        id: row.name,
        name: row.name,
        kind,
        size: row.size_human,
        projectId: row.example_id,
        projectTitle: row.example_title || row.example_id,
        defaultStrategy: row.strategy || mapKindToStrategy(kind),
      }
    })
  }, [data])

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase()
    return allDatasets.filter(ds => {
      const byKind = kind === 'all' || ds.kind === kind
      const byTerm = term
        ? ds.name.toLowerCase().includes(term) ||
          ds.projectTitle.toLowerCase().includes(term)
        : true
      return byKind && byTerm
    })
  }, [allDatasets, kind, search])

  const selectedObj = useMemo(
    () =>
      filtered.find(d => d.uid === selected) ||
      allDatasets.find(d => d.uid === selected),
    [filtered, allDatasets, selected]
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-3xl lg:max-w-4xl -translate-y-[50%] md:-translate-y-[55%] h-[100dvh] sm:h-auto max-h-[100vh] md:max-h-[85vh] overflow-hidden grid grid-rows-[auto_1fr_auto]"
        onOpenAutoFocus={e => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>Import sample dataset</DialogTitle>
        </DialogHeader>
        {/* Middle scrollable region */}
        <div className="grid grid-rows-[auto_1fr] gap-3 min-h-0">
          <Input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search datasets or projects"
            aria-label="Search datasets or projects"
            className="text-sm focus-visible:ring-1 focus-visible:ring-primary"
          />
          {/* Filters temporarily hidden */}
          <div className="hidden" aria-hidden="true" />
          {/* Scroll region: always reserve scrollbar space to avoid layout jump */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 min-h-0 overflow-y-scroll items-stretch">
            {isLoading ? (
              <div className="col-span-full text-sm text-muted-foreground p-2">
                Loading sample datasets…
              </div>
            ) : isError ? (
              <div className="col-span-full text-sm text-muted-foreground p-2">
                Could not load sample datasets. Ensure the server is running.
                <div className="mt-2">
                  <button
                    type="button"
                    className="px-3 py-1.5 rounded-md border border-input hover:bg-accent/30"
                    onClick={() => refetch()}
                  >
                    Retry
                  </button>
                </div>
              </div>
            ) : filtered.length === 0 ? (
              <div className="col-span-full text-sm text-muted-foreground p-2 self-start">
                No datasets match your search.
              </div>
            ) : (
              filtered.map(ds => {
                const isSelected = ds.uid === selected
                return (
                  <button
                    key={ds.uid}
                    type="button"
                    className={`w-full h-28 text-left rounded-md border p-3 transition-colors flex flex-col ${
                      isSelected
                        ? 'border-primary bg-accent/30'
                        : 'border-input bg-card hover:bg-accent/20'
                    }`}
                    onClick={() => setSelected(ds.uid)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="text-sm font-medium">{ds.name}</div>
                      {ds.kind ? (
                        <Badge size="sm" className="rounded-xl capitalize">
                          {ds.kind}
                        </Badge>
                      ) : null}
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      From {ds.projectTitle}
                    </div>
                    <div className="mt-auto pt-1 text-xs text-foreground/80 flex items-center gap-2">
                      <span className="font-mono">{ds.size || '—'}</span>
                      <span className="text-muted-foreground">•</span>
                      <span>Default strategy: {ds.defaultStrategy}</span>
                    </div>
                  </button>
                )
              })
            )}
          </div>
        </div>
        <DialogFooter className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                className="accent-current"
                checked={includeStrategy}
                onChange={e => setIncludeStrategy(e.target.checked)}
              />
              Include processing strategy
            </label>
            {selectedObj ? (
              <div className="text-xs text-muted-foreground">
                Selected: {selectedObj.name}
              </div>
            ) : null}
          </div>
          <div className="flex flex-col sm:flex-row items-stretch gap-2 justify-end sm:justify-start w-full sm:w-auto mt-2 sm:mt-0">
            <Button
              variant="secondary"
              onClick={() => onOpenChange(false)}
              className="w-full sm:w-auto"
            >
              Cancel
            </Button>
            <Button
              disabled={!selectedObj}
              onClick={() => {
                if (!selectedObj) return
                const rag = includeStrategy
                  ? selectedObj.defaultStrategy
                  : 'default'
                onImport({
                  name: selectedObj.name,
                  rag_strategy: rag,
                  sourceProjectId: selectedObj.projectId,
                })
              }}
              className="w-full sm:w-auto"
            >
              Import
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default ImportSampleDatasetModal
