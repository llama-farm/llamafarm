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
import { getFileBasedDemos } from '../../config/demos'

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

function ImportSampleDatasetModal({ open, onOpenChange, onImport }: Props) {
  const [search, setSearch] = useState('')
  const [kind] = useState<'all' | Kind>('all')
  const [selected, setSelected] = useState<string>('')

  // Reset transient state whenever the modal opens
  useEffect(() => {
    if (open) {
      setSearch('')
      setSelected('')
    }
  }, [open])

  // Transform file-based demos into dataset entries (no API needed!)
  const allDatasets: FlattenedDataset[] = useMemo(() => {
    const fileBasedDemos = getFileBasedDemos()
    return fileBasedDemos.map(demo => {
      // Infer kind from file types
      let kind: Kind | undefined = undefined
      if (demo.files.length > 0) {
        const firstType = demo.files[0].type.toLowerCase()
        if (firstType.includes('pdf')) {
          kind = 'pdf'
        } else if (firstType.includes('markdown') || demo.files[0].filename.endsWith('.md')) {
          kind = 'markdown'
        } else if (firstType.includes('csv')) {
          kind = 'csv'
        } else if (firstType.includes('json')) {
          kind = 'json'
        } else if (firstType.includes('image')) {
          kind = 'images'
        }
      }

      // Calculate approximate size (for display only)
      const fileCount = demo.files.length
      const size = fileCount === 1 ? '~1 file' : `~${fileCount} files`

      return {
        uid: `${demo.id}:${demo.datasetName}`,
        id: demo.datasetName,
        name: demo.datasetName,
        kind,
        size,
        projectId: demo.id,
        projectTitle: demo.displayName,
        defaultStrategy: 'markdown_encyclopedia_processor', // Will be loaded from demo config
      }
    })
  }, [])

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
          <p className="text-sm text-muted-foreground pt-1">
            Choose a sample dataset to import
          </p>
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
          {/* Scroll region: always reserve scrollbar space to avoid layout jump */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 min-h-0 overflow-y-scroll items-stretch">
            {filtered.length === 0 ? (
              <div className="col-span-full text-sm text-muted-foreground p-2 self-start">
                No demo datasets match your search.
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
            <label className="flex items-center gap-2 text-sm opacity-60 cursor-not-allowed">
              <input
                type="checkbox"
                className="accent-current"
                checked={true}
                disabled={true}
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
                // Always include strategy for demo datasets
                const rag = selectedObj.defaultStrategy
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
