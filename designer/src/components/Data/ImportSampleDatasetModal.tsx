import { useMemo, useState, useEffect, useCallback } from 'react'
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
import { useHFDatasetSearch } from '../../hooks/useHFDatasets'
import { getDatasetConfigs } from '../../api/huggingface'
import type { HFDatasetSearchResult, SelectedHFDataset } from '../../types/huggingface'
import { Search, Loader2, ExternalLink, Download } from 'lucide-react'

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
  onImportHF?: (dataset: SelectedHFDataset) => void
}

type TabType = 'samples' | 'huggingface'

function ImportSampleDatasetModal({ open, onOpenChange, onImport, onImportHF }: Props) {
  const [activeTab, setActiveTab] = useState<TabType>('samples')
  const [search, setSearch] = useState('')
  const [kind] = useState<'all' | Kind>('all')
  const [selected, setSelected] = useState<string>('')

  // HF-specific state
  const [hfSearch, setHfSearch] = useState('')
  const [debouncedHfSearch, setDebouncedHfSearch] = useState('')
  const [selectedHF, setSelectedHF] = useState<SelectedHFDataset | null>(null)
  const [loadingConfigFor, setLoadingConfigFor] = useState<string | null>(null)

  // Debounce HF search
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedHfSearch(hfSearch), 300)
    return () => clearTimeout(timer)
  }, [hfSearch])

  const { data: hfResults, isLoading: hfLoading, error: hfError } = useHFDatasetSearch(debouncedHfSearch)

  const handleSelectHF = useCallback(async (dataset: HFDatasetSearchResult) => {
    // Fetch actual configs for this dataset
    setLoadingConfigFor(dataset.id)
    try {
      const configs = await getDatasetConfigs(dataset.id)
      const firstConfig = configs[0]
      // Prefer 'train' split if available, otherwise first split
      const split = firstConfig.splits.includes('train')
        ? 'train'
        : firstConfig.splits[0] || 'train'

      setSelectedHF({
        id: dataset.id,
        name: dataset.cardData?.pretty_name || dataset.id.split('/').pop() || dataset.id,
        rowCount: 1000,
        config: firstConfig.config,
        split,
      })
    } catch (err) {
      console.error('Failed to get dataset configs:', err)
      // Fallback to default values
      setSelectedHF({
        id: dataset.id,
        name: dataset.cardData?.pretty_name || dataset.id.split('/').pop() || dataset.id,
        rowCount: 1000,
        config: 'default',
        split: 'train',
      })
    } finally {
      setLoadingConfigFor(null)
    }
  }, [])

  // Reset transient state whenever the modal opens
  useEffect(() => {
    if (open) {
      setSearch('')
      setSelected('')
      setHfSearch('')
      setSelectedHF(null)
      setActiveTab('samples')
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
          <DialogTitle>Import dataset</DialogTitle>
          <p className="text-sm text-muted-foreground pt-1">
            Choose a sample dataset or search Hugging Face
          </p>
        </DialogHeader>

        {/* Tab switcher */}
        <div className="flex gap-1 p-1 bg-muted rounded-lg w-fit">
          <button
            type="button"
            onClick={() => setActiveTab('samples')}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
              activeTab === 'samples'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Sample Datasets
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('huggingface')}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors flex items-center gap-1.5 ${
              activeTab === 'huggingface'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <span className="text-base">🤗</span>
            Hugging Face
          </button>
        </div>

        {/* Middle scrollable region */}
        <div className="grid grid-rows-[auto_1fr] gap-3 min-h-0">
          {activeTab === 'samples' ? (
            <>
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
            </>
          ) : (
            <>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  value={hfSearch}
                  onChange={e => setHfSearch(e.target.value)}
                  placeholder="Search datasets (e.g., customer support, FAQ, reviews)"
                  aria-label="Search Hugging Face datasets"
                  className="pl-10 text-sm focus-visible:ring-1 focus-visible:ring-primary"
                />
              </div>
              {/* HF results */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 min-h-0 overflow-y-scroll items-stretch">
                {hfLoading && debouncedHfSearch && (
                  <div className="col-span-full flex items-center justify-center py-8">
                    <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                    <span className="ml-2 text-sm text-muted-foreground">Searching...</span>
                  </div>
                )}
                {hfError && (
                  <div className="col-span-full text-sm text-destructive p-2">
                    Failed to search datasets. Please try again.
                  </div>
                )}
                {!hfLoading && !hfError && hfResults?.length === 0 && debouncedHfSearch && (
                  <div className="col-span-full text-sm text-muted-foreground p-2">
                    No datasets found. Try a different search term.
                  </div>
                )}
                {!hfLoading && !hfError && !debouncedHfSearch && (
                  <div className="col-span-full flex flex-col items-center py-4 text-center">
                    <p className="text-sm text-muted-foreground mb-4">
                      Search for datasets on Hugging Face to get started.
                    </p>
                    <a
                      href="https://huggingface.co/datasets"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 px-4 py-2 text-sm text-muted-foreground hover:text-foreground border border-border rounded-lg hover:bg-accent/30 transition-colors"
                    >
                      <span className="text-base">🤗</span>
                      Browse Hugging Face datasets
                      <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                    <p className="text-xs text-muted-foreground/70 mt-2">
                      Browse and copy dataset names to search above
                    </p>
                  </div>
                )}
                {!hfLoading && !hfError && hfResults?.map(ds => {
                  const displayName = ds.cardData?.pretty_name || ds.id.split('/').pop() || ds.id
                  const isSelected = selectedHF?.id === ds.id
                  const isLoadingThis = loadingConfigFor === ds.id
                  return (
                    <button
                      key={ds.id}
                      type="button"
                      disabled={isLoadingThis}
                      className={`w-full min-h-[7rem] text-left rounded-md border p-3 transition-colors flex flex-col ${
                        isSelected
                          ? 'border-primary bg-accent/30'
                          : isLoadingThis
                            ? 'border-primary/50 bg-accent/20 cursor-wait'
                            : 'border-input bg-card hover:bg-accent/20'
                      }`}
                      onClick={() => handleSelectHF(ds)}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="text-sm font-medium truncate flex-1">{displayName}</div>
                        <a
                          href={`https://huggingface.co/datasets/${ds.id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={e => e.stopPropagation()}
                          className="flex-shrink-0 text-muted-foreground hover:text-foreground"
                        >
                          <ExternalLink className="h-4 w-4" />
                        </a>
                      </div>
                      {ds.description && (
                        <div className="mt-1 text-xs text-muted-foreground line-clamp-2">
                          {ds.description}
                        </div>
                      )}
                      <div className="mt-auto pt-2 text-xs text-foreground/80 flex items-center gap-3">
                        {isLoadingThis ? (
                          <span className="flex items-center gap-1 text-primary">
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Loading config...
                          </span>
                        ) : (
                          <>
                            <span className="flex items-center gap-1">
                              <Download className="h-3 w-3" />
                              {ds.downloads.toLocaleString()}
                            </span>
                            {ds.cardData?.size_categories?.[0] && (
                              <span>{ds.cardData.size_categories[0]}</span>
                            )}
                          </>
                        )}
                      </div>
                    </button>
                  )
                })}
              </div>
            </>
          )}
        </div>
        <DialogFooter className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            {activeTab === 'samples' && (
              <label className="flex items-center gap-2 text-sm opacity-60 cursor-not-allowed">
                <input
                  type="checkbox"
                  className="accent-current"
                  checked={true}
                  disabled={true}
                />
                Include processing strategy
              </label>
            )}
            {activeTab === 'samples' && selectedObj ? (
              <div className="text-xs text-muted-foreground">
                Selected: {selectedObj.name}
              </div>
            ) : null}
            {activeTab === 'huggingface' && selectedHF ? (
              <div className="text-xs text-muted-foreground">
                Selected: {selectedHF.name} ({selectedHF.rowCount.toLocaleString()} rows)
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
              disabled={activeTab === 'samples' ? !selectedObj : !selectedHF || !onImportHF}
              onClick={() => {
                if (activeTab === 'samples') {
                  if (!selectedObj) return
                  // Always include strategy for demo datasets
                  const rag = selectedObj.defaultStrategy
                  onImport({
                    name: selectedObj.name,
                    rag_strategy: rag,
                    sourceProjectId: selectedObj.projectId,
                  })
                } else {
                  if (!selectedHF || !onImportHF) return
                  onImportHF(selectedHF)
                }
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
