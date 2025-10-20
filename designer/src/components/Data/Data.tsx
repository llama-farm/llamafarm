import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import type { Mode } from '../ModeToggle'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '../ui/dropdown-menu'
import { defaultStrategies } from '../Rag/strategies'
import type { RagStrategy } from '../Rag/strategies'
import { getStoredSet, setStoredSet } from '../../utils/storage'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogTrigger,
  DialogClose,
} from '../ui/dialog'
import { Button } from '../ui/button'
import ImportSampleDatasetModal from './ImportSampleDatasetModal'
import { useImportExampleDataset } from '../../hooks/useExamples'
import PageActions from '../common/PageActions'
import { Input } from '../ui/input'
import { Textarea } from '../ui/textarea'
import { Badge } from '../ui/badge'
import { useToast } from '../ui/toast'
import { useLocation, useNavigate } from 'react-router-dom'
import { useActiveProject } from '../../hooks/useActiveProject'
import {
  useListDatasets,
  useCreateDataset,
  useDeleteDataset,
} from '../../hooks/useDatasets'
import type { UIFile } from '../../types/datasets'

type RawFile = UIFile

const Data = () => {
  const [isDragging, setIsDragging] = useState(false)
  const [isDropped, setIsDropped] = useState(false)
  const [rawFiles, setRawFiles] = useState<RawFile[]>(() => {
    try {
      const stored = localStorage.getItem('lf_raw_files')
      return stored ? (JSON.parse(stored) as RawFile[]) : []
    } catch {
      return []
    }
  })
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [mode, setMode] = useState<Mode>('designer')

  const navigate = useNavigate()
  const location = useLocation()
  const { toast } = useToast()

  // Get current active project for API calls
  const activeProject = useActiveProject()

  // Use React Query hooks for datasets with localStorage fallback
  const {
    data: apiDatasets,
    isLoading: isDatasetsLoading,
    error: datasetsError,
  } = useListDatasets(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject?.namespace && !!activeProject?.project }
  )
  const createDatasetMutation = useCreateDataset()
  const deleteDatasetMutation = useDeleteDataset()
  const importExampleDataset = useImportExampleDataset()

  // Local demo datasets change counter (forces recompute when we mutate localStorage)
  const [localDatasetsVersion, setLocalDatasetsVersion] = useState(0)

  // Convert API datasets to UI format; provide demo fallback if none
  const datasets = useMemo(() => {
    const apiList = (apiDatasets?.datasets || []).map(dataset => ({
      id: dataset.name,
      name: dataset.name,
      // No rag_strategy on datasets; server provides data_processing_strategy/database
      database: (dataset as any).database,
      processingStrategy: (() => {
        const strategy = (dataset as any).data_processing_strategy
        if (!strategy) {
          try {
            console.warn(
              `Warning: 'data_processing_strategy' is missing for dataset '${dataset.name}'. Defaulting to 'default'.`
            )
          } catch {}
          return 'default'
        }
        return strategy
      })(),
      files: Array.isArray((dataset as any).details?.files_metadata)
        ? (dataset as any).details.files_metadata
        : (dataset as any).files,
      lastRun: new Date(),
      embedModel: 'text-embedding-3-large',
      // Estimate chunk count numerically for display
      numChunks: Array.isArray((dataset as any).details?.files_metadata)
        ? Math.max(
            0,
            ((dataset as any).details.files_metadata.length || 0) * 100
          )
        : Array.isArray((dataset as any).files)
          ? Math.max(
              0,
              (Array.isArray((dataset as any).files)
                ? (dataset as any).files.length
                : 0) * 100
            )
          : 0,
      processedPercent: 100,
      version: 'v1',
      description: '',
    }))

    // Load locally persisted demo/imported datasets
    let localList: any[] = []
    try {
      const stored = localStorage.getItem('lf_demo_datasets')
      const parsed = stored ? JSON.parse(stored) : []
      if (Array.isArray(parsed)) localList = parsed
    } catch {}

    // If no API datasets, fall back to local list or seed demo entries
    if (apiList.length === 0) {
      if (localList.length > 0) return localList
      const demo = [
        {
          id: 'demo-arxiv',
          name: 'arxiv-papers',
          database: 'main_database',
          processingStrategy: 'universal_processor',
          files: [],
          lastRun: new Date(),
          embedModel: 'text-embedding-3-large',
          numChunks: 12800,
          processedPercent: 100,
          version: 'v1',
          description: 'Demo dataset of academic PDFs',
        },
        {
          id: 'demo-handbook',
          name: 'company-handbook',
          database: 'main_database',
          processingStrategy: 'universal_processor',
          files: [],
          lastRun: new Date(),
          embedModel: 'text-embedding-3-large',
          numChunks: 4200,
          processedPercent: 100,
          version: 'v2',
          description: 'Demo employee handbook and policies',
        },
      ]
      try {
        localStorage.setItem('lf_demo_datasets', JSON.stringify(demo))
      } catch {}
      return demo
    }

    // Merge API and local lists when API returns entries, keeping API as source of truth
    const mergedById = new Map<string, any>()
    for (const ds of apiList) mergedById.set(ds.id, ds)
    for (const ds of localList)
      if (!mergedById.has(ds.id)) mergedById.set(ds.id, ds)
    return Array.from(mergedById.values())
  }, [apiDatasets, localDatasetsVersion])

  // If navigated with ?dataset= query, auto-redirect to that dataset's detail if it exists
  const hasRedirectedFromQuery = useRef(false)
  useEffect(() => {
    if (hasRedirectedFromQuery.current) return
    const params = new URLSearchParams(location.search)
    const datasetParam = params.get('dataset')
    if (!datasetParam) return
    const found = datasets.find(d => d.id === datasetParam)
    if (found) {
      hasRedirectedFromQuery.current = true
      navigate(`/chat/data/${found.id}`, { replace: true })
    }
  }, [location.search, datasets, navigate])

  // Map of fileKey -> array of dataset ids
  const [fileAssignments] = useState<Record<string, string[]>>(() => {
    try {
      const stored = localStorage.getItem('lf_file_assignments')
      return stored ? (JSON.parse(stored) as Record<string, string[]>) : {}
    } catch {
      return {}
    }
  })

  // Processing strategies state management ----------------------------------
  const [metaTick, setMetaTick] = useState(0)
  const [strategyEditOpen, setStrategyEditOpen] = useState(false)
  const [strategyEditId, setStrategyEditId] = useState<string>('')
  const [strategyEditName, setStrategyEditName] = useState('')
  const [strategyEditDescription, setStrategyEditDescription] = useState('')
  const [strategyCreateOpen, setStrategyCreateOpen] = useState(false)
  const [strategyCreateName, setStrategyCreateName] = useState('')
  const [strategyCreateDescription, setStrategyCreateDescription] = useState('')
  const [strategyCopyFromId, setStrategyCopyFromId] = useState('')

  // Validate that an object is a well-formed RagStrategy
  const isValidRagStrategy = (s: any): s is RagStrategy => {
    return (
      !!s &&
      typeof s.id === 'string' &&
      typeof s.name === 'string' &&
      typeof s.description === 'string' &&
      typeof s.isDefault === 'boolean' &&
      typeof s.datasetsUsing === 'number'
    )
  }

  const getCustomStrategies = (): RagStrategy[] => {
    try {
      const raw = localStorage.getItem('lf_custom_strategies')
      if (!raw) return []
      const arr = JSON.parse(raw) as RagStrategy[]
      if (!Array.isArray(arr)) return []
      return arr.filter(isValidRagStrategy)
    } catch {
      return []
    }
  }

  const saveCustomStrategies = (list: RagStrategy[]) => {
    try {
      localStorage.setItem('lf_custom_strategies', JSON.stringify(list))
    } catch {}
  }

  const addCustomStrategy = (s: RagStrategy) => {
    const list = getCustomStrategies()
    const exists = list.some(x => x.id === s.id)
    if (exists) {
      toast({ message: 'Strategy id already exists', variant: 'destructive' })
      return
    }
    list.push(s)
    saveCustomStrategies(list)
    setMetaTick(t => t + 1)
  }

  const removeCustomStrategy = (id: string) => {
    const list = getCustomStrategies().filter(s => s.id !== id)
    saveCustomStrategies(list)
  }

  const getDeletedSet = (): Set<string> => getStoredSet('lf_strategy_deleted')
  const saveDeletedSet = (s: Set<string>) =>
    setStoredSet('lf_strategy_deleted', s)

  const markDeleted = (id: string) => {
    const set = getDeletedSet()
    set.add(id)
    saveDeletedSet(set)
    setMetaTick(t => t + 1)
  }

  // Derive display strategies with local overrides
  const displayStrategies = useMemo(() => {
    const deleted = getDeletedSet()
    const all = [...defaultStrategies, ...getCustomStrategies()]
    return all
      .filter(s => !deleted.has(s.id))
      .map(s => {
        let { name, description } = s
        try {
          const n = localStorage.getItem(`lf_strategy_name_override_${s.id}`)
          if (typeof n === 'string' && n.trim().length > 0) {
            name = n.trim()
          }
          const d = localStorage.getItem(`lf_strategy_description_${s.id}`)
          if (typeof d === 'string' && d.trim().length > 0) {
            description = d.trim()
          }
        } catch {}
        return { ...s, name, description }
      })
  }, [metaTick])

  // Build mapping of strategy name -> dataset names
  const datasetsByStrategyName = useMemo(() => {
    const map = new Map<string, string[]>()

    // Use the datasets from the useMemo above
    for (const d of datasets) {
      const strategyName = (d as any).processingStrategy
      const datasetName = d.name
      if (typeof strategyName === 'string' && typeof datasetName === 'string') {
        const arr = map.get(strategyName) || []
        arr.push(datasetName)
        map.set(strategyName, arr)
      }
    }

    return map
  }, [datasets, metaTick])

  const getParsersCount = (sid: string): number => {
    try {
      const raw = localStorage.getItem(`lf_strategy_parsers_${sid}`)
      if (!raw) return 7 // default seed
      const arr = JSON.parse(raw)
      return Array.isArray(arr) ? arr.length : 7
    } catch {
      return 7
    }
  }

  const getExtractorsCount = (sid: string): number => {
    try {
      const raw = localStorage.getItem(`lf_strategy_extractors_${sid}`)
      if (!raw) return 8 // default seed
      const arr = JSON.parse(raw)
      return Array.isArray(arr) ? arr.length : 8
    } catch {
      return 8
    }
  }

  // Refresh on processing changes (parsers/extractors add/edit/delete)
  useEffect(() => {
    const handler = (_e: Event) => {
      setMetaTick(t => t + 1)
    }
    try {
      window.addEventListener('lf:processingUpdated', handler as EventListener)
    } catch {}
    return () => {
      try {
        window.removeEventListener(
          'lf:processingUpdated',
          handler as EventListener
        )
      } catch {}
    }
  }, [])

  // (initial state is loaded from localStorage)

  // Persist data when it changes
  useEffect(() => {
    try {
      localStorage.setItem('lf_raw_files', JSON.stringify(rawFiles))
    } catch {}
  }, [rawFiles])

  useEffect(() => {
    try {
      localStorage.setItem(
        'lf_file_assignments',
        JSON.stringify(fileAssignments)
      )
    } catch {}
  }, [fileAssignments])

  // Dataset persistence is handled in the setDatasets function for localStorage fallback

  // Create dataset dialog state
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [isImportOpen, setIsImportOpen] = useState(false)
  const [newDatasetName, setNewDatasetName] = useState('')
  const [newDatasetDescription, setNewDatasetDescription] = useState('')
  const [newDatasetDatabase, setNewDatasetDatabase] = useState('')
  const [
    newDatasetDataProcessingStrategy,
    setNewDatasetDataProcessingStrategy,
  ] = useState('')

  // Simple edit modal state
  const [isEditOpen, setIsEditOpen] = useState(false)
  const [editDatasetId, setEditDatasetId] = useState<string>('')
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [isConfirmDeleteOpen, setIsConfirmDeleteOpen] = useState(false)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string>('')
  const [confirmDeleteName, setConfirmDeleteName] = useState<string>('')

  // Ensure setters are considered used even in builds that elide menu handlers
  useEffect(() => {
    // no-op referencing setters to satisfy strict noUnusedLocals in some CI builds
  }, [setConfirmDeleteId, setConfirmDeleteName])

  const handleCreateDataset = async () => {
    const name = newDatasetName.trim()
    if (!name) return

    if (!activeProject?.namespace || !activeProject?.project) {
      toast({
        message: 'No active project selected',
        variant: 'destructive',
      })
      return
    }

    try {
      await createDatasetMutation.mutateAsync({
        namespace: activeProject.namespace,
        project: activeProject.project,
        name,
        data_processing_strategy: newDatasetDataProcessingStrategy || 'default',
        database: newDatasetDatabase || 'default',
      })
      toast({ message: 'Dataset created successfully', variant: 'default' })
      setIsCreateOpen(false)
      setNewDatasetName('')
      setNewDatasetDescription('')
      setNewDatasetDatabase('')
      setNewDatasetDataProcessingStrategy('')
    } catch (error) {
      console.error('Failed to create dataset:', error)
      toast({
        message: 'Failed to create dataset. Please try again.',
        variant: 'destructive',
      })
    }
  }

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDropped(true)

    setTimeout(() => {
      setIsDragging(false)
      setIsDropped(false)
    }, 1000)

    const files = Array.from(e.dataTransfer.files)
    setTimeout(() => {
      const converted: RawFile[] = files.map(f => ({
        id: `${f.name}:${f.size}:${f.lastModified}`,
        name: f.name,
        size: f.size,
        lastModified: f.lastModified,
        type: f.type,
      }))
      setRawFiles(prev => {
        const existingIds = new Set(prev.map(r => r.id))
        const deduped = converted.filter(r => !existingIds.has(r.id))
        return [...prev, ...deduped]
      })
    }, 4000)

    // console.log('Dropped files:', files)
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : []
    if (files.length === 0) return

    setIsDropped(true)

    setTimeout(() => {
      const converted: RawFile[] = files.map(f => ({
        id: `${f.name}:${f.size}:${f.lastModified}`,
        name: f.name,
        size: f.size,
        lastModified: f.lastModified,
        type: f.type,
      }))
      setRawFiles(prev => {
        const existingIds = new Set(prev.map(r => r.id))
        const deduped = converted.filter(r => !existingIds.has(r.id))
        return [...prev, ...deduped]
      })
      setIsDropped(false)
    }, 4000)

    // console.log('Selected files:', files)
  }

  const formatLastRun = (d: Date) => {
    if (!(d instanceof Date) || isNaN(d.getTime())) {
      return '-'
    }
    return new Intl.DateTimeFormat('en-US', {
      month: 'numeric',
      day: 'numeric',
      year: '2-digit',
    }).format(d)
  }

  return (
    <div
      className="h-full w-full flex flex-col"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="w-full flex items-center justify-between mb-4 flex-shrink-0">
        <h2 className="text-2xl ">
          {mode === 'designer' ? 'Data' : 'Config editor'}
        </h2>
        <PageActions mode={mode} onModeChange={setMode} />
      </div>
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        multiple
        onChange={handleFileSelect}
      />
      <div className="w-full flex flex-col flex-1 min-h-0">
        {mode === 'designer' && (
          <>
            {/* Processing strategies section */}
            <div className="flex items-center justify-between mt-0 mb-3">
              <div>
                <div className="font-medium">Processing strategies</div>
                <div className="h-1" />
                <div className="text-xs text-muted-foreground">
                  Processing strategies are applied to datasets.
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  onClick={() => {
                    setStrategyCreateName('')
                    setStrategyCreateDescription('')
                    setStrategyCopyFromId('')
                    setStrategyCreateOpen(true)
                  }}
                >
                  Create new
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-8">
              {displayStrategies.map(s => {
                const assigned = datasetsByStrategyName.get(s.name) || []
                return (
                  <div
                    key={s.id}
                    className={`w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 hover:cursor-pointer transition-colors ${displayStrategies.length === 1 ? 'md:col-span-2' : ''}`}
                    onClick={() => navigate(`/chat/data/strategies/${s.id}`)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        navigate(`/chat/data/strategies/${s.id}`)
                      }
                    }}
                  >
                    <div className="absolute top-2 right-2">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button
                            className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30"
                            onClick={e => e.stopPropagation()}
                          >
                            <FontIcon type="overflow" className="w-4 h-4" />
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="end"
                          className="min-w-[12rem] w-[12rem]"
                        >
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              setStrategyEditId(s.id)
                              setStrategyEditName(s.name)
                              setStrategyEditDescription(s.description)
                              setStrategyEditOpen(true)
                            }}
                          >
                            Rename
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              setStrategyCreateName(`${s.name} (copy)`)
                              setStrategyCreateDescription(s.description)
                              setStrategyCopyFromId(s.id)
                              setStrategyCreateOpen(true)
                            }}
                          >
                            Duplicate
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive focus:text-destructive"
                            onClick={e => {
                              e.stopPropagation()
                              const ok = confirm(
                                'Delete this processing strategy?'
                              )
                              if (!ok) return
                              try {
                                removeCustomStrategy(s.id)
                                const set = getDeletedSet()
                                set.add(s.id)
                                saveDeletedSet(set)
                                setMetaTick(t => t + 1)
                              } catch {}
                            }}
                          >
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    <div className="text-sm font-medium">{s.name}</div>
                    <div className="text-xs text-primary text-left w-fit">
                      {s.description ||
                        'Unified processor for PDFs, Word docs, CSVs, Markdown, and text files.'}
                    </div>
                    <div className="flex items-center gap-2 flex-wrap pt-0.5">
                      {assigned.slice(0, 4).map(name => (
                        <Badge
                          key={name}
                          variant="default"
                          size="sm"
                          className="rounded-xl bg-teal-600 text-white dark:bg-teal-500 dark:text-slate-900"
                        >
                          {name}
                        </Badge>
                      ))}
                      {assigned.length > 4 ? (
                        <Badge
                          variant="secondary"
                          size="sm"
                          className="rounded-xl"
                        >
                          +{assigned.length - 4}
                        </Badge>
                      ) : null}
                      {assigned.length === 0 ? (
                        <Badge
                          variant="secondary"
                          size="sm"
                          className="rounded-xl"
                        >
                          No datasets yet
                        </Badge>
                      ) : null}
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <div className="text-xs text-muted-foreground">
                        {getParsersCount(s.id)} parsers •{' '}
                        {getExtractorsCount(s.id)} extractors
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="px-3 h-7"
                        onClick={e => {
                          e.stopPropagation()
                          navigate(`/chat/data/strategies/${s.id}`)
                        }}
                      >
                        Configure
                      </Button>
                    </div>
                  </div>
                )
              })}
            </div>
            {/* End processing strategies section */}

            {/* Divider */}
            <div className="border-t border-border mb-6"></div>

            {/* Datasets section */}
            <div className="mb-2 flex flex-row gap-2 justify-between items-end flex-shrink-0">
              <div className="font-medium">Datasets</div>
              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setIsImportOpen(true)}
                >
                  Import sample dataset
                </Button>
                <Dialog
                  open={isCreateOpen}
                  onOpenChange={open => {
                    // Prevent closing dialog during mutation
                    if (!createDatasetMutation.isPending) {
                      setIsCreateOpen(open)
                    }
                  }}
                >
                  <DialogTrigger asChild>
                    <Button variant="default" size="sm">
                      Create new
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>New dataset</DialogTitle>
                    </DialogHeader>
                    <div className="flex flex-col gap-3">
                      <div className="flex flex-col gap-1">
                        <label className="text-xs text-muted-foreground">
                          Name
                        </label>
                        <Input
                          autoFocus
                          value={newDatasetName}
                          onChange={e => setNewDatasetName(e.target.value)}
                          placeholder="Enter dataset name"
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <label className="text-xs text-muted-foreground">
                          Description
                        </label>
                        <Textarea
                          value={newDatasetDescription}
                          onChange={e =>
                            setNewDatasetDescription(e.target.value)
                          }
                          placeholder="Optional description"
                          rows={3}
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <label className="text-xs text-muted-foreground">
                          Data Processing Strategy
                        </label>
                        <Input
                          value={newDatasetDataProcessingStrategy}
                          onChange={e =>
                            setNewDatasetDataProcessingStrategy(e.target.value)
                          }
                          placeholder="e.g., PDF Simple, Markdown"
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <label className="text-xs text-muted-foreground">
                          Database
                        </label>
                        <Input
                          value={newDatasetDatabase}
                          onChange={e => setNewDatasetDatabase(e.target.value)}
                          placeholder="e.g., default_db"
                        />
                      </div>
                    </div>
                    <DialogFooter>
                      <DialogClose
                        asChild
                        disabled={createDatasetMutation.isPending}
                      >
                        <Button variant="secondary">Cancel</Button>
                      </DialogClose>
                      <Button
                        onClick={handleCreateDataset}
                        disabled={
                          !newDatasetName.trim() ||
                          !newDatasetDataProcessingStrategy.trim() ||
                          !newDatasetDatabase.trim() ||
                          createDatasetMutation.isPending
                        }
                      >
                        {createDatasetMutation.isPending
                          ? 'Creating...'
                          : 'Create'}
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </div>
            </div>
          </>
        )}
        {mode !== 'designer' ? (
          <div className="flex-1 min-h-0 overflow-hidden pb-6">
            <ConfigEditor className="h-full" />
          </div>
        ) : (
          <div className="flex-1 min-h-0 overflow-auto pb-6">
            {isDragging ? (
              <div
                className={`w-full h-full flex flex-col items-center justify-center border border-dashed rounded-lg p-4 gap-2 transition-colors border-input`}
              >
                <div className="flex flex-col items-center justify-center gap-4 text-center my-[56px] text-primary">
                  {isDropped ? (
                    <Loader />
                  ) : (
                    <FontIcon
                      type="upload"
                      className="w-10 h-10 text-blue-200 dark:text-white"
                    />
                  )}
                  <div className="text-xl text-foreground">Drop data here</div>
                </div>
                <p className="max-w-[527px] text-sm text-muted-foreground text-center mb-10">
                  You can upload PDFs, explore various list formats, or draw
                  inspiration from other data sources to enhance your project
                  with LlaMaFarm.
                </p>
              </div>
            ) : (
              <div>
                {mode === 'designer' && isDatasetsLoading ? (
                  <div className="w-full mb-6 flex items-center justify-center rounded-lg py-4 text-primary text-center bg-primary/10">
                    <Loader size={32} className="mr-2" />
                    Loading datasets...
                  </div>
                ) : mode === 'designer' && datasets.length <= 0 ? (
                  <div className="w-full mb-6 flex items-center justify-center rounded-lg py-4 text-primary text-center bg-primary/10">
                    {datasetsError
                      ? 'Unable to load datasets. Using local storage.'
                      : 'No datasets found. Create one to get started.'}
                  </div>
                ) : (
                  mode === 'designer' && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-2 mb-6">
                      {datasets.map(ds => (
                        <div
                          key={ds.id}
                          className="w-full bg-card rounded-lg border border-border flex flex-col gap-3 p-4 relative hover:bg-accent/20 cursor-pointer transition-colors"
                          onClick={() => navigate(`/chat/data/${ds.id}`)}
                          role="button"
                          tabIndex={0}
                          onKeyDown={e => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              navigate(`/chat/data/${ds.id}`)
                            }
                          }}
                        >
                          <div className="absolute right-3 top-3">
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <button
                                  className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30"
                                  onClick={e => e.stopPropagation()}
                                  aria-label="Dataset actions"
                                >
                                  <FontIcon
                                    type="overflow"
                                    className="w-4 h-4"
                                  />
                                </button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent
                                align="end"
                                className="min-w-[10rem] w-[10rem]"
                              >
                                <DropdownMenuItem
                                  onClick={e => {
                                    e.stopPropagation()
                                    // open simple edit modal
                                    setEditDatasetId(ds.id)
                                    setEditName(ds.name)
                                    setEditDescription(ds.description || '')
                                    setIsEditOpen(true)
                                  }}
                                >
                                  Edit
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onClick={e => {
                                    e.stopPropagation()
                                    navigate(`/chat/data/${ds.id}`)
                                  }}
                                >
                                  View
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  className="text-destructive focus:text-destructive"
                                  onClick={e => {
                                    e.stopPropagation()
                                    setConfirmDeleteId(ds.id)
                                    setConfirmDeleteName(ds.name)
                                    setIsConfirmDeleteOpen(true)
                                  }}
                                >
                                  Delete
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </div>
                          <div className="text-sm font-medium">{ds.name}</div>
                          <div className="text-xs text-muted-foreground">
                            Last run on {formatLastRun(ds.lastRun)}
                          </div>
                          <div className="flex flex-row gap-2 items-center flex-wrap">
                            {(ds as any).database && (
                              <Badge
                                variant="default"
                                size="sm"
                                className="rounded-xl bg-teal-600 text-white dark:bg-teal-500 dark:text-slate-900"
                              >
                                {(ds as any).database}
                              </Badge>
                            )}
                            <Badge
                              variant="default"
                              size="sm"
                              className="rounded-xl"
                            >
                              {(ds as any).processingStrategy || 'default'}
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {ds.numChunks.toLocaleString()} chunks •{' '}
                            {ds.processedPercent}% processed • {ds.version}
                          </div>
                        </div>
                      ))}
                    </div>
                  )
                )}

                {/* Project-level raw files UI removed: files now only exist within datasets. */}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Edit dataset dialog */}
      <ImportSampleDatasetModal
        open={isImportOpen}
        onOpenChange={setIsImportOpen}
        onImport={async ({ name, sourceProjectId }) => {
          try {
            if (!activeProject?.namespace || !activeProject?.project) {
              toast({
                message: 'No active project selected',
                variant: 'destructive',
              })
              return
            }
            await importExampleDataset.mutateAsync({
              exampleId: sourceProjectId,
              namespace: activeProject.namespace,
              project: activeProject.project,
              dataset: name,
              include_strategies: true,
              process: true,
            })
            toast({
              message: `Dataset "${name}" importing…`,
              variant: 'default',
            })
            setIsImportOpen(false)
            navigate(`/chat/data/${name}`)
          } catch (error: any) {
            console.error('Import failed', error)
            try {
              const serverMessage =
                (error?.response?.data?.detail as string) ||
                (error?.message as string) ||
                'Unknown error'
              toast({
                message: `Failed to import dataset: ${serverMessage}`,
                variant: 'destructive',
              })
            } catch {}
            // Local fallback to make import work without server: persist into demo datasets
            try {
              const raw = localStorage.getItem('lf_demo_datasets')
              const arr = raw ? JSON.parse(raw) : []
              const newEntry = {
                id: name,
                name,
                files: [],
                lastRun: new Date(),
                embedModel: 'text-embedding-3-large',
                numChunks: 0,
                processedPercent: 0,
                version: 'v1',
                description: 'Imported sample dataset (local)',
              }
              const exists =
                Array.isArray(arr) && arr.some((d: any) => d.id === name)
              const updated = exists ? arr : [...arr, newEntry]
              localStorage.setItem('lf_demo_datasets', JSON.stringify(updated))
              setLocalDatasetsVersion(v => v + 1)
              setIsImportOpen(false)
              toast({
                message: `Dataset "${name}" imported (local)`,
                variant: 'default',
              })
              navigate(`/chat/data?dataset=${encodeURIComponent(name)}`)
            } catch {
              toast({
                message: 'Failed to import dataset',
                variant: 'destructive',
              })
            }
          }
        }}
      />

      {/* Edit dataset dialog */}
      <Dialog open={isConfirmDeleteOpen} onOpenChange={setIsConfirmDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete dataset</DialogTitle>
          </DialogHeader>
          <div className="text-sm text-muted-foreground">
            Are you sure you want to delete this{' '}
            {confirmDeleteName ? `"${confirmDeleteName}"` : 'dataset'}?
          </div>
          <div className="mt-4 flex items-center justify-end gap-2">
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button
              variant="destructive"
              onClick={async () => {
                const id = confirmDeleteId
                setIsConfirmDeleteOpen(false)
                if (!id) return
                if (!activeProject?.namespace || !activeProject?.project) {
                  // No active project: perform local deletion fallback
                  try {
                    const raw = localStorage.getItem('lf_demo_datasets')
                    const arr = raw ? JSON.parse(raw) : []
                    const updated = Array.isArray(arr)
                      ? arr.filter((d: any) => d.id !== id)
                      : []
                    localStorage.setItem(
                      'lf_demo_datasets',
                      JSON.stringify(updated)
                    )
                    setLocalDatasetsVersion(v => v + 1)
                    toast({
                      message: 'Dataset deleted (local)',
                      variant: 'default',
                    })
                  } catch {
                    toast({
                      message: 'Failed to delete dataset',
                      variant: 'destructive',
                    })
                  }
                  return
                }
                try {
                  await deleteDatasetMutation.mutateAsync({
                    namespace: activeProject.namespace,
                    project: activeProject.project,
                    dataset: id,
                  })
                  toast({ message: 'Dataset deleted', variant: 'default' })
                } catch (err) {
                  console.error('Delete failed', err)
                  // Local fallback removal
                  try {
                    const raw = localStorage.getItem('lf_demo_datasets')
                    const arr = raw ? JSON.parse(raw) : []
                    const updated = Array.isArray(arr)
                      ? arr.filter((d: any) => d.id !== id)
                      : []
                    localStorage.setItem(
                      'lf_demo_datasets',
                      JSON.stringify(updated)
                    )
                    setLocalDatasetsVersion(v => v + 1)
                    toast({
                      message: 'Dataset deleted (local)',
                      variant: 'default',
                    })
                  } catch {
                    toast({
                      message: 'Failed to delete dataset',
                      variant: 'destructive',
                    })
                  }
                }
              }}
            >
              Delete
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Edit dataset dialog */}
      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit dataset</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">Name</label>
              <Input
                autoFocus
                value={editName}
                onChange={e => setEditName(e.target.value)}
                placeholder="Dataset name"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">
                Description
              </label>
              <Textarea
                value={editDescription}
                onChange={e => setEditDescription(e.target.value)}
                placeholder="Optional description"
                rows={3}
              />
            </div>
          </div>
          <div className="mt-4 flex items-center justify-end gap-2">
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button
              onClick={() => {
                const id = editDatasetId.trim()
                const name = editName.trim()
                if (!id || !name) return
                try {
                  localStorage.setItem(`lf_dataset_name_${id}`, name)
                  localStorage.setItem(
                    `lf_dataset_description_${id}`,
                    editDescription
                  )
                } catch {}
                setIsEditOpen(false)
              }}
              disabled={!editName.trim()}
            >
              Save
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Edit Strategy Modal */}
      <Dialog open={strategyEditOpen} onOpenChange={setStrategyEditOpen}>
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Edit strategy
            </DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3 pt-1">
            <div>
              <label className="text-xs text-muted-foreground">
                Strategy name
              </label>
              <input
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Enter name"
                value={strategyEditName}
                onChange={e => setStrategyEditName(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">
                Description
              </label>
              <textarea
                rows={4}
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Add a brief description"
                value={strategyEditDescription}
                onChange={e => setStrategyEditDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter className="flex items-center justify-between gap-2">
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
              onClick={() => {
                if (!strategyEditId) return
                const ok = confirm(
                  'Are you sure you want to delete this strategy?'
                )
                if (ok) {
                  try {
                    localStorage.removeItem(
                      `lf_strategy_name_override_${strategyEditId}`
                    )
                    localStorage.removeItem(
                      `lf_strategy_description_${strategyEditId}`
                    )
                  } catch {}
                  removeCustomStrategy(strategyEditId)
                  markDeleted(strategyEditId)
                  setStrategyEditOpen(false)
                  toast({ message: 'Strategy deleted', variant: 'default' })
                }
              }}
              type="button"
            >
              Delete
            </button>
            <div className="flex items-center gap-2 ml-auto">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => setStrategyEditOpen(false)}
                type="button"
              >
                Cancel
              </button>
              <button
                className={`px-3 py-2 rounded-md text-sm ${strategyEditName.trim().length > 0 ? 'bg-primary text-primary-foreground hover:opacity-90' : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'}`}
                onClick={() => {
                  if (!strategyEditId || strategyEditName.trim().length === 0)
                    return
                  try {
                    localStorage.setItem(
                      `lf_strategy_name_override_${strategyEditId}`,
                      strategyEditName.trim()
                    )
                    localStorage.setItem(
                      `lf_strategy_description_${strategyEditId}`,
                      strategyEditDescription
                    )
                  } catch {}
                  setStrategyEditOpen(false)
                  setMetaTick(t => t + 1)
                }}
                disabled={strategyEditName.trim().length === 0}
                type="button"
              >
                Save
              </button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create Strategy Modal */}
      <Dialog
        open={strategyCreateOpen}
        onOpenChange={open => {
          setStrategyCreateOpen(open)
          if (!open) {
            setStrategyCreateName('')
            setStrategyCreateDescription('')
            setStrategyCopyFromId('')
          }
        }}
      >
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Create new processing strategy
            </DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3 pt-1">
            <div>
              <label className="text-xs text-muted-foreground">
                Strategy name
              </label>
              <input
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Enter name"
                value={strategyCreateName}
                onChange={e => setStrategyCreateName(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">
                Copy from existing
              </label>
              <select
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                value={strategyCopyFromId}
                onChange={e => {
                  const v = e.target.value
                  setStrategyCopyFromId(v)
                  const found = displayStrategies.find(x => x.id === v)
                  if (found) {
                    setStrategyCreateDescription(found.description || '')
                    if (strategyCreateName.trim().length === 0) {
                      setStrategyCreateName(`${found.name} (copy)`)
                    }
                  }
                }}
              >
                <option value="">None</option>
                {displayStrategies.map(s => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground">
                Description
              </label>
              <textarea
                rows={4}
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Add a brief description"
                value={strategyCreateDescription}
                onChange={e => setStrategyCreateDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter className="flex items-center justify-between gap-2">
            <button
              className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
              onClick={() => setStrategyCreateOpen(false)}
              type="button"
            >
              Cancel
            </button>
            <button
              className={`px-3 py-2 rounded-md text-sm ${strategyCreateName.trim().length > 0 ? 'bg-primary text-primary-foreground hover:opacity-90' : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'}`}
              onClick={() => {
                const name = strategyCreateName.trim()
                if (name.length === 0) return
                const slugify = (str: string) =>
                  str
                    .toLowerCase()
                    .replace(/[^a-z0-9]+/g, '-')
                    .replace(/^-+|-+$/g, '')
                const baseId = `custom-${slugify(name)}`
                const existingIds = new Set(
                  [...defaultStrategies, ...getCustomStrategies()].map(
                    s => s.id
                  )
                )
                let newId = baseId
                if (existingIds.has(newId)) {
                  newId = `${baseId}-${Date.now()}`
                }
                const newStrategy: RagStrategy = {
                  id: newId,
                  name,
                  description: strategyCreateDescription,
                  isDefault: false,
                  datasetsUsing: 0,
                }
                addCustomStrategy(newStrategy)
                try {
                  localStorage.setItem(
                    `lf_strategy_name_override_${newId}`,
                    name
                  )
                  localStorage.setItem(
                    `lf_strategy_description_${newId}`,
                    strategyCreateDescription
                  )
                } catch {}
                setStrategyCreateOpen(false)
                setStrategyCreateName('')
                setStrategyCreateDescription('')
                setStrategyCopyFromId('')
                setMetaTick(t => t + 1)
                toast({ message: 'Strategy created', variant: 'default' })
              }}
              disabled={strategyCreateName.trim().length === 0}
              type="button"
            >
              Create
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default Data
