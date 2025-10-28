import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import { useModeWithReset } from '../../hooks/useModeWithReset'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '../ui/dropdown-menu'
import type { RagStrategy } from '../Rag/strategies'
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
import { Badge } from '../ui/badge'
import { useToast } from '../ui/toast'
import { useLocation, useNavigate } from 'react-router-dom'
import { useActiveProject } from '../../hooks/useActiveProject'
import {
  useListDatasets,
  useCreateDataset,
  useDeleteDataset,
  useAvailableStrategies,
} from '../../hooks/useDatasets'
import { useProject } from '../../hooks/useProjects'
import { useDataProcessingStrategies } from '../../hooks/useDataProcessingStrategies'
const Data = () => {
  const [isDragging, setIsDragging] = useState(false)
  const [isDropped, setIsDropped] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [mode, setMode] = useModeWithReset('designer')

  const navigate = useNavigate()
  const location = useLocation()
  const { toast } = useToast()

  // Get current active project for API calls
  const activeProject = useActiveProject()

  // File type mapping for parser creation (centralized to avoid duplication)
  const fileTypeMapping = useMemo(
    () => [
      { type: 'PDF', parser: 'PDFParser_LlamaIndex', extensions: ['*.pdf'] },
      {
        type: 'Docx',
        parser: 'DOCXParser_LlamaIndex',
        extensions: ['*.docx'],
      },
      { type: 'Text', parser: 'TEXTParser_LlamaIndex', extensions: ['*.txt'] },
      { type: 'CSV', parser: 'CSVParser_Pandas', extensions: ['*.csv'] },
      {
        type: 'Markdown',
        parser: 'MARKDOWNParser_LlamaIndex',
        extensions: ['*.md', '*.markdown'],
      },
    ],
    []
  )

  // Load project config to get strategies (source of truth)
  const { data: projectResp } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject
  )

  // Use React Query hooks for datasets
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

  // Fetch available strategies and databases from API
  const { data: availableOptions } = useAvailableStrategies(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject?.namespace && !!activeProject?.project }
  )

  // Hook for managing data processing strategies
  const strategies = useDataProcessingStrategies(
    activeProject?.namespace || '',
    activeProject?.project || ''
  )

  // Convert API datasets to UI format - only show real datasets from the API
  const datasets = useMemo(() => {
    // Only return datasets from the API, no localStorage, no placeholders
    return (apiDatasets?.datasets || []).map(dataset => ({
      id: dataset.name,
      name: dataset.name,
      database: dataset.database,
      data_processing_strategy: dataset.data_processing_strategy,
      files: dataset.details?.files_metadata || [],
      // Only show fields that actually come from the API
    }))
  }, [apiDatasets])

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

  // Map of fileKey -> array of dataset ids (transient UI state)
  // const [fileAssignments] = useState<Record<string, string[]>>({})

  // Processing strategies state management ----------------------------------
  const [strategyEditOpen, setStrategyEditOpen] = useState(false)
  const [strategyEditId, setStrategyEditId] = useState<string>('')
  const [strategyEditName, setStrategyEditName] = useState('')
  const [strategyEditDescription, setStrategyEditDescription] = useState('')
  const [strategyCreateOpen, setStrategyCreateOpen] = useState(false)
  const [strategyCreateName, setStrategyCreateName] = useState('')
  const [strategyCreateDescription, setStrategyCreateDescription] = useState('')
  const [strategyCopyFromId, setStrategyCopyFromId] = useState('')
  const [strategyCreateFileTypes, setStrategyCreateFileTypes] = useState<
    Set<string>
  >(new Set())

  // Load strategies from config (source of truth - NO hardcoding)
  const displayStrategies = useMemo((): RagStrategy[] => {
    const projectConfig = (projectResp as any)?.project?.config
    if (!projectConfig?.rag?.data_processing_strategies) {
      // Return empty array if config not loaded yet
      return []
    }

    const configStrategies = projectConfig.rag.data_processing_strategies || []

    // Convert config strategies to UI format
    return configStrategies.map(
      (strategy: any) =>
        ({
          id: `processing-${strategy.name.replace(/_/g, '-')}`, // Convert snake_case to kebab-case
          name: strategy.name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, (c: string) => c.toUpperCase()),
          description: strategy.description || '',
          isDefault: false, // All strategies are equal - no hardcoded defaults
          datasetsUsing: 0, // Will be calculated from datasetsByStrategyName
          configName: strategy.name, // Store original config name for API calls
        }) as RagStrategy
    )
  }, [projectResp])

  // Build mapping of strategy name -> dataset names
  const datasetsByStrategyName = useMemo(() => {
    const map = new Map<string, string[]>()

    // Use the datasets from API
    for (const d of datasets) {
      const strategyName = (d as any).data_processing_strategy
      const datasetName = d.name
      if (typeof strategyName === 'string' && typeof datasetName === 'string') {
        const arr = map.get(strategyName) || []
        arr.push(datasetName)
        map.set(strategyName, arr)
      }
    }

    return map
  }, [datasets])

  const getParsersCount = (strategyName: string): number => {
    const projectConfig = (projectResp as any)?.project?.config
    if (!projectConfig?.rag?.data_processing_strategies) return 0

    const strategy = projectConfig.rag.data_processing_strategies.find(
      (s: any) => s.name === strategyName
    )
    return strategy?.parsers?.length || 0
  }

  const getExtractorsCount = (strategyName: string): number => {
    const projectConfig = (projectResp as any)?.project?.config
    if (!projectConfig?.rag?.data_processing_strategies) return 0

    const strategy = projectConfig.rag.data_processing_strategies.find(
      (s: any) => s.name === strategyName
    )
    return strategy?.extractors?.length || 0
  }

  // rawFiles and fileAssignments are transient UI state - no persistence needed

  // Create dataset dialog state
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [isImportOpen, setIsImportOpen] = useState(false)
  const [newDatasetName, setNewDatasetName] = useState('')
  const [newDatasetDatabase, setNewDatasetDatabase] = useState('')
  const [
    newDatasetDataProcessingStrategy,
    setNewDatasetDataProcessingStrategy,
  ] = useState('')

  // Set default values when dialog opens and options are available
  useEffect(() => {
    if (isCreateOpen && availableOptions) {
      if (
        !newDatasetDataProcessingStrategy &&
        availableOptions.data_processing_strategies?.[0]
      ) {
        setNewDatasetDataProcessingStrategy(
          availableOptions.data_processing_strategies[0]
        )
      }
      if (!newDatasetDatabase && availableOptions.databases?.[0]) {
        setNewDatasetDatabase(availableOptions.databases[0])
      }
    }
  }, [
    isCreateOpen,
    availableOptions,
    newDatasetDataProcessingStrategy,
    newDatasetDatabase,
  ])

  // Simple edit modal state
  // Edit dataset removed - API doesn't support updating datasets
  const [isConfirmDeleteOpen, setIsConfirmDeleteOpen] = useState(false)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string>('')
  const [confirmDeleteName, setConfirmDeleteName] = useState<string>('')

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

    // File drop handling removed - files are now uploaded directly to datasets
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : []
    if (files.length === 0) return

    setIsDropped(true)
    setTimeout(() => setIsDropped(false), 1000)

    // console.log('Selected files:', files)
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
                const assigned =
                  datasetsByStrategyName.get(s.configName || '') || []
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

                              // Prevent deleting default strategy
                              if (s.isDefault) {
                                toast({
                                  message: 'Cannot delete default strategy',
                                  variant: 'destructive',
                                })
                                return
                              }

                              const ok = confirm(
                                'Delete this processing strategy?'
                              )
                              if (!ok) return

                              const projectConfig = (projectResp as any)
                                ?.project?.config
                              if (!projectConfig || !s.configName) {
                                toast({
                                  message: 'Unable to delete strategy',
                                  variant: 'destructive',
                                })
                                return
                              }

                              strategies.deleteStrategy.mutate(
                                {
                                  strategyName: s.configName,
                                  projectConfig,
                                },
                                {
                                  onSuccess: () => {
                                    toast({
                                      message: 'Strategy deleted',
                                      variant: 'default',
                                    })
                                  },
                                  onError: (error: any) => {
                                    toast({
                                      message:
                                        error.message ||
                                        'Failed to delete strategy',
                                      variant: 'destructive',
                                    })
                                  },
                                }
                              )
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
                        {getParsersCount(s.configName || '')} parsers •{' '}
                        {getExtractorsCount(s.configName || '')} extractors
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
                      if (!open) {
                        // Reset form when closing
                        setNewDatasetName('')
                        setNewDatasetDatabase('')
                        setNewDatasetDataProcessingStrategy('')
                      }
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
                          Data Processing Strategy
                        </label>
                        <select
                          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                          value={newDatasetDataProcessingStrategy}
                          onChange={e =>
                            setNewDatasetDataProcessingStrategy(e.target.value)
                          }
                        >
                          <option value="">Select a strategy...</option>
                          {availableOptions?.data_processing_strategies?.map(
                            strategy => (
                              <option key={strategy} value={strategy}>
                                {strategy}
                              </option>
                            )
                          )}
                        </select>
                      </div>
                      <div className="flex flex-col gap-1">
                        <label className="text-xs text-muted-foreground">
                          Database
                        </label>
                        <select
                          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                          value={newDatasetDatabase}
                          onChange={e => setNewDatasetDatabase(e.target.value)}
                        >
                          <option value="">Select a database...</option>
                          {availableOptions?.databases?.map(database => (
                            <option key={database} value={database}>
                              {database}
                            </option>
                          ))}
                        </select>
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
                                {/* Edit removed - API doesn't support updating datasets */}
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
                          <div className="flex flex-row gap-2 items-center flex-wrap mt-2">
                            {ds.database && (
                              <Badge
                                variant="default"
                                size="sm"
                                className="rounded-xl bg-teal-600 text-white dark:bg-teal-500 dark:text-slate-900"
                              >
                                {ds.database}
                              </Badge>
                            )}
                            <Badge
                              variant="default"
                              size="sm"
                              className="rounded-xl"
                            >
                              {ds.data_processing_strategy}
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground mt-2">
                            {ds.files.length}{' '}
                            {ds.files.length === 1 ? 'file' : 'files'}
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
            // Import failed - show error (no localStorage fallback)
            toast({
              message: 'Failed to import dataset',
              variant: 'destructive',
            })
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
                  toast({
                    message: 'No active project selected',
                    variant: 'destructive',
                  })
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
                  toast({
                    message: 'Failed to delete dataset',
                    variant: 'destructive',
                  })
                }
              }}
            >
              Delete
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

                const strategy = displayStrategies.find(
                  s => s.id === strategyEditId
                )
                if (!strategy || strategy.isDefault) {
                  toast({
                    message: strategy?.isDefault
                      ? 'Cannot delete default strategy'
                      : 'Strategy not found',
                    variant: 'destructive',
                  })
                  return
                }

                const ok = confirm(
                  'Are you sure you want to delete this strategy?'
                )
                if (!ok) return

                const projectConfig = (projectResp as any)?.project?.config
                if (!projectConfig || !strategy.configName) {
                  toast({
                    message: 'Unable to delete strategy',
                    variant: 'destructive',
                  })
                  return
                }

                strategies.deleteStrategy.mutate(
                  {
                    strategyName: strategy.configName,
                    projectConfig,
                  },
                  {
                    onSuccess: () => {
                      setStrategyEditOpen(false)
                      toast({ message: 'Strategy deleted', variant: 'default' })
                    },
                    onError: (error: any) => {
                      toast({
                        message: error.message || 'Failed to delete strategy',
                        variant: 'destructive',
                      })
                    },
                  }
                )
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

                  const strategy = displayStrategies.find(
                    s => s.id === strategyEditId
                  )
                  if (!strategy || !strategy.configName) {
                    toast({
                      message: 'Strategy not found',
                      variant: 'destructive',
                    })
                    return
                  }

                  const projectConfig = (projectResp as any)?.project?.config
                  if (!projectConfig) {
                    toast({
                      message: 'Unable to update strategy',
                      variant: 'destructive',
                    })
                    return
                  }

                  // Convert UI name back to snake_case for config
                  const newConfigName = strategyEditName
                    .trim()
                    .toLowerCase()
                    .replace(/\s+/g, '_')

                  strategies.updateStrategy.mutate(
                    {
                      strategyName: strategy.configName,
                      updates: {
                        name: newConfigName,
                        description: strategyEditDescription,
                      },
                      projectConfig,
                    },
                    {
                      onSuccess: () => {
                        setStrategyEditOpen(false)
                        toast({
                          message: 'Strategy updated',
                          variant: 'default',
                        })
                      },
                      onError: (error: any) => {
                        toast({
                          message: error.message || 'Failed to update strategy',
                          variant: 'destructive',
                        })
                      },
                    }
                  )
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
              <label className="text-xs text-muted-foreground mb-2 block">
                What type of files do you plan on uploading?
              </label>
              <div className="flex flex-wrap gap-2">
                {fileTypeMapping.map(({ type }) => {
                  const isSelected = strategyCreateFileTypes.has(type)
                  return (
                    <button
                      key={type}
                      type="button"
                      onClick={() => {
                        const newSet = new Set(strategyCreateFileTypes)
                        if (isSelected) {
                          newSet.delete(type)
                        } else {
                          newSet.add(type)
                        }
                        setStrategyCreateFileTypes(newSet)
                      }}
                      className={`px-3 py-1.5 rounded-md text-sm border transition-colors ${
                        isSelected
                          ? 'bg-primary text-primary-foreground border-primary'
                          : 'border-input hover:bg-accent hover:text-accent-foreground'
                      }`}
                    >
                      {type}
                    </button>
                  )
                })}
              </div>
            </div>
            <div>
              <label className="text-xs text-muted-foreground">
                Copy from existing (optional)
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
              onClick={() => {
                setStrategyCreateOpen(false)
                setStrategyCreateName('')
                setStrategyCreateDescription('')
                setStrategyCopyFromId('')
                setStrategyCreateFileTypes(new Set())
              }}
              type="button"
            >
              Cancel
            </button>
            <button
              className={`px-3 py-2 rounded-md text-sm ${strategyCreateName.trim().length > 0 && !strategies.isUpdating ? 'bg-primary text-primary-foreground hover:opacity-90' : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'}`}
              onClick={async () => {
                const displayName = strategyCreateName.trim()
                if (displayName.length === 0) return

                // Convert display name to snake_case for config
                const strategyName = displayName
                  .toLowerCase()
                  .replace(/[^a-z0-9]+/g, '_')
                  .replace(/^_+|_+$/g, '')

                if (!projectResp) {
                  toast({
                    message: 'Project config not loaded',
                    variant: 'destructive',
                  })
                  return
                }

                const projectConfig = (projectResp as any)?.project?.config

                // Get parsers from copyFrom strategy, selected file types, or use defaults
                const copyFrom = displayStrategies.find(
                  s => s.id === strategyCopyFromId
                )
                let parsers: any[] = []
                let extractors: any[] = [] // Always start with no extractors

                if (copyFrom && projectConfig) {
                  // Find the source strategy in config
                  const sourceStrategy =
                    projectConfig.rag?.data_processing_strategies?.find(
                      (s: any) =>
                        s.name ===
                        copyFrom.name.toLowerCase().replace(/[^a-z0-9]+/g, '_')
                    )
                  if (sourceStrategy) {
                    parsers = sourceStrategy.parsers || []
                    // Don't copy extractors - let user add them manually
                  }
                }

                // If no copy source but file types selected, create parsers from file types
                if (parsers.length === 0 && strategyCreateFileTypes.size > 0) {
                  parsers = Array.from(strategyCreateFileTypes).map(
                    fileType => {
                      const mapping = fileTypeMapping.find(
                        m => m.type === fileType
                      )
                      return {
                        type: mapping?.parser || 'PDFParser_LlamaIndex',
                        config: {},
                        file_include_patterns: mapping?.extensions || ['*.pdf'],
                        priority: 50,
                      }
                    }
                  )
                }

                // If no copy source and no file types selected, create with a default parser
                if (parsers.length === 0) {
                  parsers = [
                    {
                      type: 'PDFParser_LlamaIndex',
                      config: {},
                      file_include_patterns: ['*.pdf'],
                      priority: 50,
                    },
                  ]
                }

                try {
                  // Build strategy object, only including valid fields
                  const description =
                    strategyCreateDescription.trim() || displayName
                  const strategy: any = {
                    name: strategyName,
                    parsers,
                  }

                  // Only include description if it meets the 10 character minimum (schema requirement)
                  if (description.length >= 10) {
                    strategy.description = description
                  }

                  // Only include extractors if there are any
                  if (extractors.length > 0) {
                    strategy.extractors = extractors
                  }

                  await strategies.createStrategy.mutateAsync({
                    strategy,
                    projectConfig,
                  })

                  setStrategyCreateOpen(false)
                  setStrategyCreateName('')
                  setStrategyCreateDescription('')
                  setStrategyCopyFromId('')
                  setStrategyCreateFileTypes(new Set())
                  toast({
                    message: 'Strategy created successfully',
                    variant: 'default',
                  })
                } catch (error) {
                  console.error('Failed to create strategy:', error)
                  toast({
                    message:
                      error instanceof Error
                        ? error.message
                        : 'Failed to create strategy',
                    variant: 'destructive',
                  })
                }
              }}
              disabled={
                strategyCreateName.trim().length === 0 || strategies.isUpdating
              }
              type="button"
            >
              {strategies.isUpdating ? 'Creating...' : 'Create'}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default Data
