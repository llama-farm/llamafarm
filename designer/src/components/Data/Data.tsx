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

  // Local demo datasets change counter (forces recompute when we mutate localStorage)
  const [localDatasetsVersion, setLocalDatasetsVersion] = useState(0)

  // Convert API datasets to UI format; provide demo fallback if none
  const datasets = useMemo(() => {
    if (apiDatasets?.datasets && apiDatasets.datasets.length > 0) {
      const apiList = apiDatasets.datasets.map(dataset => ({
        id: dataset.name,
        name: dataset.name,
        // No rag_strategy on datasets; server provides data_processing_strategy/database
        files: (dataset as any).files,
        lastRun: new Date(),
        embedModel: 'text-embedding-3-large',
        // Estimate chunk count numerically for display
        numChunks: Array.isArray((dataset as any).files)
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

      // Merge any locally created demo datasets (e.g., offline imports) without duplicating API items
      try {
        const stored = localStorage.getItem('lf_demo_datasets')
        const localArr: any[] = stored ? JSON.parse(stored) : []
        const apiIds = new Set(apiList.map(d => d.id))
        const localAsUi = Array.isArray(localArr)
          ? localArr
              .filter(d => !apiIds.has(d.id))
              .map(d => ({
                id: d.id,
                name: d.name,
                files: d.files || [],
                lastRun: new Date(d.lastRun || Date.now()),
                embedModel: d.embedModel || 'text-embedding-3-large',
                numChunks: d.numChunks ?? 0,
                processedPercent: d.processedPercent ?? 0,
                version: d.version || 'v1',
                description: d.description || 'Local dataset',
              }))
          : []
        return [...apiList, ...localAsUi]
      } catch {
        return apiList
      }
    }

    // Demo fallback datasets to populate the grid when API has none
    try {
      const stored = localStorage.getItem('lf_demo_datasets')
      if (stored) {
        const parsed = JSON.parse(stored)
        if (Array.isArray(parsed) && parsed.length > 0) return parsed
      }
    } catch {}

    const demo = [
      {
        id: 'demo-arxiv',
        name: 'arxiv-papers',
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
          <div className="mb-2 flex flex-row gap-2 justify-between items-end flex-shrink-0">
            <div>Datasets</div>
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
                        onChange={e => setNewDatasetDescription(e.target.value)}
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
                    <div className="grid grid-cols-2 gap-2 mb-6">
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
                          <div className="flex flex-row gap-2 items-center">
                            <Badge
                              variant="default"
                              size="sm"
                              className="rounded-xl"
                            >
                              {ds.embedModel}
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
        onImport={async ({ name, rag_strategy }) => {
          try {
            if (!activeProject?.namespace || !activeProject?.project) {
              toast({
                message: 'No active project selected',
                variant: 'destructive',
              })
              return
            }
            await createDatasetMutation.mutateAsync({
              namespace: activeProject.namespace,
              project: activeProject.project,
              name,
              data_processing_strategy: rag_strategy || 'default',
              database: 'default',
            })
            toast({ message: `Dataset "${name}" imported`, variant: 'default' })
            setIsImportOpen(false)
            navigate(`/chat/data/${name}`)
          } catch (error) {
            console.error('Import failed', error)
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
    </div>
  )
}

export default Data
