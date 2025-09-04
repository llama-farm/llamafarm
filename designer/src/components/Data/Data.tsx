import { useState, useCallback, useRef, useEffect } from 'react'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import ModeToggle, { Mode } from '../ModeToggle'
import ConfigEditor from '../ConfigEditor'
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
import { Input } from '../ui/input'
import { Textarea } from '../ui/textarea'
import { Badge } from '../ui/badge'
import { useToast } from '../ui/toast'
import { useNavigate } from 'react-router-dom'

type Dataset = {
  id: string
  name: string
  lastRun: Date
  embedModel: string
  numChunks: number
  processedPercent: number // 0-100
  version: string
  description?: string
}

type RawFile = {
  id: string // stable key (name:size:lastModified)
  name: string
  size: number
  lastModified: number
  type?: string
}

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
  const { toast } = useToast()

  // Datasets state (ensure at least one dataset exists)
  const [datasets, setDatasets] = useState<Dataset[]>(() => {
    try {
      const stored = localStorage.getItem('lf_datasets')
      if (stored) {
        const parsed = JSON.parse(stored) as Array<
          Omit<Dataset, 'lastRun'> & { lastRun: string }
        >
        return parsed.map(d => ({ ...d, lastRun: new Date(d.lastRun) }))
      }
    } catch {}
    return [
      {
        id: 'default',
        name: 'default-dataset',
        lastRun: new Date(),
        embedModel: 'text-embedding-3-large',
        numChunks: 28500,
        processedPercent: 100,
        version: 'v2',
        description: '',
      },
    ]
  })

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

  useEffect(() => {
    try {
      const serializable = datasets.map(d => ({
        ...d,
        lastRun: d.lastRun.toISOString(),
      }))
      localStorage.setItem('lf_datasets', JSON.stringify(serializable))
    } catch {}
  }, [datasets])

  // Create dataset dialog state
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [newDatasetName, setNewDatasetName] = useState('')
  const [newDatasetDescription, setNewDatasetDescription] = useState('')

  // Edit dataset dialog state (from overflow menu)
  const [isEditOpen, setIsEditOpen] = useState(false)
  const [editDatasetId, setEditDatasetId] = useState<string>('')
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')

  // Delete dataset dialog state
  const [isDeleteDatasetOpen, setIsDeleteDatasetOpen] = useState(false)
  const [datasetToDelete, setDatasetToDelete] = useState<Dataset | null>(null)

  const openEditDataset = (ds: Dataset) => {
    setEditDatasetId(ds.id)
    setEditName(ds.name)
    setEditDescription(ds.description || '')
    setIsEditOpen(true)
  }

  const saveEditDataset = () => {
    const id = editDatasetId
    if (!id || !editName.trim()) return
    setDatasets(prev =>
      prev.map(d =>
        d.id === id
          ? { ...d, name: editName.trim(), description: editDescription }
          : d
      )
    )
    setIsEditOpen(false)
  }

  const deleteDataset = (id: string) => {
    setDatasets(prev => prev.filter(d => d.id !== id))
    try {
      // Clean up file assignments, and delete project files not used elsewhere
      const storedAssignments = localStorage.getItem('lf_file_assignments')
      const storedRaw = localStorage.getItem('lf_raw_files')
      const assignments: Record<string, string[]> = storedAssignments
        ? JSON.parse(storedAssignments)
        : {}
      let rawFiles: Array<{
        id: string
        name: string
        size: number
        lastModified: number
        type?: string
      }> = storedRaw ? JSON.parse(storedRaw) : []

      const remainingAssignments: Record<string, string[]> = {}
      const keepRawIds = new Set<string>()
      for (const [fileId, arr] of Object.entries(assignments)) {
        const nextArr = arr.filter(x => x !== id)
        if (nextArr.length > 0) {
          remainingAssignments[fileId] = nextArr
          keepRawIds.add(fileId)
        }
      }
      // Filter raw files to only those still referenced
      rawFiles = rawFiles.filter(f => keepRawIds.has(f.id))
      localStorage.setItem(
        'lf_file_assignments',
        JSON.stringify(remainingAssignments)
      )
      localStorage.setItem('lf_raw_files', JSON.stringify(rawFiles))

      // Remove any dataset-scoped keys
      try {
        localStorage.removeItem(`lf_dataset_strategy_name_${id}`)
      } catch {}
      try {
        localStorage.removeItem(`lf_dataset_versions_${id}`)
      } catch {}
      try {
        localStorage.removeItem(`lf_dataset_selected_version_${id}`)
      } catch {}
    } catch {}
    toast({ message: 'Dataset deleted', variant: 'default' })
  }

  const slugify = (value: string) =>
    value
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')

  const handleCreateDataset = () => {
    const name = newDatasetName.trim()
    if (!name) return
    const baseId = slugify(name) || 'dataset'
    let id = baseId
    let counter = 1
    const existingIds = new Set(datasets.map(d => d.id))
    while (existingIds.has(id)) {
      id = `${baseId}-${counter++}`
    }
    const created: Dataset = {
      id,
      name,
      description: newDatasetDescription.trim(),
      lastRun: new Date(),
      embedModel: datasets[0]?.embedModel || 'text-embedding-3-large',
      numChunks: 0,
      processedPercent: 0,
      version: 'v1',
    }
    setDatasets(prev => [...prev, created])
    setIsCreateOpen(false)
    setNewDatasetName('')
    setNewDatasetDescription('')
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
      className="h-full w-full flex flex-col gap-2 pb-32"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="w-full flex items-center justify-between mb-4">
        <h2 className="text-2xl ">
          {mode === 'designer' ? 'Data' : 'Config editor'}
        </h2>
        <div className="flex items-center gap-3">
          <ModeToggle mode={mode} onToggle={setMode} />
          <button className="opacity-50 cursor-not-allowed text-sm px-3 py-2 rounded-lg border border-input text-muted-foreground">
            Deploy
          </button>
        </div>
      </div>
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        multiple
        onChange={handleFileSelect}
      />
      <div className="w-full flex flex-col h-full">
        {mode === 'designer' && (
          <div className="mb-2 flex flex-row gap-2 justify-between items-end">
            <div>Datasets</div>
            <div className="flex items-center gap-2">
              <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
                <DialogTrigger asChild>
                  <Button variant="secondary" size="sm">
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
                  </div>
                  <DialogFooter>
                    <DialogClose asChild>
                      <Button variant="secondary">Cancel</Button>
                    </DialogClose>
                    <Button
                      onClick={handleCreateDataset}
                      disabled={!newDatasetName.trim()}
                    >
                      Create
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        )}
        {mode !== 'designer' ? (
          <ConfigEditor />
        ) : isDragging ? (
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
              inspiration from other data sources to enhance your project with
              LlaMaFarm.
            </p>
          </div>
        ) : (
          <div>
            {mode === 'designer' && (
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
                            <FontIcon type="overflow" className="w-4 h-4" />
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="end"
                          className="min-w-[10rem] w-[10rem]"
                        >
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              openEditDataset(ds)
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
                              setDatasetToDelete(ds)
                              setIsDeleteDatasetOpen(true)
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
                      <Badge variant="default" size="sm" className="rounded-xl">
                        {(() => {
                          try {
                            return (
                              localStorage.getItem(
                                `lf_dataset_strategy_name_${ds.id}`
                              ) || 'PDF Simple'
                            )
                          } catch {
                            return 'PDF Simple'
                          }
                        })()}
                      </Badge>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {ds.numChunks.toLocaleString()} chunks •{' '}
                      {ds.processedPercent}% processed • {ds.version}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {/* Project-level raw files UI removed: files now only exist within datasets. */}
          </div>
        )}
      </div>

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
            <Button onClick={saveEditDataset} disabled={!editName.trim()}>
              Save
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete dataset dialog */}
      <Dialog open={isDeleteDatasetOpen} onOpenChange={setIsDeleteDatasetOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete dataset?</DialogTitle>
          </DialogHeader>
          <div className="text-sm">
            Are you sure you want to delete this dataset and all the files
            within it?
            <div className="mt-2 font-medium">{datasetToDelete?.name}</div>
            <div className="text-xs text-muted-foreground mt-1">
              This action cannot be undone.
            </div>
          </div>
          <div className="mt-4 flex items-center justify-end gap-2">
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button
              variant="destructive"
              onClick={() => {
                if (datasetToDelete) {
                  deleteDataset(datasetToDelete.id)
                }
                setIsDeleteDatasetOpen(false)
                setDatasetToDelete(null)
              }}
            >
              Yes, delete
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default Data
