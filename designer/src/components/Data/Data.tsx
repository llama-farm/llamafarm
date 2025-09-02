import { useState, useCallback, useRef, useMemo, useEffect } from 'react'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import LoadingSteps from '../../common/LoadingSteps'
import ModeToggle, { Mode } from '../ModeToggle'
import ConfigEditor from '../ConfigEditor'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
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
import SearchInput from '../ui/search-input'
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
  const [isLoading, setIsLoading] = useState(false)
  const [rawFiles, setRawFiles] = useState<RawFile[]>(() => {
    try {
      const stored = localStorage.getItem('lf_raw_files')
      return stored ? (JSON.parse(stored) as RawFile[]) : []
    } catch {
      return []
    }
  })
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [searchValue, setSearchValue] = useState('')
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
  const [fileAssignments, setFileAssignments] = useState<
    Record<string, string[]>
  >(() => {
    try {
      const stored = localStorage.getItem('lf_file_assignments')
      return stored ? (JSON.parse(stored) as Record<string, string[]>) : {}
    } catch {
      return {}
    }
  })

  const [uploadStatus, setUploadStatus] = useState<
    Record<string, 'processing' | 'done'>
  >({})

  // Delete file (project-wide) dialog state
  const [isDeleteOpen, setIsDeleteOpen] = useState(false)
  const [fileToDelete, setFileToDelete] = useState<RawFile | null>(null)

  const openDeleteProjectFile = (file: RawFile) => {
    setFileToDelete(file)
    setIsDeleteOpen(true)
  }

  const confirmDeleteProjectFile = () => {
    if (!fileToDelete) return
    const id = fileToDelete.id
    // remove from raw files
    setRawFiles(prev => prev.filter(f => f.id !== id))
    // remove from assignments
    setFileAssignments(prev => {
      const next = { ...prev }
      delete (next as any)[id]
      return next
    })
    setIsDeleteOpen(false)
    setFileToDelete(null)
    toast({ message: 'File removed from project', variant: 'default' })
  }

  const getFileKey = useCallback((file: RawFile) => {
    return file.id
  }, [])

  const toggleFileDataset = useCallback(
    (file: RawFile, datasetId: string) => {
      const key = getFileKey(file)
      setFileAssignments(prev => {
        const current = prev[key] ?? []
        const isAssigned = current.includes(datasetId)
        const next = isAssigned
          ? current.filter(id => id !== datasetId)
          : [...current, datasetId]
        return { ...prev, [key]: next }
      })
    },
    [getFileKey]
  )

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
    setIsLoading(true)

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
      setUploadStatus(prev => ({
        ...prev,
        ...Object.fromEntries(
          converted.map(rf => [rf.id, 'processing' as const])
        ),
      }))
      setRawFiles(prev => {
        const existingIds = new Set(prev.map(r => r.id))
        const deduped = converted.filter(r => !existingIds.has(r.id))
        return [...prev, ...deduped]
      })
      setIsLoading(false)
      setTimeout(() => {
        setUploadStatus(prev => ({
          ...prev,
          ...Object.fromEntries(converted.map(rf => [rf.id, 'done' as const])),
        }))
        // remove the status after a short delay for a brief fade-out effect
        setTimeout(() => {
          setUploadStatus(prev => {
            const next = { ...prev }
            for (const rf of converted) delete (next as any)[rf.id]
            return next
          })
        }, 1500)
      }, 1500)
    }, 4000)

    // console.log('Dropped files:', files)
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : []
    if (files.length === 0) return

    setIsDropped(true)
    setIsLoading(true)

    setTimeout(() => {
      const converted: RawFile[] = files.map(f => ({
        id: `${f.name}:${f.size}:${f.lastModified}`,
        name: f.name,
        size: f.size,
        lastModified: f.lastModified,
        type: f.type,
      }))
      setUploadStatus(prev => ({
        ...prev,
        ...Object.fromEntries(
          converted.map(rf => [rf.id, 'processing' as const])
        ),
      }))
      setRawFiles(prev => {
        const existingIds = new Set(prev.map(r => r.id))
        const deduped = converted.filter(r => !existingIds.has(r.id))
        return [...prev, ...deduped]
      })
      setIsDropped(false)
      setIsLoading(false)
      setTimeout(() => {
        setUploadStatus(prev => ({
          ...prev,
          ...Object.fromEntries(converted.map(rf => [rf.id, 'done' as const])),
        }))
        setTimeout(() => {
          setUploadStatus(prev => {
            const next = { ...prev }
            for (const rf of converted) delete (next as any)[rf.id]
            return next
          })
        }, 1500)
      }, 1500)
    }, 4000)

    // console.log('Selected files:', files)
  }

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setSearchValue(value)
  }

  const filteredFiles = useMemo(
    () =>
      rawFiles.filter(file =>
        file.name.toLowerCase().includes(searchValue.toLowerCase())
      ),
    [rawFiles, searchValue]
  )

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
      className="h-full w-full flex flex-col gap-2 pb-20"
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
            {mode === 'designer' && rawFiles.length <= 0 ? (
              <div className="w-full mb-6 flex items-center justify-center rounded-lg py-4 text-primary text-center bg-primary/10">
                Datasets will appear here when they’re ready
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
                      <button
                        className="absolute right-3 top-3 text-xs bg-transparent text-primary hover:opacity-80 rounded-lg px-2 py-1"
                        onClick={e => {
                          e.stopPropagation()
                          navigate(`/chat/data/${ds.id}`)
                        }}
                      >
                        View
                      </button>
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
            {mode === 'designer' && (
              <div className="mb-4 flex items-center justify-between">
                <div>Raw data files</div>
                <Button size="sm" onClick={() => fileInputRef.current?.click()}>
                  Upload data
                </Button>
              </div>
            )}
            {isLoading && rawFiles.length <= 0 ? (
              <div className="w-full flex flex-col items-center justify-center border border-solid rounded-lg p-4 gap-2 transition-colors border-input">
                <div className="flex flex-col items-center justify-center gap-4 text-center my-[40px]">
                  <div className="text-xl text-foreground">
                    Processing your data...
                  </div>
                  <Loader size={72} className="border-primary" />
                  <LoadingSteps />
                </div>
              </div>
            ) : (
              mode === 'designer' &&
              rawFiles.length <= 0 && (
                <div className="w-full flex flex-col items-center justify-center border border-dashed rounded-lg p-4 gap-2 transition-colors border-input">
                  <div className="flex flex-col items-center justify-center gap-4 text-center my-[56px]">
                    <FontIcon
                      type="upload"
                      className="w-10 h-10 text-foreground"
                    />
                    <div className="text-xl text-foreground">
                      Drop data here to start
                    </div>
                    <button
                      className="text-sm py-2 px-6 border border-solid border-foreground rounded-lg hover:bg-secondary hover:border-secondary hover:text-secondary-foreground"
                      onClick={() => {
                        fileInputRef.current?.click()
                      }}
                    >
                      Or choose files
                    </button>
                  </div>
                  <p className="max-w-[527px] text-sm text-muted-foreground text-center mb-10">
                    You can upload PDFs, explore various list formats, or draw
                    inspiration from other data sources to enhance your project
                    with LlaMaFarm.
                  </p>
                </div>
              )
            )}
            {mode === 'designer' && filteredFiles.length > 0 && (
              <div>
                <div className="w-full flex flex-row gap-2">
                  <div className="w-3/4">
                    <SearchInput
                      placeholder="Search files"
                      value={searchValue}
                      onChange={handleSearch}
                    />
                  </div>
                  <div className="w-1/4 text-sm text-foreground flex items-center bg-card rounded-lg px-3 justify-between border border-input">
                    <div>All datasets</div>
                    <FontIcon
                      type="chevron-down"
                      className="w-4 h-4 text-foreground"
                    />
                  </div>
                </div>
                <div className="rounded-md border border-input bg-background p-0 text-xs mt-2 mb-20">
                  <ul>
                    {filteredFiles.map((file, i) => (
                      <li
                        key={i}
                        className="flex items-center justify-between px-3 py-3 border-b last:border-b-0 border-border/60"
                      >
                        <span className="font-mono text-xs text-muted-foreground truncate max-w-[60%]">
                          {file.name}
                        </span>
                        <div className="w-1/2 grid grid-cols-[1fr_88px_auto] items-center gap-4">
                          {/* Dataset assignment dropdown */}
                          <div className="flex items-center gap-2">
                            {(() => {
                              const key = getFileKey(file)
                              const assignedIds = fileAssignments[key] ?? []
                              const assignedNames = assignedIds
                                .map(
                                  id => datasets.find(d => d.id === id)?.name
                                )
                                .filter(Boolean) as string[]
                              const label =
                                assignedNames.length === 0
                                  ? 'Unassigned'
                                  : assignedNames.length === 1
                                    ? assignedNames[0]
                                    : `${assignedNames.length} datasets`
                              return (
                                <DropdownMenu>
                                  <DropdownMenuTrigger asChild>
                                    <button className="text-xs flex items-center gap-2 border border-input rounded-md px-2 py-1 bg-card text-foreground">
                                      <span className="truncate max-w-[160px]">
                                        {label}
                                      </span>
                                      <FontIcon
                                        type="chevron-down"
                                        className="w-3 h-3 text-muted-foreground"
                                      />
                                    </button>
                                  </DropdownMenuTrigger>
                                  <DropdownMenuContent className="w-56">
                                    <div className="px-2 py-1.5 text-xs text-muted-foreground">
                                      Assign to datasets
                                    </div>
                                    {datasets.map(ds => (
                                      <DropdownMenuCheckboxItem
                                        key={ds.id}
                                        checked={(
                                          fileAssignments[key] ?? []
                                        ).includes(ds.id)}
                                        onCheckedChange={() =>
                                          toggleFileDataset(file, ds.id)
                                        }
                                      >
                                        {ds.name}
                                      </DropdownMenuCheckboxItem>
                                    ))}
                                  </DropdownMenuContent>
                                </DropdownMenu>
                              )
                            })()}
                          </div>
                          {/* File size column */}
                          <div className="text-xs text-muted-foreground">
                            {Math.ceil((file as any).size / 1024)} KB
                          </div>
                          {/* Right actions: status + trash */}
                          <div className="flex items-center gap-6">
                            {(() => {
                              const key = getFileKey(file)
                              const status = uploadStatus[key]
                              if (status === 'processing') {
                                return (
                                  <div className="flex items-center gap-1 text-muted-foreground">
                                    <FontIcon type="fade" className="w-4 h-4" />
                                    <div className="text-xs">Processing</div>
                                  </div>
                                )
                              }
                              if (status === 'done') {
                                return (
                                  <FontIcon
                                    type="checkmark-outline"
                                    className="w-4 h-4 text-teal-600 dark:text-teal-400"
                                  />
                                )
                              }
                              return null
                            })()}
                            <button
                              className="w-4 h-4 grid place-items-center text-muted-foreground hover:text-foreground"
                              onClick={() => openDeleteProjectFile(file)}
                              aria-label={`Remove ${file.name} from project`}
                            >
                              <FontIcon type="trashcan" className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
                {/* Delete project file dialog */}
                <Dialog open={isDeleteOpen} onOpenChange={setIsDeleteOpen}>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Remove file from project</DialogTitle>
                    </DialogHeader>
                    <div className="text-sm">
                      <div className="mb-2 text-muted-foreground">
                        This file will be removed from the project and from all
                        datasets it belongs to. This action cannot be undone.
                      </div>
                      <div className="font-mono text-xs break-all">
                        {fileToDelete?.name}
                      </div>
                      {fileToDelete &&
                        (fileAssignments[fileToDelete.id] ?? []).length > 0 && (
                          <div className="mt-3">
                            <div className="text-xs text-muted-foreground mb-1">
                              Currently in datasets:
                            </div>
                            <ul className="list-disc pl-5 text-xs">
                              {(fileAssignments[fileToDelete.id] ?? [])
                                .map(
                                  id => datasets.find(d => d.id === id)?.name
                                )
                                .filter(Boolean)
                                .map(name => (
                                  <li key={name as string}>{name}</li>
                                ))}
                            </ul>
                          </div>
                        )}
                    </div>
                    <div className="mt-4 flex items-center justify-end gap-2">
                      <DialogClose asChild>
                        <Button variant="secondary">Cancel</Button>
                      </DialogClose>
                      <Button
                        variant="destructive"
                        onClick={confirmDeleteProjectFile}
                      >
                        Yes, remove
                      </Button>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default Data
