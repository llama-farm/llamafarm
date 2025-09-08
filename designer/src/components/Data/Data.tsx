import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import ModeToggle, { Mode } from '../ModeToggle'
import ConfigEditor from '../ConfigEditor'
// Dropdown menu no longer used here
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
import { useActiveProject } from '../../hooks/useActiveProject'
import { useListDatasets, useCreateDataset } from '../../hooks/useDatasets'
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
  const { toast } = useToast()

  // Get current active project for API calls
  const activeProject = useActiveProject()

  // Use React Query hooks for datasets with localStorage fallback
  const { data: apiDatasets, isLoading: isDatasetsLoading, error: datasetsError } = useListDatasets(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject?.namespace && !!activeProject?.project }
  )
  const createDatasetMutation = useCreateDataset()
  // const deleteDatasetMutation = useDeleteDataset() // TODO: Implement dataset deletion with API

  // Convert API datasets to UI format
  const datasets = useMemo(() => {
    if (apiDatasets?.datasets) {
      return apiDatasets.datasets.map((dataset) => ({
        id: dataset.name,
        name: dataset.name,
        rag_strategy: dataset.rag_strategy,
        files: dataset.files,
        lastRun: new Date(),
        embedModel: 'text-embedding-3-large',
        numChunks: dataset.files.length ? `~${dataset.files.length * 100}` : 'unknown', // Estimated, or 'unknown' if unavailable
        processedPercent: 100,
        version: 'v1',
        description: '',
      }))
    }
    return []
  }, [apiDatasets])





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
  const [newDatasetName, setNewDatasetName] = useState('')
  const [newDatasetDescription, setNewDatasetDescription] = useState('')



  const handleCreateDataset = async () => {
    const name = newDatasetName.trim()
    if (!name) return

    if (!activeProject?.namespace || !activeProject?.project) {
      toast({ 
        message: 'No active project selected', 
        variant: 'destructive' 
      })
      return
    }

    try {
      await createDatasetMutation.mutateAsync({
        namespace: activeProject.namespace,
        project: activeProject.project,
        name,
        rag_strategy: 'default', // Default strategy
      })
      toast({ message: 'Dataset created successfully', variant: 'default' })
      setIsCreateOpen(false)
      setNewDatasetName('')
      setNewDatasetDescription('')
    } catch (error) {
      console.error('Failed to create dataset:', error)
      toast({ 
        message: 'Failed to create dataset. Please try again.', 
        variant: 'destructive' 
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
              <Dialog 
                open={isCreateOpen} 
                onOpenChange={(open) => {
                  // Prevent closing dialog during mutation
                  if (!createDatasetMutation.isPending) {
                    setIsCreateOpen(open)
                  }
                }}
              >
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
                    <DialogClose asChild disabled={createDatasetMutation.isPending}>
                      <Button variant="secondary">Cancel</Button>
                    </DialogClose>
                    <Button
                      onClick={handleCreateDataset}
                      disabled={!newDatasetName.trim() || createDatasetMutation.isPending}
                    >
                      {createDatasetMutation.isPending ? 'Creating...' : 'Create'}
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
            {mode === 'designer' && isDatasetsLoading ? (
              <div className="w-full mb-6 flex items-center justify-center rounded-lg py-4 text-primary text-center bg-primary/10">
                <Loader size={32} className="mr-2" />
                Loading datasets...
              </div>
            ) : mode === 'designer' && datasets.length <= 0 ? (
              <div className="w-full mb-6 flex items-center justify-center rounded-lg py-4 text-primary text-center bg-primary/10">
                {datasetsError ? 'Unable to load datasets. Using local storage.' : 'No datasets found. Create one to get started.'}
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
            
            {/* Project-level raw files UI removed: files now only exist within datasets. */}
          </div>
        )}
      </div>

      {/* Edit/Delete dialogs removed in favor of simple View navigation */}
    </div>
  )
}

export default Data
