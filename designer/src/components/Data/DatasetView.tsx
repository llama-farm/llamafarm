import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import SearchInput from '../ui/search-input'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogClose,
} from '../ui/dialog'
import { Textarea } from '../ui/textarea'
import { useToast } from '../ui/toast'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProjectSwitchNavigation } from '../../hooks/useProjectSwitchNavigation'
import {
  useUploadFileToDataset,
  useReIngestDataset,
  useTaskStatus,
  useListDatasets,
  useDeleteDatasetFile,
  useDeleteDataset,
} from '../../hooks/useDatasets'

type Dataset = {
  id: string
  name: string
  lastRun: string | Date
  embedModel: string
  numChunks: number
  processedPercent: number
  version: string
  description?: string
  files?: string[] // Array of file hashes from API
}

function DatasetView() {
  const navigate = useNavigate()
  const { datasetId } = useParams()
  const { toast } = useToast()

  // Get current active project for API calls
  const activeProject = useActiveProject()

  // Handle automatic navigation when project changes
  useProjectSwitchNavigation()
  const uploadMutation = useUploadFileToDataset()

  // Fetch datasets from API to get file information
  const { data: datasetsResponse } = useListDatasets(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject }
  )

  // Task tracking state and hooks
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null)
  const reIngestMutation = useReIngestDataset()
  const deleteFileMutation = useDeleteDatasetFile()
  const deleteDatasetMutation = useDeleteDataset()
  const { data: taskStatus } = useTaskStatus(
    activeProject?.namespace || '',
    activeProject?.project || '',
    currentTaskId,
    { enabled: !!currentTaskId && !!activeProject }
  )

  const [dataset, setDataset] = useState<Dataset | null>(null)
  const datasetName = useMemo(
    () => dataset?.name || datasetId || 'dataset',
    [dataset?.name, datasetId]
  )

  type RawFile = {
    id: string
    name: string
    size: number
    lastModified: number
    type?: string
    fullHash?: string // For API files, stores the complete hash
  }

  // Get current dataset from API response
  const currentApiDataset = useMemo(() => {
    if (!datasetsResponse?.datasets || !datasetId) return null
    return datasetsResponse.datasets.find(d => d.name === datasetId)
  }, [datasetsResponse, datasetId])

  // Files from API data only
  const files = useMemo(() => {
    if (!currentApiDataset?.files) return []
    return currentApiDataset.files.map((fileObj: any) => {
      const fileHash =
        typeof fileObj === 'string' ? fileObj : fileObj?.id || fileObj || ''
      const size =
        typeof fileObj === 'object' &&
        fileObj !== null &&
        'size' in fileObj &&
        fileObj.size !== undefined
          ? fileObj.size
          : 'unknown'
      const lastModified =
        typeof fileObj === 'object' &&
        fileObj !== null &&
        'lastModified' in fileObj &&
        fileObj.lastModified !== undefined
          ? fileObj.lastModified
          : 'unknown'
      return {
        id: fileHash,
        name: `${fileHash.substring(0, 12)}...${fileHash.substring(fileHash.length - 8)}`, // Show first 12 and last 8 chars
        size,
        lastModified,
        type: 'unknown',
        fullHash: fileHash, // Store full hash for operations
      }
    })
  }, [currentApiDataset])

  const [isEditOpen, setIsEditOpen] = useState(false)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [showDeleteConfirmation, setShowDeleteConfirmation] = useState(false)
  const [pendingDeleteFileHash, setPendingDeleteFileHash] = useState<
    string | null
  >(null)
  const [showDeleteFileConfirmation, setShowDeleteFileConfirmation] =
    useState(false)
  const [copyStatus, setCopyStatus] = useState<{
    [id: string]: string | undefined
  }>({})
  const [searchValue, setSearchValue] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploadStatus, setUploadStatus] = useState<
    Record<string, 'processing' | 'done'>
  >({})

  type DatasetVersion = {
    id: string // e.g., v1, v2
    createdAt: string // ISO
  }
  const [versions, setVersions] = useState<DatasetVersion[]>([])

  // Load dataset metadata from API or create fallback
  useEffect(() => {
    if (!datasetId) return

    if (currentApiDataset) {
      // Use API data when available
      setDataset({
        id: datasetId,
        name: currentApiDataset.name,
        lastRun: new Date(),
        embedModel: 'text-embedding-3-large',
        numChunks: currentApiDataset.files?.length || 0,
        processedPercent: 100,
        version: 'v1',
        description: '',
        files: currentApiDataset.files,
      })
    } else {
      // Fallback to minimal dataset object
      setDataset({
        id: datasetId,
        name: datasetId,
        lastRun: new Date(),
        embedModel: 'text-embedding-3-large',
        numChunks: 0,
        processedPercent: 100,
        version: 'v1',
        description: '',
        files: [],
      })
    }
  }, [datasetId, currentApiDataset])

  // Handle task completion
  useEffect(() => {
    if (!taskStatus) return

    if (taskStatus.state === 'SUCCESS') {
      // Task completed successfully
      setCurrentTaskId(null)
      toast({
        message: 'Dataset reprocessing completed successfully',
        variant: 'default',
      })

      // Create a new version
      if (datasetId) {
        const nextNum = (versions.length || 0) + 1
        const nextId = `v${nextNum}`
        const next: DatasetVersion = {
          id: nextId,
          createdAt: new Date().toISOString(),
        }
        const list = [...versions, next]
        setVersions(list)

        // Update dataset version/lastRun
        try {
          const stored = localStorage.getItem('lf_datasets')
          const arr = stored ? (JSON.parse(stored) as Dataset[]) : []
          const updated = arr.map(d =>
            d.id === datasetId
              ? {
                  ...d,
                  version: nextId,
                  lastRun: new Date().toISOString(),
                }
              : d
          )
          localStorage.setItem('lf_datasets', JSON.stringify(updated))
          setDataset(updated.find(d => d.id === datasetId) || null)
        } catch {}
      }
    } else if (taskStatus.state === 'FAILURE') {
      // Task failed
      setCurrentTaskId(null)
      const errorMessage = taskStatus.error || 'Unknown error occurred'
      toast({
        message: `Dataset reprocessing failed: ${errorMessage}`,
        variant: 'destructive',
      })
    }
  }, [taskStatus?.state, taskStatus?.error, datasetId, versions, toast])

  const openEdit = () => {
    setEditName(dataset?.name ?? '')
    setEditDescription(dataset?.description ?? '')
    setIsEditOpen(true)
  }

  const handleSaveEdit = () => {
    if (!dataset || !datasetId) return
    // TODO: Replace with API call to update dataset
    const updatedDataset = {
      ...dataset,
      name: editName.trim() || dataset.name,
      description: editDescription,
    }
    setDataset(updatedDataset)
    setIsEditOpen(false)
    console.warn('Dataset edit saved locally - should use API instead')
  }

  const handleDeleteClick = () => {
    setShowDeleteConfirmation(true)
  }

  const handleConfirmDelete = async () => {
    if (!activeProject?.namespace || !activeProject?.project || !datasetId)
      return

    try {
      await deleteDatasetMutation.mutateAsync({
        namespace: activeProject.namespace,
        project: activeProject.project,
        dataset: datasetId,
      })

      setShowDeleteConfirmation(false)
      setIsEditOpen(false)

      toast({
        message: 'Dataset deleted successfully',
        variant: 'default',
      })

      // Navigate away since the dataset no longer exists
      navigate('/chat/data')
    } catch (error) {
      console.error('Dataset deletion failed:', error)
      toast({
        message: 'Failed to delete dataset. Please try again.',
        variant: 'destructive',
      })
    }
  }

  const handleCancelDelete = () => {
    setShowDeleteConfirmation(false)
  }

  const handleDeleteFile = (fileHash: string) => {
    if (!activeProject?.namespace || !activeProject?.project || !datasetId)
      return
    setPendingDeleteFileHash(fileHash)
    setShowDeleteFileConfirmation(true)
  }

  const handleConfirmDeleteFile = async () => {
    if (
      !activeProject?.namespace ||
      !activeProject?.project ||
      !datasetId ||
      !pendingDeleteFileHash
    ) {
      setShowDeleteFileConfirmation(false)
      setPendingDeleteFileHash(null)
      return
    }

    try {
      await deleteFileMutation.mutateAsync({
        namespace: activeProject.namespace,
        project: activeProject.project,
        dataset: datasetId,
        fileHash: pendingDeleteFileHash,
        removeFromDisk: true,
      })

      toast({
        message: 'File deleted successfully',
        variant: 'default',
      })
    } catch (error) {
      console.error('File deletion failed:', error)
      toast({
        message: 'Failed to delete file. Please try again.',
        variant: 'destructive',
      })
    } finally {
      setShowDeleteFileConfirmation(false)
      setPendingDeleteFileHash(null)
    }
  }

  const handleCancelDeleteFile = () => {
    setShowDeleteFileConfirmation(false)
    setPendingDeleteFileHash(null)
  }

  // Helper function to check if a specific file is being deleted
  const isFileDeleting = (fileHash: string) => {
    return (
      deleteFileMutation.isPending &&
      deleteFileMutation.variables?.fileHash === fileHash
    )
  }

  // Derived RAG strategy info for this dataset
  const strategyName = useMemo(() => {
    if (!datasetId) return 'PDF Simple'
    try {
      return (
        localStorage.getItem(`lf_dataset_strategy_name_${datasetId}`) ||
        'PDF Simple'
      )
    } catch {
      return 'PDF Simple'
    }
  }, [datasetId])

  const slugify = (v: string) =>
    v
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')

  const strategyId = useMemo(() => slugify(strategyName), [strategyName])

  const docType = useMemo(() => {
    const id = strategyId
    if (id.includes('pdf')) return 'PDFs'
    if (id.includes('csv')) return 'CSVs'
    if (id.includes('chat')) return 'Conversations'
    if (id.includes('json')) return 'JSON'
    return 'Documents'
  }, [strategyId])

  const embeddingModel = useMemo(() => {
    try {
      return (
        localStorage.getItem(`lf_strategy_embedding_model_${strategyId}`) ||
        'text-embedding-3-large'
      )
    } catch {
      return 'text-embedding-3-large'
    }
  }, [strategyId])

  return (
    <div className="h-full w-full flex flex-col gap-3 pb-40">
      <nav className="text-sm md:text-base flex items-center gap-1.5 mb-3">
        <button
          className="text-teal-600 dark:text-teal-400 hover:underline"
          onClick={() => navigate('/chat/data')}
        >
          Data
        </button>
        <span className="text-muted-foreground px-1">\</span>
        <span className="text-foreground">{datasetName}</span>
      </nav>

      {/* Header row */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <h2 className="text-xl md:text-2xl font-medium">{datasetName}</h2>
              <button
                className="p-1 rounded-md hover:bg-accent text-muted-foreground"
                onClick={openEdit}
                aria-label="Edit dataset"
                title="Edit dataset"
              >
                <FontIcon type="edit" className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-muted-foreground max-w-[640px]">
              {dataset?.description && dataset.description.trim().length > 0
                ? dataset.description
                : 'Add a short description so teammates know what this dataset is for.'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" size="sm" className="rounded-xl">
              {dataset?.numChunks?.toLocaleString?.() || '—'} chunks •{' '}
              {dataset?.processedPercent ?? 0}% processed
            </Badge>
            <Button
              size="sm"
              onClick={async () => {
                if (
                  !datasetId ||
                  !activeProject?.namespace ||
                  !activeProject?.project
                )
                  return

                try {
                  const result = await reIngestMutation.mutateAsync({
                    namespace: activeProject.namespace,
                    project: activeProject.project,
                    dataset: datasetId,
                  })

                  // Extract task ID from task_uri (e.g., "http://localhost:8000/v1/projects/ns/proj/tasks/abc-123" -> "abc-123")
                  const taskId = result.task_uri.split('/').pop()
                  if (taskId) {
                    setCurrentTaskId(taskId)
                    toast({
                      message: 'Dataset reprocessing started...',
                      variant: 'default',
                    })
                  }
                } catch (error) {
                  console.error('Failed to start reprocessing:', error)
                  toast({
                    message: 'Failed to start reprocessing. Please try again.',
                    variant: 'destructive',
                  })
                }
              }}
              disabled={reIngestMutation.isPending || !!currentTaskId}
            >
              {reIngestMutation.isPending
                ? 'Starting...'
                : currentTaskId && taskStatus?.state === 'PENDING'
                  ? 'Processing...'
                  : 'Reprocess'}
            </Button>
          </div>
        </div>
      </div>

      {/* RAG Strategy card */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">RAG strategy</h3>
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(`/chat/rag/${strategyId}`)}
          >
            Configure
          </Button>
        </div>
        <div className="flex items-center gap-2 flex-wrap mb-2">
          <Badge variant="default" size="sm" className="rounded-xl">
            {strategyName}
          </Badge>
          <Badge variant="secondary" size="sm" className="rounded-xl">
            Last processed{' '}
            {dataset?.lastRun
              ? new Date(dataset.lastRun).toLocaleString()
              : '—'}
          </Badge>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Document type</div>
            <Input value={docType} readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Search</div>
            <Input value="Hybrid" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Embedding model</div>
            <Input value={embeddingModel} readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Chunks</div>
            <Input
              value={dataset?.numChunks?.toLocaleString?.() || '—'}
              readOnly
              className="bg-background"
            />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Results</div>
            <Input value="8" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Avg query time</div>
            <Input value="180ms" readOnly className="bg-background" />
          </div>
        </div>
      </section>

      <Dialog
        open={isEditOpen}
        onOpenChange={open => {
          setIsEditOpen(open)
          if (!open) {
            // Reset confirmation state when modal closes
            setShowDeleteConfirmation(false)
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {showDeleteConfirmation ? 'Delete Dataset' : 'Edit dataset'}
            </DialogTitle>
          </DialogHeader>

          {showDeleteConfirmation ? (
            <div className="space-y-4">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-red-600">
                  Confirm Deletion
                </h3>
                <p className="mt-2 text-sm text-muted-foreground">
                  Are you sure you want to delete the dataset "{datasetName}"?
                </p>
                <p className="mt-1 text-xs text-red-500">
                  This action cannot be undone. All files and data will be
                  permanently deleted.
                </p>
              </div>

              <div className="flex gap-3 justify-center">
                <Button
                  variant="secondary"
                  onClick={handleCancelDelete}
                  disabled={deleteDatasetMutation.isPending}
                >
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  onClick={handleConfirmDelete}
                  disabled={deleteDatasetMutation.isPending}
                >
                  {deleteDatasetMutation.isPending ? (
                    <>
                      <span className="inline-block animate-spin mr-2">⟳</span>
                      Deleting...
                    </>
                  ) : (
                    'Delete Dataset'
                  )}
                </Button>
              </div>
            </div>
          ) : (
            <>
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
              <div className="mt-4 flex flex-col-reverse gap-2 sm:flex-row sm:items-center sm:justify-between">
                <Button variant="destructive" onClick={handleDeleteClick}>
                  Delete
                </Button>
                <div className="flex items-center gap-2">
                  <DialogClose asChild>
                    <Button variant="secondary">Cancel</Button>
                  </DialogClose>
                  <Button onClick={handleSaveEdit} disabled={!editName.trim()}>
                    Save
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* File deletion confirmation modal */}
      {showDeleteFileConfirmation && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-background p-6 rounded-lg shadow-lg max-w-md w-full mx-4 border border-border">
            <h3 className="text-lg font-semibold text-red-600 mb-2">
              Delete File
            </h3>
            <p className="text-muted-foreground mb-4">
              Are you sure you want to delete this file?
            </p>
            <p className="text-sm text-muted-foreground mb-6 font-mono bg-muted p-2 rounded">
              {pendingDeleteFileHash?.substring(0, 20)}...
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={handleCancelDeleteFile}
                className="px-4 py-2 border border-border rounded hover:bg-muted"
                disabled={deleteFileMutation.isPending}
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDeleteFile}
                disabled={deleteFileMutation.isPending}
                className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 disabled:opacity-50"
              >
                {deleteFileMutation.isPending ? 'Deleting...' : 'Delete File'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Processing strategy */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">Processing strategy</h3>
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm">
              Change
            </Button>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Parsing</div>
            <Input value="PDF-aware" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Chunk size</div>
            <Input value="800" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Overlap</div>
            <Input value="100" readOnly className="bg-background" />
          </div>
        </div>
      </section>

      {/* Embedding model */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">Embedding model</h3>
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm">
              Change
            </Button>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Badge variant="default" size="sm" className="rounded-xl">
            text-embedding-3-large
          </Badge>
          <Badge variant="secondary" size="sm" className="rounded-xl">
            Active
          </Badge>
          <Badge variant="secondary" size="sm" className="rounded-xl">
            Run v3
          </Badge>
        </div>
        <div className="mt-2 text-xs">
          <a className="text-primary hover:underline" href="#">
            OpenAI API (source here)
          </a>
        </div>
      </section>

      {/* Raw data */}
      <section className="rounded-lg border border-border bg-card p-4 mb-40">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">Raw data</h3>
          <Button size="sm" onClick={() => fileInputRef.current?.click()}>
            Upload data
          </Button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          multiple
          onChange={async e => {
            if (
              !datasetId ||
              !activeProject?.namespace ||
              !activeProject?.project
            ) {
              toast({
                message: 'Missing required information for upload',
                variant: 'destructive',
              })
              return
            }

            const list = e.target.files ? Array.from(e.target.files) : []
            if (list.length === 0) return

            // Convert to RawFile shape for UI
            const converted: RawFile[] = list.map(f => ({
              id: `${f.name}:${f.size}:${f.lastModified}`,
              name: f.name,
              size: f.size,
              lastModified: f.lastModified,
              type: f.type,
            }))

            try {
              // Set processing status for UI feedback
              setUploadStatus(prev => ({
                ...prev,
                ...Object.fromEntries(
                  converted.map(rf => [rf.id, 'processing' as const])
                ),
              }))

              // Upload each file to the API
              for (const file of list) {
                try {
                  await uploadMutation.mutateAsync({
                    namespace: activeProject.namespace,
                    project: activeProject.project,
                    dataset: datasetId,
                    file,
                  })
                } catch (error) {
                  console.error(`Failed to upload ${file.name}:`, error)
                  toast({
                    message: `Failed to upload ${file.name}`,
                    variant: 'destructive',
                  })
                  // Remove from processing status on error
                  const fileId = `${file.name}:${file.size}:${file.lastModified}`
                  setUploadStatus(prev => {
                    const next = { ...prev }
                    delete (next as any)[fileId]
                    return next
                  })
                  continue
                }
              }

              // Files will be updated via API refresh after upload completes

              // Show success status
              setTimeout(() => {
                setUploadStatus(prev => ({
                  ...prev,
                  ...Object.fromEntries(
                    converted.map(rf => [rf.id, 'done' as const])
                  ),
                }))
                toast({
                  message: `Successfully uploaded ${list.length} file${list.length > 1 ? 's' : ''}`,
                  variant: 'default',
                })

                // Clear status after delay
                setTimeout(() => {
                  setUploadStatus(prev => {
                    const next = { ...prev }
                    for (const rf of converted) delete (next as any)[rf.id]
                    return next
                  })
                }, 1500)
              }, 500)
            } catch (error) {
              console.error('Upload error:', error)
              toast({
                message: 'Failed to upload files',
                variant: 'destructive',
              })
              // Clear processing status on error
              setUploadStatus(prev => {
                const next = { ...prev }
                for (const rf of converted) delete (next as any)[rf.id]
                return next
              })
            }

            // Reset input so same files can be picked again
            e.currentTarget.value = ''
          }}
        />
        <div className="flex items-center gap-2 mb-2">
          <div className="w-1/2">
            <SearchInput
              placeholder="Search raw files"
              value={searchValue}
              onChange={e => setSearchValue(e.target.value)}
            />
          </div>
        </div>
        <div className="rounded-md border border-input bg-background p-0 text-xs">
          {files.length === 0 ? (
            <div className="p-3 text-muted-foreground">
              No files assigned yet.
              {/* Debug info */}
              {process.env.NODE_ENV === 'development' && (
                <div className="mt-2 text-xs">
                  <div>
                    API Dataset: {currentApiDataset ? 'Found' : 'Not found'}
                  </div>
                  <div>API Files: {currentApiDataset?.files?.length || 0}</div>
                </div>
              )}
            </div>
          ) : (
            <div>
              <div className="p-3 border-b border-border/60 bg-muted/20">
                <div className="text-xs font-medium">
                  {files.length} file{files.length !== 1 ? 's' : ''}
                </div>
              </div>
              <ul>
                {files
                  .filter(f =>
                    f.name.toLowerCase().includes(searchValue.toLowerCase())
                  )
                  .map(f => (
                    <li
                      key={f.id}
                      className="flex items-center justify-between px-3 py-3 border-b last:border-b-0 border-border/60"
                    >
                      <div className="font-mono text-xs text-muted-foreground truncate max-w-[60%] flex flex-col gap-1">
                        <span>{f.fullHash ? f.name : f.name}</span>
                        {f.fullHash && (
                          <>
                            <button
                              onClick={async () => {
                                try {
                                  await navigator.clipboard.writeText(
                                    f.fullHash!
                                  )
                                  setCopyStatus(prev => ({
                                    ...prev,
                                    [f.id]: 'Copied!',
                                  }))
                                } catch (err) {
                                  setCopyStatus(prev => ({
                                    ...prev,
                                    [f.id]: 'Failed to copy',
                                  }))
                                }
                                setTimeout(() => {
                                  setCopyStatus(prev => ({
                                    ...prev,
                                    [f.id]: undefined,
                                  }))
                                }, 1500)
                              }}
                              className="text-xs text-blue-600 hover:text-blue-800 text-left"
                              title="Click to copy full hash"
                            >
                              Copy full hash
                            </button>
                            {copyStatus?.[f.id] && (
                              <span
                                className={`ml-2 text-xs ${copyStatus[f.id] === 'Copied!' ? 'text-green-600' : 'text-red-600'}`}
                              >
                                {copyStatus[f.id]}
                              </span>
                            )}
                          </>
                        )}
                      </div>
                      <div className="w-1/2 flex items-center justify-between gap-4">
                        <div className="text-xs text-muted-foreground">
                          {f.size === 'unknown' || f.fullHash
                            ? 'N/A'
                            : `${Math.ceil(f.size / 1024)} KB`}
                        </div>
                        <div className="flex items-center gap-6">
                          {uploadStatus[f.id] === 'processing' && (
                            <div className="flex items-center gap-1 text-muted-foreground">
                              <FontIcon type="fade" className="w-4 h-4" />
                              <span className="text-xs">Processing</span>
                            </div>
                          )}
                          {uploadStatus[f.id] === 'done' && (
                            <FontIcon
                              type="checkmark-outline"
                              className="w-4 h-4 text-teal-600 dark:text-teal-400"
                            />
                          )}
                          <button
                            className="w-4 h-4 grid place-items-center text-muted-foreground hover:text-red-600 disabled:opacity-50"
                            onClick={() =>
                              f.fullHash && handleDeleteFile(f.fullHash)
                            }
                            disabled={
                              f.fullHash ? isFileDeleting(f.fullHash) : true
                            }
                            aria-label={`Delete ${f.name} from dataset`}
                            title="Delete file"
                          >
                            {f.fullHash && isFileDeleting(f.fullHash) ? (
                              <span className="text-xs">...</span>
                            ) : (
                              <FontIcon type="trashcan" className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                      </div>
                    </li>
                  ))}
              </ul>
            </div>
          )}
        </div>
      </section>

      {/* File deletion now handled directly via API calls with confirmation dialog */}
    </div>
  )
}

export default DatasetView
