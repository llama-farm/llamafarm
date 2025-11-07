import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip'
import SearchInput from '../ui/search-input'
import { useModeWithReset } from '../../hooks/useModeWithReset'
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
import { useProject } from '../../hooks/useProjects'
import {
  useUploadFileToDataset,
  useProcessDataset,
  useTaskStatus,
  useListDatasets,
  useDeleteDatasetFile,
  useDeleteDataset,
} from '../../hooks/useDatasets'
import { DatasetFile } from '../../types/datasets'
import PageActions from '../common/PageActions'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import { useConfigPointer } from '../../hooks/useConfigPointer'
import type { ProjectConfig } from '../../types/config'

type Dataset = {
  id: string
  name: string
  lastRun: string | Date
  embedModel: string
  numChunks: number
  processedPercent: number
  version: string
  description?: string
  files?: string[] | DatasetFile[] // Flexible file format from API
}

function DatasetView() {
  const navigate = useNavigate()
  const { datasetId } = useParams()
  const { toast } = useToast()
  const [mode, setMode] = useModeWithReset('designer')

  // Get current active project for API calls
  const activeProject = useActiveProject()
  const { data: projectResp } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject
  )

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
  const [processingResult, setProcessingResult] = useState<any>(null)
  const processMutation = useProcessDataset()
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

  const projectConfig = (projectResp as any)?.project?.config as ProjectConfig | undefined
  const getDatasetLocation = useCallback(() => {
    const targetName = dataset?.name || datasetId
    if (targetName) {
      return { type: 'dataset' as const, datasetName: targetName }
    }
    return { type: 'datasets' as const }
  }, [dataset?.name, datasetId])
  const { configPointer, handleModeChange } = useConfigPointer({
    mode,
    setMode,
    config: projectConfig,
    getLocation: getDatasetLocation,
  })

  // Get current dataset from API response
  const currentApiDataset = useMemo(() => {
    if (!datasetsResponse?.datasets || !datasetId) return null
    return datasetsResponse.datasets.find(d => d.name === datasetId)
  }, [datasetsResponse, datasetId])

  // Files from API data only
  const files = useMemo(() => {
    const filesSource: any[] = Array.isArray(
      (currentApiDataset as any)?.details?.files_metadata
    )
      ? (currentApiDataset as any).details.files_metadata
      : (currentApiDataset as any)?.files || []
    if (!filesSource) return []
    return filesSource.map((fileObj: any) => {
      // Handle new API response format with file details
      if (
        typeof fileObj === 'object' &&
        fileObj !== null &&
        ('original_filename' in fileObj || 'original_file_name' in fileObj)
      ) {
        return {
          id: fileObj.hash,
          name: fileObj.original_filename || fileObj.original_file_name,
          size: fileObj.size,
          lastModified: new Date(fileObj.timestamp * 1000).toLocaleString(),
          type: fileObj.mime_type,
          fullHash: fileObj.hash, // Store full hash for operations
        }
      }

      // Fallback for legacy format (file hash strings)
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
  const directoryInputRef = useRef<HTMLInputElement>(null)

  // Set directory attributes on mount
  useEffect(() => {
    const input = directoryInputRef.current
    if (input) {
      input.setAttribute('webkitdirectory', '')
      input.setAttribute('directory', '')
      input.setAttribute('mozdirectory', '')
    }
  }, [])
  type FileUploadStatus = {
    id: string
    name: string
    status: 'pending' | 'uploading' | 'success' | 'error'
    error?: string
  }
  const [fileUploadStatuses, setFileUploadStatuses] = useState<
    FileUploadStatus[]
  >([])

  // File type filtering - flexible input for any file extension
  const [fileTypeFilter, setFileTypeFilter] = useState<string>('')
  const [showFileTypeFilter, setShowFileTypeFilter] = useState(false)
  const [uploadStats, setUploadStats] = useState<{
    total: number
    filtered: number
    uploading: number
  } | null>(null)

  // Common file type suggestions (non-exhaustive)
  const commonFileTypes = [
    '.pdf',
    '.txt',
    '.md',
    '.csv',
    '.json',
    '.xml',
    '.html',
    '.docx',
    '.xlsx',
    '.pptx',
  ]

  const addFileTypeToFilter = (ext: string) => {
    const current = fileTypeFilter
      .split(',')
      .map(s => s.trim())
      .filter(Boolean)

    if (!current.includes(ext)) {
      const newFilter = [...current, ext].join(', ')
      setFileTypeFilter(newFilter)
    }
  }

  const clearFileTypeFilter = () => {
    setFileTypeFilter('')
  }

  const filterFilesByType = (files: File[]): File[] => {
    if (!fileTypeFilter.trim()) {
      // If no filter specified, accept all files
      return files
    }

    // Parse extensions from comma-separated input
    const extensions = fileTypeFilter
      .split(',')
      .map(ext => ext.trim().toLowerCase())
      .filter(Boolean)
      .map(ext => (ext.startsWith('.') ? ext : `.${ext}`)) // Ensure leading dot

    if (extensions.length === 0) {
      return files
    }

    return files.filter(file => {
      const fileName = file.name.toLowerCase()
      return extensions.some(ext => fileName.endsWith(ext))
    })
  }

  // Drag-and-drop state
  const [isDragging, setIsDragging] = useState(false)
  const [isDropped, setIsDropped] = useState(false)

  // Note: Custom strategies now come from API via project config, not localStorage

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleFilesUpload = async (list: File[]) => {
    if (!datasetId || !activeProject?.namespace || !activeProject?.project) {
      toast({
        message: 'Missing required information for upload',
        variant: 'destructive',
      })
      return
    }
    if (list.length === 0) return

    // Apply file type filtering
    const totalFiles = list.length
    const filteredFiles = filterFilesByType(list)
    const filteredCount = filteredFiles.length

    // Show stats if filtering occurred
    if (totalFiles !== filteredCount) {
      setUploadStats({
        total: totalFiles,
        filtered: filteredCount,
        uploading: filteredCount,
      })
      toast({
        message: `Filtered ${filteredCount} of ${totalFiles} files based on file type filter`,
        variant: 'default',
      })
    }

    if (filteredFiles.length === 0) {
      console.error('❌ All files filtered out by file type filter!')
      toast({
        message: 'No files match the selected file type filter',
        variant: 'destructive',
      })
      return
    }

    // Note: Filename stripping for folder uploads is now handled in the API layer
    // (see datasets.ts uploadFileToDataset function)

    // Initialize upload statuses
    const initialStatuses: FileUploadStatus[] = filteredFiles.map(f => ({
      id: `${f.name}:${f.size}:${f.lastModified}`,
      name: f.name.split('/').pop() || f.name, // Display basename only
      status: 'pending',
    }))
    setFileUploadStatuses(initialStatuses)

    await Promise.all(
      filteredFiles.map(async (file, idx) => {
        setFileUploadStatuses(prev =>
          prev.map((s, i) => (i === idx ? { ...s, status: 'uploading' } : s))
        )
        try {
          await uploadMutation.mutateAsync({
            namespace: activeProject.namespace!,
            project: activeProject.project!,
            dataset: datasetId!,
            file,
          })
          setFileUploadStatuses(prev =>
            prev.map((s, i) => (i === idx ? { ...s, status: 'success' } : s))
          )
        } catch (error: any) {
          console.error(`Failed to upload ${file.name}:`, error)
          toast({
            message: `Failed to upload ${file.name}`,
            variant: 'destructive',
          })
          setFileUploadStatuses(prev =>
            prev.map((s, i) =>
              i === idx
                ? {
                    ...s,
                    status: 'error',
                    error: error?.message || 'Upload failed',
                  }
                : s
            )
          )
        }
      })
    )

    // Clear stats after upload completes
    setTimeout(() => setUploadStats(null), 5000)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDropped(true)
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files)
    handleFilesUpload(files).finally(() => setIsDropped(false))
  }

  // Load dataset metadata from API only - no hardcoded values
  useEffect(() => {
    if (!datasetId) return

    if (currentApiDataset) {
      // Use only real API data
      setDataset({
        id: datasetId,
        name: currentApiDataset.name,
        lastRun: '', // Not available from API
        embedModel: '', // Not available from API
        numChunks: 0, // Not available from API - will be in processing results
        processedPercent: 0, // Not available from API
        version: '', // Not available from API
        description: '',
        files: currentApiDataset.files,
      })
    } else {
      // Minimal fallback
      setDataset({
        id: datasetId,
        name: datasetId,
        lastRun: '',
        embedModel: '',
        numChunks: 0,
        processedPercent: 0,
        version: '',
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

      // Store the processing result
      if (taskStatus.result) {
        setProcessingResult(taskStatus.result)
      }

      toast({
        message: taskStatus.result?.processed_files
          ? `✅ Processing complete! ${taskStatus.result.processed_files} file(s) processed`
          : 'Dataset processing completed successfully',
        variant: 'default',
      })
    } else if (taskStatus.state === 'FAILURE') {
      // Task failed
      console.error('Task failed:', taskStatus.error, taskStatus.traceback)
      setCurrentTaskId(null)
      setProcessingResult(null)
      const errorMessage = taskStatus.error || 'Unknown error occurred'
      toast({
        message: `❌ Processing failed: ${errorMessage}`,
        variant: 'destructive',
      })
    }
  }, [taskStatus?.state, taskStatus?.error, taskStatus?.result, toast, currentTaskId])

  const openEdit = () => {
    setEditName(dataset?.name ?? '')
    setEditDescription(dataset?.description ?? '')
    setIsEditOpen(true)
  }

  const handleSaveEdit = () => {
    if (!dataset || !datasetId) return
    // Note: Dataset name/description updates are local-only until backend supports PATCH endpoint
    const updatedDataset = {
      ...dataset,
      name: editName.trim() || dataset.name,
      description: editDescription,
    }
    setDataset(updatedDataset)
    setIsEditOpen(false)
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

  // Processing strategy comes from API dataset response
  const currentStrategy =
    (currentApiDataset as any)?.data_processing_strategy ||
    'universal_processor'

  // Note: Parsers/extractors info now comes from project config API, not localStorage
  // This is a placeholder - should be fetched from the strategy config if needed
  const parsersSummary = 'Configured via strategy settings'
  const extractorsSummary = 'Configured via strategy settings'

  // Removed unused derived values

  return (
    <div
      className={`h-full w-full flex flex-col ${mode === 'designer' ? 'gap-3 pb-40' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {mode === 'designer' ? (
        <div className="flex items-center justify-between mb-3">
          <nav className="text-sm md:text-base flex items-center gap-1.5">
            <button
              className="text-teal-600 dark:text-teal-400 hover:underline"
              onClick={() => navigate('/chat/data')}
            >
              Data
            </button>
            <span className="text-muted-foreground px-1">\</span>
            <span className="text-foreground">{datasetName}</span>
          </nav>
          <PageActions mode={mode} onModeChange={handleModeChange} />
        </div>
      ) : (
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-2xl">Config editor</h2>
          <PageActions mode={mode} onModeChange={handleModeChange} />
        </div>
      )}

      {mode !== 'designer' ? (
        <div className="flex-1 min-h-0 overflow-hidden">
          <ConfigEditor className="h-full" initialPointer={configPointer} />
        </div>
      ) : (
        <>
          {/* Header row */}
          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-start justify-between gap-4">
              <div className="flex flex-col gap-1 flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h2 className="text-xl md:text-2xl font-medium">
                    {datasetName}
                  </h2>
                  <button
                    className="p-1 rounded-md hover:bg-accent text-muted-foreground"
                    onClick={openEdit}
                    aria-label="Edit dataset"
                    title="Edit dataset"
                  >
                    <FontIcon type="edit" className="w-4 h-4" />
                  </button>
                </div>
                {/* Status shown inline on mobile */}
                {currentTaskId && taskStatus && (
                  <div className="sm:hidden mt-1 mb-2">
                    <Badge
                      variant="secondary"
                      size="sm"
                      className="rounded-xl w-max flex items-center gap-1.5"
                    >
                      {taskStatus.state === 'PENDING' && (
                        <>
                          <span className="inline-block animate-spin" aria-label="Loading" role="status">⟳</span>
                          Queued...
                        </>
                      )}
                      {taskStatus.state !== 'PENDING' &&
                        taskStatus.state !== 'SUCCESS' &&
                        taskStatus.state !== 'FAILURE' && (
                          <>
                            <span className="inline-block animate-spin" aria-label="Loading" role="status">⟳</span>
                            {taskStatus.meta?.progress
                              ? `${taskStatus.meta.progress}%`
                              : taskStatus.meta?.current && taskStatus.meta?.total
                                ? `${taskStatus.meta.current}/${taskStatus.meta.total}`
                                : 'Processing...'}
                          </>
                        )}
                    </Badge>
                  </div>
                )}
                <p className="text-xs text-muted-foreground max-w-[640px]">
                  {dataset?.description && dataset.description.trim().length > 0
                    ? dataset.description
                    : 'Add a short description so teammates know what this dataset is for.'}
                </p>
              </div>
              <div className="flex items-center gap-2">
                {/* Processing status badge */}
                {currentTaskId && taskStatus && (
                  <div className="flex items-center gap-2">
                    <Badge
                      variant="secondary"
                      size="sm"
                      className="rounded-xl w-max flex items-center gap-1.5"
                    >
                      {taskStatus.state === 'PENDING' && (
                        <>
                          <span className="inline-block animate-spin" aria-label="Loading" role="status">⟳</span>
                          Queued...
                        </>
                      )}
                      {taskStatus.state !== 'PENDING' &&
                        taskStatus.state !== 'SUCCESS' &&
                        taskStatus.state !== 'FAILURE' && (
                        <>
                          <span className="inline-block animate-spin" aria-label="Loading" role="status">⟳</span>
                          {taskStatus.meta?.progress
                            ? `Processing ${taskStatus.meta.progress}%`
                            : taskStatus.meta?.current && taskStatus.meta?.total
                              ? `Processing ${taskStatus.meta.current}/${taskStatus.meta.total}`
                              : 'Processing...'}
                        </>
                      )}
                    </Badge>
                    {taskStatus.meta?.message && (
                      <span className="text-xs text-muted-foreground max-w-[200px] truncate hidden sm:inline">
                        {taskStatus.meta.message}
                      </span>
                    )}
                  </div>
                )}
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
                      // Clear previous results
                      setProcessingResult(null)

                      const result = await processMutation.mutateAsync({
                        namespace: activeProject.namespace,
                        project: activeProject.project,
                        dataset: datasetId,
                      })

                      // The process endpoint returns task_id directly
                      if (result.task_id) {
                        setCurrentTaskId(result.task_id)
                        toast({
                          message: 'Dataset processing started...',
                          variant: 'default',
                        })
                      }
                    } catch (error) {
                      console.error('Failed to start processing:', error)
                      toast({
                        message:
                          'Failed to start processing. Please try again.',
                        variant: 'destructive',
                      })
                    }
                  }}
                  disabled={processMutation.isPending || !!currentTaskId}
                >
                  {processMutation.isPending
                    ? 'Starting...'
                    : currentTaskId && taskStatus
                      ? 'Processing...'
                      : 'Process Dataset'}
                </Button>
              </div>
            </div>
          </div>

          {/* Connected database card */}
          {(currentApiDataset as any)?.database && (
            <section className="rounded-lg border border-border bg-card p-4">
              <div className="flex items-center justify-between">
                <div className="flex flex-col gap-1">
                  <h3 className="text-sm font-medium">Connected database</h3>
                  <p className="text-xs text-muted-foreground">
                    This dataset is connected to the following database
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Badge
                    variant="default"
                    size="sm"
                    className="rounded-xl bg-teal-600 text-white dark:bg-teal-500 dark:text-slate-900"
                  >
                    {(currentApiDataset as any).database}
                  </Badge>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => navigate('/chat/databases')}
                  >
                    Go to database
                  </Button>
                </div>
              </div>
            </section>
          )}

          {/* Processing Strategy card */}
          <section className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium">Processing strategy</h3>
            </div>
            <div className="flex items-center gap-2 flex-wrap mb-2">
              <Badge variant="default" size="sm" className="rounded-xl">
                {currentStrategy}
              </Badge>
              {processingResult && (
                <Badge variant="secondary" size="sm" className="rounded-xl">
                  Last processed: {processingResult.processed_files || 0} files
                </Badge>
              )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <div className="text-xs text-muted-foreground">Parsers</div>
                <Input
                  value={parsersSummary}
                  readOnly
                  className="bg-background"
                />
              </div>
              <div className="flex flex-col gap-1">
                <div className="text-xs text-muted-foreground">Extractors</div>
                <Input
                  value={extractorsSummary}
                  readOnly
                  className="bg-background"
                />
              </div>
            </div>
          </section>

          {/* Processing Results */}
          {processingResult && (
            <section className="rounded-lg border border-border bg-card p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium">Processing Results</h3>
                <button
                  onClick={() => setProcessingResult(null)}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  Clear
                </button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                <div className="rounded-md border border-border p-3">
                  <div className="text-xs text-muted-foreground mb-1">
                    Processed Files
                  </div>
                  <div className="text-2xl font-semibold text-green-600">
                    {processingResult.processed_files || 0}
                  </div>
                </div>
                <div className="rounded-md border border-border p-3">
                  <div className="text-xs text-muted-foreground mb-1">
                    Skipped Files
                  </div>
                  <div className="text-2xl font-semibold text-yellow-600">
                    {processingResult.skipped_files || 0}
                  </div>
                </div>
                <div className="rounded-md border border-border p-3">
                  <div className="text-xs text-muted-foreground mb-1">
                    Failed Files
                  </div>
                  <div className="text-2xl font-semibold text-red-600">
                    {processingResult.failed_files || 0}
                  </div>
                </div>
              </div>
              {processingResult.details &&
                processingResult.details.length > 0 && (
                  <div className="mt-3">
                    <div className="text-xs font-medium text-muted-foreground mb-2">
                      File Processing Details
                    </div>
                    <div className="rounded-md border border-border max-h-96 overflow-auto">
                      {processingResult.details.map(
                        (fileResult: any, idx: number) => {
                          const details = fileResult.details || {}
                          const result = details.result || {}
                          const isSkipped = result.status === 'skipped' || details.status === 'skipped'
                          const isFailed = !fileResult.success
                          const isSuccess = fileResult.success && !isSkipped

                          // Get filename from result (actual name) or fall back to hash
                          const displayFilename = result.filename || details.filename || fileResult.file_hash
                          const isHashFilename = displayFilename === fileResult.file_hash || !result.filename

                          // Get file extension for icon
                          const getFileExtension = (filename: string) => {
                            const parts = filename.split('.')
                            return parts.length > 1 ? parts[parts.length - 1].toLowerCase() : ''
                          }
                          const fileExt = getFileExtension(displayFilename)
                          const fileIcon = ['pdf', 'doc', 'docx', 'txt', 'md'].includes(fileExt) ? '📄' : '📁'

                          // Calculate total chunks if available
                          const totalChunks = result.document_count || details.chunks || 0
                          const storedChunks = result.stored_count || 0
                          const skippedChunks = result.skipped_count || 0

                          return (
                            <div
                              key={idx}
                              className="p-4 border-b last:border-b-0 hover:bg-muted/30 transition-colors"
                            >
                              {/* File header with status */}
                              <div className="flex items-start justify-between gap-3 mb-2">
                                <div className="flex items-center gap-2.5 flex-1 min-w-0">
                                  {/* File type icon */}
                                  <span className="text-xl flex-shrink-0">{fileIcon}</span>

                                  {/* Status icon */}
                                  {isSuccess && (
                                    <FontIcon
                                      type="checkmark-filled"
                                      className="w-5 h-5 text-green-600 flex-shrink-0"
                                    />
                                  )}
                                  {isSkipped && (
                                    <div className="w-5 h-5 rounded-full bg-yellow-500 flex items-center justify-center flex-shrink-0">
                                      <span className="text-white text-xs font-bold">!</span>
                                    </div>
                                  )}
                                  {isFailed && (
                                    <FontIcon
                                      type="close"
                                      className="w-5 h-5 text-red-600 flex-shrink-0"
                                    />
                                  )}

                                  {/* Filename */}
                                  <div className="flex flex-col flex-1 min-w-0">
                                    <span className="text-sm font-medium truncate">
                                      {displayFilename}
                                    </span>
                                    {isHashFilename && fileResult.file_hash && (
                                      <TooltipProvider>
                                        <Tooltip>
                                          <TooltipTrigger asChild>
                                            <span className="text-xs text-muted-foreground font-mono cursor-pointer">
                                              Hash: {fileResult.file_hash.substring(0, 12)}...
                                            </span>
                                          </TooltipTrigger>
                                          <TooltipContent>
                                            <p className="font-mono text-xs">{fileResult.file_hash}</p>
                                          </TooltipContent>
                                        </Tooltip>
                                      </TooltipProvider>
                                    )}
                                  </div>
                                </div>

                                {/* Status badge */}
                                <Badge
                                  variant={
                                    isSuccess
                                      ? 'default'
                                      : isSkipped
                                        ? 'secondary'
                                        : 'outline'
                                  }
                                  size="sm"
                                  className="rounded-xl flex-shrink-0 font-medium"
                                >
                                  {isSuccess && '✓ SUCCESS'}
                                  {isSkipped && `⚠️ SKIPPED${result.reason ? ` (${result.reason})` : ''}`}
                                  {isFailed && '✗ FAILED'}
                                </Badge>
                              </div>

                              {/* Processing stats */}
                              <div className="ml-9 space-y-2 text-xs">
                                {/* Chunks info - enhanced display */}
                                {totalChunks > 0 && (
                                  <div className="flex items-center gap-3 bg-muted/40 rounded px-3 py-2">
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-base">📦</span>
                                      <span className="font-semibold text-foreground">
                                        {totalChunks}
                                      </span>
                                      <span className="text-muted-foreground">
                                        chunk{totalChunks !== 1 ? 's' : ''} created
                                      </span>
                                    </div>
                                    {storedChunks > 0 && (
                                      <div className="flex items-center gap-1 text-green-600 dark:text-green-500">
                                        <span className="font-semibold">✓ {storedChunks}</span>
                                        <span>stored</span>
                                      </div>
                                    )}
                                    {skippedChunks > 0 && (
                                      <div className="flex items-center gap-1 text-yellow-600 dark:text-yellow-500">
                                        <span className="font-semibold">⊘ {skippedChunks}</span>
                                        <span>skipped</span>
                                      </div>
                                    )}
                                  </div>
                                )}

                                {/* Processing details grid */}
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
                                  {/* Parser info */}
                                  {(details.parser || result.parsers_used?.length > 0) && (
                                    <div className="text-muted-foreground">
                                      <span className="font-medium text-foreground">Parser:</span>{' '}
                                      <span className="font-mono text-xs">
                                        {result.parsers_used?.join(', ') || details.parser}
                                      </span>
                                    </div>
                                  )}

                                  {/* Embedder */}
                                  {(details.embedder || result.embedder) && (
                                    <div className="text-muted-foreground">
                                      <span className="font-medium text-foreground">Embedder:</span>{' '}
                                      <span className="font-mono text-xs">
                                        {result.embedder || details.embedder}
                                      </span>
                                    </div>
                                  )}
                                </div>

                                {/* Extractors - full width */}
                                {(details.extractors?.length > 0 || result.extractors_applied?.length > 0) && (
                                  <div className="text-muted-foreground">
                                    <span className="font-medium text-foreground">Extractors:</span>{' '}
                                    <div className="inline-flex flex-wrap gap-1 mt-1">
                                      {(result.extractors_applied || details.extractors || []).map((ext: string, i: number) => (
                                        <Badge
                                          key={i}
                                          variant="outline"
                                          size="sm"
                                          className="rounded font-mono text-xs"
                                        >
                                          {ext}
                                        </Badge>
                                      ))}
                                    </div>
                                  </div>
                                )}

                                {/* Document IDs if stored */}
                                {result.document_ids && result.document_ids.length > 0 && (
                                  <div className="text-muted-foreground pt-1">
                                    <span className="font-medium text-foreground">Document IDs:</span>{' '}
                                    <span className="font-mono text-xs">
                                      {result.document_ids.length} stored in vector database
                                    </span>
                                  </div>
                                )}

                                {/* Reason for skip/failure - highlighted */}
                                {(result.reason || details.reason) && (
                                  <div className={`mt-2 p-2 rounded ${
                                    isSkipped
                                      ? 'bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-800'
                                      : 'bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800'
                                  }`}>
                                    <span className={`font-medium ${isSkipped ? 'text-yellow-800 dark:text-yellow-300' : 'text-red-800 dark:text-red-300'}`}>
                                      Reason:
                                    </span>{' '}
                                    <span className={isSkipped ? 'text-yellow-700 dark:text-yellow-400' : 'text-red-700 dark:text-red-400'}>
                                      {result.reason || details.reason}
                                    </span>
                                  </div>
                                )}

                                {/* Error message for failures */}
                                {isFailed && (fileResult.error || details.error) && (
                                  <div className="mt-2 p-3 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded">
                                    <span className="font-medium text-red-800 dark:text-red-300">Error:</span>{' '}
                                    <span className="text-red-700 dark:text-red-400">
                                      {fileResult.error || details.error}
                                    </span>
                                  </div>
                                )}
                              </div>
                            </div>
                          )
                        }
                      )}
                    </div>
                  </div>
                )}
            </section>
          )}

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
                      Are you sure you want to delete the dataset "{datasetName}
                      "?
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
                          <span className="inline-block animate-spin mr-2">
                            ⟳
                          </span>
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
                      <label className="text-xs text-muted-foreground">
                        Name
                      </label>
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
                      <Button
                        onClick={handleSaveEdit}
                        disabled={!editName.trim()}
                      >
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
                    {deleteFileMutation.isPending
                      ? 'Deleting...'
                      : 'Delete File'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Live Processing Progress - shown while task is running */}
          {currentTaskId && taskStatus && taskStatus.meta?.files && (
            <section className="rounded-lg border border-border bg-card p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium">Processing Progress</h3>
                <Badge variant="secondary" size="sm" className="rounded-xl">
                  {taskStatus.meta.current || 0} / {taskStatus.meta.total || 0} files
                </Badge>
              </div>
              <div className="rounded-md border border-border max-h-80 overflow-auto">
                {taskStatus.meta.files.map((file: any, idx: number) => (
                  <div
                    key={file.task_id || idx}
                    className="p-3 border-b last:border-b-0 text-xs"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2 flex-1 min-w-0">
                        {/* Status icon */}
                        {file.state === 'pending' && (
                          <div className="w-4 h-4 rounded-full border-2 border-muted-foreground/30 flex-shrink-0" />
                        )}
                        {file.state === 'processing' && (
                          <div className="w-4 h-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin flex-shrink-0" />
                        )}
                        {file.state === 'success' && (
                          <FontIcon
                            type="checkmark-filled"
                            className="w-4 h-4 text-green-600 flex-shrink-0"
                          />
                        )}
                        {file.state === 'failure' && (
                          <FontIcon
                            type="close"
                            className="w-4 h-4 text-red-600 flex-shrink-0"
                          />
                        )}

                        {/* Filename */}
                        <span className="font-mono text-muted-foreground truncate">
                          {file.filename || `File ${idx + 1}`}
                        </span>
                      </div>

                      {/* Status badge */}
                      <Badge
                        variant={
                          file.state === 'success'
                            ? 'default'
                            : file.state === 'failure'
                              ? 'outline'
                              : 'secondary'
                        }
                        size="sm"
                        className="rounded-xl flex-shrink-0"
                      >
                        {file.state === 'pending' && 'Queued'}
                        {file.state === 'processing' && 'Processing'}
                        {file.state === 'success' && 'Complete'}
                        {file.state === 'failure' && 'Failed'}
                      </Badge>
                    </div>

                    {/* Additional info */}
                    {file.chunks && (
                      <div className="text-muted-foreground mt-1 ml-6">
                        {file.chunks} chunks created
                      </div>
                    )}
                    {file.error && (
                      <div className="text-red-600 mt-1 ml-6 text-xs">
                        Error: {file.error}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Processing strategy and Embedding model sections removed per request */}

          {isDragging && (
            <div className="w-full h-full flex flex-col items-center justify-center border border-dashed rounded-lg p-4 gap-2 transition-colors border-input">
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
                You can upload PDFs, CSVs, or other documents directly to this
                dataset.
              </p>
            </div>
          )}

          {/* Raw data */}
          <section className="rounded-lg border border-border bg-card p-4 mb-40">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium">Raw data</h3>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setShowFileTypeFilter(!showFileTypeFilter)}
                >
                  {showFileTypeFilter ? 'Hide' : 'Filter'}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    directoryInputRef.current?.click()
                  }}
                >
                  Upload folder
                </Button>
                <Button size="sm" onClick={() => fileInputRef.current?.click()}>
                  Upload files
                </Button>
              </div>
            </div>

            {/* File type filter section */}
            {showFileTypeFilter && (
              <div className="mb-3 p-3 bg-muted/20 rounded-md border border-border/60">
                <div className="flex flex-col gap-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-muted-foreground">
                      File type filter (comma-separated)
                    </label>
                    {fileTypeFilter && (
                      <button
                        onClick={clearFileTypeFilter}
                        className="text-xs text-blue-600 hover:text-blue-800"
                      >
                        Clear filter
                      </button>
                    )}
                  </div>
                  <Input
                    value={fileTypeFilter}
                    onChange={e => setFileTypeFilter(e.target.value)}
                    placeholder="e.g., .pdf, .txt, .docx (leave empty for all files)"
                    className="text-sm"
                  />
                  <div className="flex flex-wrap gap-1.5">
                    <span className="text-xs text-muted-foreground">
                      Quick add:
                    </span>
                    {commonFileTypes.map(ext => (
                      <button
                        key={ext}
                        onClick={() => addFileTypeToFilter(ext)}
                        className="text-xs px-2 py-0.5 rounded bg-background border border-border hover:bg-accent"
                      >
                        {ext}
                      </button>
                    ))}
                  </div>
                  {fileTypeFilter && (
                    <p className="text-xs text-muted-foreground">
                      Will only upload files matching:{' '}
                      <span className="font-mono font-medium">
                        {fileTypeFilter}
                      </span>
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Upload stats */}
            {uploadStats && (
              <div className="mb-3 p-2 bg-blue-50 dark:bg-blue-950/20 rounded border border-blue-200 dark:border-blue-800">
                <p className="text-xs text-blue-700 dark:text-blue-300">
                  Uploading {uploadStats.filtered} of {uploadStats.total} files
                  (filtered by file type)
                </p>
              </div>
            )}

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

                // Reset input immediately to allow re-selecting same files
                const target = e.currentTarget
                try {
                  await handleFilesUpload(list)
                } catch (error) {
                  console.error('File upload error:', error)
                  toast({
                    message: 'Failed to upload files. See console for details.',
                    variant: 'destructive',
                  })
                } finally {
                  // Reset input value after upload completes
                  if (target) {
                    target.value = ''
                  }
                }
              }}
            />
            <input
              ref={directoryInputRef}
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

                if (list.length === 0) {
                  toast({
                    message: 'No files selected',
                    variant: 'destructive',
                  })
                  return
                }

                // Store reference to avoid null issues after async operation
                const target = e.currentTarget
                try {
                  await handleFilesUpload(list)
                } catch (error) {
                  console.error('Folder upload error:', error)
                  toast({
                    message:
                      'Failed to upload folder. See console for details.',
                    variant: 'destructive',
                  })
                } finally {
                  // Reset input value after upload completes
                  if (target) {
                    target.value = ''
                  }
                }
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
                              {fileUploadStatuses.find(s => s.id === f.id)
                                ?.status === 'uploading' && (
                                <div className="flex items-center gap-1 text-muted-foreground">
                                  <FontIcon type="fade" className="w-4 h-4" />
                                  <span className="text-xs">Processing</span>
                                </div>
                              )}
                              {fileUploadStatuses.find(s => s.id === f.id)
                                ?.status === 'success' && (
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
                                  <FontIcon
                                    type="trashcan"
                                    className="w-4 h-4"
                                  />
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
        </>
      )}
    </div>
  )
}

export default DatasetView
